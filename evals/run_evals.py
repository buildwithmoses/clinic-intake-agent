"""Run the 30-case eval suite with LangSmith `evaluate()`.

    uv run python evals/run_evals.py                       # full suite
    uv run python evals/run_evals.py --kinds triage        # one family
    uv run python evals/run_evals.py --prefix candidate-haiku

Needs OPENROUTER_API_KEY (or the key for whichever provider CLINIC_MODEL uses) and LANGSMITH_API_KEY. Writes evals/results/latest.json,
which evals/gate.py compares against evals/baseline.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langsmith import Client, evaluate

from clinic_intake_agent import config
from clinic_intake_agent.agent import build_agent, build_specialist
from clinic_intake_agent.evals.scoring import score_example, summarize

HERE = Path(__file__).resolve().parent
DATASET_FILE = HERE / "dataset.jsonl"
RESULTS_DIR = HERE / "results"


def load_cases(kinds: set[str] | None = None) -> list[dict]:
    cases = [json.loads(line) for line in DATASET_FILE.read_text().splitlines() if line.strip()]
    return [c for c in cases if not kinds or c["inputs"]["kind"] in kinds]


def sync_dataset(client: Client, name: str, cases: list[dict]) -> str:
    """Create the LangSmith dataset on first run. Bump --dataset name when cases change."""
    if client.has_dataset(dataset_name=name):
        return name
    ds = client.create_dataset(dataset_name=name, description="Clinic intake agent: extraction, policy, triage")
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {"inputs": c["inputs"], "outputs": c["outputs"], "metadata": {**c["metadata"], "case_id": c["id"]}}
            for c in cases
        ],
    )
    print(f"Created LangSmith dataset {name!r} with {len(cases)} examples")
    return name


# Built once, reused across examples. Triage runs the full lead agent with an
# in-memory checkpointer so HITL interrupts work; each example gets its own thread.
# evaluate() runs examples on parallel threads, so build under a lock: two threads
# racing here used to build two agents (each with its own checkpointer), and one
# would then read state from the other's empty checkpointer.
_extractor = _policy = _lead = None
_build_lock = threading.Lock()


def _lead_agent():
    global _lead
    with _build_lock:
        if _lead is None:
            _lead = build_agent(checkpointer=InMemorySaver(), store=InMemoryStore())
        return _lead


def target(inputs: dict) -> dict:
    global _extractor, _policy
    kind = inputs["kind"]
    if kind == "extraction":
        _extractor = _extractor or build_specialist("intake-extractor")
        res = _extractor.invoke(
            {"messages": [{"role": "user", "content": f"Extract structured patient info from intake form {inputs['form_id']}."}]}
        )
        return {"intake": res["structured_response"].model_dump()}
    if kind == "policy":
        _policy = _policy or build_specialist("policy-answerer")
        res = _policy.invoke({"messages": [{"role": "user", "content": inputs["question"]}]})
        return {"policy": res["structured_response"].model_dump()}
    if kind == "triage":
        lead = _lead_agent()
        cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}
        res = lead.invoke({"messages": [{"role": "user", "content": inputs["message"]}]}, cfg)
        escalated = False
        for intr in res.get("__interrupt__", []) or []:
            value = intr.value if isinstance(intr.value, dict) else {}
            escalated |= any(a.get("name") == "escalate_to_clinician" for a in value.get("action_requests", []))
        state = lead.get_state(cfg).values
        last = state["messages"][-1]
        return {
            "escalated": escalated,
            "red_flags": sorted({f["category"] for f in state.get("red_flags", [])}),
            "final": last.content if isinstance(last.content, str) else str(last.content),
        }
    raise ValueError(f"unknown kind {kind!r}")


def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    s = score_example(inputs["kind"], outputs, reference_outputs)
    return {"key": "correctness", "score": s["score"], "comment": inputs["kind"]}


def urgent_recall(inputs: list[dict], outputs: list[dict], reference_outputs: list[dict]) -> dict:
    pairs = [
        (bool((o or {}).get("escalated")), r["should_escalate"])
        for i, o, r in zip(inputs, outputs, reference_outputs)
        if i["kind"] == "triage"
    ]
    positives = [esc for esc, should in pairs if should]
    return {"key": "urgent_recall", "score": (sum(positives) / len(positives)) if positives else None}


def main() -> int:
    config.load_env()
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="clinic-intake-evals-v1")
    p.add_argument("--prefix", default="clinic-intake")
    p.add_argument("--kinds", nargs="*", choices=["extraction", "policy", "triage"])
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    missing = config.missing_model_keys() + [k for k in ("LANGSMITH_API_KEY",) if not os.getenv(k)]
    if missing:
        print(f"Missing env vars: {', '.join(missing)}. Copy .env.example to .env and fill them in.")
        return 2

    client = Client()
    cases = load_cases(set(args.kinds) if args.kinds else None)
    dataset = sync_dataset(client, args.dataset, load_cases())
    data = client.list_examples(dataset_name=dataset)
    if args.kinds:
        data = [ex for ex in data if ex.inputs["kind"] in set(args.kinds)]

    results = evaluate(
        target,
        data=data,
        evaluators=[correctness],
        summary_evaluators=[urgent_recall],
        experiment_prefix=args.prefix,
        metadata={"lead_model": config.lead_model_id(), "subagent_model": config.subagent_model_id()},
        max_concurrency=args.concurrency,
        client=client,
    )

    rows = []
    for row in results:
        ex, run = row["example"], row["run"]
        s = score_example(ex.inputs["kind"], run.outputs or {}, ex.outputs)
        rows.append({"case_id": (ex.metadata or {}).get("case_id"), "kind": ex.inputs["kind"], **s})
    metrics = summarize(rows)
    report = {
        "experiment": results.experiment_name,
        "dataset": dataset,
        "lead_model": config.lead_model_id(),
        "subagent_model": config.subagent_model_id(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_cases_requested": len(cases),
        "metrics": metrics,
        "rows": rows,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{results.experiment_name}.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    (RESULTS_DIR / "latest.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {out.relative_to(HERE.parent)} and evals/results/latest.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
