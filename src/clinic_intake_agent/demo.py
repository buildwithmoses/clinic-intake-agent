"""Terminal demo: process one intake form and approve escalations by hand.

    uv run clinic-demo "Please process intake form intake-002."

Same graph as `langgraph dev`, but with a local InMemorySaver + InMemoryStore so
the human-in-the-loop interrupt can be resumed from the terminal.
"""

from __future__ import annotations

import json
import os
import sys
import uuid

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from clinic_intake_agent import config
from clinic_intake_agent.agent import build_agent
from clinic_intake_agent.tools import ESCALATION_NAMESPACE
from clinic_intake_agent.tracing import configure_trace_redaction


def _review(action: dict) -> dict:
    print("\n=== HUMAN REVIEW REQUIRED ===")
    print(f"Tool: {action['name']}")
    print(json.dumps(action["args"], indent=2))
    choice = input("[a]pprove / [r]eject? ").strip().lower()
    if choice.startswith("a"):
        return {"type": "approve"}
    reason = input("Reason for rejecting: ").strip() or "Rejected by staff."
    return {"type": "reject", "message": reason}


def main() -> int:
    config.load_env()
    if any(a in {"-h", "--help"} for a in sys.argv[1:]):
        print(__doc__)
        return 0
    missing = config.missing_model_keys()
    if missing:
        print(f"{', '.join(missing)} not set. Copy .env.example to .env and add your key.")
        return 2
    configure_trace_redaction()
    message = " ".join(sys.argv[1:]) or "Please process intake form intake-002."
    store = InMemoryStore()
    agent = build_agent(checkpointer=InMemorySaver(), store=store)
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    result = agent.invoke({"messages": [{"role": "user", "content": message}]}, cfg)
    while result.get("__interrupt__"):
        request = result["__interrupt__"][0].value
        decisions = [_review(a) for a in request["action_requests"]]
        result = agent.invoke(Command(resume={"decisions": decisions}), cfg)

    print("\n=== AGENT ===")
    print(result["messages"][-1].content)
    queued = store.search(ESCALATION_NAMESPACE)
    print(f"\nEscalations in clinician queue: {len(queued)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
