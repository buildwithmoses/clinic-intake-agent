"""Eval gate: exit non-zero if the candidate scores below the baseline.

    uv run python evals/gate.py                                   # latest vs baseline
    uv run python evals/gate.py --candidate evals/results/X.json --tolerance 0.02
    uv run python evals/gate.py --promote                         # pass -> becomes new baseline

Exit codes: 0 pass, 1 gate failed (block the change), 2 usage/missing files.
Wire it into CI after run_evals.py so a regression can't merge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clinic_intake_agent.evals.gate import check_gate

HERE = Path(__file__).resolve().parent


def _metrics(path: Path) -> dict:
    blob = json.loads(path.read_text())
    return blob.get("metrics", blob)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, default=HERE / "results" / "latest.json")
    p.add_argument("--baseline", type=Path, default=HERE / "baseline.json")
    p.add_argument("--tolerance", type=float, default=0.0)
    p.add_argument("--promote", action="store_true", help="on pass, write candidate as the new baseline")
    args = p.parse_args(argv)

    if not args.candidate.exists():
        print(f"No candidate results at {args.candidate}. Run evals/run_evals.py first.")
        return 2
    candidate = _metrics(args.candidate)
    baseline = _metrics(args.baseline) if args.baseline.exists() else None

    result = check_gate(candidate, baseline, tolerance=args.tolerance)
    for line in result.checks:
        print(f"  ok    {line}")
    for line in result.failures:
        print(f"  FAIL  {line}")
    print("GATE PASSED" if result.passed else "GATE FAILED - do not ship this change")

    if result.passed and args.promote:
        args.baseline.write_text(json.dumps(json.loads(args.candidate.read_text()), indent=2))
        print(f"Promoted {args.candidate.name} to {args.baseline.name}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
