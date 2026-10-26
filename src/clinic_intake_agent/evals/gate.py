"""Release gate: block a candidate that scores below the current baseline.

Two kinds of checks:
- relative: each tracked metric must be >= baseline - tolerance
- absolute floors: safety metrics (urgent_recall) must meet a fixed bar no matter
  what the baseline was. Safety never averages away.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_TRACKED = ("overall", "extraction_accuracy", "policy_accuracy", "triage_accuracy")
DEFAULT_FLOORS = {"urgent_recall": 1.0}


@dataclass
class GateResult:
    passed: bool
    failures: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)


def check_gate(
    candidate: dict,
    baseline: dict | None,
    *,
    tolerance: float = 0.0,
    tracked: tuple[str, ...] = DEFAULT_TRACKED,
    floors: dict[str, float] | None = None,
) -> GateResult:
    floors = DEFAULT_FLOORS if floors is None else floors
    result = GateResult(passed=True)

    for metric, floor in floors.items():
        value = candidate.get(metric)
        if value is None:
            result.failures.append(f"{metric}: missing from candidate (floor {floor})")
        elif value < floor:
            result.failures.append(f"{metric}: {value:.3f} < hard floor {floor:.3f}")
        else:
            result.checks.append(f"{metric}: {value:.3f} >= floor {floor:.3f}")

    if not baseline:
        result.checks.append("no baseline yet: relative checks skipped (promote one with --promote)")
    else:
        for metric in tracked:
            base, cand = baseline.get(metric), candidate.get(metric)
            if base is None:
                continue
            if cand is None:
                result.failures.append(f"{metric}: missing from candidate (baseline {base:.3f})")
            elif cand < base - tolerance:
                result.failures.append(f"{metric}: {cand:.3f} < baseline {base:.3f} (tolerance {tolerance})")
            else:
                result.checks.append(f"{metric}: {cand:.3f} vs baseline {base:.3f} ok")

    result.passed = not result.failures
    return result
