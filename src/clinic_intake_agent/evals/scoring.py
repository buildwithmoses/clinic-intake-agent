"""Scoring for the three eval families.

- extraction: per-field accuracy against the reference (0..1)
- policy: correct citation + required facts present, or a correct "not in policy" (0..1)
- triage: did the agent put an escalation in front of a human when it should (0/1)

Safety metrics (urgent_recall) are reported separately so a gate can hold them
to a hard floor instead of letting them average away.
"""

from __future__ import annotations

import re
from statistics import mean
from typing import Any


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _digits(value: Any) -> str:
    return re.sub(r"\D", "", str(value))


def _names(items: list[Any], key: str) -> set[str]:
    out = set()
    for item in items or []:
        name = item.get(key, "") if isinstance(item, dict) else str(item)
        out.add(_norm(name.split()[0]) if name else "")
    return {n for n in out if n}


def _field_match(field: str, expected: Any, actual: Any) -> bool:
    if field in {"current_medications", "allergies"}:
        key = "name" if field == "current_medications" else "substance"
        return _names(actual or [], key) == {_norm(str(e).split()[0]) for e in expected}
    if field == "phone":
        return _digits(expected) == _digits(actual)
    if isinstance(expected, bool):
        return bool(actual) is expected
    if field in {"insurance_provider", "preferred_language"}:
        # "Aetna" matches "Aetna (PPO)"; "Spanish" matches "Spanish (interpreter requested)".
        return _norm(expected) in _norm(actual or "")
    return _norm(expected) == _norm(actual or "")


def score_extraction(outputs: dict, reference: dict) -> dict:
    expected_fields: dict = reference["fields"]
    actual = outputs.get("intake") or {}
    per_field = {f: _field_match(f, exp, actual.get(f)) for f, exp in expected_fields.items()}
    return {"score": sum(per_field.values()) / len(per_field), "per_field": per_field}


def score_policy(outputs: dict, reference: dict) -> dict:
    ans = outputs.get("policy") or {}
    text = (ans.get("answer") or "").lower()
    if reference.get("expected_source") is None:  # out-of-scope question
        ok = ans.get("found_in_policy") is False
        return {"score": 1.0 if ok else 0.0, "cited": None, "facts": ok}
    cited = any(c.get("source") == reference["expected_source"] for c in ans.get("citations", []))
    facts = all(
        any(alt.lower() in text for alt in (kw if isinstance(kw, list) else [kw]))
        for kw in reference.get("must_include", [])
    )
    return {"score": (0.5 * cited) + (0.5 * facts), "cited": cited, "facts": facts}


def score_triage(outputs: dict, reference: dict) -> dict:
    escalated = bool(outputs.get("escalated"))
    should = bool(reference["should_escalate"])
    return {"score": 1.0 if escalated == should else 0.0, "escalated": escalated, "should_escalate": should}


SCORERS = {"extraction": score_extraction, "policy": score_policy, "triage": score_triage}


def score_example(kind: str, outputs: dict, reference: dict) -> dict:
    return SCORERS[kind](outputs or {}, reference)


def summarize(rows: list[dict]) -> dict:
    """rows: [{'kind': str, 'score': float, 'escalated'?: bool, 'should_escalate'?: bool}]"""
    by_kind: dict[str, list[float]] = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["score"])
    triage = [r for r in rows if r["kind"] == "triage"]
    positives = [r for r in triage if r["should_escalate"]]
    predicted = [r for r in triage if r["escalated"]]
    metrics: dict[str, float | int | None] = {
        "n": len(rows),
        "overall": round(mean(r["score"] for r in rows), 4) if rows else None,
        **{f"{k}_accuracy": round(mean(v), 4) for k, v in sorted(by_kind.items())},
        "urgent_recall": round(sum(r["escalated"] for r in positives) / len(positives), 4) if positives else None,
        "urgent_precision": round(sum(r["should_escalate"] for r in predicted) / len(predicted), 4)
        if predicted
        else None,
    }
    return metrics
