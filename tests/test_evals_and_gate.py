import json
import runpy
import sys
from pathlib import Path

import pytest

from clinic_intake_agent.evals.gate import check_gate
from clinic_intake_agent.evals.scoring import score_extraction, score_policy, score_triage, summarize

ROOT = Path(__file__).resolve().parents[1]


def test_gate_blocks_regression_like_075_vs_0958():
    """The pattern from Henry's site: a change scoring 0.75 vs a 0.958 baseline is blocked."""
    baseline = {"overall": 0.958, "urgent_recall": 1.0}
    candidate = {"overall": 0.75, "urgent_recall": 1.0}
    result = check_gate(candidate, baseline)
    assert not result.passed
    assert any("overall" in f for f in result.failures)


def test_gate_passes_equal_or_better():
    assert check_gate({"overall": 0.96, "urgent_recall": 1.0}, {"overall": 0.958}).passed


def test_gate_tolerance():
    assert check_gate({"overall": 0.95, "urgent_recall": 1.0}, {"overall": 0.958}, tolerance=0.01).passed
    assert not check_gate({"overall": 0.94, "urgent_recall": 1.0}, {"overall": 0.958}, tolerance=0.01).passed


def test_safety_floor_cannot_average_away():
    # Better overall score, but it missed one urgent case: still blocked.
    result = check_gate({"overall": 0.99, "urgent_recall": 0.83}, {"overall": 0.9, "urgent_recall": 1.0})
    assert not result.passed
    assert any("urgent_recall" in f for f in result.failures)


def test_gate_without_baseline_still_enforces_floor():
    assert check_gate({"overall": 0.5, "urgent_recall": 1.0}, None).passed
    assert not check_gate({"overall": 0.5, "urgent_recall": 0.5}, None).passed


def _run_gate_cli(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["gate.py", *args])
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(str(ROOT / "evals" / "gate.py"), run_name="__main__")
    return exc.value.code


def test_gate_cli_exit_codes(tmp_path, monkeypatch):
    base, cand = tmp_path / "baseline.json", tmp_path / "cand.json"
    base.write_text(json.dumps({"metrics": {"overall": 0.958, "urgent_recall": 1.0}}))
    cand.write_text(json.dumps({"metrics": {"overall": 0.75, "urgent_recall": 1.0}}))
    assert _run_gate_cli(monkeypatch, "--candidate", str(cand), "--baseline", str(base)) == 1
    cand.write_text(json.dumps({"metrics": {"overall": 0.97, "urgent_recall": 1.0}}))
    assert _run_gate_cli(monkeypatch, "--candidate", str(cand), "--baseline", str(base), "--promote") == 0
    assert json.loads(base.read_text())["metrics"]["overall"] == 0.97
    assert _run_gate_cli(monkeypatch, "--candidate", str(tmp_path / "missing.json")) == 2


def test_score_extraction():
    ref = {"fields": {"full_name": "Riley Sampleton", "date_of_birth": "1979-05-09", "phone": "5550100401",
                      "is_new_patient": False, "insurance_provider": "UnitedHealthcare",
                      "allergies": ["Penicillin"], "current_medications": ["Amoxicillin", "Lisinopril"]}}
    out = {"intake": {"full_name": "Riley Sampleton", "date_of_birth": "1979-05-09", "phone": "(555) 010-0401",
                      "is_new_patient": False, "insurance_provider": "UnitedHealthcare (Choice Plus PPO)",
                      "allergies": [{"substance": "Penicillin", "reaction": "hives"}],
                      "current_medications": [{"name": "Amoxicillin"}, {"name": "Lisinopril"}]}}
    assert score_extraction(out, ref)["score"] == 1.0
    out["intake"]["date_of_birth"] = "05/09/1979"  # not ISO
    out["intake"]["current_medications"] = [{"name": "Lisinopril"}]  # missed one
    assert score_extraction(out, ref)["score"] == pytest.approx(5 / 7)


def test_score_policy():
    ref = {"expected_source": "cancellation-and-no-show.md", "must_include": [["$50", "50"]]}
    good = {"policy": {"answer": "There is a $50 fee.", "found_in_policy": True,
                       "citations": [{"source": "cancellation-and-no-show.md", "section": "x", "quote": "y"}]}}
    assert score_policy(good, ref)["score"] == 1.0
    uncited = {"policy": {"answer": "There is a $50 fee.", "found_in_policy": True, "citations": []}}
    assert score_policy(uncited, ref)["score"] == 0.5
    oos = {"expected_source": None, "must_include": []}
    assert score_policy({"policy": {"answer": "I don't know", "found_in_policy": False}}, oos)["score"] == 1.0
    assert score_policy({"policy": {"answer": "Yes, $300", "found_in_policy": True}}, oos)["score"] == 0.0


def test_score_triage_and_summary():
    rows = []
    for esc, should in [(True, True), (False, True), (False, False), (True, False)]:
        rows.append({"kind": "triage", **score_triage({"escalated": esc}, {"should_escalate": should})})
    rows.append({"kind": "policy", "score": 1.0})
    m = summarize(rows)
    assert m["urgent_recall"] == 0.5
    assert m["urgent_precision"] == 0.5
    assert m["triage_accuracy"] == 0.5
    assert m["n"] == 5
