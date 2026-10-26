import json
from pathlib import Path

import pytest

from clinic_intake_agent import data, policy_search
from clinic_intake_agent.schemas import PatientIntake, PolicyAnswer, TriageResult
from clinic_intake_agent.tools import list_intake_forms, search_clinic_policies

ROOT = Path(__file__).resolve().parents[1]
DATASET = [json.loads(l) for l in (ROOT / "evals" / "dataset.jsonl").read_text().splitlines() if l.strip()]


def test_every_sample_is_marked_fictional():
    for path in list((ROOT / "data" / "intake").glob("*.md")) + list((ROOT / "data" / "policies").glob("*.md")):
        text = path.read_text().lower()
        assert "fictional" in text, path.name


def test_fake_identifiers_only():
    for fid in data.list_form_ids():
        text = data.load_intake(fid)
        assert "900-00-" in text  # 9xx SSNs are never issued
        assert "(555) 010-" in text  # 555-01xx is reserved for fiction
        assert "@example.com" in text


@pytest.mark.parametrize("bad", ["../../etc/passwd", "intake-1", "intake-999"])
def test_load_intake_rejects_bad_ids(bad):
    with pytest.raises((ValueError, FileNotFoundError)):
        data.load_intake(bad)


def test_list_tool():
    out = list_intake_forms.invoke({})
    assert "intake-001" in out and "intake-008" in out


def test_patient_schema_has_no_ssn_field():
    assert not any("ssn" in f or "social" in f for f in PatientIntake.model_fields)


def test_schemas_validate():
    PatientIntake(form_id="intake-001", full_name="Avery Testcase", date_of_birth="1988-03-14",
                  is_new_patient=True, reason_for_visit="Annual physical")
    PolicyAnswer(answer="We are closed Sundays.", found_in_policy=True,
                 citations=[{"source": "hours-and-locations.md", "section": "Clinic hours", "quote": "closed on Sundays"}])
    TriageResult(patient_ref="intake-002", urgency="urgent", rationale="x", recommended_action="escalate",
                 findings=[{"category": "chest_pain", "evidence": "chest pain on stairs"}])
    with pytest.raises(Exception):
        TriageResult(patient_ref="x", urgency="maybe", rationale="x", recommended_action="x")


POLICY_CASES = [c for c in DATASET if c["inputs"]["kind"] == "policy" and c["outputs"]["expected_source"]]


@pytest.mark.parametrize("case", POLICY_CASES, ids=[c["id"] for c in POLICY_CASES])
def test_retrieval_finds_expected_source_in_top3(case):
    """Offline retrieval eval: the right document must be retrievable before the LLM ever runs."""
    hits = policy_search.search(case["inputs"]["question"], k=3)
    assert case["outputs"]["expected_source"] in {c.source for c, _ in hits}


def test_search_tool_formats_citations():
    out = search_clinic_policies.invoke({"query": "late cancellation fee"})
    assert "cancellation-and-no-show.md" in out and "$50" in out


def test_dataset_shape():
    assert len(DATASET) == 30
    kinds = [c["inputs"]["kind"] for c in DATASET]
    assert kinds.count("extraction") == 8 and kinds.count("policy") == 10 and kinds.count("triage") == 12
    assert len({c["id"] for c in DATASET}) == 30
    policy_files = {p.name for p in (ROOT / "data" / "policies").glob("*.md")}
    for c in DATASET:
        if c["inputs"]["kind"] == "extraction":
            data.load_intake(c["inputs"]["form_id"])
        if c["inputs"]["kind"] == "policy" and c["outputs"]["expected_source"]:
            assert c["outputs"]["expected_source"] in policy_files
    assert sum(c["outputs"].get("should_escalate", False) for c in DATASET) == 6
