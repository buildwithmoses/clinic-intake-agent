import pytest

from clinic_intake_agent.data import list_form_ids, load_intake
from clinic_intake_agent.redflags import find_allergy_conflicts, screen_intake_document, screen_text

EXPECTED = {
    "intake-001": set(),
    "intake-002": {"chest_pain"},
    "intake-003": {"suicidal_ideation"},
    "intake-004": {"allergy_conflict"},
    "intake-005": set(),  # "Denies chest pain." must NOT fire
    "intake-006": set(),
    "intake-007": {"stroke_symptoms"},
    "intake-008": set(),
}


def test_all_forms_covered():
    assert set(list_form_ids()) == set(EXPECTED)


@pytest.mark.parametrize("form_id,expected", sorted(EXPECTED.items()))
def test_intake_screen(form_id, expected):
    flags = screen_intake_document(load_intake(form_id), form_id)
    assert {f.category for f in flags} == expected
    assert all(f.patient_ref == form_id for f in flags)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("I've had crushing chest pressure since this morning", {"chest_pain"}),
        ("Patient denies chest pain.", set()),
        ("No chest pain, just a cough", set()),
        ("Chest pain: No", set()),
        ("I'm not sure why, but I've had chest pain all day", {"chest_pain"}),
        ("Sometimes I think about killing myself", {"suicidal_ideation"}),
        ("My speech was slurred for 10 minutes", {"stroke_symptoms"}),
        ("My throat is closing after eating shrimp", {"anaphylaxis"}),
        ("Can I reschedule to Friday?", set()),
        ("Do you accept Cigna?", set()),
    ],
)
def test_screen_text(text, expected):
    assert {f.category for f in screen_text(text)} == expected


def test_allergy_conflicts():
    assert find_allergy_conflicts(["Penicillin — hives"], ["Amoxicillin 500 mg"])[0][2] == "penicillin"
    assert find_allergy_conflicts(["Sulfa drugs — rash"], ["Lisinopril 20 mg"]) == []
    assert find_allergy_conflicts(["Ibuprofen"], ["Naproxen 220 mg"])[0][2] == "nsaid"
    assert find_allergy_conflicts(["Latex"], ["Cetirizine"]) == []


def test_flag_ids_are_stable():
    a = screen_text("chest pain", patient_ref="intake-002")[0]
    assert a.id == "intake-002:chest_pain"
