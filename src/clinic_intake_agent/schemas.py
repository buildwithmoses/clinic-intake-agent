"""Pydantic schemas: the contract between the agent and the clinic's systems."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RedFlagCategory = Literal[
    "chest_pain",
    "suicidal_ideation",
    "stroke_symptoms",
    "breathing_emergency",
    "severe_bleeding",
    "anaphylaxis",
    "allergy_conflict",
]


class Medication(BaseModel):
    name: str = Field(description="Drug name only, e.g. 'Lisinopril'.")
    dose: str | None = Field(default=None, description="e.g. '10 mg'.")
    frequency: str | None = Field(default=None, description="e.g. 'once daily'.")


class Allergy(BaseModel):
    substance: str = Field(description="What the patient is allergic to, e.g. 'Penicillin'.")
    reaction: str | None = Field(default=None, description="e.g. 'hives'.")


class EmergencyContact(BaseModel):
    name: str | None = None
    relationship: str | None = None
    phone: str | None = None


class PatientIntake(BaseModel):
    """Structured patient info extracted from one intake form.

    Deliberately has NO Social Security Number field: the clinic's front desk
    does not need it from the agent, and PII middleware redacts it before the
    model ever sees the form.
    """

    form_id: str = Field(description="The form ID, e.g. 'intake-003'.")
    full_name: str
    date_of_birth: str = Field(description="ISO 8601 date, YYYY-MM-DD.")
    phone: str | None = None
    email: str | None = None
    preferred_language: str | None = None
    interpreter_requested: bool = False
    is_new_patient: bool
    guardian_name: str | None = Field(default=None, description="For minors only.")
    insurance_provider: str | None = Field(
        default=None, description="Carrier and plan as written on the form, or 'Self-pay'."
    )
    insurance_member_id: str | None = None
    emergency_contact: EmergencyContact | None = None
    reason_for_visit: str = Field(description="Patient's reason for visit, close to their own words.")
    current_medications: list[Medication] = Field(default_factory=list)
    allergies: list[Allergy] = Field(default_factory=list)
    medical_history: list[str] = Field(default_factory=list)
    missing_or_unclear_fields: list[str] = Field(
        default_factory=list, description="Fields that were blank, illegible or ambiguous."
    )


class Citation(BaseModel):
    source: str = Field(description="Policy file name, e.g. 'insurance-and-billing.md'.")
    section: str = Field(description="Section heading the answer came from.")
    quote: str = Field(description="Short verbatim quote supporting the answer.")


class PolicyAnswer(BaseModel):
    answer: str = Field(description="Plain-language answer for a patient or staff member.")
    # No default on purpose: a defaulted field is optional in the JSON schema, and
    # smaller models (Haiku) then omit it entirely. Required forces a decision.
    citations: list[Citation] = Field(
        description="REQUIRED when found_in_policy is true: at least one citation per fact, "
        "using the exact file name returned by search_clinic_policies. Empty only if found_in_policy is false.",
    )
    found_in_policy: bool = Field(
        description="False if the clinic's documents do not answer the question."
    )


class RedFlag(BaseModel):
    """A deterministic screening hit. Stored in graph state, not just in prose."""

    category: RedFlagCategory
    evidence: str = Field(description="The text that triggered the flag.")
    source: str = Field(description="Where it was found: a form ID or 'conversation'.")
    patient_ref: str = Field(description="Form ID, or 'conversation' for chat messages.")

    @property
    def id(self) -> str:
        return f"{self.patient_ref}:{self.category}"

    def to_state(self) -> dict:
        return {"id": self.id, **self.model_dump()}


class TriageFinding(BaseModel):
    category: RedFlagCategory
    evidence: str


class TriageResult(BaseModel):
    patient_ref: str = Field(description="Form ID, or 'conversation'.")
    urgency: Literal["routine", "urgent"]
    findings: list[TriageFinding] = Field(default_factory=list)
    rationale: str
    recommended_action: str = Field(
        description="What staff should do next. Never a diagnosis or treatment plan."
    )
