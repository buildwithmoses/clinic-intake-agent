"""Deterministic red-flag screening.

This is code, not a prompt. The LLM triage specialist adds judgement on top,
but these rules are the floor: if a rule fires, the flag lands in graph state
and the RedFlagGuard middleware will not let the lead agent finish until a
human has reviewed an escalation for it.

Deliberately biased toward recall: a false positive costs a staff member a
click; a false negative can cost a life.
"""

from __future__ import annotations

import re

from clinic_intake_agent.data import bullet_items, parse_sections
from clinic_intake_agent.schemas import RedFlag

PATTERNS: dict[str, list[str]] = {
    "chest_pain": [
        r"chest (pain|pressure|tightness|discomfort)",
        r"crushing (pain|pressure)",
        r"pain (going |that goes |radiating )?(down|into) (my |the )?(left )?arm",
    ],
    "suicidal_ideation": [
        r"suicid\w*",
        r"kill(ing)? myself",
        r"end(ing)? (my|it) (life|all)",
        r"want(ed)? to die",
        r"better off dead",
        r"self[- ]harm",
        r"(hurt|harm)(ing)? myself",
    ],
    "stroke_symptoms": [
        r"(face|facial)\b.{0,20}\bdroop\w*",
        r"droop\w*.{0,20}\b(face|mouth)",
        r"slurred",
        r"(sudden )?(weakness|numbness) (on|in) one side",
    ],
    "breathing_emergency": [
        r"can'?t breathe",
        r"cannot breathe",
        r"(struggling|gasping) (to|for) breath",
        r"lips? (are |is |turning )?blue",
    ],
    "severe_bleeding": [
        r"(won'?t|will not) stop bleeding",
        r"(vomiting|coughing up) blood",
        r"heavy bleeding",
    ],
    "anaphylaxis": [
        r"anaphyla\w*",
        r"throat (is )?(closing|swelling|tight)",
    ],
}
_COMPILED = {cat: [re.compile(p, re.I) for p in pats] for cat, pats in PATTERNS.items()}

# A negation cue within this many words before a match cancels it
# ("Denies chest pain", "No shortness of breath").
_NEGATION = re.compile(r"\b(denies|denied|no|not|never|without|negative for)\b", re.I)
_NEGATION_WINDOW_WORDS = 4
# "Chest pain: No" style checkbox answers.
_ANSWER_NO = re.compile(r":\s*(no|none|n/?a|denies)\.?\s*$", re.I)

# Allergy class -> drugs that conflict with it. Small on purpose; a real
# deployment would use the clinic's drug database (RxNorm / First Databank).
ALLERGY_CLASSES: dict[str, set[str]] = {
    "penicillin": {"penicillin", "amoxicillin", "ampicillin", "augmentin", "dicloxacillin", "nafcillin", "piperacillin"},
    "sulfa": {"sulfamethoxazole", "bactrim", "septra", "sulfasalazine"},
    "nsaid": {"ibuprofen", "naproxen", "aspirin", "celecoxib", "diclofenac", "ketorolac", "meloxicam"},
    "opioid": {"codeine", "hydrocodone", "oxycodone", "morphine", "tramadol"},
    "cephalosporin": {"cephalexin", "cefdinir", "ceftriaxone", "cefuroxime"},
}
_ALLERGY_ALIASES = {
    "sulfa drugs": "sulfa",
    "sulfonamides": "sulfa",
    "nsaids": "nsaid",
    "ibuprofen": "nsaid",
    "aspirin": "nsaid",
    "codeine": "opioid",
    "amoxicillin": "penicillin",
}


def _sentences(text: str) -> list[str]:
    out: list[str] = []
    for line in text.splitlines():
        line = line.strip().lstrip("-*> ").strip()
        if not line or _ANSWER_NO.search(line):
            continue
        out.extend(s for s in re.split(r"(?<=[.!?;])\s+", line) if s)
    return out


def _negated(sentence: str, start: int) -> bool:
    prefix_words = sentence[:start].split()[-_NEGATION_WINDOW_WORDS:]
    return bool(_NEGATION.search(" ".join(prefix_words)))


def screen_text(text: str, *, patient_ref: str = "conversation", source: str | None = None) -> list[RedFlag]:
    """Keyword/phrase screen with simple negation handling. One flag per category."""
    found: dict[str, RedFlag] = {}
    for sentence in _sentences(text):
        for category, patterns in _COMPILED.items():
            if category in found:
                continue
            for pat in patterns:
                m = pat.search(sentence)
                if m and not _negated(sentence, m.start()):
                    found[category] = RedFlag(
                        category=category,  # type: ignore[arg-type]
                        evidence=sentence[:300],
                        source=source or patient_ref,
                        patient_ref=patient_ref,
                    )
                    break
    return list(found.values())


def _allergy_class(allergy: str) -> str | None:
    a = allergy.lower()
    for alias, cls in _ALLERGY_ALIASES.items():
        if alias in a:
            return cls
    for cls in ALLERGY_CLASSES:
        if cls in a:
            return cls
    return None


def find_allergy_conflicts(allergies: list[str], medications: list[str]) -> list[tuple[str, str, str]]:
    """Return (allergy, medication, drug_class) for each conflicting pair."""
    conflicts = []
    for allergy in allergies:
        cls = _allergy_class(allergy)
        if cls is None:
            continue
        for med in medications:
            med_l = med.lower()
            if any(re.search(rf"\b{re.escape(drug)}\b", med_l) for drug in ALLERGY_CLASSES[cls]):
                conflicts.append((allergy, med, cls))
    return conflicts


def screen_intake_document(markdown: str, form_id: str) -> list[RedFlag]:
    """Screen a full intake form: symptom phrases + allergy/medication conflicts."""
    flags = screen_text(markdown, patient_ref=form_id, source=form_id)
    sections = parse_sections(markdown)
    allergies = bullet_items(sections.get("Allergies", ""))
    meds = bullet_items(sections.get("Current Medications", ""))
    conflicts = find_allergy_conflicts(allergies, meds)
    if conflicts:
        allergy, med, cls = conflicts[0]
        flags.append(
            RedFlag(
                category="allergy_conflict",
                evidence=f"Allergy '{allergy}' ({cls} class) conflicts with current medication '{med}'.",
                source=form_id,
                patient_ref=form_id,
            )
        )
    return flags


def format_flags(flags: list[RedFlag]) -> str:
    if not flags:
        return "[RED-FLAG SCREEN] No rule-based red flags found."
    lines = ["[RED-FLAG SCREEN] URGENT - rule-based screen found:"]
    lines += [f"  - {f.category} (patient_ref={f.patient_ref}): {f.evidence}" for f in flags]
    lines.append("  These must be escalated for human review via escalate_to_clinician.")
    return "\n".join(lines)
