"""LangSmith trace redaction.

Tracing itself is enabled purely by env vars (LANGSMITH_TRACING=true,
LANGSMITH_API_KEY, LANGSMITH_PROJECT). This module installs a global LangSmith
client with a regex anonymizer so phone numbers, emails, SSNs and dates of birth
are masked in uploaded traces, while the agent still sees real values.
"""

from __future__ import annotations

import os
import re

_RULES = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "<ssn>"),
    (r"\(?\b\d{3}\)?[\s.-]?\d{3}-\d{4}\b", "<phone>"),
    (r"[\w.+-]+@[\w-]+\.[\w.]+", "<email>"),
    (r"\b\d{1,2}/\d{1,2}/\d{4}\b", "<date>"),
    (r"\b(19|20)\d{2}-\d{2}-\d{2}\b", "<date>"),
]


def anonymizer_rules() -> list[dict]:
    return [{"pattern": re.compile(p), "replace": r} for p, r in _RULES]


def redact_text(text: str) -> str:
    """Same rules applied to a plain string (used in tests and local logs)."""
    for rule in anonymizer_rules():
        text = rule["pattern"].sub(rule["replace"], text)
    return text


def configure_trace_redaction() -> bool:
    """Install the anonymizing client globally. Returns True if installed.

    Opt out with CLINIC_REDACT_TRACES=false (e.g. for debugging on synthetic data).
    """
    if os.getenv("CLINIC_REDACT_TRACES", "true").lower() in {"0", "false", "no"}:
        return False
    import langsmith as ls
    from langsmith.anonymizer import create_anonymizer

    ls.configure(client=ls.Client(anonymizer=create_anonymizer(anonymizer_rules())))
    return True
