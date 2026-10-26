"""Loading the clinic's documents from disk."""

from __future__ import annotations

import re
from pathlib import Path

from clinic_intake_agent.config import data_dir

FORM_ID_RE = re.compile(r"^intake-\d{3}$")


def intake_dir() -> Path:
    return data_dir() / "intake"


def policy_dir() -> Path:
    return data_dir() / "policies"


def list_form_ids() -> list[str]:
    return sorted(p.stem for p in intake_dir().glob("intake-*.md"))


def load_intake(form_id: str) -> str:
    form_id = form_id.strip().removesuffix(".md")
    if not FORM_ID_RE.match(form_id):  # also blocks path traversal
        raise ValueError(f"Invalid form id {form_id!r}; expected like 'intake-001'.")
    path = intake_dir() / f"{form_id}.md"
    if not path.exists():
        raise FileNotFoundError(f"No intake form {form_id!r}. Known: {', '.join(list_form_ids())}")
    return path.read_text(encoding="utf-8")


def parse_sections(markdown: str) -> dict[str, str]:
    """Split a markdown doc into {'## heading': body} (H2 sections only)."""
    sections: dict[str, str] = {}
    current = "_preamble"
    buf: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("## "):
            sections[current] = "\n".join(buf).strip()
            current, buf = line[3:].strip(), []
        else:
            buf.append(line)
    sections[current] = "\n".join(buf).strip()
    return sections


def bullet_items(section_body: str) -> list[str]:
    """Return '- item' lines from a section, skipping 'None' / 'No known ...'."""
    items = []
    for line in section_body.splitlines():
        line = line.strip()
        if not line.startswith("- "):
            continue
        item = line[2:].strip()
        if re.match(r"^(none|no known|n/?a)\b", item, re.I):
            continue
        items.append(item)
    return items
