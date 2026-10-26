"""Tools. Each does one narrow thing; consequential actions are interrupt-gated."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from langchain.tools import ToolRuntime, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from clinic_intake_agent import data, policy_search
from clinic_intake_agent.redflags import format_flags, screen_intake_document, screen_text

ESCALATION_NAMESPACE = ("escalations",)


@tool
def list_intake_forms() -> str:
    """List the IDs of intake forms waiting to be processed."""
    ids = data.list_form_ids()
    return "Intake forms: " + ", ".join(ids) if ids else "No intake forms found."


@tool
def read_intake_form(form_id: str, runtime: ToolRuntime) -> Command | str:
    """Read one patient intake form by ID (e.g. 'intake-003').

    Also runs the clinic's rule-based red-flag screen and records any flags in
    the agent state so they cannot be silently dropped.
    """
    try:
        text = data.load_intake(form_id)
    except (ValueError, FileNotFoundError) as e:
        return f"Error: {e}"
    flags = screen_intake_document(text, form_id.strip().removesuffix(".md"))
    content = f"{text}\n\n---\n{format_flags(flags)}"
    return Command(
        update={
            "red_flags": [f.to_state() for f in flags],
            "messages": [ToolMessage(content=content, tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def screen_for_red_flags(text: str, runtime: ToolRuntime, patient_ref: str = "conversation") -> Command:
    """Run the rule-based red-flag screen on free text (e.g. a patient's chat message).

    patient_ref: the form ID the text belongs to, or 'conversation' for chat.
    """
    flags = screen_text(text, patient_ref=patient_ref)
    return Command(
        update={
            "red_flags": [f.to_state() for f in flags],
            "messages": [ToolMessage(content=format_flags(flags), tool_call_id=runtime.tool_call_id)],
        }
    )


@tool
def search_clinic_policies(query: str) -> str:
    """Search the clinic's policy documents (hours, insurance, cancellations,
    new patients, refills). Returns the top matching sections with citations."""
    hits = policy_search.search(query, k=3)
    if not hits:
        return "No matching policy sections found."
    blocks = [
        f"[source: {chunk.source} | section: {chunk.section} | score: {score:.2f}]\n{chunk.text}"
        for chunk, score in hits
    ]
    return "\n\n".join(blocks)


@tool
def escalate_to_clinician(patient_ref: str, categories: list[str], summary: str, runtime: ToolRuntime) -> str:
    """Send an urgent case to the on-call clinician's review queue.

    This tool is gated by a human-in-the-loop interrupt: a staff member must
    approve, edit or reject it before it runs. Use it for every red flag.

    patient_ref: the form ID (e.g. 'intake-002') or 'conversation'.
    categories: red-flag categories, e.g. ['chest_pain'].
    summary: 1-3 sentences of facts for the clinician. No diagnosis.
    """
    record = {
        "id": str(uuid.uuid4()),
        "patient_ref": patient_ref,
        "categories": categories,
        "summary": summary,
        "status": "approved_by_staff",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    if runtime.store is not None:  # LangGraph Store: persists across threads
        runtime.store.put(ESCALATION_NAMESPACE, record["id"], record)
    return (
        f"Escalation {record['id']} queued for the on-call clinician "
        f"(patient_ref={patient_ref}, categories={', '.join(categories)})."
    )
