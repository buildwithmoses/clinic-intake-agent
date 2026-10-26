"""Assemble the lead agent and its specialists.

Henry's pattern: a lead agent plans, specialists each do one job, and a human
approves anything with consequences.
"""

from __future__ import annotations

from typing import Any

from deepagents import SubAgent, create_deep_agent
from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.agents.structured_output import ProviderStrategy
from langchain_core.language_models import BaseChatModel

from clinic_intake_agent import config
from clinic_intake_agent.middleware import RedFlagGuardMiddleware, build_guardrails
from clinic_intake_agent.prompts import EXTRACTOR_PROMPT, LEAD_PROMPT, POLICY_PROMPT, TRIAGE_PROMPT
from clinic_intake_agent.schemas import PatientIntake, PolicyAnswer, TriageResult
from clinic_intake_agent.state import ClinicState
from clinic_intake_agent.tools import (
    escalate_to_clinician,
    list_intake_forms,
    read_intake_form,
    screen_for_red_flags,
    search_clinic_policies,
)

# Approve / edit / reject — no free-text "respond": an escalation is either sent,
# corrected, or consciously declined by a human.
ESCALATION_REVIEW = {"allowed_decisions": ["approve", "edit", "reject"]}
INTERRUPT_ON = {"escalate_to_clinician": ESCALATION_REVIEW}

SPECIALISTS: dict[str, dict[str, Any]] = {
    "intake-extractor": {
        "description": "Extracts structured patient info (demographics, insurance, meds, allergies, "
        "reason for visit) from ONE intake form. Give it the form ID.",
        "system_prompt": EXTRACTOR_PROMPT,
        "tools": [read_intake_form],
        "response_format": ProviderStrategy(PatientIntake),
    },
    "policy-answerer": {
        "description": "Answers a question about clinic policy (hours, insurance, cancellation fees, "
        "new-patient steps, refills) from the clinic's documents, with citations.",
        "system_prompt": POLICY_PROMPT,
        "tools": [search_clinic_policies],
        "response_format": ProviderStrategy(PolicyAnswer),
    },
    "triage": {
        "description": "Checks ONE intake form (give the form ID) or a patient's message (give the exact "
        "text) for urgent red flags. Returns urgency and findings. Does not escalate.",
        "system_prompt": TRIAGE_PROMPT,
        "tools": [read_intake_form, screen_for_red_flags],
        "response_format": ProviderStrategy(TriageResult),
    },
}


def subagent_specs(model: BaseChatModel) -> list[SubAgent]:
    return [
        SubAgent(name=name, model=model, middleware=build_guardrails(), **spec)  # type: ignore[typeddict-item]
        for name, spec in SPECIALISTS.items()
    ]


def build_agent(
    *,
    model: BaseChatModel | str | None = None,
    subagent_model: BaseChatModel | str | None = None,
    checkpointer: Any = None,
    store: Any = None,
):
    """Build the lead deep agent.

    Leave checkpointer/store as None under `langgraph dev` / LangSmith Deployment
    (the platform provides persistence and rejects custom ones in local dev).
    Pass an InMemorySaver for scripts and evals: HITL interrupts need a checkpointer.
    """
    lead = model if isinstance(model, BaseChatModel) else config.get_model(model or config.lead_model_id())
    sub = (
        subagent_model
        if isinstance(subagent_model, BaseChatModel)
        else config.get_model(subagent_model or config.subagent_model_id())
    )
    return create_deep_agent(
        model=lead,
        tools=[list_intake_forms, escalate_to_clinician],
        system_prompt=LEAD_PROMPT,
        subagents=subagent_specs(sub),
        middleware=[
            *build_guardrails(),
            ToolCallLimitMiddleware(
                tool_name="escalate_to_clinician",
                run_limit=config.max_escalations_per_run(),
                exit_behavior="continue",
            ),
            RedFlagGuardMiddleware(),
        ],
        interrupt_on=INTERRUPT_ON,
        state_schema=ClinicState,
        checkpointer=checkpointer,
        store=store,
        name="clinic-intake-agent",
    )


def build_specialist(name: str, *, model: BaseChatModel | str | None = None, checkpointer: Any = None):
    """Run one specialist on its own (same prompt, tools, schema, guardrails).

    Used by the eval suite to score extraction and policy answers in isolation,
    which makes regressions attributable to a single component.
    """
    spec = SPECIALISTS[name]
    llm = model if isinstance(model, BaseChatModel) else config.get_model(model or config.subagent_model_id())
    return create_agent(
        llm,
        tools=spec["tools"],
        system_prompt=spec["system_prompt"],
        response_format=spec["response_format"],
        middleware=build_guardrails(),
        state_schema=ClinicState,
        checkpointer=checkpointer,
        name=name,
    )
