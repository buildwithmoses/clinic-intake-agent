"""Guarantees enforced in code, not prompts.

- Retries with backoff on model and read-only tool failures.
- Hard per-run caps on model and tool calls (runaway-loop / cost protection).
- PII redaction (SSN, card numbers) before the model sees user input or tool results.
- RedFlagGuard: the lead agent cannot finish a turn while a red flag in state has
  no escalation proposed for human review. It nudges the model, then fails closed.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import (
    AgentMiddleware,
    ModelCallLimitMiddleware,
    ModelRetryMiddleware,
    PIIMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
    hook_config,
)
from langchain_core.messages import AIMessage, HumanMessage

from clinic_intake_agent import config
from clinic_intake_agent.redflags import screen_text
from clinic_intake_agent.state import ClinicState

SSN_REGEX = r"\b\d{3}-\d{2}-\d{4}\b"
GUARD_NAME = "red_flag_guard"
READ_ONLY_TOOLS = ["read_intake_form", "list_intake_forms", "search_clinic_policies", "screen_for_red_flags"]


_TRANSIENT_NAMES = {
    "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError",
    "OverloadedError", "ServiceUnavailableError", "TimeoutError", "ConnectionError",
}


def is_transient(exc: Exception) -> bool:
    """Retry only what can succeed on retry: timeouts, 429s, 5xx/overloaded.

    Provider-agnostic (matches by class name / HTTP status) so swapping CLINIC_MODEL
    to another provider keeps working. Auth errors and bad requests fail fast.
    """
    status = getattr(exc, "status_code", None) or getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return status == 429 or status >= 500
    return any(cls.__name__ in _TRANSIENT_NAMES for cls in type(exc).__mro__)


def build_guardrails() -> list[AgentMiddleware]:
    """Fresh middleware instances (they hold per-agent state; don't share across agents)."""
    return [
        # on_failure="error": an outage surfaces as a failed run, never as a fake answer.
        ModelRetryMiddleware(
            max_retries=3, retry_on=is_transient, on_failure="error", backoff_factor=2.0, initial_delay=1.0
        ),
        # Escalations are deliberately NOT retried: never double-page a clinician.
        ToolRetryMiddleware(max_retries=2, tools=READ_ONLY_TOOLS),
        ModelCallLimitMiddleware(run_limit=config.max_model_calls_per_run(), exit_behavior="end"),
        ToolCallLimitMiddleware(run_limit=config.max_tool_calls_per_run(), exit_behavior="continue"),
        PIIMiddleware(
            "ssn", detector=SSN_REGEX, strategy="redact", apply_to_input=True, apply_to_tool_results=True
        ),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True, apply_to_tool_results=True),
    ]


def _is_guard_message(msg: Any) -> bool:
    return isinstance(msg, HumanMessage) and msg.name == GUARD_NAME


def _last_real_human_index(messages: list) -> int:
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage) and not _is_guard_message(messages[i]):
            return i
    return -1


def _message_text(msg: Any) -> str:
    content = msg.content
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))


def escalated_refs(messages: list) -> set[str]:
    """patient_refs for which an escalate_to_clinician call was proposed (and therefore
    shown to a human via the HITL interrupt)."""
    refs = set()
    for msg in messages:
        if isinstance(msg, AIMessage):
            for tc in msg.tool_calls:
                if tc["name"] == "escalate_to_clinician":
                    refs.add(str(tc["args"].get("patient_ref", "")).strip())
    return refs


def unresolved_flags(state: dict) -> list[dict]:
    done = escalated_refs(state.get("messages", []))
    return [f for f in state.get("red_flags", []) if f["patient_ref"] not in done]


class RedFlagGuardMiddleware(AgentMiddleware[ClinicState]):
    """Code-level backstop for the "never miss an urgent flag" requirement."""

    state_schema = ClinicState

    def __init__(self, max_nudges: int = 2) -> None:
        super().__init__()
        self.max_nudges = max_nudges

    def before_agent(self, state: ClinicState, runtime: Any) -> dict | None:
        messages = state.get("messages", [])
        idx = _last_real_human_index(messages)
        if idx < 0:
            return None
        flags = screen_text(_message_text(messages[idx]), patient_ref="conversation")
        return {"red_flags": [f.to_state() for f in flags]} if flags else None

    @hook_config(can_jump_to=["model", "end"])
    def after_model(self, state: ClinicState, runtime: Any) -> dict | None:
        messages = state.get("messages", [])
        if not messages:
            return None
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return None  # still working
        pending = unresolved_flags(state)
        if not pending:
            return None
        nudges = sum(1 for m in messages[_last_real_human_index(messages) + 1 :] if _is_guard_message(m))
        listing = "\n".join(f"- patient_ref={f['patient_ref']} category={f['category']}: {f['evidence']}" for f in pending)
        if nudges < self.max_nudges:
            return {
                "messages": [
                    HumanMessage(
                        name=GUARD_NAME,
                        content=(
                            "[Automated safety check] These red flags have no escalation yet. "
                            "Call escalate_to_clinician for each patient_ref before finishing:\n" + listing
                        ),
                    )
                ],
                "jump_to": "model",
            }
        # Fail closed: end the run with an explicit hand-off to staff.
        return {
            "messages": [
                AIMessage(
                    content=(
                        "URGENT - automated escalation could not be completed. A staff member must review "
                        "these red flags now:\n" + listing
                    )
                )
            ],
            "jump_to": "end",
        }
