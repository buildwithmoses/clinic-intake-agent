"""End-to-end graph behaviour with a scripted (fake) model: no API keys, no network."""

import json
import uuid
from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from clinic_intake_agent.agent import SPECIALISTS, build_agent, build_specialist
from clinic_intake_agent.middleware import GUARD_NAME, escalated_refs, unresolved_flags
from clinic_intake_agent.tools import ESCALATION_NAMESPACE

from conftest import ScriptedChatModel, ai_text, ai_tool

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return {"configurable": {"thread_id": str(uuid.uuid4())}}


def test_langgraph_json_points_at_compiled_graph():
    spec = json.loads((ROOT / "langgraph.json").read_text())
    path, var = spec["graphs"]["clinic_intake"].split(":")
    assert (ROOT / path).exists() and var == "agent"
    from clinic_intake_agent.graph import agent  # builds with the real OpenRouter model class (dummy key from conftest)

    nodes = set(agent.get_graph().nodes)
    assert "HumanInTheLoopMiddleware.after_model" in nodes
    assert "RedFlagGuardMiddleware.after_model" in nodes
    assert "PIIMiddleware[ssn].before_model" in nodes
    assert agent.checkpointer is None  # langgraph dev supplies persistence


def test_default_model_is_claude_haiku_5_5_via_openrouter(monkeypatch):
    monkeypatch.delenv("CLINIC_MODEL", raising=False)
    from clinic_intake_agent import config

    model = config.get_model(config.lead_model_id())
    assert type(model).__name__ == "ChatOpenRouter"
    assert model.model_name == "anthropic/claude-haiku-5.5"
    assert model.app_title == config.OPENROUTER_APP_TITLE
    # Direct providers still work.
    monkeypatch.setenv("CLINIC_MODEL", "anthropic:claude-haiku-5-5")
    assert config.get_model(config.lead_model_id()).model == "claude-haiku-5-5"


def test_missing_model_keys_follows_provider(monkeypatch):
    from clinic_intake_agent import config

    monkeypatch.delenv("CLINIC_MODEL", raising=False)
    monkeypatch.delenv("CLINIC_SUBAGENT_MODEL", raising=False)
    assert config.missing_model_keys() == []
    monkeypatch.delenv("OPENROUTER_API_KEY")
    assert config.missing_model_keys() == ["OPENROUTER_API_KEY"]
    monkeypatch.setenv("CLINIC_SUBAGENT_MODEL", "anthropic:claude-haiku-5-5")
    assert config.missing_model_keys() == ["ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]


def test_specialists_compile():
    for name in SPECIALISTS:
        build_specialist(name, model=ScriptedChatModel(responses=[]))


def test_urgent_form_interrupts_for_human_approval_and_guard_forces_escalation():
    triage_result = {
        "patient_ref": "intake-002", "urgency": "urgent", "rationale": "exertional chest pain",
        "recommended_action": "escalate to on-call clinician today",
        "findings": [{"category": "chest_pain", "evidence": "chest pain when I walk up stairs"}],
    }
    model = ScriptedChatModel(
        responses=[
            ai_tool("task", {"description": "Triage intake form intake-002", "subagent_type": "triage"}, "t1"),
            ai_tool("read_intake_form", {"form_id": "intake-002"}, "s1"),  # subagent
            ai_text(json.dumps(triage_result)),  # subagent native structured output (JSON)
            ai_text("All done, routine visit booked."),  # lead tries to finish WITHOUT escalating
            ai_tool("escalate_to_clinician", {"patient_ref": "intake-002", "categories": ["chest_pain"],
                                              "summary": "Exertional chest pain radiating to left arm x2 days."}, "e1"),
            ai_text("Escalated intake-002 to the on-call clinician."),
        ]
    )
    store = InMemoryStore()
    agent = build_agent(model=model, subagent_model=model, checkpointer=InMemorySaver(), store=store)
    cfg = _cfg()
    result = agent.invoke({"messages": [{"role": "user", "content": "Please process intake form intake-002."}]}, cfg)

    # 1. The rule-based screen put the flag in state (not just prose).
    state = agent.get_state(cfg).values
    assert {f["id"] for f in state["red_flags"]} == {"intake-002:chest_pain"}
    # 2. The guard refused to let the lead finish and nudged it.
    assert any(isinstance(m, HumanMessage) and m.name == GUARD_NAME for m in state["messages"])
    # 3. The escalation is paused for a human, not executed.
    assert result.get("__interrupt__"), "expected a HITL interrupt"
    request = result["__interrupt__"][0].value
    assert request["action_requests"][0]["name"] == "escalate_to_clinician"
    assert store.search(ESCALATION_NAMESPACE) == []

    # 4. Staff approves -> tool runs -> record lands in the LangGraph Store.
    result = agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), cfg)
    assert not result.get("__interrupt__")
    assert len(store.search(ESCALATION_NAMESPACE)) == 1
    assert "Escalated" in result["messages"][-1].content


def test_rejected_escalation_does_not_run():
    model = ScriptedChatModel(
        responses=[
            ai_tool("escalate_to_clinician", {"patient_ref": "conversation", "categories": ["chest_pain"],
                                              "summary": "Patient reports chest pressure."}, "e1"),
            ai_text("A staff member reviewed this; please call 911 if symptoms are severe."),
        ]
    )
    store = InMemoryStore()
    agent = build_agent(model=model, subagent_model=model, checkpointer=InMemorySaver(), store=store)
    cfg = _cfg()
    result = agent.invoke({"messages": [{"role": "user", "content": "I've had crushing chest pressure all morning"}]}, cfg)
    assert result["__interrupt__"]
    result = agent.invoke(Command(resume={"decisions": [{"type": "reject", "message": "Already called patient."}]}), cfg)
    assert store.search(ESCALATION_NAMESPACE) == []
    tool_msgs = [m for m in result["messages"] if isinstance(m, ToolMessage) and m.name == "escalate_to_clinician"]
    assert tool_msgs and "Already called patient" in str(tool_msgs[-1].content)


def test_guard_fails_closed_when_model_never_escalates():
    model = ScriptedChatModel(responses=[ai_text("Sure, Thursday works!")] * 3)
    agent = build_agent(model=model, subagent_model=model, checkpointer=InMemorySaver())
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "I've had crushing chest pressure since this morning. Thursday?"}]},
        _cfg(),
    )
    assert model.cursor == 3  # first answer + 2 nudges
    assert result["messages"][-1].content.startswith("URGENT")
    assert unresolved_flags(result)[0]["category"] == "chest_pain"


def test_routine_question_finishes_normally():
    model = ScriptedChatModel(responses=[ai_text("Yes, we accept Cigna Open Access Plus.")])
    agent = build_agent(model=model, subagent_model=model, checkpointer=InMemorySaver())
    result = agent.invoke({"messages": [{"role": "user", "content": "Do you accept Cigna?"}]}, _cfg())
    assert result["messages"][-1].content.startswith("Yes")
    assert not result.get("__interrupt__") and not result.get("red_flags")


def test_ssn_is_redacted_before_the_model_sees_the_form():
    intake = {"form_id": "intake-004", "full_name": "Riley Sampleton", "date_of_birth": "1979-05-09",
              "is_new_patient": False, "reason_for_visit": "Sinus infection"}
    model = ScriptedChatModel(
        responses=[ai_tool("read_intake_form", {"form_id": "intake-004"}, "r1"), ai_text(json.dumps(intake))]
    )
    extractor = build_specialist("intake-extractor", model=model)
    result = extractor.invoke({"messages": [{"role": "user", "content": "Extract intake-004"}]})
    seen = "\n".join(str(m.content) for m in model.calls[-1])
    assert "Riley Sampleton" in seen  # the model still gets what it needs
    assert "900-00-0401" not in seen and "[REDACTED_SSN]" in seen
    assert result["structured_response"].full_name == "Riley Sampleton"
    assert {f["category"] for f in result["red_flags"]} == {"allergy_conflict"}


def test_escalated_refs_helper():
    msgs = [ai_tool("escalate_to_clinician", {"patient_ref": "intake-003"}, "x")]
    assert escalated_refs(msgs) == {"intake-003"}
    state = {"messages": msgs, "red_flags": [
        {"id": "intake-003:suicidal_ideation", "patient_ref": "intake-003", "category": "suicidal_ideation", "evidence": ""},
        {"id": "intake-002:chest_pain", "patient_ref": "intake-002", "category": "chest_pain", "evidence": ""},
    ]}
    assert [f["patient_ref"] for f in unresolved_flags(state)] == ["intake-002"]
