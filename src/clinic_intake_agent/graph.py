"""Entry point for `langgraph dev` / LangSmith Deployment (see langgraph.json)."""

from __future__ import annotations

from clinic_intake_agent import config

from clinic_intake_agent.agent import build_agent
from clinic_intake_agent.tracing import configure_trace_redaction

config.load_env()
configure_trace_redaction()

# No checkpointer/store here: the LangGraph server injects its own persistence.
agent = build_agent()
