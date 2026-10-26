"""Clinic Intake Agent: a Deep Agents deployment for a (fictional) family clinic.

A lead agent plans; specialist subagents each do one job (intake extraction,
policy answers with citations, urgent triage); a human approves anything with
consequences (clinician escalations) through a LangGraph interrupt.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
