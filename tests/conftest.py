"""Shared fixtures. Every test runs with NO API keys and tracing off."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


@pytest.fixture(autouse=True)
def _no_keys(monkeypatch):
    for key in ("ANTHROPIC_API_KEY", "LANGSMITH_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    # ChatOpenRouter validates the key exists at construction. A dummy lets the
    # graph compile; no test ever calls a real model.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-dummy-not-a-real-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")


class ScriptedChatModel(BaseChatModel):
    """Plays back a fixed list of AIMessages and records what it was shown."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []
    cursor: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any):  # tools are ignored; the script decides
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        if self.cursor >= len(self.responses):
            raise AssertionError(f"Script exhausted after {self.cursor} calls")
        msg = self.responses[self.cursor]
        self.cursor += 1
        return ChatResult(generations=[ChatGeneration(message=msg)])


def ai_tool(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}])


def ai_text(text: str) -> AIMessage:
    return AIMessage(content=text)
