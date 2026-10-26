"""Graph state extensions."""

from __future__ import annotations

from typing import Annotated

from deepagents import DeepAgentState
from typing_extensions import NotRequired


def merge_red_flags(left: list[dict] | None, right: list[dict] | None) -> list[dict]:
    """Idempotent union keyed by flag id.

    Deep Agents passes parent state into a subagent and writes the subagent's
    (full) state back, so a plain `operator.add` reducer would duplicate flags.
    """
    merged: dict[str, dict] = {f["id"]: f for f in (left or [])}
    for flag in right or []:
        merged.setdefault(flag["id"], flag)
    return list(merged.values())


class ClinicState(DeepAgentState):
    red_flags: NotRequired[Annotated[list[dict], merge_red_flags]]
