"""LLM output contracts only (do not duplicate world models here)."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from rpg_story.models.turn import NPCDialogueLine, WorldUpdates


class SafetyFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refuse: bool
    reason: Optional[str] = None


class TurnOutput(BaseModel):
    """Structured output expected from the LLM each turn."""

    model_config = ConfigDict(extra="forbid")

    narration: str
    npc_dialogue: list[NPCDialogueLine]
    world_updates: WorldUpdates
    memory_summary: str
    safety: SafetyFlag


def validate_turn_output(data: dict) -> TurnOutput:
    return TurnOutput.model_validate(data)
