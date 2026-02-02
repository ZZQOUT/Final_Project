"""Schema validation for per-turn outputs."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class NPCDialogueLine(BaseModel):
    npc_id: str
    line: str


class NPCMove(BaseModel):
    npc_id: str
    from_location: str
    to_location: str
    reason: str
    trigger: str = Field(description="player_instruction|story_event")
    permanence: str = Field(description="temporary|until_further_notice|permanent")
    confidence: float = Field(0.0, ge=0.0, le=1.0)


class WorldUpdates(BaseModel):
    player_location: Optional[str] = None
    npc_moves: List[NPCMove] = []
    location_facts_add: List[str] = []
    location_facts_remove: List[str] = []


class SafetyFlag(BaseModel):
    refusal: bool
    reason: str


class TurnOutput(BaseModel):
    narration: str
    npc_dialogue: List[NPCDialogueLine]
    world_updates: WorldUpdates
    memory_summary: str
    safety: SafetyFlag


def validate_turn_output(data: dict) -> TurnOutput:
    return TurnOutput.model_validate(data)
