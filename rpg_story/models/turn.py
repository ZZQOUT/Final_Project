"""Canonical turn and log contracts (shared models)."""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, ConfigDict


class NPCDialogueLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    npc_id: str
    text: str


class NPCMove(BaseModel):
    model_config = ConfigDict(extra="forbid")

    npc_id: str
    from_location: str
    to_location: str
    trigger: Literal["player_instruction", "story_event", "system"]
    reason: str
    permanence: Literal["temporary", "permanent"]
    confidence: float = Field(..., ge=0.0, le=1.0)


class WorldUpdates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_location: Optional[str] = None
    npc_moves: List[NPCMove] = Field(default_factory=list)
    flags_delta: Dict[str, bool] = Field(default_factory=dict)
    quest_updates: Dict[str, str] = Field(default_factory=dict)


class TurnLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: int
    timestamp: str
    player_text: str
    location_id: str
    selected_npc_id: Optional[str] = None
    output: Dict[str, Any] = Field(default_factory=dict)
