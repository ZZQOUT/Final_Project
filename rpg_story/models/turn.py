"""Canonical turn and log contracts (shared models)."""
from __future__ import annotations

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator, field_validator


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


class QuestProgressUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    quest_id: str
    title: Optional[str] = None
    category: Optional[Literal["main", "side"]] = None
    status: Optional[Literal["available", "active", "completed", "failed"]] = None
    objective: Optional[str] = None
    guidance: Optional[str] = None
    giver_npc_id: Optional[str] = None
    reward_hint: Optional[str] = None
    required_items: Dict[str, int] = Field(default_factory=dict)
    reward_items: Dict[str, int] = Field(default_factory=dict)
    collected_items_delta: Dict[str, int] = Field(default_factory=dict)

    @field_validator("required_items", "reward_items", mode="before")
    @classmethod
    def _normalize_required_items(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, list):
            result: Dict[str, int] = {}
            for item in v:
                if not isinstance(item, dict):
                    continue
                key = item.get("item") or item.get("name")
                count = item.get("count") or item.get("qty") or item.get("quantity") or 1
                if not key:
                    continue
                try:
                    amount = int(float(count))
                except Exception:
                    amount = 1
                if amount > 0:
                    result[str(key)] = amount
            return result
        if isinstance(v, dict):
            result: Dict[str, int] = {}
            for key, raw in v.items():
                try:
                    amount = int(float(raw))
                except Exception:
                    continue
                if amount > 0:
                    result[str(key)] = amount
            return result
        return {}

    @field_validator("collected_items_delta", mode="before")
    @classmethod
    def _normalize_collected_delta(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, dict):
            result: Dict[str, int] = {}
            for key, raw in v.items():
                try:
                    amount = int(float(raw))
                except Exception:
                    continue
                if amount != 0:
                    result[str(key)] = amount
            return result
        return {}


class WorldUpdates(BaseModel):
    model_config = ConfigDict(extra="forbid")

    player_location: Optional[str] = None
    npc_moves: List[NPCMove] = Field(default_factory=list)
    flags_delta: Dict[str, bool] = Field(default_factory=dict)
    quest_updates: Dict[str, str] = Field(default_factory=dict)
    quest_progress_updates: List[QuestProgressUpdate] = Field(default_factory=list)
    inventory_delta: Dict[str, int] = Field(default_factory=dict)

    @field_validator("npc_moves", mode="before")
    @classmethod
    def _normalize_npc_moves(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, dict):
            if not v:
                return []
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError("npc_moves must be a list (use [] if none)")

    @field_validator("quest_updates", mode="before")
    @classmethod
    def _normalize_quest_updates(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, list):
            if len(v) == 0:
                return {}
            # Try to convert list of objects into dict by quest_id
            result: Dict[str, str] = {}
            for item in v:
                if not isinstance(item, dict):
                    raise ValueError("quest_updates list items must be objects")
                quest_id = item.get("quest_id")
                status = item.get("status")
                if not quest_id or status is None:
                    raise ValueError("quest_updates list items must include quest_id and status")
                result[str(quest_id)] = str(status)
            return result
        if isinstance(v, dict):
            return v
        raise ValueError("quest_updates must be an object (dict) or []")

    @field_validator("quest_progress_updates", mode="before")
    @classmethod
    def _normalize_quest_progress_updates(cls, v: Any) -> Any:
        if v is None:
            return []
        if isinstance(v, dict):
            if not v:
                return []
            return [v]
        if isinstance(v, list):
            return v
        raise ValueError("quest_progress_updates must be a list (use [] if none)")

    @field_validator("inventory_delta", mode="before")
    @classmethod
    def _normalize_inventory_delta(cls, v: Any) -> Any:
        if v is None:
            return {}
        if isinstance(v, list):
            if not v:
                return {}
            result: Dict[str, int] = {}
            for item in v:
                if not isinstance(item, dict):
                    raise ValueError("inventory_delta list items must be objects")
                key = item.get("item") or item.get("name")
                delta = item.get("delta")
                if key is None or delta is None:
                    raise ValueError("inventory_delta list items must include item/name and delta")
                try:
                    result[str(key)] = int(float(delta))
                except Exception as exc:
                    raise ValueError("inventory_delta delta must be numeric") from exc
            return result
        if isinstance(v, dict):
            result: Dict[str, int] = {}
            for key, delta in v.items():
                try:
                    result[str(key)] = int(float(delta))
                except Exception as exc:
                    raise ValueError("inventory_delta delta must be numeric") from exc
            return result
        raise ValueError("inventory_delta must be an object (dict) or []")


class SafetyFlag(BaseModel):
    model_config = ConfigDict(extra="forbid")

    refuse: bool
    reason: Optional[str] = None


class TurnOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    narration: str
    npc_dialogue: List[NPCDialogueLine]
    world_updates: WorldUpdates
    memory_summary: str
    safety: SafetyFlag

    @model_validator(mode="before")
    @classmethod
    def _normalize_safety(cls, data: Any) -> Any:
        if isinstance(data, dict) and "safety" in data:
            safety_val = data["safety"]
            if isinstance(safety_val, bool):
                data = dict(data)
                data["safety"] = {"refuse": safety_val, "reason": None if not safety_val else "unspecified"}
        return data


class TurnLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    turn_id: int
    timestamp: str
    player_text: str
    location_id: str
    selected_npc_id: Optional[str] = None
    output: Dict[str, Any] = Field(default_factory=dict)
