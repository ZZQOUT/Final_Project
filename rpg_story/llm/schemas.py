"""LLM output contracts only (do not duplicate world models here)."""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict

from rpg_story.models.turn import TurnOutput


def validate_turn_output(data: dict) -> TurnOutput:
    return TurnOutput.model_validate(data)
