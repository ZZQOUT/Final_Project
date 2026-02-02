"""Turn result container."""
from __future__ import annotations

from dataclasses import dataclass
from rpg_story.engine.state import GameState


@dataclass
class TurnResult:
    narration: str
    updated_state: GameState
    raw_response: dict
