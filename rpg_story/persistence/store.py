"""Persistence helpers for sessions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from rpg_story.engine.state import GameState


def session_dir(session_id: str) -> Path:
    return Path("data/sessions") / session_id


def save_state(state: GameState) -> Path:
    sdir = session_dir(state.session_id)
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / "state.json"
    state.save(path)
    return path


def load_state(session_id: str) -> GameState:
    path = session_dir(session_id) / "state.json"
    return GameState.load(path)


def append_turn_log(session_id: str, record: Dict[str, Any]) -> Path:
    sdir = session_dir(session_id)
    sdir.mkdir(parents=True, exist_ok=True)
    path = sdir / "turns.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
