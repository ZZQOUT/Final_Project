"""Persistence helpers for sessions (Milestone 1)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import os
import re
from datetime import datetime
import secrets

from rpg_story.config import AppConfig
from rpg_story.models.world import GameState

_SAFE_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_session_id(session_id: str) -> None:
    """Validate session_id to prevent path traversal."""
    if not session_id:
        raise ValueError("session_id must not be empty")
    if "/" in session_id or "\\" in session_id:
        raise ValueError("session_id must not contain path separators")
    if ".." in session_id:
        raise ValueError("session_id must not contain '..'")
    if not _SAFE_SESSION_RE.match(session_id):
        raise ValueError("session_id contains invalid characters")


def generate_session_id() -> str:
    """Generate a filesystem-safe session id: YYYYMMDD_HHMMSS_<6-8hex>."""
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(4)  # 8 hex chars
    return f"{timestamp}_{suffix}"


def default_sessions_root(config: Optional[AppConfig]) -> Path:
    """Resolve sessions root from config or return data/sessions."""
    if config is None:
        return Path("data/sessions")
    return config.app.sessions_dir


def get_session_dir(session_id: str, sessions_root: Path) -> Path:
    """Return the session directory path without creating it."""
    validate_session_id(session_id)
    return sessions_root / session_id


def ensure_session_dir(session_id: str, sessions_root: Path) -> Path:
    """Create the session directory if missing, return its path."""
    validate_session_id(session_id)
    path = get_session_dir(session_id, sessions_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_state(session_id: str, state: GameState, sessions_root: Path) -> Path:
    """Atomically save GameState to state.json and return its path."""
    validate_session_id(session_id)
    session_dir = ensure_session_dir(session_id, sessions_root)
    target = session_dir / "state.json"
    tmp = session_dir / "state.json.tmp"
    data = state.model_dump()
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, target)
    return target


def load_state(session_id: str, sessions_root: Path) -> GameState:
    """Load GameState from state.json. Raise FileNotFoundError if missing."""
    validate_session_id(session_id)
    path = get_session_dir(session_id, sessions_root) / "state.json"
    if not path.exists():
        raise FileNotFoundError(f"state.json not found for session_id={session_id}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"state.json is invalid JSON for session_id={session_id}") from exc
    return GameState.model_validate(raw)


def append_turn_log(session_id: str, record: Dict[str, Any], sessions_root: Path) -> Path:
    """Append one JSON record to turns.jsonl and return its path."""
    validate_session_id(session_id)
    session_dir = ensure_session_dir(session_id, sessions_root)
    path = session_dir / "turns.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def read_turn_logs(session_id: str, sessions_root: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    """Read turns.jsonl into list of dicts, skipping corrupt lines."""
    validate_session_id(session_id)
    path = get_session_dir(session_id, sessions_root) / "turns.jsonl"
    if not path.exists():
        return []
    results: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(results) >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return results


def append_story_summary(record: Dict[str, Any], sessions_root: Path) -> Path:
    """Append one story summary record to stories.jsonl under the data root."""
    root = Path(sessions_root)
    history_path = root.parent / "stories.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return history_path


def read_story_summaries(sessions_root: Path, limit: int | None = 50) -> List[Dict[str, Any]]:
    """Read stories.jsonl records in reverse chronological order."""
    root = Path(sessions_root)
    history_path = root.parent / "stories.jsonl"
    if not history_path.exists():
        return []
    records: List[Dict[str, Any]] = []
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    records.reverse()
    if limit is not None:
        return records[: max(0, int(limit))]
    return records
