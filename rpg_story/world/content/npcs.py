"""Lightweight helpers for NPC content."""
from __future__ import annotations

import re


def normalize_npc_id(npc_id: str) -> str:
    return re.sub(r"\s+", "_", npc_id.strip().lower())


def default_traits() -> list[str]:
    return ["curious", "guarded"]
