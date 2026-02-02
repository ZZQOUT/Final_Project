"""Schema validation tests."""
from __future__ import annotations

from rpg_story.llm.schemas import validate_llm_output


def test_schema_minimal():
    payload = {
        "narration": "ok",
        "npc_dialogue": [],
        "world_updates": {},
        "memory_summary": "",
        "safety": {"refusal": False, "reason": ""},
    }
    validate_llm_output(payload)
