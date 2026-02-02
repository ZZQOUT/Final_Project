"""LLM API client wrapper (OpenAI-compatible)."""
from __future__ import annotations

from typing import Any, Dict

from rpg_story.config import Config
from rpg_story.world.schemas import WorldBibleRules


class LLMClient:
    def __init__(self, config: Config) -> None:
        self.config = config
        # TODO: initialize OpenAI-compatible client using base_url + api_key

    def generate_world(self, world_prompt: str) -> Dict[str, Any]:
        # TODO: call LLM API for world generation (strict JSON)
        return {}

    def generate_turn(self, prompt_payload: Dict[str, Any]) -> Dict[str, Any]:
        # TODO: call LLM API for turn output (strict JSON)
        return {
            "narration": "(stub response)",
            "npc_dialogue": [],
            "world_updates": {},
            "memory_summary": "",
            "safety": {"refusal": False, "reason": ""},
        }

    def fix_json(self, raw: Any, schema_name: str) -> Dict[str, Any]:
        # TODO: call LLM with fix-json prompt
        return {}

    def rewrite_to_world(self, text: str, bible: WorldBibleRules) -> str:
        # TODO: ask LLM to rewrite text to fit world rules
        return text
