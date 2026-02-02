"""World generation pipeline using an API LLM."""
from __future__ import annotations

from typing import Any, Dict

from rpg_story.llm.client import LLMClient
from rpg_story.world.schemas import WorldSpec


class WorldGenPipeline:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def generate(self, world_prompt: str) -> WorldSpec:
        """Generate a world spec from a user prompt.

        Strategy:
        - call LLM for strict JSON
        - if invalid, attempt fix-json
        - if still invalid, return minimal default
        """
        raw = self.llm.generate_world(world_prompt)
        try:
            return WorldSpec.model_validate(raw)
        except Exception:
            fixed = self.llm.fix_json(raw, schema_name="WorldSpec")
            try:
                return WorldSpec.model_validate(fixed)
            except Exception:
                return WorldSpec.minimal_default()
