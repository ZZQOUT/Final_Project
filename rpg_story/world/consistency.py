"""World consistency guard: inject rules + validate for anachronisms."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from rpg_story.llm.client import LLMClient
from rpg_story.world.schemas import WorldBibleRules


@dataclass
class WorldConsistencyGuard:
    bible: WorldBibleRules
    forbidden_terms: List[str]

    def inject_rules(self, prompt: str) -> str:
        rules = (
            f"World rules: tech_level={self.bible.tech_level}; "
            f"magic_rules={self.bible.magic_rules}; "
            f"tone={self.bible.tone}; "
            f"taboos={self.bible.taboos}; "
            f"do_not_mention={self.bible.do_not_mention}."
        )
        return rules + "\n" + prompt

    def has_violation(self, text: str) -> bool:
        lowered = text.lower()
        return any(term.lower() in lowered for term in self.forbidden_terms)

    def repair(self, llm: LLMClient, text: str) -> str:
        """Ask the LLM to rewrite text to fit world rules."""
        return llm.rewrite_to_world(text, self.bible)
