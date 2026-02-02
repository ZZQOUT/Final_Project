"""Retrieval logic with strict injection policy and metadata."""
from __future__ import annotations

from typing import Dict, List


class RetrievalPolicy:
    """Defines required docs per turn."""

    def __init__(self, recent_turns: int = 3, top_k: int = 5) -> None:
        self.recent_turns = recent_turns
        self.top_k = top_k


class Retriever:
    def __init__(self, policy: RetrievalPolicy) -> None:
        self.policy = policy
        # TODO: initialize vector store client

    def fetch_always_include(self, session_id: str, location_id: str, npc_id: str | None) -> List[Dict]:
        """Always include world bible, location doc, and npc profile."""
        # TODO: query vector store by doc_type + metadata
        # doc_type: world_bible, location, npc_profile
        return []

    def fetch_recent_turn_summaries(self, session_id: str) -> List[Dict]:
        """Always include last N turn summaries."""
        # TODO: get last N turn_summary docs
        return []

    def fetch_optional_memories(self, session_id: str, location_id: str, npc_id: str | None) -> List[Dict]:
        """Optional top_k memories filtered by location_id and npc_id."""
        # TODO: similarity search with filters
        return []

    def build_context(self, session_id: str, location_id: str, npc_id: str | None) -> List[Dict]:
        required = self.fetch_always_include(session_id, location_id, npc_id)
        recent = self.fetch_recent_turn_summaries(session_id)
        optional = self.fetch_optional_memories(session_id, location_id, npc_id)
        return required + recent + optional
