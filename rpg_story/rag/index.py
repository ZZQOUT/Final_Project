"""RAG index wrapper."""
from __future__ import annotations

from typing import List

from rpg_story.models.world import WorldSpec
from rpg_story.rag.types import Document, dedupe_docs
from rpg_story.rag.stores.base import BaseStore
from rpg_story.rag.sources.world_docs import build_world_bible_doc
from rpg_story.rag.sources.location_docs import build_location_doc
from rpg_story.rag.sources.npc_docs import build_npc_profile_doc


class RAGIndex:
    def __init__(self, store: BaseStore) -> None:
        self.store = store

    def upsert(self, docs: List[Document]) -> None:
        if not docs:
            return
        self.store.add(dedupe_docs(docs))

    def build_default(self, session_id: str, world: WorldSpec) -> None:
        docs: List[Document] = [build_world_bible_doc(world, session_id)]
        for loc in world.locations:
            docs.append(build_location_doc(world, loc.location_id, session_id))
        for npc in world.npcs:
            docs.append(build_npc_profile_doc(world, npc.npc_id, session_id))
        self.upsert(docs)
