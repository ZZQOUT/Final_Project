"""RAG retriever and forced context pack builder."""
from __future__ import annotations

from typing import Any, Dict, List

from rpg_story.models.world import GameState, WorldSpec
from rpg_story.rag.index import RAGIndex
from rpg_story.rag.types import Document
from rpg_story.rag.sources.world_docs import build_world_bible_doc
from rpg_story.rag.sources.location_docs import build_location_doc
from rpg_story.rag.sources.npc_docs import build_npc_profile_doc
from rpg_story.rag.sources.summaries import build_summary_docs_from_turn_logs
from rpg_story.rag.sources.memories import build_memory_docs_from_turn_logs
from rpg_story.rag.sources.npc_memories import build_npc_memory_docs_from_turn_logs
from rpg_story.rag.types import dedupe_docs


class RAGRetriever:
    def __init__(self, index: RAGIndex) -> None:
        self.index = index

    def get_forced_context_pack(
        self,
        session_id: str,
        world: WorldSpec,
        state: GameState,
        npc_id: str | None,
        sessions_root,
        last_n_summaries: int,
        top_k: int,
        query_text: str,
    ) -> Dict[str, Any]:
        always_include: List[Document] = []
        world_doc = build_world_bible_doc(world, session_id)
        always_include.append(world_doc)

        location_doc = build_location_doc(world, state.player_location, session_id)
        always_include.append(location_doc)

        if npc_id:
            npc_doc = build_npc_profile_doc(world, npc_id, session_id)
            always_include.append(npc_doc)
            npc_memory_docs = build_npc_memory_docs_from_turn_logs(
                session_id,
                sessions_root,
                npc_id,
                limit=min(last_n_summaries, 3),
            )
            always_include.extend(npc_memory_docs)

        summary_docs = build_summary_docs_from_turn_logs(session_id, sessions_root, last_n_summaries)
        always_include.extend(summary_docs)

        memory_docs = build_memory_docs_from_turn_logs(session_id, sessions_root, top_k * 3)
        self.index.upsert(dedupe_docs(memory_docs))

        retrieved, tier_filters = self._retrieve_with_fallbacks(
            session_id=session_id,
            npc_id=npc_id,
            location_id=state.player_location,
            query_text=query_text,
            top_k=top_k,
        )

        always_include = _dedupe(always_include)
        retrieved = _dedupe([doc for doc in retrieved if doc.id not in {d.id for d in always_include}])

        debug = {
            "filters": tier_filters,
            "counts": {
                "always_include": len(always_include),
                "retrieved": len(retrieved),
                "store_total": self.index.store.count(),
            },
        }
        return {
            "always_include": always_include,
            "retrieved": retrieved,
            "debug": debug,
        }

    def _retrieve_with_fallbacks(
        self,
        session_id: str,
        npc_id: str | None,
        location_id: str,
        query_text: str,
        top_k: int,
    ) -> tuple[List[Document], List[dict]]:
        doc_type_filter = ["memory", "summary"]
        tiers = []
        if npc_id:
            tiers.append({"session_id": session_id, "npc_id": npc_id, "doc_type": doc_type_filter})
            tiers.append({"session_id": session_id, "location_id": location_id, "doc_type": doc_type_filter})
            tiers.append({"session_id": session_id, "doc_type": doc_type_filter})
        else:
            tiers.append({"session_id": session_id, "location_id": location_id, "doc_type": doc_type_filter})
            tiers.append({"session_id": session_id, "doc_type": doc_type_filter})

        results: List[Document] = []
        seen = set()
        for filters in tiers:
            if len(results) >= top_k:
                break
            batch = self.index.store.query(query_text, top_k, filters)
            for doc in batch:
                if doc.id in seen:
                    continue
                seen.add(doc.id)
                results.append(doc)
                if len(results) >= top_k:
                    break
        return results, tiers


def _dedupe(docs: List[Document]) -> List[Document]:
    seen = set()
    result: List[Document] = []
    for doc in docs:
        if doc.id in seen:
            continue
        seen.add(doc.id)
        result.append(doc)
    return result
