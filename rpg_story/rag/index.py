"""Vector store initialization and canonical doc ingestion."""
from __future__ import annotations

from typing import Dict, List


class RAGIndex:
    def __init__(self) -> None:
        # TODO: initialize Chroma or other vector store
        pass

    def ingest_world_docs(self, session_id: str, docs: List[Dict]) -> None:
        """Ingest world bible, locations, and NPC profiles as canonical docs."""
        # TODO: upsert docs with metadata: doc_type, location_id, npc_id, session_id
        pass

    def ingest_turn_summary(self, session_id: str, doc: Dict) -> None:
        # TODO: upsert summary doc
        pass
