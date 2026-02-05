"""NPC-specific memory document builders from turn logs."""
from __future__ import annotations

from typing import List

from rpg_story.persistence.store import read_turn_logs
from rpg_story.rag.types import Document, make_doc_id, normalize_metadata
from rpg_story.rag.sources.memories import _memory_text_from_record


def build_npc_memory_docs_from_turn_logs(
    session_id: str,
    sessions_root,
    npc_id: str,
    limit: int,
) -> List[Document]:
    if limit <= 0:
        return []
    logs = read_turn_logs(session_id, sessions_root)
    if not logs:
        return []
    filtered = [record for record in logs if record.get("npc_id") == npc_id]
    if not filtered:
        return []
    tail = filtered[-limit:]
    docs: List[Document] = []
    for record in tail:
        turn_id = record.get("turn_index")
        timestamp = record.get("timestamp")
        if turn_id is None or not timestamp:
            continue
        text = _memory_text_from_record(record)
        if not text:
            continue
        metadata = normalize_metadata(
            {
                "doc_type": "memory",
                "session_id": session_id,
                "turn_id": turn_id,
                "timestamp": timestamp,
                "location_id": record.get("location_id"),
                "npc_id": npc_id,
            }
        )
        doc_id = make_doc_id(metadata, text)
        docs.append(Document(id=doc_id, text=text, metadata=metadata))
    return docs
