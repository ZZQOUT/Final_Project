"""Summary document builders from turn logs."""
from __future__ import annotations

from typing import List

from rpg_story.persistence.store import read_turn_logs
from rpg_story.rag.types import Document, make_doc_id, normalize_metadata


def build_summary_docs_from_turn_logs(
    session_id: str,
    sessions_root,
    limit: int,
) -> List[Document]:
    if limit <= 0:
        return []
    logs = read_turn_logs(session_id, sessions_root)
    if not logs:
        return []
    tail = logs[-limit:]
    docs: List[Document] = []
    for record in tail:
        output = record.get("output", {})
        summary = output.get("memory_summary") or ""
        turn_id = record.get("turn_index")
        timestamp = record.get("timestamp")
        if not summary or turn_id is None or not timestamp:
            continue
        metadata = normalize_metadata(
            {
                "doc_type": "summary",
                "session_id": session_id,
                "turn_id": turn_id,
                "timestamp": timestamp,
            }
        )
        doc_id = make_doc_id(metadata, summary)
        docs.append(Document(id=doc_id, text=summary, metadata=metadata))
    return docs
