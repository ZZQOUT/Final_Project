"""Memory document builders from turn logs."""
from __future__ import annotations

from typing import List

from rpg_story.persistence.store import read_turn_logs
from rpg_story.rag.types import Document, make_doc_id, normalize_metadata


def _memory_text_from_record(record: dict) -> str:
    player_text = record.get("player_text", "")
    output = record.get("output", {})
    narration = output.get("narration", "")
    summary = output.get("memory_summary", "")
    move_rejections = record.get("move_rejections", [])
    move_refusals = record.get("move_refusals", [])
    npc_dialogue = output.get("npc_dialogue", [])

    parts = []
    if player_text:
        parts.append(f"Player: {player_text}")
    for line in npc_dialogue:
        npc_id = line.get("npc_id", "")
        text = line.get("text", "")
        if text:
            prefix = f"NPC[{npc_id}]" if npc_id else "NPC"
            parts.append(f"{prefix}: {text}")
    if narration:
        parts.append(f"Narration: {narration}")
    if summary:
        parts.append(f"Summary: {summary}")
    if move_rejections:
        for event in move_rejections:
            npc_id = event.get("npc_id", "")
            reason = event.get("reason", "rejected")
            to_loc = event.get("to_location", "")
            parts.append(f"Move rejected: npc={npc_id} to={to_loc} reason={reason}")
    if move_refusals:
        for event in move_refusals:
            npc_id = event.get("npc_id", "")
            reason = event.get("reason", "refused")
            parts.append(f"Move refused: npc={npc_id} reason={reason}")
    return "\n".join([p for p in parts if p])


def build_memory_docs_from_turn_logs(
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
                "npc_id": record.get("npc_id"),
            }
        )
        doc_id = make_doc_id(metadata, text)
        docs.append(Document(id=doc_id, text=text, metadata=metadata))
    return docs
