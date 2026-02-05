"""NPC profile document builders."""
from __future__ import annotations

from rpg_story.models.world import WorldSpec
from rpg_story.rag.types import Document, make_doc_id, normalize_metadata


def build_npc_profile_doc(world: WorldSpec, npc_id: str, session_id: str) -> Document:
    npc = next((n for n in world.npcs if n.npc_id == npc_id), None)
    if npc is None:
        text = f"Unknown NPC: {npc_id}"
    else:
        parts = [
            f"NPC: {npc.name} ({npc.profession})",
            f"Traits: {', '.join(npc.traits)}",
            f"Goals: {', '.join(npc.goals)}",
            f"Starting location: {npc.starting_location}",
            f"Obedience: {npc.obedience_level}",
            f"Stubbornness: {npc.stubbornness}",
            f"Risk tolerance: {npc.risk_tolerance}",
            f"Disposition to player: {npc.disposition_to_player}",
            f"Refusal style: {npc.refusal_style}",
        ]
        text = "\n".join([p for p in parts if p])

    metadata = normalize_metadata(
        {
            "doc_type": "npc_profile",
            "session_id": session_id,
            "npc_id": npc_id,
        }
    )
    return Document(id=make_doc_id(metadata, text), text=text, metadata=metadata)
