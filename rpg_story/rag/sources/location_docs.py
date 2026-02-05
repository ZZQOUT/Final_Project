"""Location document builders."""
from __future__ import annotations

from rpg_story.models.world import WorldSpec
from rpg_story.rag.types import Document, make_doc_id, normalize_metadata


def build_location_doc(world: WorldSpec, location_id: str, session_id: str) -> Document:
    loc = world.get_location(location_id)
    if loc is None:
        text = f"Unknown location: {location_id}"
    else:
        parts = [
            f"Location: {loc.name}",
            f"Kind: {loc.kind}",
            f"Description: {loc.description}",
        ]
        if loc.tags:
            parts.append("Tags: " + ", ".join(loc.tags))
        text = "\n".join([p for p in parts if p])

    metadata = normalize_metadata(
        {
            "doc_type": "location",
            "session_id": session_id,
            "location_id": location_id,
        }
    )
    return Document(id=make_doc_id(metadata, text), text=text, metadata=metadata)
