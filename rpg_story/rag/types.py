"""RAG document primitives and helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import hashlib


ALLOWED_METADATA_KEYS = {
    "doc_type",
    "session_id",
    "location_id",
    "npc_id",
    "turn_id",
    "timestamp",
    "tags",
}

KNOWN_DOC_TYPES = {
    "world_bible",
    "location",
    "npc_profile",
    "summary",
    "memory",
}

REQUIRED_KEYS_BASE = {"doc_type", "session_id"}
REQUIRED_KEYS_BY_TYPE = {
    "summary": {"turn_id", "timestamp"},
    "memory": {"turn_id", "timestamp"},
    "location": {"location_id"},
    "npc_profile": {"npc_id"},
    "world_bible": set(),
}

@dataclass(frozen=True)
class Document:
    id: str
    text: str
    metadata: Dict[str, Any]


def normalize_metadata(metadata: Dict[str, Any], *, strict: bool = True) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key in ALLOWED_METADATA_KEYS:
        value = metadata.get(key)
        if value is None:
            continue
        normalized[key] = value
    if strict:
        doc_type = normalized.get("doc_type")
        if doc_type not in KNOWN_DOC_TYPES:
            raise ValueError(f"Unknown doc_type: {doc_type}")
        _require_keys(normalized, REQUIRED_KEYS_BASE)
        extra_required = REQUIRED_KEYS_BY_TYPE.get(doc_type, set())
        _require_keys(normalized, extra_required)
    return normalized


def dedupe_docs(docs: list[Document]) -> list[Document]:
    seen = set()
    result: list[Document] = []
    for doc in docs:
        if doc.id in seen:
            continue
        seen.add(doc.id)
        result.append(doc)
    return result


def _require_keys(metadata: Dict[str, Any], keys: set[str]) -> None:
    for key in keys:
        value = metadata.get(key)
        if value is None or value == "":
            raise ValueError(f"Missing required metadata key: {key}")


def make_doc_id(metadata: Dict[str, Any], text: str) -> str:
    session_id = metadata.get("session_id", "-")
    doc_type = metadata.get("doc_type", "-")
    npc_id = metadata.get("npc_id", "-")
    location_id = metadata.get("location_id", "-")
    turn_id = metadata.get("turn_id", "-")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return f"{session_id}:{doc_type}:{npc_id}:{location_id}:{turn_id}:{digest}"
