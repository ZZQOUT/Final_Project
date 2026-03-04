"""Lore document builders for external knowledge ingestion."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from rpg_story.rag.types import Document, make_doc_id, normalize_metadata


def build_lore_doc(
    *,
    session_id: str,
    text: str,
    source: str,
    tags: list[str] | None = None,
) -> Document:
    normalized_text = (text or "").strip()
    metadata = normalize_metadata(
        {
            "doc_type": "lore",
            "session_id": session_id,
            "source": source,
            "tags": tags or [],
        }
    )
    return Document(
        id=make_doc_id(metadata, normalized_text),
        text=normalized_text,
        metadata=metadata,
    )


def build_lore_docs_from_paths(
    *,
    session_id: str,
    paths: Iterable[Path],
    tags: list[str] | None = None,
) -> List[Document]:
    docs: List[Document] = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue
        docs.append(
            build_lore_doc(
                session_id=session_id,
                text=text,
                source=str(path),
                tags=tags,
            )
        )
    return docs
