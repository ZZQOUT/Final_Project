"""Chunking utilities for RAG documents."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List
import re

from rpg_story.rag.types import Document, make_doc_id, normalize_metadata


@dataclass(frozen=True)
class ChunkConfig:
    max_chars: int = 700
    overlap_chars: int = 120
    min_chunk_chars: int = 80


def chunk_text(text: str, cfg: ChunkConfig | None = None) -> List[str]:
    cfg = cfg or ChunkConfig()
    cleaned = _normalize_text(text)
    if not cleaned:
        return []
    if len(cleaned) <= cfg.max_chars:
        return [cleaned]

    chunks: List[str] = []
    start = 0
    total = len(cleaned)
    while start < total:
        hard_end = min(total, start + cfg.max_chars)
        end = _find_soft_boundary(cleaned, start, hard_end)
        if end <= start:
            end = hard_end
        piece = cleaned[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= total:
            break
        # Keep contextual overlap to preserve continuity.
        next_start = max(0, end - max(0, cfg.overlap_chars))
        if next_start <= start:
            next_start = end
        start = next_start

    # Merge tiny trailing chunks to avoid noisy retrieval units.
    if len(chunks) >= 2 and len(chunks[-1]) < cfg.min_chunk_chars:
        chunks[-2] = (chunks[-2].rstrip() + " " + chunks[-1].lstrip()).strip()
        chunks.pop()
    return chunks


def chunk_document(
    doc: Document,
    cfg: ChunkConfig | None = None,
) -> List[Document]:
    cfg = cfg or ChunkConfig()
    parts = chunk_text(doc.text, cfg)
    if len(parts) <= 1:
        return [doc]

    output: List[Document] = []
    total = len(parts)
    for idx, part in enumerate(parts):
        metadata = dict(doc.metadata)
        metadata.update(
            {
                "parent_id": doc.id,
                "chunk_index": idx,
                "chunk_count": total,
            }
        )
        normalized = normalize_metadata(metadata, strict=True)
        chunk_id = make_doc_id(normalized, part)
        output.append(Document(id=chunk_id, text=part, metadata=normalized))
    return output


def chunk_documents(
    docs: Iterable[Document],
    cfg: ChunkConfig | None = None,
) -> List[Document]:
    cfg = cfg or ChunkConfig()
    result: List[Document] = []
    for doc in docs:
        result.extend(chunk_document(doc, cfg))
    return result


def _normalize_text(text: str) -> str:
    if not text:
        return ""
    # Keep paragraph signal but collapse repeated blank lines.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _find_soft_boundary(text: str, start: int, hard_end: int) -> int:
    if hard_end >= len(text):
        return len(text)
    window_start = max(start, hard_end - 180)
    snippet = text[window_start:hard_end]
    # Prefer paragraph/sentence boundaries near the hard limit.
    matches = list(re.finditer(r"[\n。！？!?；;.!?]", snippet))
    if not matches:
        return hard_end
    return window_start + matches[-1].start() + 1
