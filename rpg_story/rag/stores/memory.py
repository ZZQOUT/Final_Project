"""In-memory RAG store with simple token overlap scoring."""
from __future__ import annotations

from typing import Any, Dict, List
import re

from rpg_story.rag.types import Document
from rpg_story.rag.stores.base import BaseStore


class InMemoryStore(BaseStore):
    def __init__(self) -> None:
        self._docs: Dict[str, Document] = {}

    def add(self, docs: List[Document]) -> None:
        for doc in docs:
            if doc.id in self._docs:
                continue
            self._docs[doc.id] = doc

    def query(self, query_text: str, top_k: int, filters: Dict[str, Any]) -> List[Document]:
        candidates = [doc for doc in self._docs.values() if _matches_filters(doc, filters)]
        if not candidates or top_k <= 0:
            return []
        query_tokens = _tokenize(query_text)
        scored = []
        for doc in candidates:
            score = _score_doc(query_tokens, doc.text)
            scored.append((score, _timestamp_key(doc), doc))
        scored.sort(key=lambda item: (-item[0], item[1]))
        results = [doc for score, _ts, doc in scored if score > 0]
        return results[:top_k]

    def get(self, ids: List[str]) -> List[Document]:
        return [self._docs[doc_id] for doc_id in ids if doc_id in self._docs]

    def count(self) -> int:
        return len(self._docs)


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    lowered = text.lower()
    ascii_tokens = set(re.findall(r"[a-z0-9]+", lowered))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_tokens = set(cjk_chars)
    if len(cjk_chars) >= 2:
        for idx in range(len(cjk_chars) - 1):
            cjk_tokens.add(cjk_chars[idx] + cjk_chars[idx + 1])
    return ascii_tokens | cjk_tokens


def _score_doc(query_tokens: set[str], text: str) -> int:
    if not query_tokens:
        return 0
    doc_tokens = _tokenize(text)
    return len(query_tokens & doc_tokens)


def _timestamp_key(doc: Document) -> str:
    return str(doc.metadata.get("timestamp", ""))


def _matches_filters(doc: Document, filters: Dict[str, Any]) -> bool:
    for key, value in filters.items():
        if value is None:
            continue
        doc_val = doc.metadata.get(key)
        if isinstance(value, list):
            if doc_val not in value:
                return False
        else:
            if doc_val != value:
                return False
    return True
