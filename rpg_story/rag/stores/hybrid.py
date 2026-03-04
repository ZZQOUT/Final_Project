"""Persistent hybrid RAG store (vector + lexical)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from datetime import datetime
import json
import math
import re

from rpg_story.rag.embedder import BaseEmbedder, cosine_similarity
from rpg_story.rag.types import Document
from rpg_story.rag.stores.base import BaseStore


class PersistentHybridStore(BaseStore):
    """File-backed store with hybrid retrieval scoring."""

    VERSION = 1

    def __init__(
        self,
        store_path: Path,
        *,
        embedder: BaseEmbedder,
        lexical_weight: float = 0.35,
        vector_weight: float = 0.60,
        recency_weight: float = 0.05,
        min_score: float = 0.03,
    ) -> None:
        self._path = _resolve_store_file(store_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._lexical_weight = max(0.0, float(lexical_weight))
        self._vector_weight = max(0.0, float(vector_weight))
        self._recency_weight = max(0.0, float(recency_weight))
        self._min_score = max(0.0, float(min_score))
        self._docs: Dict[str, Document] = {}
        self._vectors: Dict[str, List[float]] = {}
        self._token_cache: Dict[str, set[str]] = {}
        self._load()

    def add(self, docs: List[Document]) -> None:
        if not docs:
            return
        new_docs = [doc for doc in docs if doc.id not in self._docs]
        if not new_docs:
            return

        for doc in new_docs:
            self._docs[doc.id] = doc
            self._token_cache[doc.id] = _tokenize(doc.text)

        vectors = self._embed_many_safe([doc.text for doc in new_docs])
        for doc, vec in zip(new_docs, vectors):
            if vec:
                self._vectors[doc.id] = vec

        self._flush()

    def query(self, query_text: str, top_k: int, filters: Dict[str, Any]) -> List[Document]:
        if top_k <= 0:
            return []
        candidates = [doc for doc in self._docs.values() if _matches_filters(doc, filters)]
        if not candidates:
            return []

        query_tokens = _tokenize(query_text or "")
        query_vector = self._embed_one_safe(query_text or "")
        recency_scores = _recency_norm_map(candidates)
        scored = []
        for doc in candidates:
            doc_tokens = self._token_cache.get(doc.id)
            if doc_tokens is None:
                doc_tokens = _tokenize(doc.text)
                self._token_cache[doc.id] = doc_tokens

            lexical = _lexical_score(query_tokens, doc_tokens)
            vector = 0.0
            vec = self._vectors.get(doc.id, [])
            if query_vector and vec:
                vector = max(0.0, cosine_similarity(query_vector, vec))
            recency = recency_scores.get(doc.id, 0.0)
            hybrid = (
                self._lexical_weight * lexical
                + self._vector_weight * vector
                + self._recency_weight * recency
            )
            if hybrid < self._min_score and lexical <= 0.0 and vector <= 0.0:
                continue
            metadata = dict(doc.metadata)
            metadata["score_lexical"] = round(lexical, 6)
            metadata["score_vector"] = round(vector, 6)
            metadata["score_hybrid"] = round(hybrid, 6)
            scored.append((hybrid, recency, Document(id=doc.id, text=doc.text, metadata=metadata)))

        scored.sort(key=lambda item: (-item[0], -item[1], item[2].id))
        return [doc for _score, _recency, doc in scored[:top_k]]

    def get(self, ids: List[str]) -> List[Document]:
        return [self._docs[doc_id] for doc_id in ids if doc_id in self._docs]

    def count(self) -> int:
        return len(self._docs)

    @property
    def backend_name(self) -> str:
        return "persistent_hybrid"

    def _embed_many_safe(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            return self._embedder.embed_many(texts)
        except Exception:
            return [[] for _ in texts]

    def _embed_one_safe(self, text: str) -> List[float]:
        if not text:
            return []
        try:
            out = self._embedder.embed_many([text])
            return out[0] if out else []
        except Exception:
            return []

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return
        if not isinstance(payload, dict):
            return
        rows = payload.get("docs", [])
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                doc_id = row.get("id")
                text = row.get("text")
                metadata = row.get("metadata")
                if not isinstance(doc_id, str) or not isinstance(text, str) or not isinstance(metadata, dict):
                    continue
                doc = Document(id=doc_id, text=text, metadata=metadata)
                self._docs[doc.id] = doc
                self._token_cache[doc.id] = _tokenize(doc.text)
        vectors = payload.get("vectors", {})
        if isinstance(vectors, dict):
            for doc_id, vec in vectors.items():
                if not isinstance(doc_id, str) or not isinstance(vec, list):
                    continue
                if all(isinstance(v, (int, float)) for v in vec):
                    self._vectors[doc_id] = [float(v) for v in vec]

    def _flush(self) -> None:
        rows = [
            {
                "id": doc.id,
                "text": doc.text,
                "metadata": doc.metadata,
            }
            for doc in self._docs.values()
        ]
        payload = {
            "version": self.VERSION,
            "docs": rows,
            "vectors": self._vectors,
        }
        self._path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _resolve_store_file(store_path: Path) -> Path:
    path = Path(store_path)
    if path.suffix.lower() == ".json":
        return path
    return path / "hybrid_store.json"


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    lowered = text.lower()
    ascii_tokens = set(re.findall(r"[a-z0-9_]+", lowered))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_tokens = set(cjk_chars)
    if len(cjk_chars) >= 2:
        for idx in range(len(cjk_chars) - 1):
            cjk_tokens.add(cjk_chars[idx] + cjk_chars[idx + 1])
    return ascii_tokens | cjk_tokens


def _lexical_score(query_tokens: set[str], doc_tokens: set[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    inter = len(query_tokens & doc_tokens)
    if inter <= 0:
        return 0.0
    # Cosine-like normalization for token sets.
    return inter / math.sqrt(len(query_tokens) * len(doc_tokens))


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


def _recency_norm_map(docs: List[Document]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for doc in docs:
        values[doc.id] = _timestamp_to_ordinal(doc.metadata.get("timestamp"))
    if not values:
        return {}
    min_v = min(values.values())
    max_v = max(values.values())
    if max_v <= min_v:
        return {doc_id: 0.0 for doc_id in values.keys()}
    return {doc_id: (val - min_v) / (max_v - min_v) for doc_id, val in values.items()}


def _timestamp_to_ordinal(raw: Any) -> float:
    text = str(raw or "").strip()
    if not text:
        return 0.0
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return datetime.fromisoformat(text).timestamp()
    except Exception:
        # Keep deterministic fallback for non-ISO timestamps used in tests (e.g. "t1").
        digits = re.findall(r"\d+", text)
        if digits:
            return float("".join(digits))
        return 0.0
