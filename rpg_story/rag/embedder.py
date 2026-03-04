"""Embedding backends for RAG retrieval."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Tuple
import hashlib
import math
import os
import re

from rpg_story.config import AppConfig

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency behavior
    OpenAI = None  # type: ignore


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_many(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError


@dataclass(frozen=True)
class HashingEmbedder(BaseEmbedder):
    embedding_dim: int = 384

    @property
    def name(self) -> str:
        return "hashing"

    @property
    def dim(self) -> int:
        return int(self.embedding_dim)

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vectors.append(_hash_embed(text or "", self.dim))
        return vectors


class OpenAICompatibleEmbedder(BaseEmbedder):
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str,
        dim_hint: int = 1536,
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package is required for OpenAICompatibleEmbedder")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._dim_hint = int(dim_hint)

    @property
    def name(self) -> str:
        return "openai_compatible"

    @property
    def dim(self) -> int:
        return self._dim_hint

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        data = sorted(response.data, key=lambda item: int(item.index))
        vectors = [list(item.embedding) for item in data]
        if vectors:
            self._dim_hint = len(vectors[0])
        return [_l2_normalize(vec) for vec in vectors]


def make_embedder(cfg: AppConfig) -> Tuple[BaseEmbedder, str]:
    provider = str(getattr(cfg.rag, "embedding_provider", "hashing") or "hashing").lower()
    dim = int(getattr(cfg.rag, "embedding_dim", 384) or 384)
    if provider in {"openai", "openai_compatible"}:
        model = str(getattr(cfg.rag, "embedding_model", "") or "").strip()
        api_key = os.getenv(cfg.llm.api_key_env or "DASHSCOPE_API_KEY", "") or cfg.llm.api_key
        if model and api_key and OpenAI is not None:
            try:
                return (
                    OpenAICompatibleEmbedder(
                        model=model,
                        base_url=cfg.llm.base_url,
                        api_key=api_key,
                        dim_hint=dim,
                    ),
                    "openai_compatible",
                )
            except Exception:
                # Fall back to deterministic local embeddings to keep gameplay available.
                pass
    return HashingEmbedder(embedding_dim=dim), "hashing"


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for av, bv in zip(a, b):
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def _hash_embed(text: str, dim: int) -> List[float]:
    if dim <= 0:
        return []
    vec = [0.0] * dim
    if not text:
        return vec
    counts = {}
    for token in _tokenize(text):
        counts[token] = counts.get(token, 0) + 1
    for token, tf in counts.items():
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        h = int(digest, 16)
        idx = h % dim
        sign = -1.0 if ((h >> 11) & 1) else 1.0
        vec[idx] += sign * (1.0 + math.log1p(tf))
    return _l2_normalize(vec)


def _tokenize(text: str) -> Iterable[str]:
    lowered = text.lower()
    for token in re.findall(r"[a-z0-9_]+", lowered):
        yield token
    cjk = re.findall(r"[\u4e00-\u9fff]", text)
    for ch in cjk:
        yield ch
    for idx in range(0, max(0, len(cjk) - 1)):
        yield cjk[idx] + cjk[idx + 1]


def _l2_normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= 0.0:
        return vec
    return [v / norm for v in vec]
