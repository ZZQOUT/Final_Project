"""Store interface for RAG documents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from rpg_story.rag.types import Document


class BaseStore(ABC):
    @abstractmethod
    def add(self, docs: List[Document]) -> None:
        raise NotImplementedError

    @abstractmethod
    def query(self, query_text: str, top_k: int, filters: Dict[str, Any]) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    def get(self, ids: List[str]) -> List[Document]:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError
