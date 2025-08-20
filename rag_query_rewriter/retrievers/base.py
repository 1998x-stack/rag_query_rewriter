"""Retriever interfaces for BM25/vector/hybrid backends."""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class SearchResult:
    """检索结果容器。"""
    doc_id: str
    score: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return {"doc_id": self.doc_id, "score": self.score, "text": self.text}


class Retriever(ABC):
    """统一检索接口。"""

    @abstractmethod
    def search(self, query: str, topk: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """执行检索并返回结果列表。"""
        raise NotImplementedError
