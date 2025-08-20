"""A mock retriever for demonstration and PRF testing (filter-aware)."""
from __future__ import annotations

from typing import List, Dict, Any, Optional
from .base import Retriever, SearchResult


class MockRetriever(Retriever):
    """演示用检索器：静态文档 + 简单计分，支持 Self-Query filters."""

    def __init__(self) -> None:
        # 简易“文档库”，附元数据供 filters 使用
        self._docs = {
            "d1": {"text": "2023 年版本发布记录，包含特性与时间线。", "year": "2023", "type": "release"},
            "d2": {"text": "2024 年次版本更新说明，修复若干问题并优化性能。", "year": "2024", "type": "release"},
            "d3": {"text": "技术规格总览：接口、限额、兼容性。", "year": "2022", "type": "spec"},
            "d4": {"text": "常见问题与解答（FAQ）。", "year": "2023", "type": "faq"},
            "d5": {"text": "对比文档：2023 与 2024 差异分析。", "year": "2024", "type": "compare"},
        }

    def _passes_filters(self, meta: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        if not filters:
            return True
        must = filters.get("must_filters") or {}
        for k, vals in must.items():
            if str(meta.get(k)) not in set(vals):
                return False
        # should/not_filters 可扩展
        not_f = filters.get("not_filters") or {}
        for k, vals in not_f.items():
            if str(meta.get(k)) in set(vals):
                return False
        return True

    def search(self, query: str, topk: int = 10,
               filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        score = []
        for doc_id, obj in self._docs.items():
            if not self._passes_filters({"year": obj["year"], "type": obj["type"]}, filters or {}):
                continue
            txt = obj["text"]
            # 简单计分：词覆盖 + 轻微加权
            uniq = set(query.split())
            sc = sum(txt.count(w) for w in uniq) + (0.1 if obj["type"] == "release" else 0.0)
            if sc > 0:
                score.append((doc_id, sc, txt))
        score.sort(key=lambda x: x[1], reverse=True)
        return [SearchResult(d, float(s), t) for d, s, t in score[:topk]]
