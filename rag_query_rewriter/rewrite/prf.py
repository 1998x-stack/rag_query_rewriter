"""PRF (Pseudo Relevance Feedback) / RM3-like expansion."""
from __future__ import annotations

from typing import List
import regex as re
from collections import Counter
from ..retrievers.base import Retriever


def rm3_expand_query(ret: Retriever, q: str, topk_initial: int,
                     expansion_terms: int, stopwords: List[str]) -> str:
    """RM3 风格的简化扩展：从首跳 top-k 文档中抽取高频关键词进行扩展。"""
    if expansion_terms <= 0:
        return q
    res = ret.search(q, topk=topk_initial)
    corpus = " ".join([r.text for r in res])
    toks = re.findall(r"[\p{L}\p{N}_]{2,}", corpus)
    sw = set(w.lower() for w in stopwords) | {"2023", "2024", "faq", "spec"}  # 演示停用
    cnt = Counter(w.lower() for w in toks if w.lower() not in sw and not w.isdigit())
    extra = [w for w, _ in cnt.most_common(expansion_terms)]
    return f"{q} " + " ".join(extra) if extra else q
