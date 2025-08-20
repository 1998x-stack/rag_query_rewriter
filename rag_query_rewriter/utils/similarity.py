"""Similarity & embedding helpers with TF-IDF for lightweight dedup and MMR."""
from __future__ import annotations

from typing import List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfEmbedder:
    """基于 TF-IDF 的简易文本嵌入器。"""

    def __init__(self) -> None:
        self._vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if not texts:
            raise ValueError("texts must be non-empty")
        return self._vec.fit_transform(texts).astype(np.float32)

    def transform(self, texts: List[str]) -> np.ndarray:
        return self._vec.transform(texts).astype(np.float32)


def dedup_texts_by_cosine(texts: List[str], thr: float) -> List[str]:
    """按余弦相似度阈值去重，保留多样性。"""
    if not texts:
        return []
    # 去除空白候选
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        return []
    emb = TfidfEmbedder().fit_transform(texts)
    keep: List[int] = []
    for i in range(len(texts)):
        if not keep:
            keep.append(i)
            continue
        sims = cosine_similarity(emb[i], emb[keep]).ravel()
        if (sims.max(initial=0.0) if sims.size else 0.0) < thr:
            keep.append(i)
    return [texts[i] for i in keep]
