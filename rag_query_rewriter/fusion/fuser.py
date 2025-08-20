"""Score fusion (RRF) and redundancy control (MMR)."""
from __future__ import annotations

from typing import List
from ..retrievers.base import SearchResult
from ..utils.similarity import TfidfEmbedder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def rrf_fuse(pools: List[List[SearchResult]], k: int = 60) -> List[SearchResult]:
    """Reciprocal Rank Fusion (RRF)."""
    if not pools:
        return []
    score = {}
    first_seen = {}

    for results in pools:
        for rank, r in enumerate(results, start=1):
            if r.doc_id is None:
                continue
            score.setdefault(r.doc_id, 0.0)
            score[r.doc_id] += 1.0 / (k + rank)
            if r.doc_id not in first_seen:
                first_seen[r.doc_id] = r

    fused = [SearchResult(doc_id=d, score=s, text=first_seen[d].text)
             for d, s in score.items()]
    fused.sort(key=lambda x: (-(x.score), x.doc_id))
    return fused


def mmr_select(query: str, docs: List[str], topk: int = 8, lamb: float = 0.7) -> List[int]:
    """Maximal Marginal Relevance 选择文档索引集合。"""
    if topk <= 0 or not docs:
        return []
    topk = min(topk, len(docs))

    emb = TfidfEmbedder()
    X = emb.fit_transform([query] + docs)  # 0: query
    qv = X[0]
    D = X[1:]

    # 相似度矩阵可能出现空向量，需安全处理
    sim_qd = cosine_similarity(qv, D).ravel()
    sim_dd = cosine_similarity(D, D)
    np.nan_to_num(sim_qd, copy=False)
    np.nan_to_num(sim_dd, copy=False)

    selected: List[int] = []
    candidates = list(range(len(docs)))

    while candidates and len(selected) < topk:
        if not selected:
            i_rel = int(np.argmax(sim_qd[candidates]))
            selected.append(candidates[i_rel])
            candidates.pop(i_rel)
            continue
        # 与已选的最大相似度
        max_sim_to_sel = np.array(
            [max(sim_dd[i][selected]) if selected else 0.0 for i in candidates]
        )
        mmr_scores = lamb * sim_qd[candidates] - (1.0 - lamb) * max_sim_to_sel
        picked = int(np.argmax(mmr_scores))
        selected.append(candidates[picked])
        candidates.pop(picked)

    return selected
