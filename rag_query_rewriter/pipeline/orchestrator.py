"""Rewrite→Retrieve→Fuse pipeline orchestrator (parallel, safer, logged)."""
from __future__ import annotations

from typing import List, Dict, Any
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..config import AppConfig
from ..utils.text_norm import normalize_text, load_alias_table
from ..llm.base import LLMClient
from ..retrievers.base import Retriever, SearchResult
from ..rewrite.cqr import cqr_rewrite
from ..rewrite.multiquery import multiquery_rewrite
from ..rewrite.decompose import decompose_into_subqueries
from ..rewrite.hyde import hyde_generate
from ..rewrite.self_query import extract_filters
from ..rewrite.prf import rm3_expand_query
from ..rewrite.router import choose_strategy
from ..fusion.fuser import rrf_fuse, mmr_select


def _retrieve_batch(retriever: Retriever, queries: List[str],
                    filters: Dict[str, Any] | None) -> List[List[SearchResult]]:
    """并行检索批次。"""
    results: List[List[SearchResult]] = [None] * len(queries)  # type: ignore
    with ThreadPoolExecutor(max_workers=min(8, max(2, len(queries)))) as ex:
        futs = {ex.submit(retriever.search, q, 10, filters): i for i, q in enumerate(queries)}
        for fut in as_completed(futs):
            i = futs[fut]
            try:
                results[i] = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("检索失败: idx={} exc={}", i, exc)
                results[i] = []
    return results


def rewrite_and_retrieve(q: str, ctx: str, cfg: AppConfig,
                         llm: LLMClient, retriever: Retriever) -> Dict[str, Any]:
    """端到端：改写→检索→融合→去冗选择。

    返回结构：
        - normalized, cqr, strategy, self_query_filters
        - candidates（所有查询候选）
        - fused_docs（RRF 后）
        - final_docs（MMR 终选）
        - metrics（耗时、候选/文档计数）
    """
    t0 = time.perf_counter()
    logger.info("原始问题: {}", q)

    # A. 规范化
    alias_table = load_alias_table(cfg.normalizer.alias_table_path)
    q_norm = normalize_text(
        q,
        alias_table=alias_table,
        case_fold=cfg.normalizer.enable_case_fold,
        punct_trim=cfg.normalizer.enable_punct_trim,
        date_normalize=cfg.normalizer.enable_date_normalize,
    )
    logger.info("规范化后: {}", q_norm)

    # B. CQR
    cqr = cqr_rewrite(q_norm, history_brief=ctx)
    logger.info("CQR 改写: {}", cqr)

    # C. 路由
    plan = choose_strategy(
        cqr,
        enable_multiquery=cfg.router.enable_multiquery,
        enable_decompose=cfg.router.enable_decompose,
        enable_hyde=cfg.router.enable_hyde,
        enable_prf=cfg.router.enable_prf,
        enable_self_query=cfg.router.enable_self_query,
    )
    logger.info("策略计划: {}", plan)

    # D. 候选生成
    candidates: List[str] = [cqr]
    if plan.use_multiquery:
        candidates.extend(
            multiquery_rewrite(llm, cqr, max_queries=cfg.router.max_queries,
                               dedup_thr=cfg.router.dedup_cosine_thr)
        )
    if plan.use_decompose:
        candidates.extend(decompose_into_subqueries(cqr))
    if plan.use_prf:
        candidates.append(rm3_expand_query(
            retriever, cqr, cfg.prf.topk_initial, cfg.prf.expansion_terms, cfg.prf.stopwords
        ))

    hyde_doc = None
    if plan.use_hyde:
        hyde_doc = hyde_generate(llm, cqr)
        # 在实际系统中：对 hyde_doc 做向量检索；这里简化为把其文本也作为查询候选
        candidates.append(hyde_doc)

    sq_filters = extract_filters(llm, cqr) if plan.use_self_query else {}
    logger.info("候选查询条数: {}", len(candidates))

    t_retr_s = time.perf_counter()
    pools = _retrieve_batch(retriever, candidates, filters=sq_filters or None)
    t_retr_e = time.perf_counter()
    logger.info("检索完成：{}ms", int((t_retr_e - t_retr_s) * 1000))

    # F. RRF 融合
    fused = rrf_fuse(pools, k=cfg.fusion.rrf_k)
    logger.info("RRF 融合候选: {}", len(fused))

    # G. MMR 去冗 + 终选
    final_idx = mmr_select(
        query=cqr,
        docs=[r.text for r in fused],
        topk=cfg.fusion.mmr_topk,
        lamb=cfg.fusion.mmr_lambda,
    )
    final_docs = [fused[i] for i in final_idx]

    t1 = time.perf_counter()

    return {
        "normalized": q_norm,
        "cqr": cqr,
        "strategy": plan.__dict__,
        "self_query_filters": sq_filters,
        "candidates": candidates,
        "fused_docs": [r.to_dict() for r in fused],
        "final_docs": [r.to_dict() for r in final_docs],
        "metrics": {
            "elapsed_ms": int((t1 - t0) * 1000),
            "retrieval_ms": int((t_retr_e - t_retr_s) * 1000),
            "num_candidates": len(candidates),
            "num_fused": len(fused),
            "num_final": len(final_docs),
        },
    }
