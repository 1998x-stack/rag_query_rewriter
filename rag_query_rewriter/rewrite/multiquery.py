"""MultiQuery rewriting with LLM and dedup."""
from __future__ import annotations

from typing import List
from loguru import logger
from ..llm.base import LLMClient
from ..utils.similarity import dedup_texts_by_cosine


_FALLBACKS = [
    "{q}",
    "{q} 详细说明",
    "{q} 相关文档",
    "{q} 时间线",
    "{q} 常见问题",
    "{q} 技术规格",
    "{q} 关键变更",
    "{q} 发行说明",
]


def multiquery_rewrite(llm: LLMClient, q: str, max_queries: int,
                       dedup_thr: float) -> List[str]:
    """基于 LLM 生成多样化等价查询，并做去重与裁剪。"""
    prompt = f'基于问题“{q}”，请给出 8 条语义等价但措辞多样的检索查询；每行一条，不要编号。'
    lines = []
    try:
        lines = llm.generate_lines(prompt, n_lines=8)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM multiquery 失败，使用回退：{}", exc)
        lines = [t.format(q=q) for t in _FALLBACKS]

    candidates = [q] + [c for c in lines if c and c.strip()]
    deduped = dedup_texts_by_cosine(candidates, thr=dedup_thr)
    kept = deduped[:max_queries]
    logger.debug("MultiQuery 生成={} 去重后={}", len(candidates), len(kept))
    return kept
