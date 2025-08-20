"""Decomposition-based rewriting."""
from __future__ import annotations

from typing import List
import regex as re


def decompose_into_subqueries(q: str) -> List[str]:
    """将复合问题拆为最少的可检索子问题列表（启发式）。"""
    # 对比/并列/时间线
    if any(x in q for x in ["对比", "差异", "分别", "与", "和", "vs"]):
        parts = re.split(r"[、，,；;以及和与]|对比|差异|分别", q)
        parts = [p.strip() for p in parts if p.strip()]
        # 末尾补关键词
        return [p if "?" in p else p for p in parts]

    if "时间线" in q or "timeline" in q:
        return [q, "该问题的时间线与关键节点"]

    # 默认：原问句 + “时间线”视角
    return [q, f"{q} 的时间线"]