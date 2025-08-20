"""Strategy router: choose rewriting strategies based on query features."""
from __future__ import annotations

from dataclasses import dataclass
import regex as re


@dataclass
class StrategyPlan:
    """改写策略计划。"""
    use_multiquery: bool = False
    use_decompose: bool = False
    use_hyde: bool = False
    use_prf: bool = False
    use_self_query: bool = False


def choose_strategy(q: str, enable_multiquery: bool, enable_decompose: bool,
                    enable_hyde: bool, enable_prf: bool,
                    enable_self_query: bool) -> StrategyPlan:
    """根据问句特征选择策略（启发式 + 边界约束）。"""
    q_len = len(q)
    has_compare = any(x in q for x in ["对比", "差异", "分别", "vs", "和", "与"])
    has_time = bool(re.search(r"\b(19|20)\d{2}\b|年|月|day|week|month", q))
    plan = StrategyPlan()

    if enable_decompose and has_compare:
        plan.use_decompose = True

    if enable_self_query and has_time:
        plan.use_self_query = True

    if enable_multiquery and q_len <= 36 and not plan.use_decompose:
        plan.use_multiquery = True

    if enable_prf and q_len > 18:
        plan.use_prf = True

    if enable_hyde and q_len < 16 and not has_compare:
        plan.use_hyde = True

    return plan
