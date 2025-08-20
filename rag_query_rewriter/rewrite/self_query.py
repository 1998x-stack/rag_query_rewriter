"""Self-Querying: extract structured filters from question."""
from __future__ import annotations

from typing import Dict, Any
from ..llm.base import LLMClient


def extract_filters(llm: LLMClient, q: str) -> Dict[str, Any]:
    """从问句中抽取结构化过滤条件（演示：调用 LLM/dummy 规则）。"""
    prompt = (
        f'从问题“{q}”中抽取结构化过滤条件，输出 JSON：'
        '{"keywords":[],"must_filters":{},"should_filters":{},"not_filters":{}}'
    )
    data = llm.generate_json(prompt)
    return data if isinstance(data, dict) else {}
