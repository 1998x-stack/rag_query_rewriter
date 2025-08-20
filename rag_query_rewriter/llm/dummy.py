"""A deterministic dummy LLM for offline development."""
from __future__ import annotations

from typing import List, Any
import json
import regex as re


class DummyLLM:
    """离线开发用的简单 LLM 模拟器。"""

    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        return "该问题涉及背景、时间与主体三方面，请参考技术文档与发布记录进行核对与比对。"

    def generate_lines(self, prompt: str, n_lines: int = 6,
                       max_tokens: int = 512) -> List[str]:
        m = re.search(r"[“\"](.+?)[”\"]", prompt)
        base = m.group(1) if m else "问题"
        seeds = [
            base,
            f"{base} 的详细说明",
            f"{base} 相关文档",
            f"{base} 时间线",
            f"{base} 常见问题",
            f"{base} 技术规格",
            f"{base} 关键变更",
            f"{base} 发行说明",
        ]
        return seeds[:n_lines]

    def generate_json(self, prompt: str, schema_hint: str | None = None,
                      max_tokens: int = 512) -> Any:
        return {
            "keywords": ["发布", "时间", "版本"],
            "must_filters": {"year": ["2023", "2024"]},
            "should_filters": {},
            "not_filters": {},
        }
