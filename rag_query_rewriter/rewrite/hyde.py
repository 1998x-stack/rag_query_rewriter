"""HyDE: Query->Hypothetical Document->Embedding Retrieval."""
from __future__ import annotations

from ..llm.base import LLMClient


def hyde_generate(llm: LLMClient, q: str) -> str:
    """生成“假想文档”摘要用于向量检索。"""
    prompt = (
        f"针对问题“{q}”，撰写一段与技术文档风格一致的假想摘要（150~220字），"
        "应覆盖背景、时间与主体信息，不要编造具体数值与专有名词，仅输出正文。"
    )
    return llm.generate(prompt, max_tokens=220)
