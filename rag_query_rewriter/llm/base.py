"""Abstract LLM client interface."""
from __future__ import annotations

from typing import List, Optional, Any
from abc import ABC, abstractmethod


class LLMClient(ABC):
    """LLM 客户端抽象类，便于替换为任何厂商/开源模型。"""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """生成文本。"""
        raise NotImplementedError

    @abstractmethod
    def generate_lines(self, prompt: str, n_lines: int = 6,
                       max_tokens: int = 512) -> List[str]:
        """生成多行，每行一个候选。"""
        raise NotImplementedError

    @abstractmethod
    def generate_json(self, prompt: str, schema_hint: Optional[str] = None,
                      max_tokens: int = 512) -> Any:
        """生成结构化 JSON。"""
        raise NotImplementedError
