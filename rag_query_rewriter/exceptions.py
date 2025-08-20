"""Custom exceptions for the RAG Query Rewriter."""
from __future__ import annotations


class RewriterError(Exception):
    """通用改写错误。"""


class ConfigError(Exception):
    """配置错误（缺少字段、非法范围等）。"""


class RetrievalError(Exception):
    """检索相关错误。"""