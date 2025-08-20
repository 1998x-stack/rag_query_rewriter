"""Global configuration models using Pydantic with validation."""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional


class RewriteRouterConfig(BaseModel):
    """策略路由配置。"""
    enable_multiquery: bool = True
    enable_decompose: bool = True
    enable_hyde: bool = True
    enable_prf: bool = True
    enable_self_query: bool = True
    max_queries: int = Field(default=6, ge=1, le=12)
    dedup_cosine_thr: float = Field(default=0.92, ge=0.0, le=1.0)


class NormalizerConfig(BaseModel):
    """规范化配置。"""
    enable_case_fold: bool = True
    enable_punct_trim: bool = True
    enable_date_normalize: bool = True
    alias_table_path: Optional[str] = None  # 术语/别名 YAML 路径


class PRFConfig(BaseModel):
    """PRF/RM3 配置。"""
    topk_initial: int = Field(default=5, ge=1, le=100)
    expansion_terms: int = Field(default=6, ge=0, le=50)
    stopwords: List[str] = Field(default_factory=lambda: ["的", "了", "and", "or", "the"])


class FusionConfig(BaseModel):
    """融合配置。"""
    rrf_k: int = Field(default=60, ge=1)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    mmr_topk: int = Field(default=8, ge=1, le=100)


class AppConfig(BaseModel):
    """应用总配置。"""
    normalizer: NormalizerConfig = NormalizerConfig()
    router: RewriteRouterConfig = RewriteRouterConfig()
    prf: PRFConfig = PRFConfig()
    fusion: FusionConfig = FusionConfig()
    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR"}
        if v.upper() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v.upper()