"""Text normalization utilities: case, punctuation, alias mapping, date unification."""
from __future__ import annotations

import regex as re
import yaml
from typing import Dict, Optional
from loguru import logger
from .time_utils import to_absolute_date

# 保留常见符号（含中文）
_PUNCT_RE = re.compile(r"[^\p{L}\p{N}\s:/\-_.·，。；、：%（）()【】\[\]-]")

# 中英混合“词边界”：
# - 英文用\b，中文用环绕断言（非字母数字或边界）
def _replace_alias_safe(text: str, alias: Dict[str, str]) -> str:
    for k, v in alias.items():
        # 同时支持纯英文别名与中文别名安全替换
        # 英文：\b ；中文：(?<![0-9A-Za-z]) / (?![0-9A-Za-z])
        pattern = (
            rf"(?:(?<![0-9A-Za-z])){re.escape(k)}(?![0-9A-Za-z])"
            if re.search(r"[^\p{Han}]", k) else rf"{re.escape(k)}"
        )
        text = re.sub(pattern, v, text, flags=re.IGNORECASE)
    return text


def load_alias_table(path: Optional[str]) -> Dict[str, str]:
    """加载术语/别名映射表（YAML）。"""
    if not path:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        # 统一小写键
        return {str(k).lower(): str(v) for k, v in data.items()}
    except Exception as exc:  # noqa: BLE001
        logger.warning("未能加载别名表: {}", exc)
        return {}


def normalize_text(text: str, alias_table: Optional[Dict[str, str]] = None,
                   case_fold: bool = True, punct_trim: bool = True,
                   date_normalize: bool = True) -> str:
    """执行轻量规范化：大小写、标点清理、日期归一、术语替换。

    中文说明：
        - 不改变语义，不创造实体；专注“检索友好”。
    """
    if not text:
        return text

    s = text.strip()
    if case_fold:
        s = s.lower()

    if date_normalize:
        s = to_absolute_date(s)

    if alias_table:
        s = _replace_alias_safe(s, alias_table)

    if punct_trim:
        s = _PUNCT_RE.sub(" ", s)
        s = re.sub(r"\s+", " ", s).strip()

    return s