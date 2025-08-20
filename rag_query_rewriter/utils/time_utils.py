"""Time utilities to convert relative phrases into absolute dates."""
from __future__ import annotations

import regex as re
import dateparser


def _replace_relative(match: re.Match) -> str:
    text = match.group(0)
    dt = dateparser.parse(text)
    return dt.strftime("%Y-%m-%d") if dt else text


def to_absolute_date(s: str) -> str:
    """归一化字符串中的相对日期表达。"""
    pattern = re.compile(
        r"\b(yesterday|today|tomorrow|last\s+week|last\s+month|"
        r"\d+\s+(days?|weeks?|months?)\s+ago|"
        r"上周|上个月|昨天|明天|今天)\b",
        re.IGNORECASE,
    )
    return pattern.sub(_replace_relative, s)
