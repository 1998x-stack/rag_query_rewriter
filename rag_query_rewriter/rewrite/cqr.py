"""Conversational Query Reformulation (CQR)."""
from __future__ import annotations

from typing import Optional
import regex as re


def cqr_rewrite(current_q: str, history_brief: Optional[str] = None) -> str:
    """将当前问题改写为自包含问题，补全指代实体。

    中文说明：
        - 从 "上文实体=xxx, 事件=yyy" 解析键值；
        - 将“它/该产品/上文/这个/此项”等替换为实体名（若提供）。
    """
    if not history_brief:
        return current_q

    kvs = dict(re.findall(r"([\p{L}\p{N}_]+)\s*=\s*([^\s,，]+)", history_brief))
    ent = kvs.get("上文实体") or kvs.get("entity") or list(kvs.values())[0] if kvs else None
    if not ent:
        return current_q

    s = current_q
    s = re.sub(r"(它|该(?:产品|系统|模型)?|上文|这个|此项|其)", ent, s)
    return s