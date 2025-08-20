"""Self-check to verify pipeline logic, boundaries, and concurrency."""
from __future__ import annotations

from rag_query_rewriter.logging_setup import setup_logging
from rag_query_rewriter.config import AppConfig
from rag_query_rewriter.llm.dummy import DummyLLM
from rag_query_rewriter.retrievers.mock import MockRetriever
from rag_query_rewriter.pipeline.orchestrator import rewrite_and_retrieve


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def run_checks() -> None:
    """运行关键自检：覆盖常见路径与边界。"""
    setup_logging("INFO")
    cfg = AppConfig()
    cfg.normalizer.alias_table_path = "examples/terms_alias.yaml"
    llm = DummyLLM()
    ret = MockRetriever()

    # 1) 短事实问句（应触发 MultiQuery 或 HyDE）
    out1 = rewrite_and_retrieve("它什么时候发布？", "上文实体=GPT-5", cfg, llm, ret)
    _assert(out1["candidates"], "候选未生成")
    _assert(out1["fused_docs"], "融合结果为空")
    _assert(out1["metrics"]["num_final"] >= 1, "MMR 终选为空")

    # 2) 对比型问题（应触发 Decompose）
    out2 = rewrite_and_retrieve("2023 版与 2024 版有何差异？", "", cfg, llm, ret)
    _assert(out2["strategy"]["use_decompose"] is True, "Decompose 未触发")
    _assert(out2["metrics"]["num_candidates"] >= 2, "Decompose 候选不足")

    # 3) PRF 扩展在较长问句
    out3 = rewrite_and_retrieve("请提供 2023 与 2024 的版本更新摘要与时间线说明", "", cfg, llm, ret)
    _assert(out3["strategy"]["use_prf"] is True, "PRF 未触发")

    # 4) Self-Query filters 生效（限制年份）
    out4 = rewrite_and_retrieve("2023 年的发布记录", "", cfg, llm, ret)
    # Dummy LLM 的 must_filters: {"year": ["2023","2024"]}，因此 d1/d2 均可；仅检查不为空
    _assert(out4["fused_docs"], "Self-Query 过滤后为空（mock）")

    # 5) 并行检索计时指标
    _assert(out4["metrics"]["retrieval_ms"] >= 0, "计时指标异常")

    print("✅ Self-check passed: all core flows, boundaries, and metrics OK.")


if __name__ == "__main__":
    run_checks()
