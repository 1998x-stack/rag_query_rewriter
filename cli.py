"""CLI for RAG Query Rewriter (v2)."""
from __future__ import annotations

import argparse
from loguru import logger
from rag_query_rewriter.logging_setup import setup_logging
from rag_query_rewriter.config import AppConfig
from rag_query_rewriter.llm.dummy import DummyLLM
from rag_query_rewriter.retrievers.mock import MockRetriever
from rag_query_rewriter.pipeline.orchestrator import rewrite_and_retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG Query Rewriter CLI")
    parser.add_argument("--log-level", default="INFO", help="DEBUG/INFO/WARNING/ERROR")
    parser.add_argument("--log-file", default=None, help="Optional log file path")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("rewrite", help="Rewrite + retrieve + fuse once")
    p1.add_argument("--q", required=True, help="User query")
    p1.add_argument("--ctx", default="", help="History brief, e.g., '上文实体=GPT-5, 事件=发布'")
    p1.add_argument("--alias", default="examples/terms_alias.yaml", help="Alias table yaml path")

    p2 = sub.add_parser("e2e", help="End-to-end demo")
    p2.add_argument("--q", required=True, help="User query")
    p2.add_argument("--ctx", default="", help="History brief")
    p2.add_argument("--alias", default="examples/terms_alias.yaml", help="Alias table yaml path")

    args = parser.parse_args()

    setup_logging(args.log_level, file_path=args.log_file)
    cfg = AppConfig()
    cfg.normalizer.alias_table_path = args.alias

    llm = DummyLLM()
    retriever = MockRetriever()

    if args.cmd in {"rewrite", "e2e"}:
        out = rewrite_and_retrieve(q=args.q, ctx=args.ctx, cfg=cfg, llm=llm, retriever=retriever)
        logger.success("改写完成：\n{}", out)


if __name__ == "__main__":
    main()
