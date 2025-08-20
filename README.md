# rag_query_rewriter

更稳、更快、更可控：
- 并行检索（ThreadPoolExecutor）
- 更安全的别名替换（中英混合边界）
- 自定义异常与边界检查
- 结构化 Self-Query 过滤适配（mock retriever 支持）
- 完整自检脚本

## Quickstart
```bash
pip install -e .
rqw rewrite --q "它什么时候发布？" --ctx "上文实体=GPT-5"
rqw e2e --q "2023 版与 2024 版有什么差异？"
python tests/self_check.py
```