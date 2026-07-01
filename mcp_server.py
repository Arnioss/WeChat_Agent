"""通用 MCP 服务入口 — 按需在各工具区追加 @mcp.tool() 函数。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dotenv import load_dotenv
load_dotenv()

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("iov-agent", host="0.0.0.0", port=8000)

# ── RAG 知识库 ────────────────────────────────────────────────
from tools.rag_tools import rag_summarize as _rag_summarize


@mcp.tool()
def rag_summarize(query: str) -> str:
    """功能：检索本地向量知识库，返回相关参考资料。
    参数：
    - query：完整问题文本，应包含关键实体与上下文。
    返回值：
    - str：与查询相关的参考资料摘要文本。
    """
    return _rag_summarize(query)


# ── 更多工具在此追加 ──────────────────────────────────────────


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--transport", default="streamable-http", choices=["stdio", "streamable-http"])
    args = p.parse_args()
    mcp.run(transport=args.transport)
