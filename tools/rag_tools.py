import os
from pathlib import Path

from rag.rag_service import RagSummarizeService
from rag.vector_store import VectorStoreService


def _rag_enabled() -> bool:
    """功能：读取 `RAG_ENABLED` 环境变量并判断是否允许调用知识库能力。
    参数：
    - 无。
    返回值：
    - bool：当配置值为 1/true/yes/on（不区分大小写）时返回 True，其余返回 False。
    """
    value = (os.getenv("RAG_ENABLED") or "").strip().lower()
    return value in ("1", "true", "yes", "on")


def rag_summarize(query: str) -> str:
    """功能：根据用户问题触发本地 RAG 检索与总结流程，返回可直接回复用户的文本。
    参数：
    - query：用户当前输入的问题或指令，用于在本地知识库中检索相关内容。
    返回值：
    - str：当 RAG 关闭时返回固定提示“RAG 未启用（RAG_ENABLED=false）”；开启时返回 `RagSummarizeService` 生成的总结结果。
    """
    if not _rag_enabled():
        return "RAG 未启用（RAG_ENABLED=false）。"
    service = RagSummarizeService(project_directory=str(Path(__file__).resolve().parents[1]))
    return service.rag_summarize(query)


def rag_rebuild_index() -> str:
    """功能：触发向量库索引重建流程，并返回本次增量构建统计信息。
    参数：
    - 无。
    返回值：
    - str：当 RAG 关闭时返回固定提示“RAG 未启用（RAG_ENABLED=false）”；开启时返回“索引完成”摘要，包含更新文件数和写入分片数。
    """
    if not _rag_enabled():
        return "RAG 未启用（RAG_ENABLED=false）。"
    stats = VectorStoreService().build_or_update_index()
    return f"索引完成：更新文件 {stats.get('indexed_files', 0)} 个，写入分片 {stats.get('indexed_chunks', 0)} 条。"

