import os
from pathlib import Path

from app.agent.tool_metadata import ToolRichMetadata
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
    """功能：根据用户问题触发本地 RAG 检索流程，返回供 Agent 组织最终回答的参考资料。
    参数：
    - query：用户当前输入的问题或指令，用于在本地知识库中检索相关内容。
    返回值：
    - str：当 RAG 关闭时返回固定提示“RAG 未启用（RAG_ENABLED=false）”；开启时返回检索到的参考资料文本。
    """
    if not _rag_enabled():
        return "RAG 未启用（RAG_ENABLED=false）。"
    service = RagSummarizeService(project_directory=str(Path(__file__).resolve().parents[1]))
    return service.rag_summarize(query)


rag_summarize.__tool_rich_metadata__ = ToolRichMetadata(
    summary="优先入口：检索本地向量知识库参考资料；工具只返回资料，不生成最终答案。",
    when_to_use=(
        "用户明确要求依据知识库、项目文档、内部资料或参考资料回答。",
        "问题明显属于内部流程、配置、版本、约定或排障步骤，且无法仅凭通用知识可靠回答。",
        "需要核对或引用文档中可能存在的条款、注意事项、示例。",
    ),
    when_not_to_use=(
        "纯寒暄、与业务/项目无关的闲聊。",
        "助手能力介绍、一般说明、通用知识问题，且用户没有要求依据内部资料。",
        "只问今天日历日、不要文档内容（用 get_current_date）。",
        "RAG 未启用时：勿伪造检索；由系统提示中的「知识作答策略」处理。",
        "用户未给出可检索的实质问题（可先请补充再调用）。",
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "面向知识库的完整问题，包含关键实体与上下文，避免过短模糊。",
            }
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    output_description="str：知识库参考资料/检索结果；无命中时返回可读提示（非异常）。",
    output_schema={"type": "string"},
    examples=(
        'rag_summarize("企业微信回调 URL 如何配置？")',
        'rag_summarize("RAG_ENABLED 关闭时会发生什么？")',
    ),
    notes=(
        "本工具只返回参考资料，不调用聊天模型，不生成最终答案；最终回答由 Agent 基于 observation 组织。",
        "若 observation 表明知识库暂无足够信息或未命中：最终回答可基于通用知识作答，并标明「通用知识补充」。",
        "若知识库已有资料：最终回答中优先基于资料组织结论，然后判断是否需要补充通用背景、解释、示例或建议；可以补充，但必须区分来源且不得与知识库冲突。",
        "若 observation 含 [KB_IMAGE:路径]：最终回答必须在对应步骤后原样插入同一行标记（勿删、勿挪到文末）。",
        "RAG 未启用时不要调用本工具；若误调用会返回说明文本，勿伪造检索，按系统提示直接通用知识作答。",
        "tool_calls arguments 必须包含 query 字符串字段。",
    ),
    priority=88,
    source="local",
)


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


rag_rebuild_index.__tool_rich_metadata__ = ToolRichMetadata(
    summary="扫描知识目录并增量更新向量索引（管理员或自动化任务使用）。",
    when_to_use=(
        "知识库文件更新后需要立即重建索引以便 rag_summarize 命中新内容。",
        "排障时确认索引是否成功构建。",
    ),
    when_not_to_use=(
        "普通用户问答（应使用 rag_summarize 而非直接重建索引）。",
        "RAG 已关闭或未配置知识目录时。",
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    },
    output_description="str：索引构建统计摘要。",
    output_schema={"type": "string"},
    examples=("rag_rebuild_index()",),
    notes=("该操作可能耗时较长；生产环境应控制调用频率。",),
    priority=40,
    source="local",
)
