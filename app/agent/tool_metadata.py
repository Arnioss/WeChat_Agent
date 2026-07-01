"""结构化工具元数据与 system prompt 统一展示模板。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# 函数上可选属性：ToolRichMetadata
TOOL_RICH_METADATA_ATTR = "__tool_rich_metadata__"


@dataclass(frozen=True)
class ToolRichMetadata:
    """功能：描述高质量工具说明，对齐 MCP 工具常见的描述维度。
    参数：
    - 无。
    返回值：
    - 无。可通过函数属性 `__tool_rich_metadata__` 或 MCP 包装层挂载。
    """

    summary: str
    when_to_use: Tuple[str, ...] = ()
    when_not_to_use: Tuple[str, ...] = ()
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_description: str = ""
    output_schema: Dict[str, Any] = field(default_factory=dict)
    examples: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    # 数值越大在工具列表中越靠前，便于模型优先看到高价值工具
    priority: int = 0
    source: str = "local"  # "local" | "mcp"
    mcp_server_name: str = ""
    mcp_remote_name: str = ""


def format_json_schema_properties_block(schema: Optional[Mapping[str, Any]]) -> str:
    """功能：将 JSON Schema 片段格式化为工具说明中的字段列表文本。
    参数：
    - schema：工具的 input JSON Schema，可为 None。
    返回值：
    - str：每行一个字段的说明文本；无 schema 时返回占位提示。
    """
    if not schema:
        return "（无固定 schema；按函数签名传参。）"
    properties = schema.get("properties")
    if not isinstance(properties, dict) or not properties:
        return "（无固定 properties；可直接传单个字符串或简单 JSON。）"

    required = schema.get("required")
    required_set = set(required) if isinstance(required, list) else set()
    lines: List[str] = []
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            prop = {}
        prop_type = prop.get("type") or "any"
        desc = str(prop.get("description") or "").strip()
        mark = "必填" if name in required_set else "选填"
        if desc:
            lines.append(f"  - {name}（{prop_type}，{mark}）：{desc}")
        else:
            lines.append(f"  - {name}（{prop_type}，{mark}）")
    return "\n".join(lines) if lines else "（schema 无可用字段说明。）"


def format_output_block(
    *,
    output_description: str,
    output_schema: Optional[Mapping[str, Any]] = None,
) -> str:
    """功能：格式化工具返回值/输出说明块。
    参数：
    - output_description：输出描述文本。
    - output_schema：可选输出 JSON Schema。
    返回值：
    - str：拼接后的输出说明文本。
    """
    parts: List[str] = []
    if output_description.strip():
        parts.append(output_description.strip())
    if output_schema:
        try:
            parts.append("JSON Schema（输出）：\n" + json.dumps(dict(output_schema), ensure_ascii=False, indent=2))
        except (TypeError, ValueError):
            parts.append(str(output_schema))
    return "\n".join(parts) if parts else "（字符串或可序列化结果，具体见工具实现。）"


def render_tool_block(
    *,
    name: str,
    signature: str,
    description_fallback: str,
    domain: str,
    channel: str,
    returns_json: bool,
    rich: Optional[ToolRichMetadata],
) -> str:
    """功能：生成单个工具在 system prompt 中的统一展示块。
    参数：
    - name：工具名称。
    - signature：函数签名字符串。
    - description_fallback：无 rich 元数据时的描述回退文本。
    - domain：业务域标签。
    - channel：渠道标签。
    - returns_json：是否以 JSON 直返。
    - rich：可选结构化工具元数据。
    返回值：
    - str：Markdown 风格的工具说明块文本。
    """
    if rich is not None:
        summary = rich.summary.strip() or description_fallback
        source_line = "本地"
        if rich.source == "mcp" and (rich.mcp_server_name or rich.mcp_remote_name):
            source_line = f"MCP（{rich.mcp_server_name}.{rich.mcp_remote_name}）"
        elif rich.source == "mcp":
            source_line = "MCP（远端）"

        lines: List[str] = [
            f"### {name}{signature}",
            f"- **来源**：{source_line}",
            f"- **功能**：{summary}",
        ]
        if rich.when_to_use:
            lines.append("- **适用场景**：")
            lines.extend(f"  - {item}" for item in rich.when_to_use)
        if rich.when_not_to_use:
            lines.append("- **不适用场景**：")
            lines.extend(f"  - {item}" for item in rich.when_not_to_use)

        lines.append("- **输入（schema 摘要）**：")
        lines.append(format_json_schema_properties_block(rich.input_schema))

        lines.append("- **返回值 / 输出**：")
        lines.append(format_output_block(output_description=rich.output_description, output_schema=rich.output_schema or None))

        if rich.examples:
            lines.append("- **工具调用示例（原生 tool_calls；arguments 按 input_schema 填写）**：")
            lines.extend(f"  - {ex}" for ex in rich.examples)

        if rich.notes:
            lines.append("- **注意事项**：")
            lines.extend(f"  - {note}" for note in rich.notes)

        meta = {
            "domain": domain,
            "channel": channel,
            "returns_json": returns_json,
            "priority": rich.priority,
        }
        lines.append(f"- **元信息**：{json.dumps(meta, ensure_ascii=False)}")
        return "\n".join(lines)

    # 无结构化元数据：保持可读的最小块，避免破坏旧工具
    meta = {"domain": domain, "channel": channel, "returns_json": returns_json, "priority": 0}
    return (
        f"### {name}{signature}\n"
        f"- **来源**：本地\n"
        f"- **说明**：{description_fallback.strip() or '（无描述）'}\n"
        f"- **元信息**：{json.dumps(meta, ensure_ascii=False)}"
    )


def summarize_routing_lines(
    specs: Sequence[Any],
) -> str:
    """功能：从已排序的 ToolSpec 序列生成简短路由提示文本。
    参数：
    - specs：已按 priority 排好序的 ToolSpec 列表。
    返回值：
    - str：面向 system prompt 的工具选择准则与工具摘要行。
    """
    lines: List[str] = [
        "- 纯问候、与项目/业务无关的闲聊：可直接输出最终答案，无需调用工具。",
        "- 需要「今天日期 / 当前日历日」且不要求业务文档：通常选择 `get_current_date()`。",
        "- 明确要求依据知识库、项目文档、内部资料或参考资料回答时：通常选择 `rag_summarize(\"清晰完整的问题\")`；"
        "收到 observation 后先呈现知识库要点，再判断是否需要结合通用模型知识补充说明，并标注非知识库部分。",
        "- RAG 未启用时：不要调用 `rag_summarize` 或 `rag_rebuild_index`，直接基于通用知识回答；需要以项目知识库为准时说明当前未启用。",
        "- 远端 MCP 工具：仅在确实需要其能力时调用；参数遵循各工具 schema，结构化参数用 dict。",
        "- 原生 tool_calls 调用工具时：函数名使用工具名，arguments 按该工具 input_schema 填 JSON object。",
    ]

    extra: List[str] = []
    for spec in specs:
        rich = getattr(spec, "rich_metadata", None)
        if rich is None:
            continue
        w = rich.when_to_use or ()
        if not w:
            continue
        head = w[0][:120] + ("…" if len(w[0]) > 120 else "")
        extra.append(f"- 工具 `{getattr(spec, 'name', '?')}`：{head}")

    if extra:
        lines.append("- **工具摘要（按优先级）**：")
        lines.extend(extra[:12])
    return "\n".join(lines)


def rich_metadata_from_mcp_definition(
    *,
    local_name: str,
    server_name: str,
    remote_name: str,
    description: str,
    input_schema: Dict[str, Any],
) -> ToolRichMetadata:
    """功能：由 MCP 工具定义构造与本地工具对齐的 ToolRichMetadata。
    参数：
    - local_name：本地映射工具名。
    - server_name：MCP 服务名称。
    - remote_name：远端工具名称。
    - description：远端工具描述。
    - input_schema：远端 input JSON Schema。
    返回值：
    - ToolRichMetadata：可用于 prompt 渲染的结构化元数据。
    """
    summary = (description or "调用远端 MCP 工具。").strip()
    when_to_use = (
        "需要该 MCP 服务提供的专用能力（地图、搜索、工单等）时。",
        "本地内置工具无法满足且问题明确依赖远端数据时。",
    )
    when_not_to_use = (
        "问题可用常识或当前对话直接回答，且不依赖远端实时数据。",
        "本地知识库即可覆盖的内部文档类问题（优先 rag_summarize）。",
        "纯日期查询（优先 get_current_date）。",
    )
    notes = (
        "通过 Streamable HTTP 调用；网络失败时会返回明确错误信息。",
        "推荐传入 dict；若 schema 仅有一个核心字段，也可传单个值（见包装层强制规则）。",
        "调用名称为本地映射名，与远端 tool 名可能不同。",
    )
    examples: Tuple[str, ...]
    props = input_schema.get("properties") if isinstance(input_schema, dict) else None
    if isinstance(props, dict) and len(props) == 1:
        only = next(iter(props))
        examples = (f'{local_name}({{"{only}": "示例值"}})',)
    elif isinstance(props, dict) and props:
        examples = (f"{local_name}({{...}})  # 按输入 schema 填写所有必填字段",)
    else:
        examples = (f'{local_name}("简短请求文本")',)

    return ToolRichMetadata(
        summary=summary,
        when_to_use=when_to_use,
        when_not_to_use=when_not_to_use,
        input_schema=dict(input_schema or {}),
        output_description="字符串：远端返回文本；若为 JSON 会格式化为可读字符串。",
        output_schema={"type": "string"},
        examples=examples,
        notes=notes,
        priority=45,
        source="mcp",
        mcp_server_name=server_name,
        mcp_remote_name=remote_name,
    )


def get_attached_rich_metadata(func: Any) -> Optional[ToolRichMetadata]:
    """功能：读取函数上挂载的 `__tool_rich_metadata__` 元数据。
    参数：
    - func：工具函数对象。
    返回值：
    - Optional[ToolRichMetadata]：存在时返回元数据，否则返回 None。
    """
    return getattr(func, TOOL_RICH_METADATA_ATTR, None)
