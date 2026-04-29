import inspect
import json
import warnings
from functools import lru_cache
from typing import Any, Callable, Dict, List

from app.mcp.registry import McpToolDefinition, McpToolRegistry


def _schema_properties(input_schema: Dict[str, Any]) -> Dict[str, Any]:
    """功能：从工具输入 schema 中提取 `properties` 字段。
    参数：
    - input_schema：MCP 工具声明的 JSON Schema。
    返回值：
    - Dict[str, Any]：可用字段定义字典；若不存在或类型不合法则返回空字典。
    """
    properties = input_schema.get("properties")
    return properties if isinstance(properties, dict) else {}


def _required_fields(input_schema: Dict[str, Any]) -> List[str]:
    """功能：解析工具输入 schema 中的必填字段列表。
    参数：
    - input_schema：MCP 工具声明的 JSON Schema。
    返回值：
    - List[str]：必填字段名列表；若未声明必填字段则返回空列表。
    """
    required = input_schema.get("required")
    if isinstance(required, list):
        return [str(item) for item in required]
    return []


def _coerce_arguments(payload: Any, tool: McpToolDefinition) -> Dict[str, Any]:
    """功能：按工具 schema 将输入 payload 规范化为参数字典。
    参数：
    - payload：调用方传入的原始参数，可为 dict、字符串或其他单值。
    - tool：当前 MCP 工具定义，包含输入 schema 与本地名称。
    返回值：
    - Dict[str, Any]：可直接发送给 MCP 服务端的参数字典。
    异常：
    - ValueError：当工具需要多个结构化字段但 payload 不是 dict 时抛出。
    """
    schema = tool.input_schema or {}
    properties = _schema_properties(schema)
    required_fields = _required_fields(schema)

    if payload is None:
        return {}
    if isinstance(payload, dict):
        return payload
    if not properties:
        if required_fields:
            if len(required_fields) == 1:
                return {required_fields[0]: payload}
            raise ValueError(
                f"MCP 工具 {tool.local_name} 需要结构化参数，请传入 dict。"
            )
        return {"input": payload}
    if len(properties) == 1:
        only_key = next(iter(properties))
        return {only_key: payload}
    if len(required_fields) == 1:
        return {required_fields[0]: payload}
    raise ValueError(
        f"MCP 工具 {tool.local_name} 需要多个参数，请传入 dict，例如 "
        f'{tool.local_name}({{"field": "value"}})。'
    )


def _schema_summary(tool: McpToolDefinition) -> str:
    """功能：把工具输入 schema 转换为可读的字段说明文本。
    参数：
    - tool：需要生成字段说明的 MCP 工具定义。
    返回值：
    - str：按字段逐行整理的说明；无固定 schema 时返回通用提示。
    """
    schema = tool.input_schema or {}
    properties = _schema_properties(schema)
    required_fields = set(_required_fields(schema))
    if not properties:
        return "该工具无固定输入 schema，可直接传字符串或简单 JSON。"

    field_lines = []
    for name, prop in properties.items():
        prop_type = prop.get("type") or "any"
        description = str(prop.get("description") or "").strip()
        required_mark = "必填" if name in required_fields else "选填"
        if description:
            field_lines.append(f"- {name}（{prop_type}，{required_mark}）：{description}")
        else:
            field_lines.append(f"- {name}（{prop_type}，{required_mark}）")
    return "\n".join(field_lines)


def _build_docstring(tool: McpToolDefinition) -> str:
    """功能：为动态生成的 MCP 工具包装函数构建文档字符串。
    参数：
    - tool：目标 MCP 工具定义。
    返回值：
    - str：包含功能、参数约定、输入 schema 与返回说明的完整 docstring。
    """
    description = tool.description or "调用远端 MCP 工具。"
    schema_summary = _schema_summary(tool)
    return (
        f"功能：通过 MCP Streamable HTTP 调用远端工具 `{tool.server_name}.{tool.remote_name}`。\n"
        f"说明：{description}\n"
        "参数：\n"
        "- payload：推荐传 dict；如果 schema 只有一个核心字段，也可直接传单个值。\n"
        "输入 schema：\n"
        f"{schema_summary}\n"
        "返回值：\n"
        "- str：远端工具返回的文本；若返回结构化结果，则会转成 JSON 字符串。"
    )


def _build_signature(tool: McpToolDefinition) -> inspect.Signature:
    """功能：根据工具 schema 生成包装函数签名。
    参数：
    - tool：目标 MCP 工具定义。
    返回值：
    - inspect.Signature：无固定字段时返回空签名，否则返回仅含 `payload` 的签名。
    """
    properties = _schema_properties(tool.input_schema or {})
    if not properties:
        return inspect.Signature()
    parameter = inspect.Parameter(
        "payload",
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        default=None,
    )
    return inspect.Signature(parameters=[parameter])


def _format_mcp_result(result: str) -> str:
    """功能：格式化 MCP 调用结果，优先输出可读 JSON 文本。
    参数：
    - result：工具调用返回的字符串结果。
    返回值：
    - str：空输入返回空串；可解析为 JSON 时返回 `ensure_ascii=False` 的 JSON 字符串，否则原样返回。
    """
    if not result:
        return ""
    try:
        parsed = json.loads(result)
    except Exception:
        return result
    return json.dumps(parsed, ensure_ascii=False)


def _build_wrapper(tool_registry: McpToolRegistry, tool: McpToolDefinition) -> Callable[..., Any]:
    """功能：为单个 MCP 工具构建可直接注册的本地包装函数。
    参数：
    - tool_registry：负责实际远端调用的 MCP 注册中心实例。
    - tool：要包装的 MCP 工具定义。
    返回值：
    - Callable[..., Any]：携带名称、签名与 docstring 的本地函数对象。
    """
    def _wrapper(payload=None):
        """功能：执行单个 MCP 工具调用并返回格式化结果。
        参数：
        - payload：调用参数，可为 dict 或单值，内部会按 schema 规范化。
        返回值：
        - str：工具执行结果文本；结构化返回会被转成 JSON 字符串。
        """
        arguments = _coerce_arguments(payload, tool)
        result = tool_registry.call_tool(tool, arguments)
        return _format_mcp_result(result)

    _wrapper.__name__ = tool.local_name
    _wrapper.__doc__ = _build_docstring(tool)
    _wrapper.__signature__ = _build_signature(tool)
    return _wrapper


@lru_cache(maxsize=8)
def _load_registry(project_directory: str) -> McpToolRegistry:
    """功能：按项目目录加载并缓存 MCP 工具注册中心实例。
    参数：
    - project_directory：项目根目录绝对路径。
    返回值：
    - McpToolRegistry：对应项目目录的注册中心对象。
    """
    return McpToolRegistry(project_directory=project_directory)


def load_mcp_tools(project_directory: str) -> List[Callable[..., Any]]:
    """功能：加载全部可用 MCP 工具并转换为本地可调用函数列表。
    参数：
    - project_directory：项目根目录绝对路径。
    返回值：
    - List[Callable[..., Any]]：去重后的工具包装函数列表；加载失败时返回空列表。
    """
    try:
        registry = _load_registry(project_directory)
        tool_definitions = registry.list_tool_definitions()
    except Exception as exc:
        warnings.warn(f"MCP 工具已跳过：{exc}")
        return []

    wrappers: List[Callable[..., Any]] = []
    seen_names = set()
    for tool in tool_definitions:
        if tool.local_name in seen_names:
            warnings.warn(f"重复的 MCP 工具名已跳过：{tool.local_name}")
            continue
        seen_names.add(tool.local_name)
        wrappers.append(_build_wrapper(registry, tool))
    return wrappers
