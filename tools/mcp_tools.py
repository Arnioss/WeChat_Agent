import inspect
import json
import time
import warnings
from functools import lru_cache
from typing import Any, Callable, Dict, List

from app.agent.tool_metadata import rich_metadata_from_mcp_definition
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
    """功能：为单个 MCP 工具构建可直接注册的本地 async 包装函数。
    参数：
    - tool_registry：负责实际远端调用的 MCP 注册中心实例。
    - tool：要包装的 MCP 工具定义。
    返回值：
    - Callable[..., Any]：携带名称、签名与 docstring 的 async 函数对象。
    """
    async def _wrapper(payload=None):
        """功能：执行单个 MCP 工具调用并返回格式化结果。
        参数：
        - payload：调用参数，可为 dict 或单值，内部会按 schema 规范化。
        返回值：
        - str：工具执行结果文本；结构化返回会被转成 JSON 字符串。
        """
        arguments = _coerce_arguments(payload, tool)
        result = await tool_registry.call_tool(tool, arguments)
        return _format_mcp_result(result)

    _wrapper.__name__ = tool.local_name
    rich = rich_metadata_from_mcp_definition(
        local_name=tool.local_name,
        server_name=tool.server_name,
        remote_name=tool.remote_name,
        description=tool.description or "",
        input_schema=dict(tool.input_schema or {}),
    )
    # 与 ToolRegistry 共用结构化元数据；保留简短 docstring 便于调试与 inspect。
    _wrapper.__tool_rich_metadata__ = rich
    _wrapper.__doc__ = rich.summary
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


@lru_cache(maxsize=8)
def _load_mcp_tool_wrappers(project_directory: str) -> tuple:
    """功能：按项目目录加载并缓存 MCP 工具包装函数元组。
    参数：
    - project_directory：项目根目录绝对路径。
    返回值：
    - tuple：去重后的工具包装函数元组；加载失败时返回空元组。
    """
    try:
        registry = _load_registry(project_directory)
        tool_definitions = registry.list_tool_definitions()
    except Exception as exc:
        warnings.warn(f"MCP 工具已跳过：{exc}")
        return ()

    wrappers: List[Callable[..., Any]] = []
    seen_names = set()
    for tool in tool_definitions:
        if tool.local_name in seen_names:
            warnings.warn(f"重复的 MCP 工具名已跳过：{tool.local_name}")
            continue
        seen_names.add(tool.local_name)
        wrappers.append(_build_wrapper(registry, tool))
    return tuple(wrappers)


def load_mcp_tools(project_directory: str) -> List[Callable[..., Any]]:
    """功能：加载全部可用 MCP 工具并转换为本地可调用函数列表。
    参数：
    - project_directory：项目根目录绝对路径。
    返回值：
    - List[Callable[..., Any]]：去重后的工具包装函数列表；加载失败时返回空列表。
    """
    return list(_load_mcp_tool_wrappers(project_directory))


def warm_mcp_tools(project_directory: str, *, force_refresh: bool = True) -> Dict[str, Any]:
    """功能：预热 MCP 工具注册中心，刷新远端工具列表并返回统计摘要。
    参数：
    - project_directory：项目根目录绝对路径。
    - force_refresh：是否强制重新发现远端工具，默认 True。
    返回值：
    - Dict[str, Any]：包含 server_count、tool_count、duration_ms 的预热结果。
    """
    started_at = time.time()
    if force_refresh:
        _load_mcp_tool_wrappers.cache_clear()
    registry = _load_registry(project_directory)
    tool_definitions = registry.list_tool_definitions(force_refresh=force_refresh)
    wrappers = _load_mcp_tool_wrappers(project_directory)
    server_names = {tool.server_name for tool in tool_definitions}
    return {
        "server_count": len(server_names),
        "tool_count": len(tool_definitions),
        "wrapper_count": len(wrappers),
        "duration_ms": round((time.time() - started_at) * 1000.0, 2),
    }
