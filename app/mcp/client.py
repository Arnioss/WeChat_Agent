import json
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List, Optional


def _import_mcp_sdk():
    """功能：动态导入 MCP 客户端依赖并返回核心对象。
    参数：
    - 无。
    返回值：
    - tuple：依次为 `ClientSession`、`streamable_http_client` 和 `httpx` 模块对象。
    异常：
    - RuntimeError：未安装 `mcp` 依赖时抛出。
    """
    try:
        import httpx  # type: ignore
        from mcp import ClientSession  # type: ignore
        try:
            from mcp.client.streamable_http import streamable_http_client  # type: ignore
        except ImportError:
            from mcp.client.streamable_http import streamablehttp_client as streamable_http_client  # type: ignore
        return ClientSession, streamable_http_client, httpx
    except Exception as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError(
            "未安装 mcp 依赖，请先执行 `python -m pip install mcp`。"
        ) from exc


def _to_plain_data(value: Any) -> Any:
    """功能：把复杂对象递归转换为可 JSON 序列化的基础结构。
    参数：
    - value：待转换的输入值。
    返回值：
    - Any：转换后的基础类型数据（dict/list/标量），无法结构化时返回字符串。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_plain_data(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(item) for item in value]
    if is_dataclass(value):
        return _to_plain_data(asdict(value))
    if hasattr(value, "model_dump"):
        try:
            return _to_plain_data(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _to_plain_data(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        data = {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
        if data:
            return _to_plain_data(data)
    return str(value)


def _collect_text_content(content_items: Any) -> List[str]:
    """功能：从 MCP 响应内容列表提取可展示文本。
    参数：
    - content_items：MCP 响应中的内容项列表。
    返回值：
    - List[str]：提取后的文本片段列表。
    """
    texts: List[str] = []
    for item in content_items or []:
        text = getattr(item, "text", None)
        if text:
            texts.append(str(text))
            continue
        plain = _to_plain_data(item)
        if plain not in (None, "", {}, []):
            texts.append(json.dumps(plain, ensure_ascii=False))
    return texts


class StreamableHttpMcpClient:
    """功能：基于 Streamable HTTP 协议调用远端 MCP 服务。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(
        self,
        *,
        server_name: str,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout_seconds: float = 20.0,
    ):
        """功能：保存 MCP 服务连接参数，供后续会话创建与工具调用复用。
        参数：
        - server_name：MCP 服务名称。
        - url：MCP 服务地址。
        - headers：HTTP 请求头字典。
        - timeout_seconds：HTTP 请求超时时间（秒）。
        返回值：
        - 无。该初始化不发起网络请求，连接在首次调用时按需建立。
        """
        self.server_name = server_name
        self.url = url
        self.headers = dict(headers or {})
        self.timeout_seconds = float(timeout_seconds)

    async def list_tools(self) -> List[Dict[str, Any]]:
        """功能：拉取远端 MCP 服务暴露的工具列表。
        参数：
        - 无。
        返回值：
        - List[Dict[str, Any]]：工具基础信息列表（名称、描述、输入 schema）。
        """
        return await self._with_session(self._list_tools)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """功能：调用指定 MCP 工具并返回文本结果。
        参数：
        - tool_name：远端工具名称。
        - arguments：调用参数字典。
        返回值：
        - str：工具返回文本或结构化内容序列化后的字符串。
        """
        return await self._with_session(self._call_tool, tool_name, arguments)

    async def _with_session(self, action, *args):
        """功能：创建 MCP 会话并执行指定异步动作。
        参数：
        - action：会话内要执行的异步函数。
        返回值：
        - Any：动作函数返回值。
        """
        ClientSession, streamable_http_client, httpx = _import_mcp_sdk()
        timeout = self.timeout_seconds if self.timeout_seconds > 0 else None
        async with httpx.AsyncClient(
            headers=self.headers or None,
            timeout=timeout,
            follow_redirects=True,
        ) as http_client:
            async with streamable_http_client(
                self.url,
                http_client=http_client,
            ) as client_streams:
                read_stream, write_stream, *_ = client_streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await action(session, *args)

    async def _list_tools(self, session) -> List[Dict[str, Any]]:
        """功能：在已初始化会话中请求工具清单并标准化输出。
        参数：
        - session：MCP 客户端会话对象。
        返回值：
        - List[Dict[str, Any]]：标准化后的工具定义列表。
        """
        response = await session.list_tools()
        tools = getattr(response, "tools", None) or []
        discovered = []
        for tool in tools:
            discovered.append(
                {
                    "name": getattr(tool, "name", ""),
                    "description": getattr(tool, "description", "") or "",
                    "input_schema": (
                        getattr(tool, "inputSchema", None)
                        or getattr(tool, "input_schema", None)
                        or {}
                    ),
                }
            )
        return discovered

    async def _call_tool(self, session, tool_name: str, arguments: Dict[str, Any]) -> str:
        """功能：在会话中执行工具调用并统一处理错误与返回格式。
        参数：
        - session：MCP 客户端会话对象。
        - tool_name：远端工具名称。
        - arguments：调用参数字典。
        返回值：
        - str：优先返回结构化内容 JSON 字符串，否则返回文本内容拼接结果。
        异常：
        - RuntimeError：远端工具返回错误时抛出。
        """
        response = await session.call_tool(tool_name, arguments=arguments)
        is_error = bool(
            getattr(response, "isError", False) or getattr(response, "is_error", False)
        )
        structured = (
            getattr(response, "structuredContent", None)
            or getattr(response, "structured_content", None)
        )
        if structured not in (None, "", {}, []):
            payload = json.dumps(_to_plain_data(structured), ensure_ascii=False)
        else:
            content = getattr(response, "content", None) or []
            texts = _collect_text_content(content)
            if texts:
                payload = "\n".join(texts)
            else:
                payload = json.dumps(_to_plain_data(response), ensure_ascii=False)

        if is_error:
            raise RuntimeError(payload or f"MCP 工具执行失败：{self.server_name}.{tool_name}")
        return payload
