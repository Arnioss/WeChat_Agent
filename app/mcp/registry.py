import asyncio
import json
import os
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.mcp.client import StreamableHttpMcpClient


@dataclass(frozen=True)
class McpServerConfig:
    """功能：描述单个 MCP 服务连接配置。
    参数：
    - 无。
    返回值：
    - 无。
    """
    name: str
    url: str
    enabled: bool = True
    headers: Dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 20.0
    tool_allowlist: List[str] = field(default_factory=list)
    tool_denylist: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class McpToolDefinition:
    """功能：描述 MCP 工具在本地运行时中的映射信息。
    参数：
    - 无。
    返回值：
    - 无。
    """
    server_name: str
    remote_name: str
    local_name: str
    description: str
    input_schema: Dict[str, Any] = field(default_factory=dict)
    server_url: str = ""
    from_cache: bool = False


def _normalize_tool_name(name: str) -> str:
    """功能：将远端工具名规范化为合法的 Python 标识符片段。
    参数：
    - name：原始工具名。
    返回值：
    - str：小写、仅含字母数字下划线的名称；空输入时返回 "tool"。
    """
    normalized = re.sub(r"[^0-9A-Za-z_]+", "_", (name or "").strip())
    normalized = normalized.strip("_").lower() or "tool"
    if normalized[0].isdigit():
        normalized = f"tool_{normalized}"
    return normalized


def _to_bool(value: Any, *, default: bool = True) -> bool:
    """功能：将配置值解析为布尔值。
    参数：
    - value：原始配置值。
    - default：value 为 None 时的默认值。
    返回值：
    - bool：解析结果。
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _unwrap_error(exc: BaseException) -> BaseException:
    """功能：递归展开 ExceptionGroup 等嵌套异常，取最内层根因。
    参数：
    - exc：原始异常对象。
    返回值：
    - BaseException：最内层异常。
    """
    nested = getattr(exc, "exceptions", None)
    if nested:
        return _unwrap_error(nested[0])
    return exc


class McpToolRegistry:
    """功能：管理 MCP 服务配置、工具发现、缓存与调用。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self, *, project_directory: str):
        """功能：加载 MCP 服务配置并准备按需创建的客户端与工具定义缓存。
        参数：
        - project_directory：项目根目录路径。
        返回值：
        - 无。仅保留已启用且配置完整的服务，异常配置会在加载阶段被跳过。
        """
        self.project_directory = Path(project_directory).resolve()
        self._configs = self._load_server_configs()
        self._clients: Dict[str, StreamableHttpMcpClient] = {}
        self._tool_definitions_cache: Optional[List[McpToolDefinition]] = None

    def list_tool_definitions(self, *, force_refresh: bool = False) -> List[McpToolDefinition]:
        """功能：汇总所有可用 MCP 工具定义并应用过滤规则，支持内存与磁盘缓存。
        参数：
        - force_refresh：为 True 时强制重新发现远端工具并刷新缓存。
        返回值：
        - List[McpToolDefinition]：最终可供本地注册的工具定义列表。
        """
        if self._tool_definitions_cache is not None and not force_refresh:
            return list(self._tool_definitions_cache)

        cache_payload = self._read_cache()
        discovered_tools: List[McpToolDefinition] = []

        for config in self._configs:
            tools: List[McpToolDefinition]
            if not force_refresh:
                tools = self._load_cached_tools(cache_payload, config)
                if not tools:
                    try:
                        tools = self._discover_server_tools(config)
                        self._write_server_cache(cache_payload, config, tools)
                    except Exception:
                        tools = []
            else:
                cached_tools = self._load_cached_tools(cache_payload, config)
                try:
                    tools = self._discover_server_tools(config)
                    self._write_server_cache(cache_payload, config, tools)
                except Exception:
                    tools = cached_tools

            discovered_tools.extend(self._apply_filters(tools, config))

        self._tool_definitions_cache = list(discovered_tools)
        return list(discovered_tools)

    async def call_tool(self, tool: McpToolDefinition, arguments: Dict[str, Any]) -> str:
        """功能：异步调用指定 MCP 工具并统一包装异常信息。
        参数：
        - tool：MCP 工具定义对象。
        - arguments：工具参数字典。
        返回值：
        - str：工具调用返回文本。
        异常：
        - RuntimeError：调用失败时抛出并附带服务/工具标识。
        """
        client = self._get_client(tool)
        try:
            return await client.call_tool(tool.remote_name, arguments)
        except Exception as exc:
            root_exc = _unwrap_error(exc)
            raise RuntimeError(
                f"MCP 工具调用失败（{tool.server_name}.{tool.remote_name}）：{root_exc}"
            ) from root_exc

    def _discover_server_tools(self, config: McpServerConfig) -> List[McpToolDefinition]:
        """功能：连接远端 MCP 服务并发现其暴露的工具列表。
        参数：
        - config：MCP 服务连接配置。
        返回值：
        - List[McpToolDefinition]：工具定义列表。
        """
        # 在无运行中事件循环的线程里调用（如 asyncio.to_thread 预热），asyncio.run 可安全执行。
        client = self._get_client_by_config(config)
        tool_rows = asyncio.run(client.list_tools())
        return [
            McpToolDefinition(
                server_name=config.name,
                remote_name=str(row.get("name") or ""),
                local_name=self._build_local_name(config.name, str(row.get("name") or "")),
                description=str(row.get("description") or ""),
                input_schema=dict(row.get("input_schema") or {}),
                server_url=config.url,
                from_cache=False,
            )
            for row in tool_rows
            if row.get("name")
        ]

    def _load_server_configs(self) -> List[McpServerConfig]:
        """功能：从环境变量或配置文件加载并过滤已启用的 MCP 服务配置。
        参数：
        - 无。
        返回值：
        - List[McpServerConfig]：已启用且配置完整的服务列表。
        """
        raw_servers = self._load_raw_server_configs()
        configs: List[McpServerConfig] = []
        for item in raw_servers:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            url = str(item.get("url") or "").strip()
            if not name or not url:
                continue
            configs.append(
                McpServerConfig(
                    name=name,
                    url=url,
                    enabled=_to_bool(item.get("enabled"), default=True),
                    headers=dict(item.get("headers") or {}),
                    timeout_seconds=float(item.get("timeout_seconds") or 20.0),
                    tool_allowlist=[str(v) for v in item.get("tool_allowlist") or []],
                    tool_denylist=[str(v) for v in item.get("tool_denylist") or []],
                )
            )
        return [config for config in configs if config.enabled]

    def _load_raw_server_configs(self) -> List[Dict[str, Any]]:
        """功能：读取 MCP 服务原始配置（优先环境变量 MCP_SERVERS_JSON）。
        参数：
        - 无。
        返回值：
        - List[Dict[str, Any]]：规范化前的服务配置字典列表。
        """
        env_payload = (os.getenv("MCP_SERVERS_JSON") or "").strip()
        if env_payload:
            return self._normalize_raw_server_configs(json.loads(env_payload))

        config_path = os.getenv("MCP_CONFIG_PATH") or str(
            self.project_directory / "config" / "mcp_servers.json"
        )
        path = Path(config_path)
        if not path.is_absolute():
            path = self.project_directory / path
        if not path.exists():
            return []
        return self._normalize_raw_server_configs(
            json.loads(path.read_text(encoding="utf-8"))
        )

    @staticmethod
    def _normalize_raw_server_configs(payload: Any) -> List[Dict[str, Any]]:
        """功能：将多种 MCP 配置格式统一为 Streamable HTTP 服务列表。
        参数：
        - payload：JSON 解析后的配置（list 或 mcpServers 字典）。
        返回值：
        - List[Dict[str, Any]]：仅含 url 配置的服务字典列表；stdio 配置会被跳过。
        """
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]

        if not isinstance(payload, dict):
            return []

        server_map = payload.get("mcpServers")
        if not isinstance(server_map, dict):
            return []

        normalized: List[Dict[str, Any]] = []
        for server_name, server_config in server_map.items():
            if not isinstance(server_config, dict):
                continue

            if server_config.get("command"):
                warnings.warn(
                    f"MCP server `{server_name}` 使用 stdio/command 配置，"
                    "当前项目仅支持 Streamable HTTP `url` 配置，已跳过。"
                )
                continue

            url = str(server_config.get("url") or "").strip()
            if not url:
                continue

            normalized.append(
                {
                    "name": str(server_name).strip(),
                    "url": url,
                    "enabled": _to_bool(server_config.get("enabled"), default=True),
                    "headers": dict(server_config.get("headers") or {}),
                    "timeout_seconds": float(server_config.get("timeout_seconds") or 20.0),
                    "tool_allowlist": [
                        str(v) for v in server_config.get("tool_allowlist") or []
                    ],
                    "tool_denylist": [
                        str(v) for v in server_config.get("tool_denylist") or []
                    ],
                }
            )
        return normalized

    def _apply_filters(
        self, tools: List[McpToolDefinition], config: McpServerConfig
    ) -> List[McpToolDefinition]:
        """功能：按服务的 allowlist/denylist 过滤工具定义。
        参数：
        - tools：待过滤的工具定义列表。
        - config：MCP 服务配置（含过滤规则）。
        返回值：
        - List[McpToolDefinition]：过滤后的工具列表。
        """
        allowlist = set(config.tool_allowlist)
        denylist = set(config.tool_denylist)
        filtered: List[McpToolDefinition] = []
        for tool in tools:
            if allowlist and tool.remote_name not in allowlist:
                continue
            if denylist and tool.remote_name in denylist:
                continue
            filtered.append(tool)
        return filtered

    def _get_client(self, tool: McpToolDefinition) -> StreamableHttpMcpClient:
        """功能：根据工具定义查找对应 MCP 服务并返回客户端。
        参数：
        - tool：MCP 工具定义对象。
        返回值：
        - StreamableHttpMcpClient：可复用的 HTTP MCP 客户端。
        异常：
        - KeyError：服务未配置时抛出。
        """
        config = next(
            (item for item in self._configs if item.name == tool.server_name),
            None,
        )
        if config is None:
            raise KeyError(f"MCP server 未配置：{tool.server_name}")
        return self._get_client_by_config(config)

    def _get_client_by_config(self, config: McpServerConfig) -> StreamableHttpMcpClient:
        """功能：按服务配置获取或创建缓存的 MCP 客户端实例。
        参数：
        - config：MCP 服务连接配置。
        返回值：
        - StreamableHttpMcpClient：该服务的 HTTP MCP 客户端。
        """
        client = self._clients.get(config.name)
        if client is None:
            client = StreamableHttpMcpClient(
                server_name=config.name,
                url=config.url,
                headers=config.headers,
                timeout_seconds=config.timeout_seconds,
            )
            self._clients[config.name] = client
        return client

    def _cache_path(self) -> Path:
        """功能：返回 MCP 工具发现结果的磁盘缓存路径。
        参数：
        - 无。
        返回值：
        - Path：`.mcp_cache/tool_cache.json` 的绝对路径。
        """
        return self.project_directory / ".mcp_cache" / "tool_cache.json"

    def _legacy_cache_path(self) -> Path:
        """功能：返回旧版 MCP 工具缓存文件路径（兼容迁移）。
        参数：
        - 无。
        返回值：
        - Path：`.mcp_tool_cache.json` 的绝对路径。
        """
        return self.project_directory / ".mcp_tool_cache.json"

    def _read_cache(self) -> Dict[str, Any]:
        """功能：读取 MCP 工具发现缓存，必要时从旧路径迁移。
        参数：
        - 无。
        返回值：
        - Dict[str, Any]：缓存 payload；读取失败时返回 `{"servers": {}}`。
        """
        path = self._cache_path()
        if not path.exists():
            legacy_path = self._legacy_cache_path()
            if legacy_path.exists():
                try:
                    payload = json.loads(legacy_path.read_text(encoding="utf-8"))
                    self._ensure_cache_dir()
                    path.write_text(
                        json.dumps(payload, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    return payload
                except Exception:
                    return {"servers": {}}
            return {"servers": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {"servers": {}}

    def _ensure_cache_dir(self) -> None:
        """功能：确保 MCP 缓存目录存在。
        参数：
        - 无。
        返回值：
        - 无。
        """
        self._cache_path().parent.mkdir(parents=True, exist_ok=True)

    def _write_server_cache(
        self,
        cache_payload: Dict[str, Any],
        config: McpServerConfig,
        tools: List[McpToolDefinition],
    ) -> None:
        """功能：将单个 MCP 服务的工具发现结果写入磁盘缓存。
        参数：
        - cache_payload：完整缓存字典（会被原地更新）。
        - config：MCP 服务配置。
        - tools：该服务发现的工具定义列表。
        返回值：
        - 无。写入失败时静默忽略。
        """
        cache_payload.setdefault("servers", {})
        cache_payload["servers"][config.name] = {
            "name": config.name,
            "url": config.url,
            "headers": config.headers,
            "timeout_seconds": config.timeout_seconds,
            "tools": [asdict(tool) for tool in tools],
        }
        try:
            self._ensure_cache_dir()
            self._cache_path().write_text(
                json.dumps(cache_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            return

    def _load_cached_tools(
        self,
        cache_payload: Dict[str, Any],
        config: McpServerConfig,
    ) -> List[McpToolDefinition]:
        """功能：从磁盘缓存加载指定 MCP 服务的工具定义。
        参数：
        - cache_payload：完整缓存字典。
        - config：MCP 服务配置。
        返回值：
        - List[McpToolDefinition]：缓存中的工具定义列表。
        """
        server_cache = (cache_payload.get("servers") or {}).get(config.name) or {}
        tools = server_cache.get("tools") or []
        loaded: List[McpToolDefinition] = []
        for item in tools:
            if not isinstance(item, dict):
                continue
            remote_name = str(item.get("remote_name") or item.get("name") or "").strip()
            if not remote_name:
                continue
            loaded.append(
                McpToolDefinition(
                    server_name=config.name,
                    remote_name=remote_name,
                    local_name=str(
                        item.get("local_name")
                        or self._build_local_name(config.name, remote_name)
                    ),
                    description=str(item.get("description") or ""),
                    input_schema=dict(item.get("input_schema") or {}),
                    server_url=config.url,
                    from_cache=True,
                )
            )
        return loaded

    @staticmethod
    def _build_local_name(server_name: str, remote_name: str) -> str:
        """功能：生成 MCP 工具在本地 Agent 中的注册名称。
        参数：
        - server_name：MCP 服务名。
        - remote_name：远端工具名。
        返回值：
        - str：形如 `mcp_{server}_{tool}` 的本地工具名。
        """
        return f"mcp_{_normalize_tool_name(server_name)}_{_normalize_tool_name(remote_name)}"
