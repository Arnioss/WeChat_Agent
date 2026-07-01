import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set

from app.agent.tool_metadata import (
    ToolRichMetadata,
    get_attached_rich_metadata,
    render_tool_block,
    summarize_routing_lines,
)


@dataclass(frozen=True)
class ToolSpec:
    """功能：描述单个工具的元信息与调用属性。
    参数：
    - 无。
    返回值：
    - 无。
    """
    name: str
    func: Callable[..., Any]
    signature: str
    description: str
    domain: str = "general"
    channel: str = "generic"
    returns_json: bool = False
    rich_metadata: Optional[ToolRichMetadata] = None


class ToolRegistry:
    """功能：管理工具注册、元数据查询、JSON 参数执行与 OpenAI schema 导出。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(
        self,
        tools: Iterable[Callable[..., Any]],
        *,
        direct_return_tools: Optional[Set[str]] = None,
        observer: Optional[Callable[[str, bool, float], None]] = None,
    ):
        """功能：注册可调用工具并生成签名、描述、领域及结构化元信息索引。
        参数：
        - tools：工具定义列表。
        - direct_return_tools：命中后可直接返回给用户的工具名集合。
        - observer：可选观测回调，接收工具名、成功状态和耗时毫秒。
        返回值：
        - 无。重复工具名会以后注册项覆盖前者，保持名称唯一映射。
        """
        self._tools: Dict[str, ToolSpec] = {}
        self.direct_return_tools = set(direct_return_tools or set())
        self.observer = observer
        self._selection_view_cache: Dict[Optional[frozenset[str]], str] = {}
        for func in tools:
            self.register(func)

    def register(
        self,
        func: Callable[..., Any],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        domain: Optional[str] = None,
        channel: Optional[str] = None,
        returns_json: Optional[bool] = None,
        rich_metadata: Optional[ToolRichMetadata] = None,
    ) -> None:
        """功能：将单个可调用对象注册为工具并更新索引缓存。
        参数：
        - func：工具函数。
        - name：可选覆盖工具名，默认取函数 __name__。
        - description：可选覆盖描述，默认取 docstring 或 rich summary。
        - domain：可选业务域标签。
        - channel：可选渠道标签。
        - returns_json：是否直返 JSON 结果。
        - rich_metadata：可选结构化工具元数据。
        返回值：
        - 无。重复工具名会覆盖前者。
        """
        tool_name = name or func.__name__
        doc = inspect.getdoc(func) or ""
        rich = rich_metadata if rich_metadata is not None else get_attached_rich_metadata(func)
        resolved_description = description if description is not None else doc
        if rich is not None and rich.summary.strip():
            resolved_description = rich.summary.strip()
        spec = ToolSpec(
            name=tool_name,
            func=func,
            signature=str(inspect.signature(func)),
            description=resolved_description or "",
            domain=domain or self._infer_domain(tool_name),
            channel=channel or self._infer_channel(tool_name),
            returns_json=(tool_name in self.direct_return_tools) if returns_json is None else returns_json,
            rich_metadata=rich,
        )
        self._tools[spec.name] = spec
        self._selection_view_cache.clear()

    @staticmethod
    def _infer_domain(tool_name: str) -> str:
        """功能：根据工具名推断业务域标签。
        参数：
        - tool_name：工具名称。
        返回值：
        - str：工具所属领域标识（如 knowledge/general）。
        """
        if tool_name.startswith("wecom_"):
            return "wecom"
        if tool_name.startswith("rag_"):
            return "knowledge"
        return "general"

    @staticmethod
    def _infer_channel(tool_name: str) -> str:
        """功能：根据工具名推断渠道标签。
        参数：
        - tool_name：工具名称。
        返回值：
        - str：渠道标识（如 wecom 或 generic）。
        """
        if tool_name.startswith("wecom_"):
            return "wecom"
        return "generic"

    def get(self, tool_name: str) -> ToolSpec:
        """功能：按名称获取工具规格定义。
        参数：
        - tool_name：工具名称。
        返回值：
        - ToolSpec：工具规格对象。
        """
        if tool_name not in self._tools:
            raise KeyError(f"工具不存在：{tool_name}")
        return self._tools[tool_name]

    def has(self, tool_name: str) -> bool:
        """功能：判断工具是否已注册。
        参数：
        - tool_name：工具名称。
        返回值：
        - bool：存在返回 True，否则返回 False。
        """
        return tool_name in self._tools

    def execute(self, tool_name: str, args: List[Any]) -> Any:
        """功能：以位置参数执行指定工具并记录观测信息。
        参数：
        - tool_name：工具名称。
        - args：传给工具的位置参数列表。
        返回值：
        - Any：工具函数返回值。
        """
        spec = self.get(tool_name)
        import time

        start = time.time()
        ok = False
        try:
            result = spec.func(*args)
            ok = True
            return result
        finally:
            if self.observer:
                self.observer(tool_name, ok, (time.time() - start) * 1000.0)

    def execute_json(self, tool_name: str, arguments: Optional[Mapping[str, Any]]) -> Any:
        """功能：将 JSON object 参数映射为调用参数后执行工具。
        参数：
        - tool_name：工具名称。
        - arguments：工具参数字典，可为 None 或空 dict。
        返回值：
        - Any：工具函数返回值。
        """
        spec = self.get(tool_name)
        args_map = arguments or {}
        positional = self._try_resolve_single_positional(spec, args_map)
        if positional is not None:
            return self.execute(tool_name, positional)
        kwargs = self.arguments_to_kwargs(spec, args_map)
        return self.execute_with_kwargs(tool_name, kwargs)

    def execute_with_kwargs(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        """功能：以关键字参数执行指定工具并记录观测信息。
        参数：
        - tool_name：工具名称。
        - kwargs：传给工具的关键字参数字典。
        返回值：
        - Any：工具函数返回值。
        """
        spec = self.get(tool_name)
        import time

        start = time.time()
        ok = False
        try:
            result = spec.func(**kwargs)
            ok = True
            return result
        finally:
            if self.observer:
                self.observer(tool_name, ok, (time.time() - start) * 1000.0)

    async def aexecute_json(self, tool_name: str, arguments: Optional[Mapping[str, Any]]) -> Any:
        """功能：execute_json 的异步版本；async 工具直接 await，sync 工具放线程池。
        参数：
        - tool_name：工具名称。
        - arguments：工具参数字典，可为 None 或空 dict。
        返回值：
        - Any：工具函数返回值。
        """
        import time
        spec = self.get(tool_name)
        args_map = arguments or {}
        positional = self._try_resolve_single_positional(spec, args_map)
        if positional is not None:
            kwargs: Dict[str, Any] = {}
            # positional 路径：包装成 kwargs 后统一处理
            # （positional 只在单参数工具出现，直接当作 to_thread 调用）
            start = time.time()
            ok = False
            try:
                if asyncio.iscoroutinefunction(spec.func):
                    result = await spec.func(*positional)
                else:
                    result = await asyncio.to_thread(spec.func, *positional)
                ok = True
                return result
            finally:
                if self.observer:
                    self.observer(tool_name, ok, (time.time() - start) * 1000.0)

        kwargs = self.arguments_to_kwargs(spec, args_map)
        start = time.time()
        ok = False
        try:
            if asyncio.iscoroutinefunction(spec.func):
                result = await spec.func(**kwargs)
            else:
                result = await asyncio.to_thread(spec.func, **kwargs)
            ok = True
            return result
        finally:
            if self.observer:
                self.observer(tool_name, ok, (time.time() - start) * 1000.0)

    @staticmethod
    def _callable_parameter_names(spec: ToolSpec) -> Dict[str, inspect.Parameter]:
        """功能：提取工具函数可接受的位置/关键字参数名映射。
        参数：
        - spec：工具规格对象。
        返回值：
        - Dict[str, inspect.Parameter]：参数名到 Parameter 的映射。
        """
        return {
            param.name: param
            for param in inspect.signature(spec.func).parameters.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        }

    @staticmethod
    def arguments_to_kwargs(spec: ToolSpec, arguments: Mapping[str, Any]) -> Dict[str, Any]:
        """功能：按函数签名将 JSON 参数转换为关键字参数字典。
        参数：
        - spec：工具规格对象。
        - arguments：模型传入的参数字典。
        返回值：
        - Dict[str, Any]：过滤后的关键字参数字典；缺少必填参数时抛出 ValueError。
        """
        accepted = ToolRegistry._callable_parameter_names(spec)
        for name, param in accepted.items():
            if param.default is inspect.Parameter.empty and name not in arguments:
                raise ValueError(f"工具 {spec.name} 缺少参数：{name}")
        return {name: arguments[name] for name in arguments if name in accepted}

    @staticmethod
    def _try_resolve_single_positional(
        spec: ToolSpec,
        arguments: Mapping[str, Any],
    ) -> Optional[List[Any]]:
        """功能：单参数工具保留位置传参；多参数工具返回 None 以走关键字传参。
        参数：
        - spec：工具规格对象。
        - arguments：模型传入的参数字典。
        返回值：
        - Optional[List[Any]]：单参数位置参数列表；无法解析时返回 None。
        """
        signature = inspect.signature(spec.func)
        params = [
            param
            for param in signature.parameters.values()
            if param.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if not params:
            return []
        if len(params) != 1:
            return None

        param = params[0]
        if param.name in arguments:
            return [arguments[param.name]]

        props: Dict[str, Any] = {}
        if spec.rich_metadata is not None:
            raw_props = spec.rich_metadata.input_schema.get("properties")
            if isinstance(raw_props, dict):
                props = raw_props
            if spec.rich_metadata.source == "mcp":
                return [dict(arguments)]
        if len(arguments) == 1:
            return [next(iter(arguments.values()))]
        if len(props) == 1:
            only_name = next(iter(props))
            if only_name in arguments:
                return [arguments[only_name]]
        if arguments and param.default is not inspect.Parameter.empty:
            return [dict(arguments)]
        return None

    def should_direct_return(self, tool_name: str) -> bool:
        """功能：判断工具结果是否应直接返回给上层调用方。
        参数：
        - tool_name：工具名称。
        返回值：
        - bool：命中直返集合时返回 True。
        """
        return tool_name in self.direct_return_tools

    def iter_specs_by_priority(self) -> List[ToolSpec]:
        """功能：按 priority 降序返回工具规格列表，便于模型优先阅读高价值工具。
        参数：
        - 无。
        返回值：
        - List[ToolSpec]：按 priority 降序排列的工具规格列表。
        """

        def sort_key(s: ToolSpec) -> int:
            """功能：提取工具 priority 作为排序键。
            参数：
            - s：工具规格对象。
            返回值：
            - int：rich_metadata.priority；无元数据时返回 0。
            """
            if s.rich_metadata is not None:
                return s.rich_metadata.priority
            return 0

        return sorted(self._tools.values(), key=sort_key, reverse=True)

    def describe_routing_hints(self) -> str:
        """功能：生成工具路由提示摘要文本。
        参数：
        - 无。
        返回值：
        - str：面向模型的工具选择准则与按优先级排列的工具摘要。
        """
        return summarize_routing_lines(self.iter_specs_by_priority())

    def describe_tools(self, enabled_tool_names: Optional[Iterable[str]] = None) -> str:
        """功能：生成可供提示词注入的统一格式工具清单文本。
        参数：
        - enabled_tool_names：可选工具名白名单；为空时展示全部工具。
        返回值：
        - str：每个工具一个展示块，含来源、场景、schema 与元信息。
        """
        enabled = set(enabled_tool_names) if enabled_tool_names is not None else None
        blocks = [
            render_tool_block(
                name=spec.name,
                signature=spec.signature,
                description_fallback=spec.description,
                domain=spec.domain,
                channel=spec.channel,
                returns_json=spec.returns_json,
                rich=spec.rich_metadata,
            )
            for spec in self.iter_specs_by_priority()
            if enabled is None or spec.name in enabled
        ]
        return "\n\n".join(blocks)

    def describe_selection_view(self, enabled_tool_names: Optional[Iterable[str]] = None) -> str:
        """功能：生成供 context selector 使用的紧凑工具元数据视图。
        参数：
        - enabled_tool_names：可选工具名白名单；为空时展示全部工具。
        返回值：
        - str：每行一个工具的摘要、来源、域与参数字段提示。
        """
        enabled = set(enabled_tool_names) if enabled_tool_names is not None else None
        cache_key = frozenset(enabled) if enabled is not None else None
        if cache_key in self._selection_view_cache:
            return self._selection_view_cache[cache_key]
        lines: List[str] = []
        for spec in self.iter_specs_by_priority():
            if enabled is not None and spec.name not in enabled:
                continue
            schema = self._input_schema_for_openai(spec)
            properties = schema.get("properties") if isinstance(schema, dict) else {}
            required = schema.get("required") if isinstance(schema, dict) else []
            required_set = set(required) if isinstance(required, list) else set()
            field_bits: List[str] = []
            if isinstance(properties, dict):
                for prop_name, prop in list(properties.items())[:6]:
                    prop = prop if isinstance(prop, dict) else {}
                    marker = "*" if prop_name in required_set else ""
                    prop_desc = str(prop.get("description") or "").strip()
                    suffix = f":{prop_desc[:48]}" if prop_desc else ""
                    field_bits.append(f"{prop_name}{marker}{suffix}")
            rich = spec.rich_metadata
            when = ""
            if rich is not None and rich.when_to_use:
                when = " use=" + "; ".join(item.strip()[:90] for item in rich.when_to_use[:1] if item.strip())
            source = rich.source if rich is not None else "local"
            summary = (spec.description or "").strip().replace("\n", " ")
            fields = f" args=[{', '.join(field_bits)}]" if field_bits else ""
            lines.append(
                f"- {spec.name}: {summary[:140]} | {source}/{spec.domain}{when}{fields}"
            )
        rendered = "\n".join(lines) if lines else "(no tools registered)"
        self._selection_view_cache[cache_key] = rendered
        return rendered

    def openai_tools_schema(self, enabled_tool_names: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        """功能：导出 OpenAI function calling 所需的 tools schema 列表。
        参数：
        - enabled_tool_names：可选工具名白名单；为空时导出全部工具。
        返回值：
        - List[Dict[str, Any]]：按 priority 排序的 function 工具定义列表。
        """
        enabled = set(enabled_tool_names) if enabled_tool_names is not None else None
        tools: List[Dict[str, Any]] = []
        for spec in self.iter_specs_by_priority():
            if enabled is not None and spec.name not in enabled:
                continue
            schema = self._input_schema_for_openai(spec)
            description = self._openai_tool_description(spec)
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": description,
                        "parameters": schema,
                    },
                }
            )
        return tools

    @staticmethod
    def _openai_tool_description(spec: ToolSpec) -> str:
        """功能：组装 OpenAI function 工具的 description 字段文本。
        参数：
        - spec：工具规格对象。
        返回值：
        - str：含摘要、适用/不适用场景与返回说明，最长 1600 字符。
        """
        parts: List[str] = [spec.description.strip() or f"调用工具 {spec.name}。"]
        rich = spec.rich_metadata
        if rich is not None:
            if rich.when_to_use:
                parts.append("适用场景：" + "；".join(item.strip() for item in rich.when_to_use if item.strip()))
            if rich.when_not_to_use:
                parts.append("不适用场景：" + "；".join(item.strip() for item in rich.when_not_to_use if item.strip()))
            if rich.output_description.strip():
                parts.append("返回：" + rich.output_description.strip())
        text = "\n".join(part for part in parts if part)
        return text[:1600]

    @staticmethod
    def _input_schema_for_openai(spec: ToolSpec) -> Dict[str, Any]:
        """功能：为 OpenAI function calling 生成 parameters JSON Schema。
        参数：
        - spec：工具规格对象。
        返回值：
        - Dict[str, Any]：优先使用 rich_metadata.input_schema，否则从函数签名推断。
        """
        if spec.rich_metadata is not None and isinstance(spec.rich_metadata.input_schema, dict):
            return ToolRegistry._sanitize_json_schema(dict(spec.rich_metadata.input_schema))

        signature = inspect.signature(spec.func)
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for param in signature.parameters.values():
            if param.kind not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                continue
            properties[param.name] = {
                "type": "string",
                "description": f"参数 {param.name}",
            }
            if param.default is inspect.Parameter.empty:
                required.append(param.name)

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }
        if required:
            schema["required"] = required
        return schema

    @staticmethod
    def _sanitize_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
        """功能：规范化 JSON Schema，确保符合 OpenAI object 参数要求。
        参数：
        - schema：原始 JSON Schema 字典。
        返回值：
        - Dict[str, Any]：含 type/properties/additionalProperties 的合法 schema。
        """
        if schema.get("type") != "object":
            schema = {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
                "description": json.dumps(schema, ensure_ascii=False, default=str),
            }
        if not isinstance(schema.get("properties"), dict):
            schema["properties"] = {}
        if "additionalProperties" not in schema:
            schema["additionalProperties"] = False
        required = schema.get("required")
        if required is not None and not isinstance(required, list):
            schema.pop("required", None)
        return schema
