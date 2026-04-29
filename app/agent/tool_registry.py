import inspect
import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Set


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


class ToolRegistry:
    """功能：管理工具注册、元数据查询与执行调度。
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
        """功能：注册可调用工具并生成签名、描述、领域等元信息索引。
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
        for func in tools:
            spec = ToolSpec(
                name=func.__name__,
                func=func,
                signature=str(inspect.signature(func)),
                description=inspect.getdoc(func) or "",
                domain=self._infer_domain(func.__name__),
                channel=self._infer_channel(func.__name__),
                returns_json=func.__name__ in self.direct_return_tools,
            )
            self._tools[spec.name] = spec

    @staticmethod
    def _infer_domain(tool_name: str) -> str:
        """功能：根据工具名推断业务域标签。
        参数：
        - tool_name：工具名称。
        返回值：
        - str：工具所属领域标识（如 borrow/knowledge/general）。
        """
        if tool_name.startswith("wecom_"):
            return "borrow"
        if "uds" in tool_name:
            return "uds"
        if tool_name.startswith("rag_"):
            return "knowledge"
        if "vehicle" in tool_name or tool_name == "get_current_date":
            return "vehicle"
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

    def execute(self, tool_name: str, args: List[Any]) -> Any:
        """功能：执行指定工具并记录观测信息。
        参数：
        - tool_name：工具名称。
        - args：传给脚本或工具的参数列表。
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

    def should_direct_return(self, tool_name: str) -> bool:
        """功能：判断工具结果是否应直接返回给上层调用方。
        参数：
        - tool_name：工具名称。
        返回值：
        - bool：命中直返集合时返回 True。
        """
        return tool_name in self.direct_return_tools

    def describe_tools(self) -> str:
        """功能：生成可供提示词注入的工具清单文本。
        参数：
        - 无。
        返回值：
        - str：每个工具一行的描述文本，附带元数据字段。
        """
        lines = []
        for spec in self._tools.values():
            meta = {
                "domain": spec.domain,
                "channel": spec.channel,
                "returns_json": spec.returns_json,
            }
            call_hint = f"{spec.name}(...)"
            lines.append(
                f"- {call_hint}: {spec.description}\n  metadata={json.dumps(meta, ensure_ascii=False)}"
            )
        return "\n".join(lines)
