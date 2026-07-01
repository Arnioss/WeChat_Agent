"""Agent 上下文选择器：用小模型为 Main Agent 筛选候选 skill 与 tool。"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Protocol

from app.agent.model_client import ModelClient, log_print
from app.agent.tool_registry import ToolRegistry
from app.skills.models import SkillMetadata
from app.skills.system import SkillSystem


BASE_CONTEXT_TOOL_NAMES = (
    "get_current_date",
    "load_skill_instructions",
    "list_skill_resources",
    "load_skill_reference",
    "run_skill_script",
    "expand_tool_context",
)


@dataclass(frozen=True)
class ContextCandidateSkill:
    """功能：表示上下文选择器输出的单个 skill 候选项。
    参数：
    - id：skill 名称/标识符。
    - confidence：模型对该 skill 相关性的置信度，范围 0.0–1.0。
    - reason：选择该 skill 的简要理由。
    返回值：
    - 无（数据类实例，供 ContextSelectionResult 聚合使用）。
    """

    id: str
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class ContextCandidateTool:
    """功能：表示上下文选择器输出的单个 tool 候选项。
    参数：
    - name：工具注册名。
    - confidence：模型对该 tool 相关性的置信度，范围 0.0–1.0。
    - reason：选择该 tool 的简要理由。
    返回值：
    - 无（数据类实例，供 ContextSelectionResult 聚合使用）。
    """

    name: str
    confidence: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class ContextSelectionResult:
    """功能：封装上下文选择器的完整输出，供 Main Agent 构建受限工具上下文。
    参数：
    - version：结果 schema 版本，默认 context_selector.v1。
    - intent_summary：对用户意图的简要概括。
    - candidate_skills：候选 skill 列表。
    - candidate_tools：候选 tool 列表。
    - coverage_confidence：候选集覆盖用户需求的置信度。
    - coverage_notes：覆盖度相关补充说明。
    - diagnostics：诊断信息（如 fallback 原因）。
    - fallback_all_tools：为 True 时主 Agent 应暴露全部工具。
    返回值：
    - 无（数据类实例，由 select/aselect 返回）。
    """

    version: str = "context_selector.v1"
    intent_summary: str = ""
    candidate_skills: tuple[ContextCandidateSkill, ...] = ()
    candidate_tools: tuple[ContextCandidateTool, ...] = ()
    coverage_confidence: float = 0.0
    coverage_notes: str = ""
    diagnostics: tuple[str, ...] = ()
    fallback_all_tools: bool = False


class ContextSelector(Protocol):
    """功能：定义上下文选择器的同步选择接口，供 Agent 运行时注入不同实现。
    参数：
    - 无（Protocol 仅声明方法签名）。
    返回值：
    - 无。
    """

    def select(
        self,
        *,
        user_input: str,
        additional_need: str = "",
        conversation_context: str = "",
        explicit_skill: Optional[str] = None,
    ) -> ContextSelectionResult:
        """功能：根据用户输入与对话上下文，同步生成候选 skill/tool 短名单。
        参数：
        - user_input：用户当前输入。
        - additional_need：主模型补充的能力需求说明。
        - conversation_context：近期对话摘要。
        - explicit_skill：用户显式指定的 skill 名（/skill 调用）。
        返回值：
        - ContextSelectionResult：候选列表及覆盖度；失败或未配置时可 fallback 全部工具。
        """
        ...


class ModelContextSelector:
    """功能：调用轻量模型，根据用户输入从已注册 skill/tool 中生成候选短名单。
    参数：
    - 无（通过 __init__ 注入 tool_registry、model_client、skill_system）。
    返回值：
    - 无。
    """

    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        model_client: ModelClient,
        skill_system: Optional[SkillSystem] = None,
    ):
        """功能：初始化基于轻量模型的上下文选择器，注入工具注册表与模型客户端。
        参数：
        - tool_registry：已注册工具的 ToolRegistry，用于列举可选 tool。
        - model_client：调用 TOOL_SELECTOR_MODEL 的 ModelClient。
        - skill_system：可选 SkillSystem，用于列举 skill 元数据；缺省时无 skill 候选。
        返回值：
        - 无。
        """
        self.tool_registry = tool_registry
        self.model_client = model_client
        self.skill_system = skill_system
        self._skill_selection_view_cache: Optional[str] = None
        self.last_prompt_chars: int = 0
        self.last_elapsed_ms: float = 0.0

    async def aselect(
        self,
        *,
        user_input: str,
        additional_need: str = "",
        conversation_context: str = "",
        explicit_skill: Optional[str] = None,
    ) -> ContextSelectionResult:
        """功能：异步版上下文选择，供 AgentRuntime.arun() 调用，不阻塞事件循环。
        参数：
        - user_input：用户当前输入。
        - additional_need：主模型补充的能力需求说明。
        - conversation_context：近期对话摘要。
        - explicit_skill：用户显式指定的 skill 名（/skill 调用）。
        返回值：
        - ContextSelectionResult：候选 skill/tool 及覆盖度；失败或未配置时 fallback_all_tools=True。
        """
        model = (os.getenv("TOOL_SELECTOR_MODEL") or "").strip()
        if explicit_skill:
            return self._explicit_skill_selection(explicit_skill, user_input=user_input)
        if not model:
            return ContextSelectionResult(
                intent_summary="selector model is not configured",
                diagnostics=("TOOL_SELECTOR_MODEL not configured; exposing all tools",),
                fallback_all_tools=True,
            )
        call_text_model = getattr(self.model_client, "call_text_model", None)
        if not callable(call_text_model):
            return ContextSelectionResult(
                intent_summary="selector model client is unavailable",
                diagnostics=("model client does not support selector calls; exposing all tools",),
                fallback_all_tools=True,
            )

        messages = self._build_messages(
            user_input=user_input,
            additional_need=additional_need,
            conversation_context=conversation_context,
        )
        self.last_prompt_chars = sum(len(str(item.get("content") or "")) for item in messages)
        timeout = self._env_float("TOOL_SELECTOR_TIMEOUT_SECONDS", default=8.0, minimum=0.5)
        started_at = time.time()
        try:
            raw = await call_text_model(
                messages,
                model=model,
                timeout_seconds=timeout,
                purpose="context_selector",
            )
            self.last_elapsed_ms = (time.time() - started_at) * 1000.0
            parsed = self._parse_selector_json(str(raw or ""))
            result = self._result_from_payload(parsed)
            return self._filter_result(result)
        except Exception as exc:
            self.last_elapsed_ms = (time.time() - started_at) * 1000.0
            log_print(f"Context selector failed; exposing all tools. reason={exc}")
            return ContextSelectionResult(
                intent_summary="selector failed",
                diagnostics=(f"selector failed: {exc}",),
                fallback_all_tools=True,
            )

    def select(
        self,
        *,
        user_input: str,
        additional_need: str = "",
        conversation_context: str = "",
        explicit_skill: Optional[str] = None,
    ) -> ContextSelectionResult:
        """功能：同步版上下文选择，调用 TOOL_SELECTOR_MODEL 生成候选 skill/tool 短名单。
        参数：
        - user_input：用户当前输入。
        - additional_need：主模型补充的能力需求说明。
        - conversation_context：近期对话摘要。
        - explicit_skill：用户显式指定的 skill 名。
        返回值：
        - ContextSelectionResult：候选列表；未配置或失败时暴露全部工具。
        """
        model = (os.getenv("TOOL_SELECTOR_MODEL") or "").strip()
        if explicit_skill:
            return self._explicit_skill_selection(explicit_skill, user_input=user_input)
        if not model:
            return ContextSelectionResult(
                intent_summary="selector model is not configured",
                diagnostics=("TOOL_SELECTOR_MODEL not configured; exposing all tools",),
                fallback_all_tools=True,
            )
        call_text_model = getattr(self.model_client, "call_text_model", None)
        if not callable(call_text_model):
            return ContextSelectionResult(
                intent_summary="selector model client is unavailable",
                diagnostics=("model client does not support selector calls; exposing all tools",),
                fallback_all_tools=True,
            )

        messages = self._build_messages(
            user_input=user_input,
            additional_need=additional_need,
            conversation_context=conversation_context,
        )
        self.last_prompt_chars = sum(len(str(item.get("content") or "")) for item in messages)
        timeout = self._env_float("TOOL_SELECTOR_TIMEOUT_SECONDS", default=8.0, minimum=0.5)
        try:
            started_at = time.time()
            raw = call_text_model(
                messages,
                model=model,
                timeout_seconds=timeout,
                purpose="context_selector",
            )
            self.last_elapsed_ms = (time.time() - started_at) * 1000.0
            parsed = self._parse_selector_json(str(raw or ""))
            result = self._result_from_payload(parsed)
            return self._filter_result(result)
        except Exception as exc:
            self.last_elapsed_ms = (time.time() - started_at) * 1000.0 if "started_at" in locals() else 0.0
            log_print(f"Context selector failed; exposing all tools. reason={exc}")
            return ContextSelectionResult(
                intent_summary="selector failed",
                diagnostics=(f"selector failed: {exc}",),
                fallback_all_tools=True,
            )

    def _explicit_skill_selection(self, explicit_skill: str, *, user_input: str) -> ContextSelectionResult:
        """功能：处理用户显式 /skill 调用，直接返回指定 skill 及其关联工具，跳过模型选择。
        参数：
        - explicit_skill：用户指定的 skill 名称。
        - user_input：用户当前输入，写入 coverage_notes。
        返回值：
        - ContextSelectionResult：单一 skill 候选、基础工具及 skill 允许的工具列表。
        """
        metadata = self._skill_metadata_by_name().get(explicit_skill)
        tools = list(BASE_CONTEXT_TOOL_NAMES)
        if metadata is not None:
            tools.extend(metadata.allowed_tools)
        return ContextSelectionResult(
            intent_summary=f"user explicitly requested skill {explicit_skill}",
            candidate_skills=(
                ContextCandidateSkill(
                    id=explicit_skill,
                    confidence=1.0,
                    reason="explicit /skill invocation",
                ),
            ),
            candidate_tools=tuple(
                ContextCandidateTool(name=name, confidence=1.0, reason="explicit skill context")
                for name in self._dedupe(tools)
            ),
            coverage_confidence=1.0,
            coverage_notes=user_input,
            diagnostics=(f"explicit skill candidate: {explicit_skill}",),
        )

    def _build_messages(
        self,
        *,
        user_input: str,
        additional_need: str,
        conversation_context: str = "",
    ) -> list[dict[str, str]]:
        """功能：组装发给 TOOL_SELECTOR_MODEL 的 system/user 消息，含 skill/tool 元数据视图。
        参数：
        - user_input：用户当前输入。
        - additional_need：主模型补充的能力需求说明。
        - conversation_context：近期对话摘要。
        返回值：
        - list[dict[str, str]]：OpenAI 风格的 messages 列表（role + content）。
        """
        need_block = f"\nAdditional capability need from main model:\n{additional_need}" if additional_need else ""
        history_block = (
            f"\nRecent conversation:\n{conversation_context}\n"
            if conversation_context
            else ""
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a fast context selector for an agent runtime. "
                    "Shortlist candidate skills and candidate tools using only the metadata provided. "
                    "Do not create tool arguments, execution plans, or final answers. "
                    "Return only valid JSON matching this shape: "
                    '{"version":"context_selector.v1","intent_summary":"...","candidate_skills":[{"id":"skill-name","confidence":0.0,"reason":"..."}],'
                    '"candidate_tools":[{"name":"tool_name","confidence":0.0,"reason":"..."}],"coverage":{"confidence":0.0,"notes":"..."}}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request:\n{user_input}{history_block}{need_block}\n\n"
                    f"Available skills:\n{self._render_skill_selection_view()}\n\n"
                    f"Available tools:\n{self.tool_registry.describe_selection_view()}\n\n"
                    "Rules:\n"
                    "- Pick only names that appear in the available lists.\n"
                    "- Include date/time helper tools when relative dates are present.\n"
                    "- Include expand_tool_context only if it exists; it lets the main model request more tools later.\n"
                    "- Prefer a small but sufficient candidate set. The main model will decide whether to use them."
                ),
            },
        ]

    def _render_skill_selection_view(self) -> str:
        """功能：将已注册 skill 元数据格式化为供选择模型阅读的文本视图，结果会缓存。
        参数：
        - 无（使用 self.skill_system 与实例级缓存）。
        返回值：
        - str：每行一个 skill 的摘要文本；无 skill 时返回 "(no skills registered)"。
        """
        if self._skill_selection_view_cache is not None:
            return self._skill_selection_view_cache
        metadata = self._skill_metadata()
        if not metadata:
            return "(no skills registered)"
        lines: list[str] = []
        for item in metadata:
            parts = [f"- {item.name}: {self._compact(item.description, 220)}"]
            if item.tags:
                parts.append(f"tags={', '.join(item.tags[:8])}")
            if item.keywords:
                parts.append(f"keywords={', '.join(item.keywords[:12])}")
            if item.aliases:
                parts.append(f"aliases={', '.join(item.aliases[:8])}")
            if item.examples:
                parts.append(f"examples={'; '.join(self._compact(ex, 120) for ex in item.examples[:3])}")
            if item.allowed_tools:
                parts.append(f"common_tools={', '.join(item.allowed_tools[:16])}")
            lines.append(" | ".join(parts))
        self._skill_selection_view_cache = "\n".join(lines)
        return self._skill_selection_view_cache

    def _filter_result(self, result: ContextSelectionResult) -> ContextSelectionResult:
        """功能：过滤模型返回的候选，仅保留已注册 skill/tool，并补全 BASE_CONTEXT_TOOL_NAMES。
        参数：
        - result：模型解析后的原始 ContextSelectionResult。
        返回值：
        - ContextSelectionResult：过滤并补全基础工具后的结果，fallback_all_tools=False。
        """
        skills = self._skill_metadata_by_name()
        tool_names = set(self._all_tool_names())
        filtered_skills = tuple(item for item in result.candidate_skills if item.id in skills)
        selected_tool_names = [item.name for item in result.candidate_tools if item.name in tool_names]
        for base in BASE_CONTEXT_TOOL_NAMES:
            if base in tool_names and base not in selected_tool_names:
                selected_tool_names.append(base)
        by_name: dict[str, ContextCandidateTool] = {}
        for item in result.candidate_tools:
            if item.name in tool_names:
                by_name[item.name] = item
        for name in selected_tool_names:
            by_name.setdefault(
                name,
                ContextCandidateTool(name=name, confidence=1.0, reason="base context tool"),
            )
        return ContextSelectionResult(
            version=result.version or "context_selector.v1",
            intent_summary=result.intent_summary,
            candidate_skills=filtered_skills,
            candidate_tools=tuple(by_name[name] for name in selected_tool_names),
            coverage_confidence=result.coverage_confidence,
            coverage_notes=result.coverage_notes,
            diagnostics=result.diagnostics,
            fallback_all_tools=False,
        )

    def _skill_metadata(self) -> tuple[SkillMetadata, ...]:
        """功能：从 SkillSystem 拉取全部 skill 元数据列表。
        参数：
        - 无（使用 self.skill_system）。
        返回值：
        - tuple[SkillMetadata, ...]：元数据元组；未配置或异常时返回空元组。
        """
        if self.skill_system is None:
            return ()
        try:
            return tuple(self.skill_system.list_skill_metadata())
        except Exception:
            return ()

    def _skill_metadata_by_name(self) -> dict[str, SkillMetadata]:
        """功能：将 skill 元数据列表索引为 name → SkillMetadata 字典，便于按名查找。
        参数：
        - 无（内部调用 _skill_metadata）。
        返回值：
        - dict[str, SkillMetadata]：skill 名称到元数据的映射。
        """
        return {item.name: item for item in self._skill_metadata()}

    def _all_tool_names(self) -> tuple[str, ...]:
        """功能：收集 ToolRegistry 中全部已注册工具名称，用于校验与过滤候选 tool。
        参数：
        - 无（使用 self.tool_registry）。
        返回值：
        - tuple[str, ...]：工具名元组；无法枚举时返回空元组。
        """
        if hasattr(self.tool_registry, "iter_specs_by_priority"):
            return tuple(spec.name for spec in self.tool_registry.iter_specs_by_priority())
        try:
            schemas = self.tool_registry.openai_tools_schema()
        except Exception:
            return ()
        names: list[str] = []
        for item in schemas:
            function = item.get("function") if isinstance(item, dict) else None
            name = (function or {}).get("name") if isinstance(function, dict) else None
            if name:
                names.append(str(name))
        return tuple(names)

    @classmethod
    def _result_from_payload(cls, payload: dict[str, Any]) -> ContextSelectionResult:
        """功能：将选择模型返回的 JSON 字典映射为 ContextSelectionResult 实例。
        参数：
        - payload：已解析的 selector JSON 对象（含 candidate_skills/tools、coverage 等）。
        返回值：
        - ContextSelectionResult：结构化选择结果，尚未经注册表过滤。
        """
        coverage = payload.get("coverage") if isinstance(payload.get("coverage"), dict) else {}
        return ContextSelectionResult(
            version=str(payload.get("version") or "context_selector.v1"),
            intent_summary=str(payload.get("intent_summary") or ""),
            candidate_skills=tuple(cls._parse_skill_items(payload.get("candidate_skills"))),
            candidate_tools=tuple(cls._parse_tool_items(payload.get("candidate_tools"))),
            coverage_confidence=cls._float(coverage.get("confidence"), default=0.0),
            coverage_notes=str(coverage.get("notes") or ""),
        )

    @classmethod
    def _parse_skill_items(cls, raw: Any) -> Iterable[ContextCandidateSkill]:
        """功能：解析 payload 中 candidate_skills 字段，支持字符串或对象列表。
        参数：
        - raw：JSON 中的 candidate_skills 原始值。
        返回值：
        - Iterable[ContextCandidateSkill]：有效 id 非空的 skill 候选项元组。
        """
        if not isinstance(raw, list):
            return ()
        items: list[ContextCandidateSkill] = []
        for row in raw:
            if isinstance(row, str):
                items.append(ContextCandidateSkill(id=row))
            elif isinstance(row, dict):
                items.append(
                    ContextCandidateSkill(
                        id=str(row.get("id") or row.get("name") or ""),
                        confidence=cls._float(row.get("confidence"), default=0.0),
                        reason=str(row.get("reason") or ""),
                    )
                )
        return tuple(item for item in items if item.id)

    @classmethod
    def _parse_tool_items(cls, raw: Any) -> Iterable[ContextCandidateTool]:
        """功能：解析 payload 中 candidate_tools 字段，支持字符串或对象列表。
        参数：
        - raw：JSON 中的 candidate_tools 原始值。
        返回值：
        - Iterable[ContextCandidateTool]：有效 name 非空的 tool 候选项元组。
        """
        if not isinstance(raw, list):
            return ()
        items: list[ContextCandidateTool] = []
        for row in raw:
            if isinstance(row, str):
                items.append(ContextCandidateTool(name=row))
            elif isinstance(row, dict):
                items.append(
                    ContextCandidateTool(
                        name=str(row.get("name") or row.get("id") or ""),
                        confidence=cls._float(row.get("confidence"), default=0.0),
                        reason=str(row.get("reason") or ""),
                    )
                )
        return tuple(item for item in items if item.name)

    @staticmethod
    def _parse_selector_json(text: str) -> dict[str, Any]:
        """功能：从模型原始文本中提取并解析 JSON 对象，兼容 markdown 代码块包裹。
        参数：
        - text：模型返回的原始字符串。
        返回值：
        - dict[str, Any]：解析后的 JSON 对象；非对象或无法解析时抛出异常。
        """
        value = (text or "").strip()
        value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
        value = re.sub(r"\s*```$", "", value)
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            start = value.find("{")
            end = value.rfind("}")
            if start < 0 or end <= start:
                raise
            parsed = json.loads(value[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("selector response is not a JSON object")
        return parsed

    @staticmethod
    def _float(value: Any, *, default: float) -> float:
        """功能：将任意值安全转换为浮点数，并钳制到 [0.0, 1.0] 区间。
        参数：
        - value：待转换的值。
        - default：转换失败时使用的默认值。
        返回值：
        - float：钳制后的置信度数值。
        """
        try:
            number = float(value)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(1.0, number))

    @staticmethod
    def _env_float(name: str, *, default: float, minimum: float) -> float:
        """功能：从环境变量读取浮点配置，失败时使用默认值并保证不低于 minimum。
        参数：
        - name：环境变量名（如 TOOL_SELECTOR_TIMEOUT_SECONDS）。
        - default：未设置或解析失败时的默认值。
        - minimum：返回值的下限。
        返回值：
        - float：有效的配置浮点数。
        """
        raw = os.getenv(name)
        try:
            value = float(str(raw).strip()) if raw is not None else default
        except (TypeError, ValueError):
            value = default
        return max(minimum, value)

    @staticmethod
    def _compact(text: str, limit: int) -> str:
        """功能：压缩空白并将文本截断到指定长度，用于 skill 选择视图中的摘要展示。
        参数：
        - text：原始文本。
        - limit：最大字符数（超出时末尾加 "..."）。
        返回值：
        - str：规范化并可能截断后的单行字符串。
        """
        value = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(value) <= limit:
            return value
        return value[: max(0, limit - 1)] + "..."

    @staticmethod
    def _dedupe(items: Iterable[str]) -> tuple[str, ...]:
        """功能：按首次出现顺序对字符串序列去重，忽略空字符串。
        参数：
        - items：可能含重复的字符串可迭代对象。
        返回值：
        - tuple[str, ...]：保序去重后的元组。
        """
        result: list[str] = []
        for item in items:
            if item and item not in result:
                result.append(item)
        return tuple(result)
