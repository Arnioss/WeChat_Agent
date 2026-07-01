from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Callable, List, Optional, Tuple

from app.agent.context_selector import ContextSelectionResult, ModelContextSelector
from app.agent.kb_body_images import build_kb_image_inline_hint
from app.agent.memory_manager import MemoryManager
from app.agent.model_client import ModelClient, log_print
from app.agent.react_events import (
    clip_text,
    emit_agent_finish,
    emit_agent_start,
    emit_error,
    emit_model_decision,
    emit_tool_end,
    emit_tool_start,
)
from app.agent.stream_context import bind_stream_emitter
from app.agent.prompt_service import PromptService
from app.agent.tool_call_protocol import ToolCallProtocolError, validate_tool_arguments
from app.agent.tool_metadata import ToolRichMetadata
from app.agent.tool_registry import ToolRegistry
from app.skills.executor import SkillToolResult
from app.skills.lifecycle import SkillSessionState
from app.skills.models import ActiveSkill
from app.skills.router import SkillActivationDecision, SkillRoutePlan
from app.skills.system import SkillSystem

class AgentRuntime:
    """功能：编排单轮请求中的 tool_calls 推理循环、工具执行与技能状态推进。
    参数：
    - 无。
    返回值：
    - 无。到达步数上限或协议错误时会提前终止，避免死循环。
    """
    def __init__(
        self,
        *,
        tool_registry: ToolRegistry,
        model_client: ModelClient,
        memory_manager: MemoryManager,
        prompt_service: PromptService,
        max_steps: int,
        skill_system: Optional[SkillSystem] = None,
        skill_session: Optional[SkillSessionState] = None,
        skill_shortlist_limit: int = 3,
        context_selector=None,
    ):
        """功能：注入运行期依赖并设置推理步数与技能候选数量限制。
        参数：
        - tool_registry：工具注册中心实例。
        - model_client：模型调用客户端。
        - memory_manager：会话记忆管理器。
        - prompt_service：系统提示词渲染服务。
        - max_steps：单次请求允许的最大推理步数。
        - skill_system：可选技能系统实例。
        - skill_session：可选技能会话状态对象。
        - skill_shortlist_limit：技能检索候选上限。
        返回值：
        - 无。`skill_system/skill_session` 可为空，此时运行时会退化为无技能模式。
        """
        self.tool_registry = tool_registry
        self.model_client = model_client
        self.memory_manager = memory_manager
        self.prompt_service = prompt_service
        self.max_steps = max_steps
        self.skill_system = skill_system
        self.skill_session = skill_session
        self.skill_shortlist_limit = skill_shortlist_limit
        self.context_selector = context_selector or ModelContextSelector(
            tool_registry=tool_registry,
            model_client=model_client,
            skill_system=skill_system,
        )
        self._enabled_tool_names: Optional[tuple[str, ...]] = None
        self._context_selection: Optional[ContextSelectionResult] = None
        self._prepared_user_input: str = ""
        self._last_skill_route_plan: Optional[SkillRoutePlan] = None
        self._register_internal_tools()

    @staticmethod
    def sanitize_final_answer(answer: str) -> str:
        """功能：清理最终回答中的 XML 控制标签与协议噪声。
        参数：
        - answer：模型输出的最终回答文本。
        返回值：
        - str：移除控制协议标签并 strip 后的文本。
        """
        return AgentRuntime._strip_control_protocol(answer)

    @staticmethod
    def sanitize_final_chunk(chunk: str) -> str:
        """功能：清理流式输出分片中的控制协议标签（保留首尾空白）。
        参数：
        - chunk：流式输出的文本分片。
        返回值：
        - str：移除 XML 控制标签后的分片文本，不强制 strip。
        """
        return AgentRuntime._strip_control_protocol(chunk, strip=False)

    def run(
        self,
        *,
        user_input: str,
        messages: List[dict],
        conversation_turns: List[Tuple[str, str]],
        stream_callback: Optional[Callable[[str, str], None]] = None,
        stop_event=None,
    ) -> str:
        """功能：驱动 tool_calls 模型与工具循环执行直到得到最终回答。
        参数：
        - user_input：当前用户输入文本。
        - messages：与模型交互的消息列表。
        - conversation_turns：会话轮次记录。
        - stream_callback：可选流式事件回调 `(event_kind, text)`。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - str：本轮请求的最终回答字符串。
        """
        messages[:] = [messages[0]] + [item for item in messages[1:] if item.get("role") != "system"]
        prepared_input = self._prepare_request_context(user_input=user_input, messages=messages)
        if prepared_input is None:
            final_answer = "未找到你显式指定的 skill，请检查 skill 名称后重试。"
            emit_error(stream_callback, final_answer)
            self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
            return final_answer

        emit_agent_start(stream_callback, "Agent 开始处理请求。")
        memory_context = self.memory_manager.build_recent_memory_context(conversation_turns)
        self._conversation_context = memory_context
        if memory_context:
            messages.append(
                {
                    "role": "user",
                    "content": f"conversation_history:\n{memory_context}",
                }
            )
        messages.append({"role": "user", "content": prepared_input})
        self.memory_manager.trim_messages(messages)

        self._context_selection = self.context_selector.select(
            user_input=user_input,
            conversation_context=memory_context,
            explicit_skill=self._last_skill_route_plan.explicit_skill if self._last_skill_route_plan else None,
        )
        self._enabled_tool_names = self._enabled_tools_from_selection(self._context_selection)
        self._emit_context_selection_decision(
            stream_callback,
            label="Selector 初筛",
            selection=self._context_selection,
            enabled_tool_names=self._enabled_tool_names,
        )
        self._inject_selector_directives(self._context_selection, messages)
        self._apply_context_selection_to_skills(self._context_selection, messages)
        self._enabled_tool_names = self._merge_allowed_tools_from_active_skills(self._enabled_tool_names)
        messages[0]["content"] = self._render_prompt(
            active_skills=(
                self.skill_system.lifecycle.visible_active_skills(self.skill_session)
                if self.skill_system and self.skill_session
                else ()
            ),
            enabled_tool_names=self._enabled_tool_names,
        )
        tools = self.tool_registry.openai_tools_schema(enabled_tool_names=self._enabled_tool_names)
        tool_call_signatures: set[str] = set()
        recovered_protocol_errors: set[str] = set()

        for step in range(1, self.max_steps + 1):
            if stop_event is not None and stop_event.is_set():
                final_answer = "请求处理超时，请稍后再试。"
                self._emit_final(stream_callback, final_answer)
                self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
                return final_answer

            result = self.model_client.call_tool_call_model(
                messages,
                tools=tools,
                stop_event=stop_event,
            )
            if result.tool_calls:
                emit_model_decision(stream_callback, f"模型请求调用 {len(result.tool_calls)} 个工具。")
                skipped_duplicate_tool = False
                messages.append(
                    {
                        "role": "assistant",
                        "content": result.content or None,
                        "tool_calls": result.tool_calls,
                    }
                )
                for tool_call in result.tool_calls:
                    tool_call_id = str(tool_call.get("id") or f"call_{step}")
                    function = tool_call.get("function") or {}
                    tool_name = str(function.get("name") or "")
                    raw_arguments = function.get("arguments") or "{}"
                    try:
                        self._validate_tool_name(tool_name, enabled_tool_names=self._enabled_tool_names)
                        tool_arguments = self._parse_tool_call_arguments(tool_name, raw_arguments)
                        self._validate_tool_call(tool_name, tool_arguments)
                        function["arguments"] = json.dumps(tool_arguments, ensure_ascii=False)
                    except ToolCallProtocolError as exc:
                        if self._is_recoverable_tool_call_error(exc):
                            recovery_signature = json.dumps(
                                {
                                    "tool_name": tool_name,
                                    "raw_arguments": raw_arguments,
                                    "reason": exc.reason,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                                default=str,
                            )
                            if recovery_signature not in recovered_protocol_errors:
                                recovered_protocol_errors.add(recovery_signature)
                                recovery_observation = (
                                    f"工具调用参数错误：{exc}。请根据该工具 input_schema 补齐必填参数后重试；"
                                    "如果当前任务不需要该工具，请基于已有观察结果直接给出最终回答。"
                                )
                                emit_error(stream_callback, recovery_observation)
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": recovery_observation,
                                    }
                                )
                                self.memory_manager.trim_messages(messages)
                                continue
                        return self._finish_with_protocol_error(
                            stream_callback,
                            conversation_turns,
                            user_input,
                            f"工具调用协议错误：{exc}",
                        )

                    call_signature = json.dumps(
                        {"tool_name": tool_name, "tool_arguments": tool_arguments},
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    if call_signature in tool_call_signatures:
                        skipped_duplicate_tool = True
                        duplicate_observation = (
                            f"重复工具调用 `{tool_name}` 已跳过；"
                            "请基于已有工具观察结果继续推理并给出最终回答。"
                        )
                        emit_model_decision(stream_callback, duplicate_observation)
                        if stream_callback is None:
                            log_print(f"\n执行过程：{duplicate_observation}")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": duplicate_observation,
                            }
                        )
                        self.memory_manager.trim_messages(messages)
                        continue
                    tool_call_signatures.add(call_signature)

                    emit_tool_start(stream_callback, tool_name, tool_arguments)
                    if stream_callback is None:
                        log_print(f"\n执行过程：调用工具 {tool_name}({self._format_tool_arguments(tool_arguments)})")
                    try:
                        with bind_stream_emitter(stream_callback):
                            if tool_name == "expand_tool_context":
                                observation = self._handle_expand_tool_context(
                                    need=str(tool_arguments.get("need") or ""),
                                    messages=messages,
                                    stream_callback=stream_callback,
                                )
                                tools = self.tool_registry.openai_tools_schema(
                                    enabled_tool_names=self._enabled_tool_names
                                )
                            else:
                                observation = self.tool_registry.execute_json(tool_name, tool_arguments)
                    except Exception as exc:
                        observation = f"工具执行错误：{exc}"

                    rendered_observation = self._apply_skill_tool_result(observation)
                    if stream_callback is None:
                        log_print(f"\n执行过程：观察结果：{rendered_observation}")
                    emit_tool_end(stream_callback, rendered_observation)

                    if self.tool_registry.should_direct_return(tool_name):
                        final_answer = self._stringify_observation(rendered_observation)
                        self._complete_visible_skills()
                        emit_agent_finish(stream_callback, "Agent 完成：工具结果直接返回。")
                        self._emit_final(stream_callback, final_answer)
                        self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
                        return final_answer

                    observation_text = self._stringify_observation(rendered_observation)
                    kb_hint = build_kb_image_inline_hint(observation_text)
                    if kb_hint:
                        observation_text = f"{observation_text}\n{kb_hint}"
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": observation_text,
                        }
                    )
                    self.memory_manager.trim_messages(messages)
                if skipped_duplicate_tool:
                    summary = "Agent 完成：检测到重复工具调用，已基于已有观察结果生成最终回答。"
                    if stream_callback is None:
                        log_print(f"\n执行过程：{summary}")
                    emit_agent_finish(stream_callback, summary)
                    self._complete_visible_skills()
                    final_answer = self._generate_final_answer(
                        messages,
                        fallback_content=result.content,
                        stream_callback=stream_callback,
                        stop_event=stop_event,
                    )
                    if not final_answer:
                        final_answer = "模型未返回最终答案，已终止本轮任务。"
                        emit_error(stream_callback, final_answer)
                    messages.append({"role": "assistant", "content": final_answer})
                    self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
                    return final_answer
                continue

            summary = (
                "Agent 完成：模型已根据工具观察结果生成最终回答。"
                if tool_call_signatures
                else "Agent 完成：模型未调用工具，直接回答。"
            )
            if stream_callback is None:
                log_print(f"\n执行过程：{summary}")
            emit_agent_finish(stream_callback, summary)
            self._complete_visible_skills()
            final_answer = self._resolve_final_answer(
                messages,
                fallback_content=result.content,
                stream_callback=stream_callback,
                stop_event=stop_event,
            )
            if not final_answer:
                final_answer = "模型未返回最终答案，已终止本轮任务。"
                emit_error(stream_callback, final_answer)
            messages.append({"role": "assistant", "content": final_answer})
            self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
            return final_answer

        final_answer = "本次任务推理步数已达上限，请缩小问题范围后重试。"
        emit_error(stream_callback, final_answer)
        self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
        return final_answer

    @staticmethod
    def _emit_final(stream_callback: Optional[Callable[[str, str], None]], final_answer: str) -> None:
        """功能：将最终回答按分片通过流式回调发送。
        参数：
        - stream_callback：流式事件回调；为 None 时不发送。
        - final_answer：待发送的最终回答文本。
        返回值：
        - 无。分片大小与间隔由环境变量 REACT_FINAL_CHUNK_SIZE、REACT_FINAL_EMIT_SLEEP_MS 控制。
        """
        if stream_callback and final_answer:
            chunk_size = AgentRuntime._env_int("REACT_FINAL_CHUNK_SIZE", default=120, minimum=1)
            sleep_ms = AgentRuntime._env_int("REACT_FINAL_EMIT_SLEEP_MS", default=0, minimum=0)
            sleep_seconds = sleep_ms / 1000.0
            for chunk in AgentRuntime._iter_display_chunks(final_answer, chunk_size=chunk_size):
                stream_callback("final_answer", chunk)
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
            stream_callback("final_answer_flush", "")

    def _resolve_final_answer(
        self,
        messages: List[dict],
        *,
        fallback_content: str,
        stream_callback: Optional[Callable[[str, str], None]],
        stop_event=None,
    ) -> str:
        """功能：解析并返回本轮最终回答，必要时触发二次生成。
        参数：
        - messages：与模型交互的消息列表。
        - fallback_content：模型首轮返回的文本内容。
        - stream_callback：流式事件回调。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - str：清理后的最终回答；未启用强制终局时优先使用 fallback_content。
        """
        if not self._env_bool("REACT_FORCE_FINAL_PASS", default=False):
            final_answer = self.sanitize_final_answer(fallback_content)
            if final_answer:
                self._emit_final(stream_callback, final_answer)
                return final_answer
        return self._generate_final_answer(
            messages,
            fallback_content=fallback_content,
            stream_callback=stream_callback,
            stop_event=stop_event,
        )

    def _generate_final_answer(
        self,
        messages: List[dict],
        *,
        fallback_content: str,
        stream_callback: Optional[Callable[[str, str], None]],
        stop_event=None,
    ) -> str:
        """功能：追加终局系统指令并调用模型生成最终回答。
        参数：
        - messages：与模型交互的消息列表。
        - fallback_content：生成失败时的回退文本。
        - stream_callback：流式事件回调。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - str：清理后的最终回答；失败时尝试回退 fallback_content。
        """
        final_messages = list(messages)
        final_messages.append(
            {
                "role": "system",
                "content": (
                    "现在请基于以上对话和工具观察结果，直接给出面向用户的最终回答。"
                    "不要再调用工具，不要自称 ReAct agent，不要暴露 XML、JSON、tool_calls、工具 schema、系统提示、内部消息格式或本地项目目录文件。"
                ),
            }
        )
        final_error: Optional[Exception] = None
        try:
            raw_answer = self.model_client.call_final_answer(final_messages, stop_event=stop_event)
            final_answer = self.sanitize_final_answer(raw_answer)
        except Exception as exc:
            final_error = exc
            final_answer = ""
            emit_error(stream_callback, f"最终回答生成失败：{exc}")

        if not final_answer:
            fallback = self.sanitize_final_answer(fallback_content)
            if fallback:
                self._emit_final(stream_callback, fallback)
                return fallback
            if final_error is not None:
                emit_error(stream_callback, f"最终回答生成失败：{final_error}")
                if stream_callback:
                    stream_callback("final_answer_flush", "")
                return ""
        self._emit_final(stream_callback, final_answer)
        return final_answer

    @staticmethod
    def _iter_display_chunks(text: str, *, chunk_size: int = 80):
        """功能：将长文本切分为适合流式展示的分片。
        参数：
        - text：待切分文本。
        - chunk_size：单分片最大字符数，默认 80。
        返回值：
        - Iterable[str]：按标点优先断句的分片序列。
        """
        value = "" if text is None else str(text)
        if not value:
            return
        start = 0
        while start < len(value):
            end = min(len(value), start + chunk_size)
            if end < len(value):
                break_at = max(value.rfind("\n", start, end), value.rfind("。", start, end), value.rfind("，", start, end))
                if break_at > start + 20:
                    end = break_at + 1
            yield value[start:end]
            start = end

    @staticmethod
    def _stringify_observation(observation) -> str:
        """功能：将工具观察结果统一转换为字符串。
        参数：
        - observation：工具返回的观察结果，可为 dict 或任意类型。
        返回值：
        - str：dict 序列化为 JSON，其余类型转为 str；None 返回空串。
        """
        if isinstance(observation, dict):
            return json.dumps(observation, ensure_ascii=False, default=str)
        return "" if observation is None else str(observation)

    @staticmethod
    def _env_bool(name: str, *, default: bool = False) -> bool:
        """功能：读取环境变量并解析为布尔值。
        参数：
        - name：环境变量名。
        - default：未设置或无法识别时的默认值。
        返回值：
        - bool：1/true/yes/on 为 True，0/false/no/off/空 为 False。
        """
        raw = os.getenv(name)
        if raw is None:
            return default
        value = raw.strip().lower()
        if value in ("1", "true", "yes", "on"):
            return True
        if value in ("0", "false", "no", "off", ""):
            return False
        return default

    @staticmethod
    def _env_int(name: str, *, default: int, minimum: Optional[int] = None) -> int:
        """功能：读取环境变量并解析为整数。
        参数：
        - name：环境变量名。
        - default：未设置或解析失败时的默认值。
        - minimum：可选下限，解析结果不会小于该值。
        返回值：
        - int：解析后的整数值。
        """
        raw = os.getenv(name)
        try:
            value = int(str(raw).strip()) if raw is not None else default
        except (TypeError, ValueError):
            value = default
        if minimum is not None:
            value = max(minimum, value)
        return value

    def _register_internal_tools(self) -> None:
        """功能：注册运行时内置的 expand_tool_context 工具。
        参数：
        - 无。
        返回值：
        - 无。工具已存在时跳过注册。
        """
        if self.tool_registry.has("expand_tool_context"):
            return

        def expand_tool_context(need: str):
            """功能：运行时内置占位工具，实际扩展逻辑由 _handle_expand_tool_context 处理。
            参数：
            - need：缺失能力或数据操作的简短描述。
            返回值：
            - str：原样返回 need 字符串（注册占位，执行时不走此路径）。
            """
            return need

        expand_tool_context.__name__ = "expand_tool_context"
        expand_tool_context.__tool_rich_metadata__ = ToolRichMetadata(
            summary="Request more candidate tools when the currently exposed tools are not enough.",
            when_to_use=(
                "The current tool list lacks a capability needed to finish the user's task.",
                "You know what capability is missing but cannot find a suitable exposed tool.",
            ),
            when_not_to_use=(
                "A currently exposed tool can already complete the next step.",
                "You only need to fix arguments for an already exposed tool.",
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "need": {
                        "type": "string",
                        "description": "Short description of the missing capability or data operation.",
                    }
                },
                "required": ["need"],
                "additionalProperties": False,
            },
            output_description="Text observation listing any newly exposed tools.",
            examples=('expand_tool_context({"need":"need a WeCom smart sheet append-records tool"})',),
            priority=95,
        )
        register = getattr(self.tool_registry, "register", None)
        if callable(register):
            register(expand_tool_context)

    def _enabled_tools_from_selection(self, selection: Optional[ContextSelectionResult]) -> Optional[tuple[str, ...]]:
        """功能：从 context selector 结果提取本轮启用的工具名列表。
        参数：
        - selection：上下文选择结果；None 或 fallback 时返回 None 表示全量工具。
        返回值：
        - Optional[tuple[str, ...]]：有序工具名元组；RAG 启用时自动补入 rag_summarize。
        """
        if selection is None or selection.fallback_all_tools:
            return None
        ordered = []
        for candidate in selection.candidate_tools:
            if candidate.name and self.tool_registry.has(candidate.name) and candidate.name not in ordered:
                ordered.append(candidate.name)
        if self._env_bool("RAG_ENABLED", default=False) and self.tool_registry.has("rag_summarize"):
            if "rag_summarize" not in ordered:
                ordered.append("rag_summarize")
        return tuple(ordered)

    def _emit_context_selection_decision(
        self,
        stream_callback: Optional[Callable[[str, str], None]],
        *,
        label: str,
        selection: Optional[ContextSelectionResult],
        enabled_tool_names: Optional[tuple[str, ...]],
    ) -> None:
        """功能：将 selector 初筛/扩展决策通过流式回调输出。
        参数：
        - stream_callback：流式事件回调。
        - label：决策标签（如「Selector 初筛」）。
        - selection：上下文选择结果。
        - enabled_tool_names：本轮实际下发的工具名列表。
        返回值：
        - 无。
        """
        if selection is None:
            emit_model_decision(stream_callback, f"{label}：无结果")
            return
        skills = ", ".join(
            self._format_context_candidate(item.id, item.confidence, item.reason)
            for item in selection.candidate_skills
        ) or "无"
        tools = ", ".join(
            self._format_context_candidate(item.name, item.confidence, item.reason)
            for item in selection.candidate_tools
        ) or "无"
        if enabled_tool_names is None:
            enabled = "全量工具（selector 未配置/失败或主动全量兜底）"
        else:
            enabled = ", ".join(enabled_tool_names) or "无"
        parts = [
            f"{label}：skills=[{skills}]",
            f"tools=[{tools}]",
            f"本轮下发工具=[{enabled}]",
        ]
        if selection.intent_summary:
            parts.append(f"意图={selection.intent_summary}")
        if selection.coverage_notes:
            parts.append(f"覆盖={selection.coverage_notes}")
        if selection.diagnostics:
            parts.append("诊断=" + "；".join(selection.diagnostics))
        metrics = self._selector_metrics()
        if metrics:
            parts.append(metrics)
        emit_model_decision(stream_callback, "；".join(parts))

    def _selector_metrics(self) -> str:
        """功能：格式化 context selector 最近一次调用的耗时与上下文规模。
        参数：
        - 无。
        返回值：
        - str：耗时与 prompt 字符数摘要；无数据时返回空串。
        """
        prompt_chars = getattr(self.context_selector, "last_prompt_chars", 0) or 0
        elapsed_ms = getattr(self.context_selector, "last_elapsed_ms", 0.0) or 0.0
        if not prompt_chars and not elapsed_ms:
            return ""
        return f"耗时={elapsed_ms:.0f}ms；选择上下文={prompt_chars} chars"

    @staticmethod
    def _format_context_candidate(name: str, confidence: float, reason: str) -> str:
        """功能：将 selector 候选项格式化为可读展示字符串。
        参数：
        - name：候选 skill 或工具名。
        - confidence：置信度分数。
        - reason：推荐理由。
        返回值：
        - str：含名称、分数与截断理由的展示文本。
        """
        value = name or "?"
        try:
            score = float(confidence)
        except (TypeError, ValueError):
            score = 0.0
        if score:
            value = f"{value}({score:.2f})"
        if reason:
            value = f"{value}: {clip_text(reason, limit=80)}"
        return value

    def _inject_selector_directives(
        self,
        selection: Optional[ContextSelectionResult],
        messages: List[dict],
    ) -> None:
        """功能：将 selector 推荐 skills/工具与 RAG 策略注入为 system 消息。
        参数：
        - selection：上下文选择结果。
        - messages：与模型交互的消息列表（就地追加）。
        返回值：
        - 无。fallback 全量工具时不注入。
        """
        if selection is None or selection.fallback_all_tools:
            return
        lines: list[str] = [
            "Selector 只提供候选上下文和工具推荐，不强制工具调用；是否调用工具仍由你根据用户意图、可用工具说明和当前上下文判断。"
        ]
        if selection.candidate_skills:
            skills = ", ".join(
                self._format_context_candidate(item.id, item.confidence, item.reason)
                for item in selection.candidate_skills
            )
            lines.append(f"推荐 skills：{skills}")
        if selection.candidate_tools:
            tools = ", ".join(
                self._format_context_candidate(item.name, item.confidence, item.reason)
                for item in selection.candidate_tools
            )
            lines.append(f"推荐工具：{tools}")
        if any(item.name == "rag_summarize" for item in selection.candidate_tools) or any(
            item.id == "knowledge-rag-answer" for item in selection.candidate_skills
        ):
            lines.append(
                "如果用户明确要求依据知识库、内部资料、项目文档或参考资料回答，通常优先调用 rag_summarize；"
                "若问题可直接可靠回答且用户未要求内部资料依据，可以直接给出最终回答。"
            )
        if selection.candidate_skills:
            skill_ids = ", ".join(item.id for item in selection.candidate_skills)
            lines.append(
                f"推荐 skill（{skill_ids}）的业务流程未在本消息展开；"
                "若摘要不足以确定步骤，请调用 load_skill_instructions(skill_name) 读取完整 SKILL.md，"
                "不要跳过 skill 直接猜测工具参数。"
            )
        messages.append(
            {
                "role": "system",
                "content": "Selector 参考建议：\n" + "\n".join(f"- {item}" for item in lines),
            }
        )

    def _apply_context_selection_to_skills(
        self,
        selection: Optional[ContextSelectionResult],
        messages: List[dict],
    ) -> None:
        """功能：根据 selector 候选 skills 执行 shortlist 与 summary 激活。
        参数：
        - selection：上下文选择结果。
        - messages：与模型交互的消息列表（当前未直接修改，保留扩展点）。
        返回值：
        - 无。无 skill 系统或 fallback 时跳过。
        """
        if selection is None or selection.fallback_all_tools:
            return
        if not self.skill_system or not self.skill_session:
            return
        for candidate in selection.candidate_skills:
            try:
                metadata = self.skill_system.get_skill_metadata(candidate.id)
            except Exception:
                continue
            if metadata.name in self.skill_session.active_skills:
                continue
            match = None
            if self._last_skill_route_plan:
                for route_candidate in self._last_skill_route_plan.summary_candidates:
                    if route_candidate.metadata.name == candidate.id:
                        match = route_candidate.match
                        break
            self.skill_system.lifecycle.shortlist(
                self.skill_session,
                metadata=metadata,
                match=match,
                reason=candidate.reason or "selected by context selector",
                trigger="runtime.context_selector",
            )
            self.skill_system.lifecycle.activate_summary(
                self.skill_session,
                skill_name=metadata.name,
                reason="summary exposed by context selector",
                trigger="runtime.context_selector",
            )

    def _merge_allowed_tools_from_active_skills(
        self,
        enabled_tool_names: Optional[tuple[str, ...]],
    ) -> Optional[tuple[str, ...]]:
        """功能：初筛激活 skill 后，补全该 skill 声明的 allowed_tools。
        参数：
        - enabled_tool_names：当前已启用工具名元组；None 表示全量工具。
        返回值：
        - Optional[tuple[str, ...]]：合并后的有序工具名元组。
        """
        if enabled_tool_names is None:
            return None
        if not self.skill_system or not self.skill_session:
            return enabled_tool_names
        ordered = list(enabled_tool_names)
        for active in self.skill_system.lifecycle.visible_active_skills(self.skill_session):
            for name in active.metadata.allowed_tools:
                if name and self.tool_registry.has(name) and name not in ordered:
                    ordered.append(name)
        return tuple(ordered)

    def _handle_expand_tool_context(
        self,
        *,
        need: str,
        messages: List[dict],
        stream_callback: Optional[Callable[[str, str], None]],
    ) -> str:
        """功能：处理 expand_tool_context 工具调用，扩展本轮可用工具集。
        参数：
        - need：缺失能力或数据操作的简短描述。
        - messages：与模型交互的消息列表（就地更新 system 提示）。
        - stream_callback：流式事件回调。
        返回值：
        - str：扩展结果的英文 observation 文本。
        """
        selection = self.context_selector.select(
            user_input=self._prepared_user_input,
            additional_need=need,
            conversation_context=getattr(self, "_conversation_context", ""),
        )
        previous = set(self._enabled_tool_names or ())
        if selection.fallback_all_tools:
            self._enabled_tool_names = None
            self._context_selection = selection
            messages[0]["content"] = self._render_prompt(
                active_skills=(
                    self.skill_system.lifecycle.visible_active_skills(self.skill_session)
                    if self.skill_system and self.skill_session
                    else ()
                ),
                enabled_tool_names=self._enabled_tool_names,
            )
            self._emit_context_selection_decision(
                stream_callback,
                label="Selector 扩展初筛",
                selection=selection,
                enabled_tool_names=self._enabled_tool_names,
            )
            return "Tool context expanded to all registered tools because selector was unavailable."

        expanded = list(previous)
        for candidate in selection.candidate_tools:
            if candidate.name and self.tool_registry.has(candidate.name) and candidate.name not in expanded:
                expanded.append(candidate.name)
        self._context_selection = selection
        self._enabled_tool_names = tuple(
            name for name in self._all_tool_names() if name in set(expanded)
        )
        self._emit_context_selection_decision(
            stream_callback,
            label="Selector 扩展初筛",
            selection=selection,
            enabled_tool_names=self._enabled_tool_names,
        )
        self._apply_context_selection_to_skills(selection, messages)
        self._enabled_tool_names = self._merge_allowed_tools_from_active_skills(self._enabled_tool_names)
        messages[0]["content"] = self._render_prompt(
            active_skills=(
                self.skill_system.lifecycle.visible_active_skills(self.skill_session)
                if self.skill_system and self.skill_session
                else ()
            ),
            enabled_tool_names=self._enabled_tool_names,
        )
        new_tools = [name for name in self._enabled_tool_names if name not in previous]
        if new_tools:
            observation = "Expanded tool context. Newly available tools: " + ", ".join(new_tools)
        else:
            observation = "No additional tools were selected for that need."
        emit_model_decision(stream_callback, observation)
        return observation

    def _all_tool_names(self) -> tuple[str, ...]:
        """功能：获取注册中心中全部工具名称。
        参数：
        - 无。
        返回值：
        - tuple[str, ...]：按优先级或 schema 顺序排列的工具名元组。
        """
        if hasattr(self.tool_registry, "iter_specs_by_priority"):
            return tuple(spec.name for spec in self.tool_registry.iter_specs_by_priority())
        try:
            schemas = self.tool_registry.openai_tools_schema()
        except Exception:
            return ()
        names = []
        for item in schemas:
            function = item.get("function") if isinstance(item, dict) else None
            name = (function or {}).get("name") if isinstance(function, dict) else None
            if name:
                names.append(str(name))
        return tuple(names)

    @staticmethod
    def _parse_tool_call_arguments(tool_name: str, raw_arguments) -> dict:
        """功能：解析并校验模型 tool_call 的 arguments 为 JSON object。
        参数：
        - tool_name：工具名称，用于错误提示。
        - raw_arguments：原始 arguments（str、dict 或 None）。
        返回值：
        - dict：解析后的参数字典。
        """
        if raw_arguments in (None, ""):
            return {}
        if isinstance(raw_arguments, dict):
            return dict(raw_arguments)
        if not isinstance(raw_arguments, str):
            raise ToolCallProtocolError(
                f"工具 {tool_name} 的 arguments 必须是 JSON object 字符串。",
                reason="invalid_tool_arguments",
            )
        try:
            parsed = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ToolCallProtocolError(
                f"工具 {tool_name} 的 arguments 不是合法 JSON：{exc}",
                reason="invalid_tool_arguments",
            ) from exc
        if parsed is None:
            return {}
        if not isinstance(parsed, dict):
            raise ToolCallProtocolError(
                f"工具 {tool_name} 的 arguments 必须解析为 JSON object。",
                reason="invalid_tool_arguments",
            )
        return parsed

    def _validate_tool_call(
        self,
        tool_name: str,
        tool_arguments: dict,
    ) -> None:
        """功能：按工具 rich_metadata.input_schema 校验调用参数。
        参数：
        - tool_name：工具名称。
        - tool_arguments：待校验的参数字典。
        返回值：
        - 无。校验失败时抛出 ToolCallProtocolError。
        """
        spec = self.tool_registry.get(tool_name)
        if spec.rich_metadata is not None:
            validate_tool_arguments(tool_name, tool_arguments, spec.rich_metadata.input_schema)

    def _validate_tool_name(
        self,
        tool_name: str,
        *,
        enabled_tool_names: Optional[tuple[str, ...]] = None,
    ) -> None:
        """功能：校验工具是否存在且在本轮 enabled 列表中。
        参数：
        - tool_name：工具名称。
        - enabled_tool_names：本轮启用的工具名白名单；None 时不校验启用范围。
        返回值：
        - 无。不存在或未启用时抛出 ToolCallProtocolError。
        """
        if not self.tool_registry.has(tool_name):
            raise ToolCallProtocolError(f"模型请求了不存在的工具：{tool_name}", reason="unknown_tool")
        if enabled_tool_names is not None and tool_name not in set(enabled_tool_names):
            raise ToolCallProtocolError(
                f"模型请求了本轮未启用的工具：{tool_name}",
                reason="tool_not_enabled",
            )

    @staticmethod
    def _is_recoverable_tool_call_error(exc: ToolCallProtocolError) -> bool:
        """功能：判断工具调用协议错误是否可通过 observation 反馈后重试。
        参数：
        - exc：ToolCallProtocolError 异常实例。
        返回值：
        - bool：参数类错误可恢复时返回 True。
        """
        return exc.reason in {
            "missing_required_tool_argument",
            "unknown_tool_argument",
            "invalid_tool_argument_type",
            "invalid_tool_arguments",
        }

    def _finish_with_protocol_error(
        self,
        stream_callback: Optional[Callable[[str, str], None]],
        conversation_turns: List[Tuple[str, str]],
        user_input: str,
        final_answer: str,
    ) -> str:
        """功能：以协议错误终止本轮请求并记录会话轮次。
        参数：
        - stream_callback：流式事件回调。
        - conversation_turns：会话轮次记录。
        - user_input：当前用户输入。
        - final_answer：面向用户的错误说明文本。
        返回值：
        - str：与 final_answer 相同的错误文本。
        """
        emit_error(stream_callback, final_answer)
        self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
        return final_answer

    @staticmethod
    def _format_tool_arguments(arguments: dict) -> str:
        """功能：将工具参数字典格式化为控制台展示字符串。
        参数：
        - arguments：工具参数字典。
        返回值：
        - str：key=value 形式、值截断后的逗号分隔文本。
        """
        if not arguments:
            return ""
        parts = []
        for key, value in arguments.items():
            parts.append(f"{key}={clip_text(value, limit=160)}")
        return ", ".join(parts)

    @staticmethod
    def _strip_control_protocol(text: str, *, strip: bool = True) -> str:
        """功能：移除文本中的 ReAct/XML 控制协议标签。
        参数：
        - text：待清理文本。
        - strip：是否在末尾 strip 空白，流式分片时可设为 False。
        返回值：
        - str：移除 action/observation/thought 等标签后的文本。
        """
        value = "" if text is None else str(text)
        value = re.sub(r"<action>.*?</action>", "", value, flags=re.DOTALL | re.IGNORECASE)
        value = re.sub(r"<observation>.*?</observation>", "", value, flags=re.DOTALL | re.IGNORECASE)
        value = re.sub(r"<final_answer>(.*?)</final_answer>", r"\1", value, flags=re.DOTALL | re.IGNORECASE)
        value = re.sub(
            r"</?(?:thought|thinking|think|action|observation|final_answer|question)\b[^>]*>",
            "",
            value,
            flags=re.IGNORECASE,
        )
        return value.strip() if strip else value

    def _prepare_request_context(self, *, user_input: str, messages: List[dict]) -> Optional[str]:
        """功能：根据输入与技能状态构建本轮请求上下文并更新 system 提示词。
        参数：
        - user_input：当前用户输入文本。
        - messages：与模型交互的消息列表。
        返回值：
        - Optional[str]：处理后的请求文本；显式 skill 不存在时返回 None。
        """
        active_skills: tuple[ActiveSkill, ...] = ()
        prepared_input = user_input
        self._last_skill_route_plan = None
        if self.skill_system and self.skill_session:
            self.skill_system.lifecycle.begin_request(self.skill_session)
            explicit_skill, prepared_input = self._extract_explicit_skill_invocation(user_input)
            if explicit_skill:
                plan = self.skill_system.route_request(
                    user_input=user_input,
                    prepared_input=prepared_input,
                    explicit_skill=explicit_skill,
                    limit=self.skill_shortlist_limit,
                    tool_available=self.tool_registry.has,
                )
                self._last_skill_route_plan = plan
                prepared_input = plan.prepared_input
                if plan.decision == SkillActivationDecision.EXPLICIT_MISSING:
                    messages[0]["content"] = self._render_prompt(active_skills=())
                    return None

                for candidate in plan.summary_candidates:
                    self.skill_system.lifecycle.shortlist(
                        self.skill_session,
                        metadata=candidate.metadata,
                        match=candidate.match,
                        reason="matched by explicit skill router",
                        trigger="runtime.explicit_skill",
                    )
                    self.skill_system.lifecycle.activate_summary(
                        self.skill_session,
                        skill_name=candidate.metadata.name,
                        reason="summary activated by explicit skill router",
                        trigger="runtime.explicit_skill",
                    )

            if explicit_skill and self._last_skill_route_plan and self._last_skill_route_plan.full_activation is not None:
                plan = self._last_skill_route_plan
                primary = plan.full_activation
                manifest = self.skill_system.load_skill_manifest(primary.metadata.name)
                self.skill_system.lifecycle.activate_full(
                    self.skill_session,
                    skill_name=primary.metadata.name,
                    manifest=manifest,
                    reason="manifest loaded by high-confidence skill route",
                    trigger="runtime.explicit_skill",
                )
            active_skills = self.skill_system.lifecycle.visible_active_skills(self.skill_session)
            full_context = self.skill_system.injector.render_full_skill_context(
                tuple(skill for skill in active_skills if skill.manifest is not None)
            )
            if full_context:
                messages.append(
                    {
                        "role": "system",
                        "content": f"已激活的 skill 详细说明：\n{full_context}",
                    }
                )
        messages[0]["content"] = self._render_prompt(active_skills=active_skills)
        self._prepared_user_input = prepared_input
        return prepared_input

    def _render_prompt(
        self,
        *,
        active_skills: tuple[ActiveSkill, ...],
        enabled_tool_names: Optional[tuple[str, ...]] = None,
    ) -> str:
        """功能：委托 PromptService 渲染当前 system 提示词。
        参数：
        - active_skills：当前激活技能列表。
        - enabled_tool_names：本轮暴露给模型的工具名集合。
        返回值：
        - str：完整 system 提示词文本。
        """
        return self.prompt_service.render_system_prompt(
            active_skills=active_skills,
            enabled_tool_names=enabled_tool_names,
        )

    @staticmethod
    def _extract_explicit_skill_invocation(user_input: str) -> tuple[Optional[str], str]:
        """功能：解析 /skill 显式激活指令并返回剩余输入。
        参数：
        - user_input：当前用户输入文本。
        返回值：
        - tuple[Optional[str], str]：二元组 (skill_name, prepared_input)。
        """
        text = (user_input or "").strip()
        match = re.match(r"^/skill\s+([a-z0-9-]+)\s*(.*)$", text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None, user_input
        skill_name = match.group(1).strip().lower()
        remainder = (match.group(2) or "").strip()
        prepared_input = remainder or f"请使用显式激活的 skill `{skill_name}` 处理当前请求。"
        return skill_name, prepared_input

    def _apply_skill_tool_result(self, observation):
        """功能：处理技能工具回传事件并转换为可注入 observation 文本。
        参数：
        - observation：工具执行观察结果。
        返回值：
        - Any：普通观察值原样返回；技能事件结果返回其 `observation_text`。
        """
        if not isinstance(observation, SkillToolResult):
            return observation
        if self.skill_system and self.skill_session:
            if observation.event_name == "reference_opened":
                self.skill_system.lifecycle.mark_references_opened(
                    self.skill_session,
                    skill_name=observation.skill_name,
                    reference_path=observation.resource_name or "",
                    reason="reference opened by tool",
                    trigger="runtime.tool_result",
                )
            elif observation.event_name == "instructions_loaded":
                manifest = self.skill_system.load_skill_manifest(observation.skill_name)
                self.skill_system.lifecycle.activate_full(
                    self.skill_session,
                    skill_name=observation.skill_name,
                    manifest=manifest,
                    reason="instructions loaded by tool",
                    trigger="runtime.tool_result",
                )
            elif observation.event_name == "script_exposed":
                self.skill_system.lifecycle.mark_scripts_exposed(
                    self.skill_session,
                    skill_name=observation.skill_name,
                    script_path=observation.resource_name or "",
                    reason="script executed by tool",
                    trigger="runtime.tool_result",
                )
        return observation.observation_text

    def _complete_visible_skills(self) -> None:
        """功能：在请求结束时将可见激活技能标记为完成态。
        参数：
        - 无。
        返回值：
        - 无。
        """
        if not self.skill_system or not self.skill_session:
            return
        for active in self.skill_system.lifecycle.visible_active_skills(self.skill_session):
            self.skill_system.lifecycle.complete(
                self.skill_session,
                skill_name=active.name,
                reason="request finished with final answer",
                trigger="runtime.complete",
            )

    # ------------------------------------------------------------------ #
    # Async interface (Phase-1 migration)                                  #
    # ------------------------------------------------------------------ #

    async def arun(
        self,
        *,
        user_input: str,
        messages: List[dict],
        conversation_turns: List[Tuple[str, str]],
        stream_callback: Optional[Callable[[str, str], None]] = None,
        stop_event=None,
    ) -> str:
        """功能：异步版 run()，所有模型与工具调用均为 await，不阻塞事件循环。
        参数：
        - user_input：当前用户输入文本。
        - messages：与模型交互的消息列表。
        - conversation_turns：会话轮次记录。
        - stream_callback：可选流式事件回调。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - str：本轮请求的最终回答字符串。
        """
        messages[:] = [messages[0]] + [item for item in messages[1:] if item.get("role") != "system"]
        prepared_input = self._prepare_request_context(user_input=user_input, messages=messages)
        if prepared_input is None:
            final_answer = "未找到你显式指定的 skill，请检查 skill 名称后重试。"
            emit_error(stream_callback, final_answer)
            self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
            return final_answer

        emit_agent_start(stream_callback, "Agent 开始处理请求。")
        memory_context = self.memory_manager.build_recent_memory_context(conversation_turns)
        self._conversation_context = memory_context
        if memory_context:
            messages.append({"role": "user", "content": f"conversation_history:\n{memory_context}"})
        messages.append({"role": "user", "content": prepared_input})
        self.memory_manager.trim_messages(messages)

        # context selector — await async 版本
        aselect = getattr(self.context_selector, "aselect", None)
        if callable(aselect):
            self._context_selection = await aselect(
                user_input=user_input,
                conversation_context=memory_context,
                explicit_skill=self._last_skill_route_plan.explicit_skill if self._last_skill_route_plan else None,
            )
        else:
            self._context_selection = self.context_selector.select(
                user_input=user_input,
                conversation_context=memory_context,
                explicit_skill=self._last_skill_route_plan.explicit_skill if self._last_skill_route_plan else None,
            )

        self._enabled_tool_names = self._enabled_tools_from_selection(self._context_selection)

        # Issue2 修复：把历史消息里已使用的工具补回本轮 enabled 列表，
        # 防止 context_selector 跨轮遗漏导致"本轮工具未下发"报错。
        if self._enabled_tool_names is not None:
            history_used = {
                (tc.get("function") or {}).get("name")
                for msg in messages
                for tc in (msg.get("tool_calls") or [])
            }
            extra = tuple(
                name for name in history_used
                if name and self.tool_registry.has(name) and name not in self._enabled_tool_names
            )
            if extra:
                self._enabled_tool_names = self._enabled_tool_names + extra

        self._emit_context_selection_decision(
            stream_callback,
            label="Selector 初筛",
            selection=self._context_selection,
            enabled_tool_names=self._enabled_tool_names,
        )
        self._inject_selector_directives(self._context_selection, messages)
        self._apply_context_selection_to_skills(self._context_selection, messages)
        self._enabled_tool_names = self._merge_allowed_tools_from_active_skills(self._enabled_tool_names)
        messages[0]["content"] = self._render_prompt(
            active_skills=(
                self.skill_system.lifecycle.visible_active_skills(self.skill_session)
                if self.skill_system and self.skill_session
                else ()
            ),
            enabled_tool_names=self._enabled_tool_names,
        )
        tools = self.tool_registry.openai_tools_schema(enabled_tool_names=self._enabled_tool_names)
        tool_call_signatures: set[str] = set()
        recovered_protocol_errors: set[str] = set()

        for step in range(1, self.max_steps + 1):
            if stop_event is not None and stop_event.is_set():
                final_answer = "请求处理超时，请稍后再试。"
                await self._aemit_final(stream_callback, final_answer)
                self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
                return final_answer

            result = await self.model_client.call_tool_call_model(
                messages,
                tools=tools,
                stop_event=stop_event,
            )

            if result.tool_calls:
                emit_model_decision(stream_callback, f"模型请求调用 {len(result.tool_calls)} 个工具。")
                skipped_duplicate_tool = False
                messages.append({
                    "role": "assistant",
                    "content": result.content or None,
                    "tool_calls": result.tool_calls,
                })
                for tool_call in result.tool_calls:
                    tool_call_id = str(tool_call.get("id") or f"call_{step}")
                    function = tool_call.get("function") or {}
                    tool_name = str(function.get("name") or "")
                    raw_arguments = function.get("arguments") or "{}"
                    try:
                        self._validate_tool_name(tool_name, enabled_tool_names=self._enabled_tool_names)
                        tool_arguments = self._parse_tool_call_arguments(tool_name, raw_arguments)
                        self._validate_tool_call(tool_name, tool_arguments)
                        function["arguments"] = json.dumps(tool_arguments, ensure_ascii=False)
                    except ToolCallProtocolError as exc:
                        if self._is_recoverable_tool_call_error(exc):
                            recovery_signature = json.dumps(
                                {"tool_name": tool_name, "raw_arguments": raw_arguments, "reason": exc.reason},
                                ensure_ascii=False, sort_keys=True, default=str,
                            )
                            if recovery_signature not in recovered_protocol_errors:
                                recovered_protocol_errors.add(recovery_signature)
                                recovery_observation = (
                                    f"工具调用参数错误：{exc}。请根据该工具 input_schema 补齐必填参数后重试；"
                                    "如果当前任务不需要该工具，请基于已有观察结果直接给出最终回答。"
                                )
                                emit_error(stream_callback, recovery_observation)
                                messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": recovery_observation})
                                self.memory_manager.trim_messages(messages)
                                continue
                        return self._finish_with_protocol_error(
                            stream_callback, conversation_turns, user_input, f"工具调用协议错误：{exc}",
                        )

                    call_signature = json.dumps(
                        {"tool_name": tool_name, "tool_arguments": tool_arguments},
                        ensure_ascii=False, sort_keys=True, default=str,
                    )
                    if call_signature in tool_call_signatures:
                        skipped_duplicate_tool = True
                        duplicate_observation = (
                            f"重复工具调用 `{tool_name}` 已跳过；"
                            "请基于已有工具观察结果继续推理并给出最终回答。"
                        )
                        emit_model_decision(stream_callback, duplicate_observation)
                        if stream_callback is None:
                            log_print(f"\n执行过程：{duplicate_observation}")
                        messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": duplicate_observation})
                        self.memory_manager.trim_messages(messages)
                        continue
                    tool_call_signatures.add(call_signature)

                    emit_tool_start(stream_callback, tool_name, tool_arguments)
                    if stream_callback is None:
                        log_print(f"\n执行过程：调用工具 {tool_name}({self._format_tool_arguments(tool_arguments)})")
                    try:
                        with bind_stream_emitter(stream_callback):
                            if tool_name == "expand_tool_context":
                                observation = await self._ahandle_expand_tool_context(
                                    need=str(tool_arguments.get("need") or ""),
                                    messages=messages,
                                    stream_callback=stream_callback,
                                )
                                tools = self.tool_registry.openai_tools_schema(
                                    enabled_tool_names=self._enabled_tool_names
                                )
                            else:
                                observation = await self.tool_registry.aexecute_json(tool_name, tool_arguments)
                    except Exception as exc:
                        observation = f"工具执行错误：{exc}"

                    rendered_observation = self._apply_skill_tool_result(observation)
                    if stream_callback is None:
                        log_print(f"\n执行过程：观察结果：{rendered_observation}")
                    emit_tool_end(stream_callback, rendered_observation)

                    if self.tool_registry.should_direct_return(tool_name):
                        final_answer = self._stringify_observation(rendered_observation)
                        self._complete_visible_skills()
                        emit_agent_finish(stream_callback, "Agent 完成：工具结果直接返回。")
                        await self._aemit_final(stream_callback, final_answer)
                        self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
                        return final_answer

                    observation_text = self._stringify_observation(rendered_observation)
                    kb_hint = build_kb_image_inline_hint(observation_text)
                    if kb_hint:
                        observation_text = f"{observation_text}\n{kb_hint}"
                    messages.append({"role": "tool", "tool_call_id": tool_call_id, "content": observation_text})
                    self.memory_manager.trim_messages(messages)

                if skipped_duplicate_tool:
                    summary = "Agent 完成：检测到重复工具调用，已基于已有观察结果生成最终回答。"
                    if stream_callback is None:
                        log_print(f"\n执行过程：{summary}")
                    emit_agent_finish(stream_callback, summary)
                    self._complete_visible_skills()
                    final_answer = await self._agenerate_final_answer(
                        messages,
                        fallback_content=result.content,
                        stream_callback=stream_callback,
                        stop_event=stop_event,
                    )
                    if not final_answer:
                        final_answer = "模型未返回最终答案，已终止本轮任务。"
                        emit_error(stream_callback, final_answer)
                    messages.append({"role": "assistant", "content": final_answer})
                    self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
                    return final_answer
                continue

            summary = (
                "Agent 完成：模型已根据工具观察结果生成最终回答。"
                if tool_call_signatures
                else "Agent 完成：模型未调用工具，直接回答。"
            )
            if stream_callback is None:
                log_print(f"\n执行过程：{summary}")
            emit_agent_finish(stream_callback, summary)
            self._complete_visible_skills()
            final_answer = await self._aresolve_final_answer(
                messages,
                fallback_content=result.content,
                stream_callback=stream_callback,
                stop_event=stop_event,
            )
            if not final_answer:
                final_answer = "模型未返回最终答案，已终止本轮任务。"
                emit_error(stream_callback, final_answer)
            messages.append({"role": "assistant", "content": final_answer})
            self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
            return final_answer

        final_answer = "本次任务推理步数已达上限，请缩小问题范围后重试。"
        emit_error(stream_callback, final_answer)
        self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
        return final_answer

    @staticmethod
    async def _aemit_final(stream_callback: Optional[Callable[[str, str], None]], final_answer: str) -> None:
        """功能：异步将最终回答按分片通过流式回调发送。
        参数：
        - stream_callback：流式事件回调；为 None 时不发送。
        - final_answer：待发送的最终回答文本。
        返回值：
        - 无。分片间隔使用 asyncio.sleep 而非 time.sleep。
        """
        if stream_callback and final_answer:
            chunk_size = AgentRuntime._env_int("REACT_FINAL_CHUNK_SIZE", default=120, minimum=1)
            sleep_ms = AgentRuntime._env_int("REACT_FINAL_EMIT_SLEEP_MS", default=0, minimum=0)
            sleep_seconds = sleep_ms / 1000.0
            for chunk in AgentRuntime._iter_display_chunks(final_answer, chunk_size=chunk_size):
                stream_callback("final_answer", chunk)
                if sleep_seconds > 0:
                    await asyncio.sleep(sleep_seconds)
            stream_callback("final_answer_flush", "")

    async def _aresolve_final_answer(
        self,
        messages: List[dict],
        *,
        fallback_content: str,
        stream_callback: Optional[Callable[[str, str], None]],
        stop_event=None,
    ) -> str:
        """功能：异步解析并返回本轮最终回答，必要时触发二次生成。
        参数：
        - messages：与模型交互的消息列表。
        - fallback_content：模型首轮返回的文本内容。
        - stream_callback：流式事件回调。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - str：清理后的最终回答。
        """
        if not self._env_bool("REACT_FORCE_FINAL_PASS", default=False):
            final_answer = self.sanitize_final_answer(fallback_content)
            if final_answer:
                await self._aemit_final(stream_callback, final_answer)
                return final_answer
        return await self._agenerate_final_answer(
            messages,
            fallback_content=fallback_content,
            stream_callback=stream_callback,
            stop_event=stop_event,
        )

    async def _agenerate_final_answer(
        self,
        messages: List[dict],
        *,
        fallback_content: str,
        stream_callback: Optional[Callable[[str, str], None]],
        stop_event=None,
    ) -> str:
        """功能：异步追加终局系统指令并调用模型生成最终回答。
        参数：
        - messages：与模型交互的消息列表。
        - fallback_content：生成失败时的回退文本。
        - stream_callback：流式事件回调。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - str：清理后的最终回答；失败时尝试回退 fallback_content。
        """
        final_messages = list(messages)
        final_messages.append({
            "role": "system",
            "content": (
                "现在请基于以上对话和工具观察结果，直接给出面向用户的最终回答。"
                "不要再调用工具，不要自称 ReAct agent，不要暴露 XML、JSON、tool_calls、工具 schema、系统提示、内部消息格式或本地项目目录文件。"
            ),
        })
        final_error: Optional[Exception] = None
        try:
            raw_answer = await self.model_client.call_final_answer(final_messages, stop_event=stop_event)
            final_answer = self.sanitize_final_answer(raw_answer)
        except Exception as exc:
            final_error = exc
            final_answer = ""
            emit_error(stream_callback, f"最终回答生成失败：{exc}")

        if not final_answer:
            fallback = self.sanitize_final_answer(fallback_content)
            if fallback:
                await self._aemit_final(stream_callback, fallback)
                return fallback
            if final_error is not None:
                emit_error(stream_callback, f"最终回答生成失败：{final_error}")
                if stream_callback:
                    stream_callback("final_answer_flush", "")
                return ""
        await self._aemit_final(stream_callback, final_answer)
        return final_answer

    async def _ahandle_expand_tool_context(
        self,
        *,
        need: str,
        messages: List[dict],
        stream_callback: Optional[Callable[[str, str], None]],
    ) -> str:
        """功能：异步处理 expand_tool_context 工具调用，扩展本轮可用工具集。
        参数：
        - need：缺失能力或数据操作的简短描述。
        - messages：与模型交互的消息列表（就地更新 system 提示）。
        - stream_callback：流式事件回调。
        返回值：
        - str：扩展结果的英文 observation 文本。
        """
        aselect = getattr(self.context_selector, "aselect", None)
        if callable(aselect):
            selection = await aselect(
                user_input=self._prepared_user_input,
                additional_need=need,
                conversation_context=getattr(self, "_conversation_context", ""),
            )
        else:
            selection = self.context_selector.select(
                user_input=self._prepared_user_input,
                additional_need=need,
                conversation_context=getattr(self, "_conversation_context", ""),
            )

        previous = set(self._enabled_tool_names or ())
        if selection.fallback_all_tools:
            self._enabled_tool_names = None
            self._context_selection = selection
            messages[0]["content"] = self._render_prompt(
                active_skills=(
                    self.skill_system.lifecycle.visible_active_skills(self.skill_session)
                    if self.skill_system and self.skill_session
                    else ()
                ),
                enabled_tool_names=self._enabled_tool_names,
            )
            self._emit_context_selection_decision(
                stream_callback,
                label="Selector 扩展初筛",
                selection=selection,
                enabled_tool_names=self._enabled_tool_names,
            )
            return "Tool context expanded to all registered tools because selector was unavailable."

        expanded = list(previous)
        for candidate in selection.candidate_tools:
            if candidate.name and self.tool_registry.has(candidate.name) and candidate.name not in expanded:
                expanded.append(candidate.name)
        self._context_selection = selection
        self._enabled_tool_names = tuple(
            name for name in self._all_tool_names() if name in set(expanded)
        )
        self._emit_context_selection_decision(
            stream_callback,
            label="Selector 扩展初筛",
            selection=selection,
            enabled_tool_names=self._enabled_tool_names,
        )
        self._apply_context_selection_to_skills(selection, messages)
        self._enabled_tool_names = self._merge_allowed_tools_from_active_skills(self._enabled_tool_names)
        messages[0]["content"] = self._render_prompt(
            active_skills=(
                self.skill_system.lifecycle.visible_active_skills(self.skill_session)
                if self.skill_system and self.skill_session
                else ()
            ),
            enabled_tool_names=self._enabled_tool_names,
        )
        new_tools = [name for name in self._enabled_tool_names if name not in previous]
        if new_tools:
            observation = "Expanded tool context. Newly available tools: " + ", ".join(new_tools)
        else:
            observation = "No additional tools were selected for that need."
        emit_model_decision(stream_callback, observation)
        return observation
