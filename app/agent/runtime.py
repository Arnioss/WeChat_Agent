from __future__ import annotations

import json
import re
from typing import Callable, List, Optional, Tuple

from app.agent.memory_manager import MemoryManager
from app.agent.model_client import ModelClient, safe_print
from app.agent.prompt_service import PromptService
from app.agent.tool_registry import ToolRegistry
from app.skills.executor import SkillToolResult
from app.skills.lifecycle import SkillSessionState
from app.skills.models import ActiveSkill, SkillMatch
from app.skills.system import SkillSystem


class AgentRuntime:
    """功能：编排单轮请求中的模型推理循环、工具执行与技能状态推进。
    参数：
    - 无。
    返回值：
    - 无。到达步数上限或动作连续失败时会提前终止，避免死循环。
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

    @staticmethod
    def sanitize_final_answer(answer: str) -> str:
        """功能：清理模型最终回答中的标签和提示噪声。
        参数：
        - answer：模型输出的最终回答文本。
        返回值：
        - 移除噪声标签后的最终回答文本。
        """
        text = (answer or "").strip()
        if not text:
            return text

        text = text.replace("</thought>", "").replace("<thought>", "")
        text = text.replace("<final_answer>", "").replace("</final_answer>", "")

        marker = "执行环境："
        idx = text.find(marker)
        if idx != -1:
            text = text[idx:]

        noise_markers = [
            "只能返回执行结果",
            "必须按以下顺序",
            "不要返回 raw_messages",
            "不要返回规则解释",
            "观察结果中已有这些字段",
        ]
        for marker_text in noise_markers:
            pos = text.find(marker_text)
            if pos != -1 and pos < 20:
                idx = text.find("执行环境：")
                if idx != -1:
                    text = text[idx:]
                break

        extra_noise = ["需要返回：", "必须按以下顺序", "不要省略、不要总结、不要解释"]
        if any(marker_text in text for marker_text in extra_noise):
            idx = text.find("执行环境：")
            if idx != -1:
                text = text[idx:]

        return text.strip()

    def run(
        self,
        *,
        user_input: str,
        messages: List[dict],
        conversation_turns: List[Tuple[str, str]],
        action_parser,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        stop_event=None,
    ) -> str:
        """功能：驱动模型与工具循环执行直到得到最终回答。
        参数：
        - user_input：当前用户输入文本。
        - messages：与模型交互的消息列表。
        - conversation_turns：会话轮次记录。
        - action_parser：用于解析 <action> 的解析器实例。
        - stream_callback：模型流式输出回调函数。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - 本轮请求的最终回答字符串。
        """
        messages[:] = [messages[0]] + [item for item in messages[1:] if item.get("role") != "system"]
        prepared_input = self._prepare_request_context(user_input=user_input, messages=messages)
        if prepared_input is None:
            final_answer = "未找到你显式指定的 skill，请检查 skill 名称后重试。"
            self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
            return final_answer

        memory_context = self.memory_manager.build_recent_memory_context(conversation_turns)
        if memory_context:
            messages.append(
                {
                    "role": "user",
                    "content": f"<conversation_history>{memory_context}</conversation_history>",
                }
            )
        messages.append({"role": "user", "content": f"<question>{prepared_input}</question>"})
        self.memory_manager.trim_messages(messages)

        repeated_action_failure_limit = 3
        last_action_failure_signature = ""
        repeated_action_failure_count = 0
        for step in range(1, self.max_steps + 1):
            content = self.model_client.call_model(
                messages,
                stream_callback=stream_callback,
                stop_event=stop_event,
            )
            messages.append({"role": "assistant", "content": content})

            if "<final_answer>" in content:
                final_answers = re.findall(r"<final_answer>(.*?)</final_answer>", content, re.DOTALL)
                if final_answers:
                    final_answer = self.sanitize_final_answer(final_answers[-1])
                    self._complete_visible_skills()
                    self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
                    return final_answer
                messages.append(
                    {
                        "role": "user",
                        "content": "你的上一条输出包含未闭合或格式错误的 <final_answer> 标签。请严格输出完整的 <final_answer>...</final_answer>。",
                    }
                )
                self.memory_manager.trim_messages(messages)
                continue

            action_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
            if not action_match:
                raise RuntimeError("模型未输出 <action>")

            action = action_match.group(1)
            try:
                tool_name, args = action_parser.parse(action)
                last_action_failure_signature = ""
                repeated_action_failure_count = 0
            except Exception as e:
                signature = f"{type(e).__name__}:{str(e).strip()}::{(action or '').strip()}"
                if signature == last_action_failure_signature:
                    repeated_action_failure_count += 1
                else:
                    last_action_failure_signature = signature
                    repeated_action_failure_count = 1

                if repeated_action_failure_count >= repeated_action_failure_limit:
                    final_answer = "当前请求的 action 连续解析失败，已自动终止以避免卡死。请重试，并只输出单行函数调用，例如 rag_summarize(\"你的问题\")。"
                    self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
                    return final_answer

                error_text = str(e)
                if "unclosed string literal or function call" in error_text:
                    rewrite_hint = (
                        "你的 <action> 存在未闭合的引号或括号。"
                        "请重写 action，仅输出一个完整可执行的函数调用，不要附带 <think>、解释或多行文本。"
                    )
                else:
                    rewrite_hint = (
                        "你的 <action> 格式不合法，必须是可执行的函数调用，"
                        "例如 get_current_date() 或 rag_summarize(\"问题\")。"
                    )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            rewrite_hint
                            + 
                            f"\n错误信息：{e}"
                        ),
                    }
                )
                self.memory_manager.trim_messages(messages)
                continue
            formatted_args = ", ".join(repr(arg) for arg in args)
            safe_print(f"\n\n🔧 Action: {tool_name}({formatted_args})")

            try:
                observation = self.tool_registry.execute(tool_name, args)
            except Exception as e:
                observation = f"工具执行错误：{e}"
            rendered_observation = self._apply_skill_tool_result(observation)
            safe_print(f"\n\n🔍 Observation：{rendered_observation}")

            if self.tool_registry.should_direct_return(tool_name):
                if isinstance(rendered_observation, dict):
                    final_answer = json.dumps(rendered_observation, ensure_ascii=False, default=str)
                else:
                    final_answer = "" if rendered_observation is None else str(rendered_observation)
                self._complete_visible_skills()
                self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
                return final_answer

            messages.append({"role": "user", "content": f"<observation>{rendered_observation}</observation>"})
            self.memory_manager.trim_messages(messages)

            if step == self.max_steps:
                final_answer = "本次任务推理步数已达上限，请缩小问题范围后重试。"
                self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
                return final_answer

        final_answer = "本次任务推理步数已达上限，请缩小问题范围后重试。"
        self.memory_manager.remember_turn(conversation_turns, user_input, final_answer)
        return final_answer

    def _prepare_request_context(self, *, user_input: str, messages: List[dict]) -> Optional[str]:
        """功能：根据输入与技能状态构建本轮请求上下文。
        参数：
        - user_input：当前用户输入文本。
        - messages：与模型交互的消息列表。
        返回值：
        - 处理后的请求文本；若显式 skill 不存在则返回 None。
        """
        active_skills: tuple[ActiveSkill, ...] = ()
        prepared_input = user_input
        if self.skill_system and self.skill_session:
            self.skill_system.lifecycle.begin_request(self.skill_session)
            explicit_skill, prepared_input = self._extract_explicit_skill_invocation(user_input)
            if explicit_skill:
                if not self.skill_system.registry.has_skill(explicit_skill):
                    messages[0]["content"] = self.prompt_service.render_system_prompt(active_skills=())
                    return None
                match = SkillMatch(
                    skill_name=explicit_skill,
                    score=100.0,
                    source="explicit",
                    match_reasons=(f"explicit invocation via /skill {explicit_skill}",),
                    allow_auto_activation=False,
                    decision="explicit",
                )
                metadata = self.skill_system.registry.get_metadata(explicit_skill)
                self.skill_system.lifecycle.shortlist(
                    self.skill_session,
                    metadata=metadata,
                    match=match,
                    reason="explicit invocation",
                    trigger="runtime.explicit",
                )
                self.skill_system.lifecycle.activate_summary(
                    self.skill_session,
                    skill_name=explicit_skill,
                    reason="explicit invocation",
                    trigger="runtime.explicit",
                )
            else:
                matches = self.skill_system.retrieval_strategy.retrieve(
                    prepared_input,
                    self.skill_system.list_skill_metadata(),
                    limit=self.skill_shortlist_limit,
                )
                for match in matches:
                    metadata = self.skill_system.registry.get_metadata(match.skill_name)
                    self.skill_system.lifecycle.shortlist(
                        self.skill_session,
                        metadata=metadata,
                        match=match,
                        reason="matched by retrieval",
                        trigger="runtime.retrieval",
                    )
                    self.skill_system.lifecycle.activate_summary(
                        self.skill_session,
                        skill_name=match.skill_name,
                        reason="summary activated for shortlisted skill",
                        trigger="runtime.retrieval",
                    )
            active_skills = self.skill_system.lifecycle.visible_active_skills(self.skill_session)
            if active_skills and (explicit_skill or len(active_skills) == 1):
                primary = active_skills[0]
                manifest = self.skill_system.load_skill_manifest(primary.name)
                self.skill_system.lifecycle.activate_full(
                    self.skill_session,
                    skill_name=primary.name,
                    manifest=manifest,
                    reason="manifest loaded for active primary skill",
                    trigger="runtime.manifest",
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
        messages[0]["content"] = self.prompt_service.render_system_prompt(active_skills=active_skills)
        return prepared_input

    @staticmethod
    def _extract_explicit_skill_invocation(user_input: str) -> tuple[Optional[str], str]:
        """功能：解析 /skill 显式激活指令并返回剩余输入。
        参数：
        - user_input：当前用户输入文本。
        返回值：
        - 二元组 (skill_name, prepared_input)。
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
