import asyncio
import os
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

import click
from tools import (
    bing_search,
    crawl_webpage,
    get_current_date,
    load_mcp_tools,
    rag_summarize,
)
from app.agent.memory_manager import MemoryManager
from app.agent.console_stream import make_console_stream_callback
from app.agent.model_client import log_print
from app.agent.model_client import AsyncModelClient, ModelClient
from app.agent.prompt_service import PromptService
from app.agent.runtime import AgentRuntime
from app.agent.tool_registry import ToolRegistry
from app.infrastructure.logging_setup import configure_project_logging
from app.infrastructure.tool_call_recorder import tool_call_observer
from app.skills.system import SkillSystem, build_skill_system


_DEFAULT_SYSTEM_PROMPT_CACHE: Dict[str, str] = {}


def _project_key(project_directory: str) -> str:
    """功能：将项目目录规范化为绝对路径字符串，用作缓存键。
    参数：
    - project_directory：项目根目录路径。
    返回值：
    - str：resolve 后的绝对路径字符串。
    """
    return str(Path(project_directory).resolve())


class ReActAgent:
    """功能：ReAct 智能体核心类，负责提示词渲染、工具调用和多轮推理控制。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(
        self,
        *,
        model: str,
        project_directory: str,
        skill_system: Optional[SkillSystem] = None,
        tools: Optional[List[Callable]] = None,
        key_channel: str = "default",
    ):
        """功能：组装 ReAct 运行栈（工具、提示词、记忆、模型、技能）并初始化会话状态。
        参数：
        - model：用于推理的模型标识（如 OpenRouter 模型名）。
        - project_directory：项目根目录路径。
        - skill_system：可选的技能系统实例；未传入时按项目目录自动构建。
        - tools：可选的工具函数列表；未传入时自动组装默认工具集合。
        - key_channel：LLM API Key 分池（wechat / web / default）。
        返回值：
        - 无。
        """
        self.project_directory = project_directory
        self.max_steps = int(os.getenv("REACT_MAX_STEPS"))
        self.max_history_messages = int(os.getenv("REACT_MAX_HISTORY_MESSAGES"))
        self.max_memory_turns = int(os.getenv("REACT_MAX_MEMORY_TURNS"))
        self.memory_text_limit = int(os.getenv("REACT_MEMORY_TEXT_LIMIT"))
        self.max_prompt_files = int(os.getenv("REACT_PROMPT_MAX_FILES"))
        self.skill_shortlist_limit = int(os.getenv("SKILL_SHORTLIST_LIMIT", "3"))
        self.conversation_turns: List[Tuple[str, str]] = []
        self.skill_system = skill_system or build_skill_system(project_directory=project_directory)
        self.skill_session = self.skill_system.create_session()
        resolved_tools = tools or build_tools(
            project_directory,
            skill_system=self.skill_system,
            skill_session=self.skill_session,
        )
        self.tool_registry = ToolRegistry(
            resolved_tools,
            direct_return_tools=set(),
            observer=tool_call_observer,
        )
        self.prompt_service = PromptService(
            project_directory=project_directory,
            tool_registry=self.tool_registry,
            max_prompt_files=self.max_prompt_files,
        )
        self.memory_manager = MemoryManager(
            max_memory_turns=self.max_memory_turns,
            memory_text_limit=self.memory_text_limit,
            max_history_messages=self.max_history_messages,
        )
        self.model_client = AsyncModelClient(model=model, key_channel=key_channel)
        self.runtime = AgentRuntime(
            tool_registry=self.tool_registry,
            model_client=self.model_client,
            memory_manager=self.memory_manager,
            prompt_service=self.prompt_service,
            max_steps=self.max_steps,
            skill_system=self.skill_system,
            skill_session=self.skill_session,
            skill_shortlist_limit=self.skill_shortlist_limit,
        )
        project_key = _project_key(project_directory)
        cached_prompt = _DEFAULT_SYSTEM_PROMPT_CACHE.get(project_key)
        if cached_prompt:
            system_prompt = cached_prompt
        else:
            system_prompt = self.render_system_prompt()
            _DEFAULT_SYSTEM_PROMPT_CACHE[project_key] = system_prompt
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    async def arun(
        self,
        user_input: str,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        stop_event=None,
    ) -> str:
        """功能：异步执行 ReAct 推理循环，模型与工具调用均 await，不阻塞事件循环。
        参数：
        - user_input：用户输入文本。
        - stream_callback：可选流式输出回调 (event_type, content)。
        - stop_event：可选停止信号。
        返回值：
        - str：Agent 最终回复文本。
        """
        return await self.runtime.arun(
            user_input=user_input,
            messages=self.messages,
            conversation_turns=self.conversation_turns,
            stream_callback=stream_callback,
            stop_event=stop_event,
        )

    def run(
        self,
        user_input: str,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> str:
        """功能：同步执行 ReAct 推理，供 CLI/Streamlit 调用；内部以 asyncio.run() 驱动 arun()。
        参数：
        - user_input：用户输入文本。
        - stream_callback：可选流式输出回调 (event_type, content)。
        - stop_event：可选线程停止事件。
        返回值：
        - str：Agent 最终回复文本。
        """
        return asyncio.run(self.arun(user_input, stream_callback, stop_event))

    def get_tool_list(self) -> str:
        """功能：生成工具说明列表，供系统提示词展示可用能力。
        参数：
        - 无。
        返回值：
        - str：工具签名与文档拼接文本。
        """
        return self.tool_registry.describe_tools()

    def render_system_prompt(self) -> str:
        """功能：渲染系统提示词模板。
        参数：
        - 无。
        返回值：
        - str：拼接后的系统提示词文本。
        """
        return self.prompt_service.render_system_prompt()

@lru_cache(maxsize=8)
def _build_base_tools(project_directory: str) -> tuple:
    """功能：构建并缓存不含 skill 会话绑定的静态工具集合。
    参数：
    - project_directory：项目根目录绝对路径。
    返回值：
    - tuple：静态工具与 MCP 工具的可复用元组。
    """
    project_directory = _project_key(project_directory)
    tools: List[Callable] = [
        get_current_date,
        bing_search,
        crawl_webpage,
    ]
    if _rag_enabled():
        tools.append(rag_summarize)
    tools.extend(load_mcp_tools(project_directory))
    return tuple(tools)


def build_tools(project_directory: str, *, skill_system: SkillSystem, skill_session) -> List[Callable]:
    """功能：组装智能体可用工具集合（基础工具、联网、RAG、MCP 与技能工具）。
    参数：
    - project_directory：项目根目录路径。
    - skill_system：技能系统实例，用于提供技能执行工具。
    - skill_session：当前技能会话状态，用于绑定技能工具上下文。
    返回值：
    - List[Callable]：可注册到工具调用框架的函数列表。
    """
    tools = list(_build_base_tools(project_directory))
    tools.extend(skill_system.executor.build_tools(skill_session))
    return tools


def warm_agent(
    *,
    model: str,
    project_directory: str,
    skill_system: Optional[SkillSystem] = None,
    key_channel: str = "default",
) -> Dict[str, Any]:
    """功能：启动前预热 ReActAgent，填充 MCP/工具/系统提示词等缓存。
    参数：
    - model：模型标识。
    - project_directory：项目根目录。
    - skill_system：可选共享技能系统实例。
    返回值：
    - Dict[str, Any]：包含 duration_ms、tool_count、prompt_chars 的摘要。
    """
    started_at = time.time()
    agent = ReActAgent(
        model=model,
        project_directory=project_directory,
        skill_system=skill_system,
        key_channel=key_channel,
    )
    prompt = agent.render_system_prompt()
    tool_count = len(getattr(agent.tool_registry, "_tools", {}) or {})
    return {
        "duration_ms": round((time.time() - started_at) * 1000.0, 2),
        "tool_count": tool_count,
        "prompt_chars": len(prompt or ""),
    }


def _rag_enabled() -> bool:
    """功能：读取 `RAG_ENABLED` 环境变量并判断是否注册 RAG 工具。
    参数：
    - 无。
    返回值：
    - bool：配置为 1/true/yes/on 时返回 True。
    """
    return (os.getenv("RAG_ENABLED") or "").strip().lower() in ("1", "true", "yes", "on")


@click.command()
@click.argument(
    'project_directory',
    required=False,
    default=".",
    type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
def main(project_directory):
    """功能：启动命令行交互式智能体入口。
    参数：
    - project_directory：项目根目录路径。
    返回值：
    - 无。
    异常：
    - ValueError：未配置 `OPENROUTER_MODEL` 环境变量时抛出。
    """
    project_dir = os.path.abspath(project_directory)
    configure_project_logging(Path(project_dir), logger_name="agent.cli")

    model_name = os.getenv("OPENROUTER_MODEL")
    if not model_name:
        raise ValueError("缺少环境变量 OPENROUTER_MODEL，请在 .env 文件中设置。")
    skill_system = build_skill_system(project_directory=project_dir)
    key_channel = (os.getenv("LLM_KEY_CHANNEL") or "default").strip() or "default"
    agent = ReActAgent(
        model=model_name,
        project_directory=project_dir,
        skill_system=skill_system,
        key_channel=key_channel,
    )

    while True:
        task = input("请输入任务（/exit退出，/clear清空上下文）：").strip()

        if task.lower() == "/exit":
            break
        if task.lower() == "/clear":
            agent.skill_session.active_skills.clear()
            agent.skill_session.completed_skills.clear()
            agent.skill_session.dismissed_skills.clear()
            agent.skill_session.current_request_id = 0
            agent.messages = [
                {"role": "system", "content": agent.render_system_prompt()}
            ]
            agent.conversation_turns = []
            log_print("上下文已清空。")
            continue

        cli_callback, _ = make_console_stream_callback()
        agent.run(task, stream_callback=cli_callback)


if __name__ == "__main__":
    main()
