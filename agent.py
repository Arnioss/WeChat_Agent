import os
import sys
import threading
from typing import Callable, List, Optional, Tuple

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

import click
from tools import (
    get_current_date,
    load_mcp_tools,
    rag_summarize,
)
from app.agent.action_parser import ActionParser
from app.agent.memory_manager import MemoryManager
from app.agent.model_client import safe_print
from app.agent.model_client import ModelClient
from app.agent.prompt_service import PromptService
from app.agent.runtime import AgentRuntime
from app.agent.tool_registry import ToolRegistry
from app.skills.system import SkillSystem, build_skill_system


class ReActAgent:
    """功能：封装 ReAct 智能体运行时，统一管理模型、工具和会话状态。
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
    ):
        """功能：组装 ReAct 运行栈（工具、提示词、记忆、模型、技能）并初始化会话状态。
        参数：
        - model：用于推理的模型标识（如 OpenRouter 模型名）。
        - project_directory：项目根目录路径。
        - skill_system：可选的技能系统实例；未传入时按项目目录自动构建。
        - tools：可选的工具函数列表；未传入时自动组装默认工具集合。
        返回值：
        - 无。初始化会读取多项环境变量限制参数，缺失配置会在构建阶段暴露。
        """
        self.project_directory = project_directory
        self.max_steps = int(os.getenv("REACT_MAX_STEPS"))
        self.max_history_messages = int(os.getenv("REACT_MAX_HISTORY_MESSAGES"))
        self.max_memory_turns = int(os.getenv("REACT_MAX_MEMORY_TURNS"))
        self.memory_text_limit = int(os.getenv("REACT_MEMORY_TEXT_LIMIT"))
        self.max_prompt_files = int(os.getenv("REACT_PROMPT_MAX_FILES"))
        self.skill_shortlist_limit = int(os.getenv("SKILL_SHORTLIST_LIMIT", "3"))
        self.conversation_turns: List[Tuple[str, str]] = []
        self.action_parser = ActionParser()
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
        self.model_client = ModelClient(model=model)
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
        self.messages = [
            {"role": "system", "content": self.render_system_prompt()}
        ]

    def run(
        self,
        user_input: str,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ):
        """功能：驱动模型与工具循环执行直到得到最终回答。
        参数：
        - user_input：当前用户输入文本。
        - stream_callback：模型流式输出回调函数。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - 本轮请求的最终回答字符串。
        """
        return self.runtime.run(
            user_input=user_input,
            messages=self.messages,
            conversation_turns=self.conversation_turns,
            action_parser=self.action_parser,
            stream_callback=stream_callback,
            stop_event=stop_event,
        )

    def get_tool_list(self) -> str:
        """功能：返回当前已注册工具的可读列表。
        参数：
        - 无。
        返回值：
        - str：工具名称与签名说明文本。
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

    def call_model(self, messages, stream_callback: Optional[Callable[[str, str], None]] = None, stop_event=None):
        """功能：调用底层模型客户端完成一次推理请求。
        参数：
        - messages：与模型交互的消息列表。
        - stream_callback：模型流式输出回调函数。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - str：模型最终输出文本。
        """
        return self.model_client.call_model(messages, stream_callback=stream_callback, stop_event=stop_event)

    def parse_action(self, code_str: str):
        """功能：解析模型输出中的动作代码块。
        参数：
        - code_str：模型返回的动作或指令文本。
        返回值：
        - object：动作解析结果对象，具体结构由 `ActionParser` 定义。
        """
        return self.action_parser.parse(code_str)


def build_tools(project_directory: str, *, skill_system: SkillSystem, skill_session) -> List[Callable]:
    """功能：组装智能体可用工具集合（基础工具、MCP 工具与技能工具）。
    参数：
    - project_directory：项目根目录路径。
    - skill_system：技能系统实例，用于提供技能执行工具。
    - skill_session：当前技能会话状态，用于绑定技能工具上下文。
    返回值：
    - List[Callable]：可注册到工具调用框架的函数列表。
    """
    tools = [
        get_current_date,
        rag_summarize,
    ]
    tools.extend(load_mcp_tools(project_directory))
    tools.extend(skill_system.executor.build_tools(skill_session))
    return tools


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

    model_name = os.getenv("OPENROUTER_MODEL")
    if not model_name:
        raise ValueError("缺少环境变量 OPENROUTER_MODEL，请在 .env 文件中设置。")
    skill_system = build_skill_system(project_directory=project_dir)
    agent = ReActAgent(
        model=model_name,
        project_directory=project_dir,
        skill_system=skill_system,
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
            safe_print("上下文已清空。")
            continue

        agent.run(task)


if __name__ == "__main__":
    main()
