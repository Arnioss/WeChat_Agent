from __future__ import annotations

import os
import platform
from pathlib import Path

from app.skills.models import ActiveSkill
from app.agent.tool_registry import ToolRegistry


class PromptService:
    """功能：组装系统提示词、环境信息与技能摘要内容。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self, *, project_directory: str, tool_registry: ToolRegistry, max_prompt_files: int = 30):
        """功能：绑定项目路径与工具注册中心，配置系统提示词文件展示上限。
        参数：
        - project_directory：项目根目录路径。
        - tool_registry：工具注册中心实例。
        - max_prompt_files：提示词中最多展示的项目根目录文件数。
        返回值：
        - 无。目录路径会被解析为绝对路径，用于后续环境信息渲染。
        """
        self.project_directory = project_directory
        self.tool_registry = tool_registry
        self.max_prompt_files = max_prompt_files
        self.root_dir = Path(project_directory).resolve()

    def render_system_prompt(self, *, active_skills: tuple[ActiveSkill, ...] = ()) -> str:
        """功能：渲染完整系统提示词文本。
        参数：
        - active_skills：当前激活技能列表。
        返回值：
        - str：由最小行为内核、工具清单、环境信息和技能摘要拼接而成的提示词。
        """
        sections = [
            self._render_minimal_kernel(),
            f"可用工具：\n{self.tool_registry.describe_tools()}",
            self._render_environment_section(),
        ]
        active_section = self.render_active_skill_summaries(active_skills)
        if active_section:
            sections.append(f"当前已激活的 skills 摘要：\n{active_section}")
        return "\n\n".join(section for section in sections if section.strip())

    @staticmethod
    def _render_minimal_kernel() -> str:
        """功能：返回 ReAct 执行约束的核心提示段落。
        参数：
        - 无。
        返回值：
        - str：基础行为规则文本。
        """
        return (
            "你是一个面向代码与项目任务的 ReAct agent。\n"
            "请严格使用以下 XML 标签：<question>、<thought>、<action>、<observation>、<final_answer>。\n"
            "每次回复必须先输出 <thought>，再输出 <action> 或 <final_answer>。\n"
            "输出 <action> 后必须立即停止，等待真实 <observation>。\n"
            "不要伪造 <observation>。\n"
            "工具调用必须是直接函数调用，不能使用 keyword arguments。\n"
            "调用 skill 工具时务必使用完整参数：list_skill_resources(skill_name)；"
            "load_skill_reference(skill_name, reference_path)。\n"
            "只有在 list_skill_resources 返回 references 非空时，才允许调用 load_skill_reference。\n"
            "如果当前已激活 skills，请优先遵循这些 skills 的摘要与后续注入说明。\n"
            "若任务不需要工具，也可以直接给出 <final_answer>。"
        )

    def render_active_skill_summaries(self, active_skills: tuple[ActiveSkill, ...]) -> str:
        """功能：渲染已激活技能摘要。
        参数：
        - active_skills：当前激活技能列表。
        返回值：
        - str：技能摘要文本；无激活技能时返回空串。
        """
        if not active_skills:
            return ""
        lines = []
        for active in active_skills:
            reasons = ""
            if active.match and active.match.match_reasons:
                reasons = f" | reasons: {'; '.join(active.match.match_reasons[:3])}"
            lines.append(f"- {active.metadata.summary_line()}{reasons}")
        return "\n".join(lines)

    @staticmethod
    def get_operating_system_name() -> str:
        """功能：获取当前操作系统显示名称。
        参数：
        - 无。
        返回值：
        - str：标准化后的系统名（macOS/Windows/Linux/Unknown）。
        """
        os_map = {
            "Darwin": "macOS",
            "Windows": "Windows",
            "Linux": "Linux",
        }
        return os_map.get(platform.system(), "Unknown")

    def _render_environment_section(self) -> str:
        """功能：渲染环境信息提示段落。
        参数：
        - 无。
        返回值：
        - str：包含操作系统与项目文件列表的文本。
        """
        return (
            f"环境信息：\n"
            f"操作系统：{self.get_operating_system_name()}\n"
            f"当前目录文件：{self.get_prompt_file_list()}"
        )

    def get_prompt_file_list(self) -> str:
        """功能：获取用于提示词展示的项目文件/目录列表。
        参数：
        - 无。
        返回值：
        - str：逗号分隔的文件清单，读取失败时返回降级提示文本。
        """
        exclude_dirs = {
            "Python3.12",
            ".git",
            ".idea",
            "__pycache__",
            "node_modules",
        }
        prefer_ext = {".py", ".md", ".json", ".txt", ".env"}
        names = []
        try:
            for name in sorted(os.listdir(self.project_directory)):
                abs_path = os.path.join(self.project_directory, name)
                if os.path.isdir(abs_path):
                    if name in exclude_dirs:
                        continue
                    names.append(f"{name}/")
                else:
                    _, ext = os.path.splitext(name)
                    if ext.lower() in prefer_ext or name == ".env":
                        names.append(name)
                if len(names) >= self.max_prompt_files:
                    break
        except Exception:
            return "(file list unavailable)"

        if not names:
            return "(no project files)"
        return ", ".join(names)
