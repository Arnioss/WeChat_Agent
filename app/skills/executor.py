from __future__ import annotations

import json
import logging
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from app.agent.tool_metadata import ToolRichMetadata
from app.skills.lifecycle import SkillSessionState
from app.skills.models import SkillLifecycleStatus
from app.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkillToolResult:
    """功能：承载技能工具调用结果文本及生命周期事件元数据。
    参数：
    - 无。
    返回值：
    - 无。`event_name/resource_name` 用于驱动运行时状态推进，而非直接面向用户展示。
    """
    observation_text: str
    skill_name: str
    event_name: str
    resource_name: str | None = None


class SkillExecutor:
    """功能：构建并执行技能工具（资源列举、引用读取、脚本执行）。
    参数：
    - 无。
    返回值：
    - 无。仅允许访问已激活技能且受路径边界校验限制。
    """
    def __init__(
        self,
        *,
        registry: SkillRegistry,
        allow_script_execution: bool = False,
        script_timeout_seconds: int = 20,
        script_output_limit: int = 4000,
    ):
        """功能：注入技能注册中心并配置脚本执行策略与输出限制。
        参数：
        - registry：技能注册中心实例。
        - allow_script_execution：是否允许执行技能脚本。
        - script_timeout_seconds：脚本执行超时秒数。
        - script_output_limit：脚本输出截断长度上限。
        返回值：
        - 无。禁用脚本执行时只暴露资源查询与引用读取工具。
        """
        self.registry = registry
        self.allow_script_execution = allow_script_execution
        self.script_timeout_seconds = script_timeout_seconds
        self.script_output_limit = script_output_limit

    def build_tools(self, session: SkillSessionState) -> list[Callable]:
        """功能：构建当前会话可暴露的 skill 工具集合。
        参数：
        - session：当前技能会话状态对象。
        返回值：
        - list[Callable]：包含资源查询、引用读取及可选脚本执行工具的函数列表。
        """
        tools = [
            self._make_load_skill_instructions_tool(session),
            self._make_list_skill_resources_tool(session),
            self._make_load_skill_reference_tool(session),
        ]
        if self.allow_script_execution:
            tools.append(self._make_run_skill_script_tool(session))
        return tools

    def _make_load_skill_instructions_tool(self, session: SkillSessionState) -> Callable:
        """功能：创建 `load_skill_instructions` 工具函数，用于读取完整 SKILL.md。
        参数：
        - session：当前技能会话状态对象。
        返回值：
        - Callable：可供 Agent 调用的 load_skill_instructions 工具函数。
        """
        def load_skill_instructions(skill_name: str):
            """功能：读取当前候选或激活 skill 的完整说明，并返回可执行指引文本。
            参数：
            - skill_name：当前候选或已激活的 skill 名称。
            返回值：
            - SkillToolResult：含 manifest 正文及 instructions_loaded 生命周期事件信息。
            """
            active = self._require_visible_skill(session, skill_name)
            manifest = self.registry.load_manifest(active.metadata.name)
            return SkillToolResult(
                observation_text=self._render_manifest_text(manifest),
                skill_name=active.metadata.name,
                event_name="instructions_loaded",
            )

        load_skill_instructions.__name__ = "load_skill_instructions"
        load_skill_instructions.__tool_rich_metadata__ = ToolRichMetadata(
            summary="读取当前候选或已激活 skill 的完整 SKILL.md 指令，实现按需 progressive disclosure。",
            when_to_use=(
                "当前 skill 摘要不足以决定具体流程、格式或约束时。",
                "系统提示显示多个候选 skill，但需要进一步确认某个 skill 的完整说明时。",
            ),
            when_not_to_use=(
                "没有候选或已激活 skill 时。",
                "普通闲聊、日期查询、无需专用技能的问题。",
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "当前候选或已激活的 skill 名称。"}
                },
                "required": ["skill_name"],
                "additionalProperties": False,
            },
            output_description="字符串：skill 名称、描述、完整指令正文和可用资源列表。",
            examples=('load_skill_instructions({"skill_name": "testcase-generator"})',),
            notes=("只能读取当前请求已 shortlisted/activated 的 skill。",),
            priority=58,
        )
        return load_skill_instructions

    def _make_list_skill_resources_tool(self, session: SkillSessionState) -> Callable:
        """功能：创建 `list_skill_resources` 工具函数，用于列出技能可访问资源。
        参数：
        - session：当前技能会话状态对象。
        返回值：
        - Callable：可供 Agent 调用的资源列表工具函数。
        """
        def list_skill_resources(skill_name: str):
            """功能：返回指定已激活技能的 references/scripts/assets 列表。
            参数：
            - skill_name：技能名称。
            返回值：
            - SkillToolResult：包含资源清单 JSON 文本及技能生命周期事件信息。
            """
            active = self._require_accessible_skill(session, skill_name)
            listing = self.registry.list_resources(active.metadata.name)
            return SkillToolResult(
                observation_text=json.dumps(
                    {
                        "skill": listing.skill,
                        "references": list(listing.references),
                        "scripts": list(listing.scripts),
                        "assets": list(listing.assets),
                    },
                    ensure_ascii=False,
                ),
                skill_name=active.metadata.name,
                event_name="resources_listed",
            )

        list_skill_resources.__name__ = "list_skill_resources"
        list_skill_resources.__tool_rich_metadata__ = ToolRichMetadata(
            summary="列出当前已激活 skill 的 references/scripts/assets 资源。",
            when_to_use=(
                "已激活 skill，且需要知道该 skill 是否有补充 reference、脚本或资产时。",
            ),
            when_not_to_use=(
                "skill 尚未候选或激活时。",
                "只需要读取完整 SKILL.md 指令时，优先用 load_skill_instructions。",
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "当前已激活的 skill 名称。"}
                },
                "required": ["skill_name"],
                "additionalProperties": False,
            },
            output_description="JSON 字符串：包含 references、scripts、assets 文件列表。",
            examples=('list_skill_resources({"skill_name": "knowledge-rag-answer"})',),
            priority=54,
        )
        return list_skill_resources

    def _make_load_skill_reference_tool(self, session: SkillSessionState) -> Callable:
        """功能：创建 `load_skill_reference` 工具函数，用于读取技能 reference 内容。
        参数：
        - session：当前技能会话状态对象。
        返回值：
        - Callable：可供 Agent 调用的 reference 读取工具函数。
        """
        def load_skill_reference(skill_name: str, reference_path: str = ""):
            """功能：读取指定技能 reference 文件，并返回截断后的文本内容。
            参数：
            - skill_name：技能名称。
            - reference_path：技能引用文件的相对路径。
            返回值：
            - SkillToolResult：包含读取结果文本或错误提示，并携带技能生命周期事件信息。
            """
            active = self._require_accessible_skill(session, skill_name)
            listing = self.registry.list_resources(active.metadata.name)
            references = list(listing.references)
            normalized_path = (reference_path or "").strip().replace("\\", "/")
            if not references:
                return SkillToolResult(
                    observation_text=(
                        "该 skill 没有可用 references 文件。"
                        "请不要继续调用 load_skill_reference，直接基于已激活 skill 摘要完成任务。"
                    ),
                    skill_name=active.metadata.name,
                    event_name="reference_opened",
                )

            if not normalized_path:
                if len(references) == 1:
                    normalized_path = references[0]
                else:
                    return SkillToolResult(
                        observation_text=(
                            "缺少 reference_path 参数。"
                            f"可用 references：{json.dumps(references, ensure_ascii=False)}"
                        ),
                        skill_name=active.metadata.name,
                        event_name="reference_opened",
                    )

            reference_root = active.metadata.skill_dir / "references"
            resolved_path = self._resolve_relative_path(reference_root, normalized_path)
            if not resolved_path.is_file():
                return SkillToolResult(
                    observation_text=(
                        f"reference 不存在：{normalized_path}。"
                        f"可用 references：{json.dumps(references, ensure_ascii=False)}"
                    ),
                    skill_name=active.metadata.name,
                    event_name="reference_opened",
                )
            text = resolved_path.read_text(encoding="utf-8")
            clipped = self._clip_output(text)
            return SkillToolResult(
                observation_text=clipped,
                skill_name=active.metadata.name,
                event_name="reference_opened",
                resource_name=normalized_path,
            )

        load_skill_reference.__name__ = "load_skill_reference"
        load_skill_reference.__tool_rich_metadata__ = ToolRichMetadata(
            summary="读取当前已激活 skill 的 references 文件内容。",
            when_to_use=(
                "完整 SKILL.md 指令提到需要查看 references 中的补充说明时。",
                "list_skill_resources 返回了相关 reference 且任务需要更细资料时。",
            ),
            when_not_to_use=(
                "skill 没有 references 文件时。",
                "reference_path 指向 skill references 目录之外时。",
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "当前已激活的 skill 名称。"},
                    "reference_path": {
                        "type": "string",
                        "description": "references 目录下的相对路径；只有一个 reference 时可省略。",
                    },
                },
                "required": ["skill_name"],
                "additionalProperties": False,
            },
            output_description="字符串：reference 文件内容，超过输出限制会截断。",
            examples=(
                'load_skill_reference({"skill_name": "knowledge-rag-answer", "reference_path": "answering-guide.md"})',
            ),
            notes=("路径会被限制在 skill 的 references 目录内。",),
            priority=52,
        )
        return load_skill_reference

    def _make_run_skill_script_tool(self, session: SkillSessionState) -> Callable:
        """功能：创建 `run_skill_script` 工具函数，用于执行技能目录中的脚本。
        参数：
        - session：当前技能会话状态对象。
        返回值：
        - Callable：可供 Agent 调用的脚本执行工具函数。
        """
        def run_skill_script(skill_name: str, script_path: str, args: Sequence[str] | None = None):
            """功能：执行技能脚本并返回退出码、stdout、stderr 组合文本。
            参数：
            - skill_name：技能名称。
            - script_path：技能脚本文件的相对路径。
            - args：传给脚本或工具的参数列表。
            返回值：
            - SkillToolResult：包含执行结果文本及对应的生命周期事件信息。
            """
            active = self._require_accessible_skill(session, skill_name)
            script_root = active.metadata.skill_dir / "scripts"
            resolved_script = self._resolve_relative_path(script_root, script_path)
            if not resolved_script.is_file():
                raise FileNotFoundError(f"script 不存在：{script_path}")
            if args is None:
                args = []
            if not isinstance(args, (list, tuple)):
                raise ValueError("script 参数必须是数组")
            safe_args = [str(item) for item in args]
            command = self._build_script_command(resolved_script, safe_args)
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(active.metadata.skill_dir),
                    capture_output=True,
                    text=True,
                    timeout=self.script_timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                self._audit_script(
                    skill_name=active.metadata.name,
                    script_path=script_path,
                    args=safe_args,
                    exit_code=None,
                    stdout=self._clip_output(exc.stdout or ""),
                    stderr=self._clip_output(exc.stderr or ""),
                    outcome="timeout",
                )
                raise RuntimeError("skill script 执行超时") from exc

            stdout = self._clip_output(completed.stdout or "")
            stderr = self._clip_output(completed.stderr or "")
            self._audit_script(
                skill_name=active.metadata.name,
                script_path=script_path,
                args=safe_args,
                exit_code=completed.returncode,
                stdout=stdout,
                stderr=stderr,
                outcome="ok" if completed.returncode == 0 else "failed",
            )
            text = f"exit_code={completed.returncode}\nstdout:\n{stdout}"
            if stderr:
                text += f"\n\nstderr:\n{stderr}"
            return SkillToolResult(
                observation_text=text,
                skill_name=active.metadata.name,
                event_name="script_exposed",
                resource_name=script_path.replace("\\", "/"),
            )

        run_skill_script.__name__ = "run_skill_script"
        run_skill_script.__tool_rich_metadata__ = ToolRichMetadata(
            summary="执行当前已激活 skill scripts 目录下的受控脚本。",
            when_to_use=(
                "SKILL.md 明确要求运行 scripts 中的脚本，且环境变量 SKILL_ENABLE_SCRIPTS 已开启时。",
            ),
            when_not_to_use=(
                "默认配置下该工具不会暴露。",
                "脚本路径不在当前 skill scripts 目录内时。",
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string", "description": "当前已激活的 skill 名称。"},
                    "script_path": {"type": "string", "description": "scripts 目录下的相对脚本路径。"},
                    "args": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "传给脚本的字符串参数列表。",
                    },
                },
                "required": ["skill_name", "script_path"],
                "additionalProperties": False,
            },
            output_description="字符串：exit_code、stdout、stderr；stdout/stderr 会按配置截断。",
            examples=('run_skill_script({"skill_name": "demo", "script_path": "check.py", "args": ["input"]})',),
            notes=("Python 脚本使用当前运行时解释器执行。", "所有执行都会写入审计日志。"),
            priority=48,
        )
        return run_skill_script

    @staticmethod
    def _resolve_relative_path(root: Path, relative_path: str) -> Path:
        """功能：解析并校验技能目录内的相对路径。
        参数：
        - root：允许访问的根目录。
        - relative_path：相对路径文本。
        返回值：
        - Path：解析后的绝对路径。
        """
        if not root.is_dir():
            raise FileNotFoundError(f"目录不存在：{root}")
        resolved = (root / relative_path).resolve()
        root_resolved = root.resolve()
        if root_resolved not in resolved.parents and resolved != root_resolved:
            raise PermissionError("不允许访问 skill 目录外的资源")
        return resolved

    def _require_visible_skill(self, session: SkillSessionState, skill_name: str):
        """功能：校验技能已候选或激活且状态允许读取说明。
        参数：
        - session：当前技能会话状态对象。
        - skill_name：技能名称。
        返回值：
        - ActiveSkill：通过权限校验后的激活技能对象。
        """
        if skill_name not in session.active_skills:
            raise PermissionError(f"skill 未候选或激活，无法访问说明：{skill_name}")
        active = session.active_skills[skill_name]
        allowed_statuses = {
            SkillLifecycleStatus.SHORTLISTED,
            SkillLifecycleStatus.ACTIVATED_SUMMARY,
            SkillLifecycleStatus.ACTIVATED_FULL,
            SkillLifecycleStatus.REFERENCES_OPENED,
            SkillLifecycleStatus.SCRIPTS_EXPOSED,
        }
        if active.status not in allowed_statuses:
            raise PermissionError(f"skill 当前状态不允许访问说明：{skill_name} ({active.status})")
        return active

    def _require_accessible_skill(self, session: SkillSessionState, skill_name: str):
        """功能：校验技能是否已激活且状态允许访问资源。
        参数：
        - session：当前技能会话状态对象。
        - skill_name：技能名称。
        返回值：
        - ActiveSkill：通过权限校验后的激活技能对象。
        """
        if skill_name not in session.active_skills:
            raise PermissionError(f"skill 未激活，无法访问资源：{skill_name}")
        active = session.active_skills[skill_name]
        allowed_statuses = {
            SkillLifecycleStatus.ACTIVATED_SUMMARY,
            SkillLifecycleStatus.ACTIVATED_FULL,
            SkillLifecycleStatus.REFERENCES_OPENED,
            SkillLifecycleStatus.SCRIPTS_EXPOSED,
        }
        if active.status not in allowed_statuses:
            raise PermissionError(f"skill 当前状态不允许访问资源：{skill_name} ({active.status})")
        return active

    @staticmethod
    def _build_script_command(script_path: Path, args: list[str]) -> list[str]:
        """功能：根据脚本后缀生成可执行命令行参数列表。
        参数：
        - script_path：技能脚本文件的相对路径。
        - args：传给脚本或工具的参数列表。
        返回值：
        - list[str]：可直接传给 `subprocess.run` 的命令数组。
        """
        suffix = script_path.suffix.lower()
        if suffix == ".py":
            return [sys.executable, str(script_path), *args]
        if suffix in {".sh", ".bash"}:
            return ["bash", str(script_path), *args]
        return [str(script_path), *args]

    @staticmethod
    def _render_manifest_text(manifest) -> str:
        """功能：将技能 manifest 格式化为 Agent 可读的说明文本。
        参数：
        - manifest：技能 manifest 对象，含名称、描述、正文与资源列表。
        返回值：
        - str：包含 Skill/Description/Instructions 及可选资源清单的多行文本。
        """
        parts = [
            f"Skill: {manifest.name}",
            f"Description: {manifest.description}",
            "Instructions:",
            manifest.body,
        ]
        if manifest.references:
            parts.append("References: " + ", ".join(manifest.references))
        if manifest.scripts:
            parts.append("Scripts: " + ", ".join(manifest.scripts))
        if manifest.assets:
            parts.append("Assets: " + ", ".join(manifest.assets))
        return "\n".join(parts)

    def _clip_output(self, text: str) -> str:
        """功能：按配置上限截断脚本输出，避免 observation 文本过长。
        参数：
        - text：待处理文本内容。
        返回值：
        - str：未超限时返回原文，超限时返回截断后的文本并追加标记。
        """
        if len(text) <= self.script_output_limit:
            return text
        return text[: self.script_output_limit] + "\n...(truncated)"

    def _audit_script(
        self,
        *,
        skill_name: str,
        script_path: str,
        args: list[str],
        exit_code: int | None,
        stdout: str,
        stderr: str,
        outcome: str,
    ) -> None:
        """功能：记录技能脚本执行审计日志。
        参数：
        - skill_name：技能名称。
        - script_path：技能脚本文件的相对路径。
        - args：传给脚本或工具的参数列表。
        - exit_code：脚本退出码，超时时可为 None。
        - stdout：标准输出文本（已截断）。
        - stderr：标准错误文本（已截断）。
        - outcome：执行结果标识（如 ok/failed/timeout）。
        返回值：
        - 无。
        """
        logger.info(
            "技能脚本执行 skill=%s script=%s args=%s outcome=%s exit_code=%s stdout=%r stderr=%r",
            skill_name,
            script_path,
            args,
            outcome,
            exit_code,
            stdout,
            stderr,
        )
