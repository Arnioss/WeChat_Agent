from __future__ import annotations

from typing import Iterable

from app.skills.models import ActiveSkill


class SkillInjector:
    """功能：把激活技能信息渲染为可注入提示词的文本。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def render_active_skill_summaries(self, active_skills: Iterable[ActiveSkill]) -> str:
        """功能：渲染激活技能摘要列表。
        参数：
        - active_skills：当前激活技能列表。
        返回值：
        - str：按行展示的技能摘要文本，包含主要匹配原因。
        """
        lines = []
        for active in active_skills:
            match_suffix = ""
            if active.match and active.match.match_reasons:
                match_suffix = f" | reasons: {'; '.join(active.match.match_reasons[:3])}"
            lines.append(f"- {active.metadata.summary_line()}{match_suffix}")
        return "\n".join(lines)

    def render_full_skill_context(self, active_skills: Iterable[ActiveSkill]) -> str:
        """功能：渲染激活技能的完整上下文（说明、正文与资源）。
        参数：
        - active_skills：当前激活技能列表。
        返回值：
        - str：拼接后的完整技能上下文文本。
        """
        sections = []
        for active in active_skills:
            if active.manifest is None:
                continue
            parts = [
                f"Skill: {active.manifest.name}",
                f"Description: {active.manifest.description}",
                "Instructions:",
                active.manifest.body,
            ]
            if active.manifest.references:
                parts.append(
                    "References: " + ", ".join(active.manifest.references)
                )
            if active.manifest.scripts:
                parts.append(
                    "Scripts: " + ", ".join(active.manifest.scripts)
                )
            sections.append("\n".join(parts))
        return "\n\n".join(sections)
