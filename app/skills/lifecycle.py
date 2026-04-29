from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from app.skills.models import (
    ActiveSkill,
    SkillLifecycleStatus,
    SkillManifest,
    SkillMatch,
    SkillMetadata,
    SkillTransition,
)


@dataclass
class SkillSessionState:
    """功能：保存单会话内技能激活、完成、取消三类生命周期状态集合。
    参数：
    - 无。
    返回值：
    - 无。`current_request_id` 用于区分请求轮次并驱动状态归档。
    """
    active_skills: dict[str, ActiveSkill] = field(default_factory=dict)
    completed_skills: dict[str, ActiveSkill] = field(default_factory=dict)
    dismissed_skills: dict[str, ActiveSkill] = field(default_factory=dict)
    current_request_id: int = 0


class SkillLifecycleManager:
    """功能：负责技能状态迁移、可见性判断与生命周期事件记录。
    参数：
    - 无。
    返回值：
    - 无。状态迁移幂等，重复迁移到同一状态会被安全忽略。
    """
    def create_session(self) -> SkillSessionState:
        """功能：创建新的技能生命周期会话状态容器。
        参数：
        - 无。
        返回值：
        - SkillSessionState：初始化后的空会话状态对象。
        """
        return SkillSessionState()

    def begin_request(self, session: SkillSessionState) -> None:
        """功能：开始新请求并清理/归档上一个请求遗留的激活技能状态。
        参数：
        - session：当前技能会话状态对象。
        返回值：
        - 无。
        """
        session.current_request_id += 1
        for active in list(session.active_skills.values()):
            if active.status == SkillLifecycleStatus.COMPLETED:
                session.completed_skills[active.metadata.name] = active
                continue
            if active.status == SkillLifecycleStatus.DISMISSED:
                session.dismissed_skills[active.metadata.name] = active
                continue
            if active.status not in {SkillLifecycleStatus.COMPLETED, SkillLifecycleStatus.DISMISSED}:
                self.dismiss(
                    session,
                    active.metadata.name,
                    reason="superseded by new request",
                    trigger="runtime.begin_request",
                )
        session.active_skills.clear()

    def shortlist(
        self,
        session: SkillSessionState,
        *,
        metadata: SkillMetadata,
        match: Optional[SkillMatch],
        reason: str,
        trigger: str,
    ) -> ActiveSkill:
        """功能：将技能加入当前请求候选列表并更新状态。
        参数：
        - session：当前技能会话状态对象。
        - metadata：技能元数据列表或对象。
        - match：可选检索命中信息。
        - reason：进入候选列表的原因说明。
        - trigger：触发来源标识。
        返回值：
        - ActiveSkill：候选技能的运行态对象。
        """
        active = self._get_or_create(session, metadata=metadata, match=match, activated_by=trigger, activation_reason=reason)
        self._transition(active, SkillLifecycleStatus.SHORTLISTED, reason=reason, trigger=trigger)
        session.active_skills[metadata.name] = active
        return active

    def activate_summary(
        self,
        session: SkillSessionState,
        *,
        skill_name: str,
        reason: str,
        trigger: str,
    ) -> ActiveSkill:
        """功能：将技能提升为摘要激活状态。
        参数：
        - session：当前技能会话状态对象。
        - skill_name：技能名称。
        - reason：状态变更原因。
        - trigger：状态变更触发来源。
        返回值：
        - ActiveSkill：更新后的激活技能对象。
        """
        active = self.require_active(session, skill_name)
        self._transition(active, SkillLifecycleStatus.ACTIVATED_SUMMARY, reason=reason, trigger=trigger)
        return active

    def activate_full(
        self,
        session: SkillSessionState,
        *,
        skill_name: str,
        manifest: SkillManifest,
        reason: str,
        trigger: str,
    ) -> ActiveSkill:
        """功能：注入完整清单并将技能设为完整激活状态。
        参数：
        - session：当前技能会话状态对象。
        - skill_name：技能名称。
        - manifest：已加载的技能清单对象。
        - reason：状态变更原因。
        - trigger：状态变更触发来源。
        返回值：
        - ActiveSkill：更新后的激活技能对象。
        """
        active = self.require_active(session, skill_name)
        active.manifest = manifest
        self._transition(active, SkillLifecycleStatus.ACTIVATED_FULL, reason=reason, trigger=trigger)
        return active

    def mark_references_opened(
        self,
        session: SkillSessionState,
        *,
        skill_name: str,
        reference_path: str,
        reason: str,
        trigger: str,
    ) -> ActiveSkill:
        """功能：记录技能引用文件已被读取，并更新生命周期状态。
        参数：
        - session：当前技能会话状态对象。
        - skill_name：技能名称。
        - reference_path：技能引用文件的相对路径。
        - reason：状态变更原因。
        - trigger：状态变更触发来源。
        返回值：
        - ActiveSkill：更新后的激活技能对象。
        """
        active = self.require_active(session, skill_name)
        if reference_path not in active.opened_references:
            active.opened_references = tuple(list(active.opened_references) + [reference_path])
        self._transition(active, SkillLifecycleStatus.REFERENCES_OPENED, reason=reason, trigger=trigger)
        return active

    def mark_scripts_exposed(
        self,
        session: SkillSessionState,
        *,
        skill_name: str,
        script_path: str,
        reason: str,
        trigger: str,
    ) -> ActiveSkill:
        """功能：记录技能脚本已被暴露/执行，并更新生命周期状态。
        参数：
        - session：当前技能会话状态对象。
        - skill_name：技能名称。
        - script_path：技能脚本文件的相对路径。
        - reason：状态变更原因。
        - trigger：状态变更触发来源。
        返回值：
        - ActiveSkill：更新后的激活技能对象。
        """
        active = self.require_active(session, skill_name)
        if script_path not in active.exposed_scripts:
            active.exposed_scripts = tuple(list(active.exposed_scripts) + [script_path])
        self._transition(active, SkillLifecycleStatus.SCRIPTS_EXPOSED, reason=reason, trigger=trigger)
        return active

    def complete(self, session: SkillSessionState, *, skill_name: str, reason: str, trigger: str) -> ActiveSkill:
        """功能：将技能标记为完成，并归档到 completed 列表。
        参数：
        - session：当前技能会话状态对象。
        - skill_name：技能名称。
        - reason：状态变更原因。
        - trigger：状态变更触发来源。
        返回值：
        - ActiveSkill：更新后的激活技能对象。
        """
        active = self.require_active(session, skill_name)
        self._transition(active, SkillLifecycleStatus.COMPLETED, reason=reason, trigger=trigger)
        session.completed_skills[skill_name] = active
        return active

    def dismiss(self, session: SkillSessionState, skill_name: str, *, reason: str, trigger: str) -> ActiveSkill:
        """功能：将技能标记为取消，并归档到 dismissed 列表。
        参数：
        - session：当前技能会话状态对象。
        - skill_name：技能名称。
        - reason：状态变更原因。
        - trigger：状态变更触发来源。
        返回值：
        - ActiveSkill：更新后的激活技能对象。
        """
        active = self.require_active(session, skill_name)
        self._transition(active, SkillLifecycleStatus.DISMISSED, reason=reason, trigger=trigger)
        session.dismissed_skills[skill_name] = active
        return active

    def require_active(self, session: SkillSessionState, skill_name: str) -> ActiveSkill:
        """功能：读取指定技能的激活态对象，不存在时抛出异常。
        参数：
        - session：当前技能会话状态对象。
        - skill_name：技能名称。
        返回值：
        - ActiveSkill：会话中已激活的技能对象。
        """
        if skill_name not in session.active_skills:
            raise KeyError(f"skill 未激活：{skill_name}")
        return session.active_skills[skill_name]

    @staticmethod
    def visible_active_skills(session: SkillSessionState) -> tuple[ActiveSkill, ...]:
        """功能：筛选当前应展示给模型的激活技能，并按优先级排序。
        参数：
        - session：当前技能会话状态对象。
        返回值：
        - tuple[ActiveSkill, ...]：可见激活技能元组（按优先级降序、名称升序）。
        """
        visible_statuses = {
            SkillLifecycleStatus.ACTIVATED_SUMMARY,
            SkillLifecycleStatus.ACTIVATED_FULL,
            SkillLifecycleStatus.REFERENCES_OPENED,
            SkillLifecycleStatus.SCRIPTS_EXPOSED,
        }
        visible = [item for item in session.active_skills.values() if item.status in visible_statuses]
        visible.sort(key=lambda item: (-item.metadata.priority, item.metadata.name))
        return tuple(visible)

    @staticmethod
    def active_skill_names(session: SkillSessionState) -> tuple[str, ...]:
        """功能：返回当前可见激活技能名称列表。
        参数：
        - session：当前技能会话状态对象。
        返回值：
        - tuple[str, ...]：可见激活技能名称元组。
        """
        return tuple(skill.name for skill in SkillLifecycleManager.visible_active_skills(session))

    def _get_or_create(
        self,
        session: SkillSessionState,
        *,
        metadata: SkillMetadata,
        match: Optional[SkillMatch],
        activated_by: str,
        activation_reason: str,
    ) -> ActiveSkill:
        """功能：获取已存在激活技能，或创建新的运行态技能对象。
        参数：
        - session：当前技能会话状态对象。
        - metadata：技能元数据列表或对象。
        - match：可选检索命中信息。
        - activated_by：激活来源标识。
        - activation_reason：首次激活原因说明。
        返回值：
        - ActiveSkill：会话中的激活技能对象。
        """
        existing = session.active_skills.get(metadata.name)
        if existing is not None:
            return existing
        active = ActiveSkill(
            metadata=metadata,
            status=SkillLifecycleStatus.DISCOVERED,
            activated_by=activated_by,
            activation_reason=activation_reason,
            match=match,
        )
        active.transitions.append(
            SkillTransition(
                from_status=None,
                to_status=SkillLifecycleStatus.DISCOVERED,
                reason="skill discovered for this session request",
                trigger="lifecycle.create",
                timestamp=time.time(),
            )
        )
        return active

    @staticmethod
    def _transition(
        active: ActiveSkill,
        to_status: SkillLifecycleStatus,
        *,
        reason: str,
        trigger: str,
    ) -> None:
        """功能：执行技能状态迁移并记录迁移日志。
        参数：
        - active：要更新状态的激活技能对象。
        - to_status：目标生命周期状态。
        - reason：状态变更原因。
        - trigger：状态变更触发来源。
        返回值：
        - 无。
        """
        if active.status == to_status:
            return
        active.transitions.append(
            SkillTransition(
                from_status=active.status,
                to_status=to_status,
                reason=reason,
                trigger=trigger,
                timestamp=time.time(),
            )
        )
        active.status = to_status
