from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Optional

from app.skills.models import SkillMatch, SkillMetadata
from app.skills.retrieval import SkillRetrievalStrategy


class SkillActivationDecision(str, Enum):
    """功能：技能路由器对当前请求的激活决策枚举。
    参数：
    - 无。
    返回值：
    - 无。
    """

    NONE = "none"
    SUMMARY_ONLY = "summary-only"
    FULL = "full"
    EXPLICIT_MISSING = "explicit-missing"
    TOOLS_UNAVAILABLE = "tools-unavailable"


@dataclass(frozen=True)
class SkillRouteCandidate:
    """功能：入选提示词暴露范围的技能候选。
    参数：
    - metadata：技能元数据。
    - match：检索匹配结果。
    返回值：
    - 无。
    """

    metadata: SkillMetadata
    match: SkillMatch


@dataclass(frozen=True)
class SkillRoutePlan:
    """功能：技能路由器输出，供运行时消费。
    参数：
    - prepared_input：预处理后的用户输入。
    - decision：激活决策。
    - summary_candidates：摘要候选列表。
    - full_activation：完整激活的候选（若有）。
    - recommended_tools：推荐工具名元组。
    - tool_allowlist：工具白名单元组。
    - diagnostics：诊断信息元组。
    - explicit_skill：显式指定的技能名（若有）。
    返回值：
    - 无。
    """

    prepared_input: str
    decision: SkillActivationDecision
    summary_candidates: tuple[SkillRouteCandidate, ...] = ()
    full_activation: Optional[SkillRouteCandidate] = None
    recommended_tools: Optional[tuple[str, ...]] = None
    tool_allowlist: Optional[tuple[str, ...]] = None
    diagnostics: tuple[str, ...] = ()
    explicit_skill: Optional[str] = None

    @property
    def has_candidates(self) -> bool:
        """功能：判断是否存在摘要候选技能。
        参数：
        - 无。
        返回值：
        - bool：summary_candidates 非空时为 True。
        """
        return bool(self.summary_candidates)


class SkillRouter:
    """功能：将用户请求路由至技能摘要或完整技能激活。
    参数：
    - 无（通过 __init__ 注入检索策略与分数阈值）。
    返回值：
    - 无。
    """

    def __init__(
        self,
        *,
        retrieval_strategy: SkillRetrievalStrategy,
        min_summary_score: float = 5.0,
        full_activation_score: float = 12.0,
        full_activation_margin: float = 4.0,
    ):
        """功能：初始化技能路由器及激活阈值配置。
        参数：
        - retrieval_strategy：技能检索策略。
        - min_summary_score：进入摘要候选的最低分数。
        - full_activation_score：触发完整激活的最低分数。
        - full_activation_margin：完整激活所需的领先分差。
        返回值：
        - 无。
        """
        self.retrieval_strategy = retrieval_strategy
        self.min_summary_score = min_summary_score
        self.full_activation_score = full_activation_score
        self.full_activation_margin = full_activation_margin

    def route(
        self,
        *,
        user_input: str,
        metadata: Iterable[SkillMetadata],
        limit: int,
        tool_available: Callable[[str], bool],
        explicit_skill: Optional[str] = None,
        prepared_input: Optional[str] = None,
    ) -> SkillRoutePlan:
        """功能：根据用户输入与技能元数据生成路由计划。
        参数：
        - user_input：原始用户输入。
        - metadata：可路由的技能元数据集合。
        - limit：检索返回的最大候选数。
        - tool_available：判断工具是否可用的回调。
        - explicit_skill：用户显式指定的技能名（可选）。
        - prepared_input：预处理后的输入；未传则使用 user_input。
        返回值：
        - SkillRoutePlan：含决策、候选与诊断信息的路由计划。
        """
        prepared = user_input if prepared_input is None else prepared_input
        items_by_name = {item.name: item for item in metadata}

        if explicit_skill:
            if explicit_skill not in items_by_name:
                return SkillRoutePlan(
                    prepared_input=prepared,
                    decision=SkillActivationDecision.EXPLICIT_MISSING,
                    diagnostics=(f"explicit skill not found: {explicit_skill}",),
                    explicit_skill=explicit_skill,
                )
            item = items_by_name[explicit_skill]
            unavailable = self._unavailable_tools(item, tool_available)
            if unavailable:
                return SkillRoutePlan(
                    prepared_input=(
                        f"用户显式指定的 skill `{explicit_skill}` 依赖当前未启用或未注册的工具，"
                        f"请不要使用该 skill，直接处理原始请求：{prepared}"
                    ),
                    decision=SkillActivationDecision.TOOLS_UNAVAILABLE,
                    diagnostics=(f"explicit skill unavailable tools: {', '.join(unavailable)}",),
                    explicit_skill=explicit_skill,
                )
            match = SkillMatch(
                skill_name=explicit_skill,
                score=100.0,
                source="explicit",
                match_reasons=(f"explicit invocation via /skill {explicit_skill}",),
                allow_auto_activation=False,
                decision="explicit",
            )
            candidate = SkillRouteCandidate(metadata=item, match=match)
            return SkillRoutePlan(
                prepared_input=prepared,
                decision=SkillActivationDecision.FULL,
                summary_candidates=(candidate,),
                full_activation=candidate,
                recommended_tools=self._recommended_tools(item),
                tool_allowlist=self._recommended_tools(item),
                diagnostics=(f"explicit full activation: {explicit_skill}",),
                explicit_skill=explicit_skill,
            )

        matches = self.retrieval_strategy.retrieve(
            prepared,
            items_by_name.values(),
            limit=limit,
        )
        candidates: list[SkillRouteCandidate] = []
        diagnostics: list[str] = []
        for match in matches:
            item = items_by_name[match.skill_name]
            if match.score < self.min_summary_score:
                diagnostics.append(f"drop {match.skill_name}: score {match.score:.2f} below {self.min_summary_score:.2f}")
                continue
            unavailable = self._unavailable_tools(item, tool_available)
            if unavailable:
                diagnostics.append(f"drop {match.skill_name}: unavailable tools {', '.join(unavailable)}")
                continue
            candidates.append(SkillRouteCandidate(metadata=item, match=match))

        if not candidates:
            return SkillRoutePlan(
                prepared_input=prepared,
                decision=SkillActivationDecision.NONE,
                diagnostics=tuple(diagnostics or ("no skill candidates",)),
            )

        top = candidates[0]
        second_score = candidates[1].match.score if len(candidates) > 1 else 0.0
        margin = top.match.score - second_score
        diagnostics.append(
            f"top={top.metadata.name} score={top.match.score:.2f} second={second_score:.2f} margin={margin:.2f}"
        )
        if top.match.score >= self.full_activation_score and margin >= self.full_activation_margin:
            return SkillRoutePlan(
                prepared_input=prepared,
                decision=SkillActivationDecision.FULL,
                summary_candidates=(top,),
                full_activation=top,
                recommended_tools=self._recommended_tools(top.metadata),
                tool_allowlist=self._recommended_tools(top.metadata),
                diagnostics=tuple(diagnostics),
            )

        diagnostics.append("using summary-only candidates due to low confidence or close scores")
        return SkillRoutePlan(
            prepared_input=prepared,
            decision=SkillActivationDecision.SUMMARY_ONLY,
            summary_candidates=tuple(candidates),
            diagnostics=tuple(diagnostics),
        )

    @staticmethod
    def _unavailable_tools(metadata: SkillMetadata, tool_available: Callable[[str], bool]) -> tuple[str, ...]:
        """功能：列出技能依赖但当前不可用的工具名。
        参数：
        - metadata：技能元数据。
        - tool_available：判断工具是否可用的回调。
        返回值：
        - tuple[str, ...]：不可用工具名元组。
        """
        return tuple(tool_name for tool_name in metadata.allowed_tools if not tool_available(tool_name))

    @staticmethod
    def _recommended_tools(metadata: SkillMetadata) -> tuple[str, ...]:
        """功能：合并技能允许工具与基础 skill 工具，生成推荐工具列表。
        参数：
        - metadata：技能元数据。
        返回值：
        - tuple[str, ...]：去重后的推荐工具名元组。
        """
        base_tools = (
            "get_current_date",
            "load_skill_instructions",
            "list_skill_resources",
            "load_skill_reference",
            "run_skill_script",
        )
        ordered: list[str] = []
        for tool_name in (*metadata.allowed_tools, *base_tools):
            if tool_name not in ordered:
                ordered.append(tool_name)
        return tuple(ordered)
