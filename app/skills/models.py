from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple


class SkillLifecycleStatus(str, Enum):
    """功能：定义技能在一次请求生命周期中的状态枚举。
    参数：
    - 无。
    返回值：
    - 无。
    """
    DISCOVERED = "discovered"
    SHORTLISTED = "shortlisted"
    ACTIVATED_SUMMARY = "activated-summary"
    ACTIVATED_FULL = "activated-full"
    REFERENCES_OPENED = "references-opened"
    SCRIPTS_EXPOSED = "scripts-exposed"
    COMPLETED = "completed"
    DISMISSED = "dismissed"


@dataclass(frozen=True)
class SkillResourceSummary:
    """功能：汇总技能资源统计信息。
    参数：
    - 无。
    返回值：
    - 无。
    """
    has_references: bool
    reference_count: int
    has_scripts: bool
    script_count: int
    has_assets: bool
    asset_count: int


@dataclass(frozen=True)
class SkillMetadata:
    """功能：描述技能的轻量元数据，用于检索与展示。
    参数：
    - 无。
    返回值：
    - 无。
    """
    name: str
    description: str
    skill_dir: Path
    manifest_path: Path
    tags: Tuple[str, ...] = ()
    keywords: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    priority: int = 0
    disable_model_invocation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    examples: Tuple[str, ...] = ()
    allowed_tools: Tuple[str, ...] = ()
    resources: SkillResourceSummary = field(
        default_factory=lambda: SkillResourceSummary(
            has_references=False,
            reference_count=0,
            has_scripts=False,
            script_count=0,
            has_assets=False,
            asset_count=0,
        )
    )

    def summary_line(self) -> str:
        """功能：生成技能摘要单行文本。
        参数：
        - 无。
        返回值：
        - str：包含技能描述及资源/限制标记的摘要行。
        """
        flags = []
        if self.disable_model_invocation:
            flags.append("explicit-only")
        if self.resources.has_references:
            flags.append(f"references={self.resources.reference_count}")
        if self.resources.has_scripts:
            flags.append(f"scripts={self.resources.script_count}")
        if self.resources.has_assets:
            flags.append(f"assets={self.resources.asset_count}")
        suffix = f" [{', '.join(flags)}]" if flags else ""
        return f"{self.name}: {self.description}{suffix}"


@dataclass(frozen=True)
class SkillManifest:
    """功能：描述技能完整清单，包含正文与资源清单。
    参数：
    - 无。
    返回值：
    - 无。
    """
    name: str
    description: str
    skill_dir: Path
    manifest_path: Path
    body: str
    tags: Tuple[str, ...] = ()
    keywords: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    priority: int = 0
    disable_model_invocation: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    examples: Tuple[str, ...] = ()
    allowed_tools: Tuple[str, ...] = ()
    references: Tuple[str, ...] = ()
    scripts: Tuple[str, ...] = ()
    assets: Tuple[str, ...] = ()

    @property
    def summary(self) -> SkillMetadata:
        """功能：将完整技能清单转换为轻量元数据对象。
        参数：
        - 无。
        返回值：
        - SkillMetadata：用于检索和排序的技能元数据。
        """
        return SkillMetadata(
            name=self.name,
            description=self.description,
            skill_dir=self.skill_dir,
            manifest_path=self.manifest_path,
            tags=self.tags,
            keywords=self.keywords,
            aliases=self.aliases,
            priority=self.priority,
            disable_model_invocation=self.disable_model_invocation,
            metadata=dict(self.metadata),
            examples=self.examples,
            allowed_tools=self.allowed_tools,
            resources=SkillResourceSummary(
                has_references=bool(self.references),
                reference_count=len(self.references),
                has_scripts=bool(self.scripts),
                script_count=len(self.scripts),
                has_assets=bool(self.assets),
                asset_count=len(self.assets),
            ),
        )


@dataclass(frozen=True)
class SkillMatch:
    """功能：保存一次技能检索命中结果及评分信息。
    参数：
    - 无。
    返回值：
    - 无。
    """
    skill_name: str
    score: float
    source: str
    match_reasons: Tuple[str, ...]
    matched_terms: Tuple[str, ...] = ()
    allow_auto_activation: bool = True
    decision: str = "candidate"


@dataclass(frozen=True)
class SkillTransition:
    """功能：记录技能状态迁移事件。
    参数：
    - 无。
    返回值：
    - 无。
    """
    from_status: Optional[SkillLifecycleStatus]
    to_status: SkillLifecycleStatus
    reason: str
    trigger: str
    timestamp: float


@dataclass
class ActiveSkill:
    """功能：保存当前请求中已激活技能的运行态信息。
    参数：
    - 无。
    返回值：
    - 无。
    """
    metadata: SkillMetadata
    status: SkillLifecycleStatus
    activated_by: str
    activation_reason: str
    match: Optional[SkillMatch] = None
    manifest: Optional[SkillManifest] = None
    opened_references: Tuple[str, ...] = ()
    exposed_scripts: Tuple[str, ...] = ()
    transitions: list[SkillTransition] = field(default_factory=list)

    @property
    def name(self) -> str:
        """功能：返回当前激活技能名称。
        参数：
        - 无。
        返回值：
        - str：技能名称。
        """
        return self.metadata.name


@dataclass(frozen=True)
class SkillResourceListing:
    """功能：按技能维度列出资源文件路径集合。
    参数：
    - 无。
    返回值：
    - 无。
    """
    skill: str
    references: Tuple[str, ...] = ()
    scripts: Tuple[str, ...] = ()
    assets: Tuple[str, ...] = ()
