from __future__ import annotations

from dataclasses import dataclass
import os

from app.skills.executor import SkillExecutor
from app.skills.injector import SkillInjector
from app.skills.lifecycle import SkillLifecycleManager, SkillSessionState
from app.skills.models import SkillManifest, SkillMetadata, SkillResourceListing
from app.skills.parser import SkillParser
from app.skills.registry import SkillRegistry
from app.skills.retrieval import RuleBasedSkillRetrievalStrategy, SkillRetrievalStrategy
from app.skills.router import SkillRoutePlan, SkillRouter


@dataclass(frozen=True)
class SkillSystem:
    """功能：聚合技能注册、检索、注入与生命周期管理能力。
    参数：
    - 无。
    返回值：
    - 无。
    """
    registry: SkillRegistry
    retrieval_strategy: SkillRetrievalStrategy
    lifecycle: SkillLifecycleManager
    injector: SkillInjector
    executor: SkillExecutor
    router: SkillRouter

    def list_skill_metadata(self) -> tuple[SkillMetadata, ...]:
        """功能：列出系统已发现的全部技能元数据。
        参数：
        - 无。
        返回值：
        - tuple[SkillMetadata, ...]：技能元数据只读集合。
        """
        return self.registry.list_metadata()

    def get_skill_metadata(self, skill_name: str) -> SkillMetadata:
        """功能：按技能名获取单个技能元数据。
        参数：
        - skill_name：技能名称。
        返回值：
        - SkillMetadata：指定技能的元数据信息。
        """
        return self.registry.get_metadata(skill_name)

    def load_skill_manifest(self, skill_name: str) -> SkillManifest:
        """功能：加载并返回指定技能的完整清单内容。
        参数：
        - skill_name：技能名称。
        返回值：
        - SkillManifest：技能清单对象（含正文与资源声明）。
        """
        return self.registry.load_manifest(skill_name)

    def list_skill_resources(self, skill_name: str) -> SkillResourceListing:
        """功能：列出指定技能可用的引用、脚本和资产资源。
        参数：
        - skill_name：技能名称。
        返回值：
        - SkillResourceListing：技能资源清单对象。
        """
        return self.registry.list_resources(skill_name)

    def create_session(self) -> SkillSessionState:
        """功能：创建新的技能生命周期会话状态。
        参数：
        - 无。
        返回值：
        - SkillSessionState：初始化后的会话状态对象。
        """
        return self.lifecycle.create_session()

    def route_request(
        self,
        *,
        user_input: str,
        limit: int,
        tool_available,
        explicit_skill: str | None = None,
        prepared_input: str | None = None,
    ) -> SkillRoutePlan:
        """功能：根据当前请求生成技能路由计划。
        参数：
        - user_input：用户原始输入。
        - limit：最多返回的候选技能数量。
        - tool_available：用于判断依赖工具是否已注册的回调。
        - explicit_skill：显式 `/skill` 指定的技能名。
        - prepared_input：去掉显式命令后的用户输入。
        返回值：
        - SkillRoutePlan：运行时可直接消费的技能路由决策。
        """
        return self.router.route(
            user_input=user_input,
            metadata=self.list_skill_metadata(),
            limit=limit,
            tool_available=tool_available,
            explicit_skill=explicit_skill,
            prepared_input=prepared_input,
        )


def build_skill_system(*, project_directory: str, skill_root_name: str = "skills") -> SkillSystem:
    """功能：构建完整技能系统（注册、检索、生命周期、注入与执行）。
    参数：
    - project_directory：项目根目录路径。
    - skill_root_name：技能目录名，默认使用项目下的 `skills` 目录。
    返回值：
    - SkillSystem：可直接用于运行时的技能系统实例。
    """
    parser = SkillParser()
    registry = SkillRegistry(
        project_directory=project_directory,
        skill_root_name=skill_root_name,
        parser=parser,
    )
    retrieval_strategy = RuleBasedSkillRetrievalStrategy()
    lifecycle = SkillLifecycleManager()
    injector = SkillInjector()
    router = SkillRouter(retrieval_strategy=retrieval_strategy)
    executor = SkillExecutor(
        registry=registry,
        allow_script_execution=(os.getenv("SKILL_ENABLE_SCRIPTS") or "").lower() in {"1", "true", "yes", "on"},
        script_timeout_seconds=int(os.getenv("SKILL_SCRIPT_TIMEOUT_SECONDS", "20")),
        script_output_limit=int(os.getenv("SKILL_SCRIPT_OUTPUT_LIMIT", "4000")),
    )
    return SkillSystem(
        registry=registry,
        retrieval_strategy=retrieval_strategy,
        lifecycle=lifecycle,
        injector=injector,
        executor=executor,
        router=router,
    )
