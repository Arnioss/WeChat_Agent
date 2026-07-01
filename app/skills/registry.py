from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from app.skills.models import SkillManifest, SkillMetadata, SkillResourceListing
from app.skills.parser import SkillParser


@dataclass(frozen=True)
class SkillDiscoveryResult:
    """功能：保存技能发现阶段产出的元数据与根目录信息。
    参数：
    - 无。
    返回值：
    - 无。
    """
    metadata: tuple[SkillMetadata, ...]
    skill_root: Path


class SkillRegistry:
    """功能：负责技能发现、清单加载与资源查询。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(
        self,
        *,
        project_directory: str,
        skill_root_name: str = "skills",
        parser: Optional[SkillParser] = None,
    ):
        """功能：确定技能根目录、注入解析器并执行一次技能发现索引构建。
        参数：
        - project_directory：项目根目录路径。
        - skill_root_name：技能目录名称，默认值为 `skills`。
        - parser：可选的技能解析器实例，未提供时使用默认 `SkillParser`。
        返回值：
        - 无。发现结果会缓存到实例内存，后续查询基于该快照执行。
        """
        self.project_directory = Path(project_directory).resolve()
        self.skill_root = self.project_directory / skill_root_name
        self.parser = parser or SkillParser()
        self._metadata_by_name: dict[str, SkillMetadata] = {}
        self._manifest_cache: dict[str, SkillManifest] = {}
        self._discovery_result = self._discover()

    def _discover(self) -> SkillDiscoveryResult:
        """功能：扫描技能目录并构建技能元数据索引。
        参数：
        - 无。
        返回值：
        - SkillDiscoveryResult：包含已发现技能列表和技能根路径。
        异常：
        - ValueError：技能根路径不是目录或发现重复技能名时抛出。
        """
        discovered: list[SkillMetadata] = []
        if not self.skill_root.exists():
            return SkillDiscoveryResult(metadata=(), skill_root=self.skill_root)
        if not self.skill_root.is_dir():
            raise ValueError(f"skills 根路径不是目录：{self.skill_root}")

        for entry in sorted(self.skill_root.iterdir(), key=lambda path: path.name.lower()):
            if not entry.is_dir():
                continue
            manifest_path = entry / "SKILL.md"
            if not manifest_path.is_file():
                continue
            metadata = self.parser.parse_metadata(entry)
            if metadata.name in self._metadata_by_name:
                raise ValueError(f"发现重复的 skill name：{metadata.name}")
            self._metadata_by_name[metadata.name] = metadata
            discovered.append(metadata)
        return SkillDiscoveryResult(metadata=tuple(discovered), skill_root=self.skill_root)

    def list_metadata(self) -> tuple[SkillMetadata, ...]:
        """功能：返回全部已发现技能元数据。
        参数：
        - 无。
        返回值：
        - tuple[SkillMetadata, ...]：技能元数据元组。
        """
        return self._discovery_result.metadata

    def iter_metadata(self) -> Iterable[SkillMetadata]:
        """功能：按迭代器形式遍历已发现技能元数据。
        参数：
        - 无。
        返回值：
        - Iterable[SkillMetadata]：技能元数据迭代器。
        """
        return iter(self._discovery_result.metadata)

    def get_metadata(self, skill_name: str) -> SkillMetadata:
        """功能：按技能名称获取元数据。
        参数：
        - skill_name：技能名称。
        返回值：
        - SkillMetadata：目标技能的元数据对象。
        异常：
        - KeyError：技能不存在时抛出。
        """
        if skill_name not in self._metadata_by_name:
            raise KeyError(f"skill 不存在：{skill_name}")
        return self._metadata_by_name[skill_name]

    def has_skill(self, skill_name: str) -> bool:
        """功能：判断技能是否已注册。
        参数：
        - skill_name：技能名称。
        返回值：
        - bool：存在返回 True，否则返回 False。
        """
        return skill_name in self._metadata_by_name

    def load_manifest(self, skill_name: str) -> SkillManifest:
        """功能：加载指定技能清单，并使用内存缓存加速重复访问。
        参数：
        - skill_name：技能名称。
        返回值：
        - SkillManifest：技能完整清单对象。
        """
        if skill_name in self._manifest_cache:
            return self._manifest_cache[skill_name]
        metadata = self.get_metadata(skill_name)
        manifest = self.parser.load_manifest(metadata.skill_dir)
        self._manifest_cache[skill_name] = manifest
        return manifest

    def list_resources(self, skill_name: str) -> SkillResourceListing:
        """功能：列出指定技能暴露的资源文件。
        参数：
        - skill_name：技能名称。
        返回值：
        - SkillResourceListing：包含 references、scripts、assets 的资源信息。
        """
        metadata = self.get_metadata(skill_name)
        return self.parser.list_resources(metadata.skill_dir, skill_name=metadata.name)

    def clear_manifest_cache(self) -> None:
        """功能：清空技能清单缓存。
        参数：
        - 无。
        返回值：
        - 无。
        """
        self._manifest_cache.clear()

    @property
    def discovery_result(self) -> SkillDiscoveryResult:
        """功能：返回最近一次技能发现结果快照。
        参数：
        - 无。
        返回值：
        - SkillDiscoveryResult：技能发现结果对象。
        """
        return self._discovery_result
