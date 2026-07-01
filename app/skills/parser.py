from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.skills.models import SkillManifest, SkillMetadata, SkillResourceListing, SkillResourceSummary


@dataclass(frozen=True)
class ParsedSkillDocument:
    """功能：表示 `SKILL.md` 解析后的 frontmatter 与正文内容。
    参数：
    - 无。
    返回值：
    - 无。该对象不可变，确保后续元数据构建过程不会意外改写原始解析结果。
    """
    frontmatter: dict[str, Any]
    body: str


class SkillParser:
    """功能：解析技能目录结构与 `SKILL.md`，产出元数据、清单和资源列表。
    参数：
    - 无。
    返回值：
    - 无。支持 PyYAML 与内置回退解析，兼容依赖缺失场景。
    """
    def parse_metadata(self, skill_dir: Path) -> SkillMetadata:
        """功能：解析技能目录并构建轻量元数据对象。
        参数：
        - skill_dir：技能目录路径。
        返回值：
        - SkillMetadata：用于检索和展示的技能元数据。
        """
        manifest_path = self._resolve_manifest_path(skill_dir)
        frontmatter = self._parse_frontmatter(self._read_frontmatter_text(manifest_path))
        resources = self._build_resource_summary(skill_dir)
        name = self._require_text(frontmatter, "name")
        description = self._require_text(frontmatter, "description")
        if skill_dir.name != name:
            raise ValueError(
                f"技能目录名与 frontmatter.name 不一致：dir={skill_dir.name}, name={name}"
            )
        return SkillMetadata(
            name=name,
            description=description,
            skill_dir=skill_dir,
            manifest_path=manifest_path,
            tags=self._tuple_of_text(frontmatter.get("tags")),
            keywords=self._tuple_of_text(frontmatter.get("keywords")),
            aliases=self._tuple_of_text(frontmatter.get("aliases")),
            priority=self._coerce_int(frontmatter.get("priority"), default=0),
            disable_model_invocation=bool(frontmatter.get("disable-model-invocation", False)),
            metadata=self._coerce_mapping(frontmatter.get("metadata")),
            examples=self._tuple_of_text(frontmatter.get("examples")),
            allowed_tools=self._tuple_of_text(frontmatter.get("allowed_tools")),
            resources=resources,
        )

    def load_manifest(self, skill_dir: Path) -> SkillManifest:
        """功能：加载技能清单并构建完整 `SkillManifest`。
        参数：
        - skill_dir：技能目录路径。
        返回值：
        - SkillManifest：包含正文、标签和资源列表的完整清单对象。
        """
        manifest_path = self._resolve_manifest_path(skill_dir)
        parsed = self._parse_skill_document(manifest_path)
        frontmatter = parsed.frontmatter
        name = self._require_text(frontmatter, "name")
        description = self._require_text(frontmatter, "description")
        references = self._list_relative_files(skill_dir / "references")
        scripts = self._list_relative_files(skill_dir / "scripts")
        assets = self._list_relative_files(skill_dir / "assets")
        return SkillManifest(
            name=name,
            description=description,
            skill_dir=skill_dir,
            manifest_path=manifest_path,
            body=parsed.body,
            tags=self._tuple_of_text(frontmatter.get("tags")),
            keywords=self._tuple_of_text(frontmatter.get("keywords")),
            aliases=self._tuple_of_text(frontmatter.get("aliases")),
            priority=self._coerce_int(frontmatter.get("priority"), default=0),
            disable_model_invocation=bool(frontmatter.get("disable-model-invocation", False)),
            metadata=self._coerce_mapping(frontmatter.get("metadata")),
            examples=self._tuple_of_text(frontmatter.get("examples")),
            allowed_tools=self._tuple_of_text(frontmatter.get("allowed_tools")),
            references=references,
            scripts=scripts,
            assets=assets,
        )

    def list_resources(self, skill_dir: Path, *, skill_name: str) -> SkillResourceListing:
        """功能：列出技能目录中的 references/scripts/assets 资源。
        参数：
        - skill_dir：技能目录路径。
        - skill_name：技能名称。
        返回值：
        - SkillResourceListing：技能资源清单对象。
        """
        return SkillResourceListing(
            skill=skill_name,
            references=self._list_relative_files(skill_dir / "references"),
            scripts=self._list_relative_files(skill_dir / "scripts"),
            assets=self._list_relative_files(skill_dir / "assets"),
        )

    @staticmethod
    def _resolve_manifest_path(skill_dir: Path) -> Path:
        """功能：定位技能清单文件 `SKILL.md` 路径。
        参数：
        - skill_dir：技能目录路径。
        返回值：
        - Path：清单文件路径。
        """
        manifest_path = skill_dir / "SKILL.md"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"未找到技能文件：{manifest_path}")
        return manifest_path

    def _parse_skill_document(self, manifest_path: Path) -> ParsedSkillDocument:
        """功能：读取并解析 SKILL.md，拆分 frontmatter 与正文后组装解析结果对象。
        参数：
        - manifest_path：技能清单文件路径。
        返回值：
        - ParsedSkillDocument：包含 frontmatter 字典与正文文本的解析结果。
        """
        text = manifest_path.read_text(encoding="utf-8")
        frontmatter_text, body = self._split_frontmatter(text)
        return ParsedSkillDocument(
            frontmatter=self._parse_frontmatter(frontmatter_text),
            body=body.strip(),
        )

    @staticmethod
    def _read_frontmatter_text(manifest_path: Path) -> str:
        """功能：从 SKILL.md 中提取 YAML frontmatter 原文（不包含分隔符行）。
        参数：
        - manifest_path：技能清单文件路径。
        返回值：
        - str：frontmatter 文本内容。
        """
        with manifest_path.open("r", encoding="utf-8") as handle:
            first_line = handle.readline()
            if first_line.strip() != "---":
                raise ValueError("SKILL.md 缺少 YAML frontmatter")
            frontmatter_lines = []
            for line in handle:
                if line.strip() == "---":
                    return "".join(frontmatter_lines)
                frontmatter_lines.append(line)
        raise ValueError("SKILL.md frontmatter 缺少结束分隔符")

    @staticmethod
    def _split_frontmatter(text: str) -> tuple[str, str]:
        """功能：按 `---` 分隔符把 SKILL.md 拆分为 frontmatter 与正文两部分。
        参数：
        - text：待处理文本内容。
        返回值：
        - tuple[str, str]：第一个元素为 frontmatter 文本，第二个元素为正文文本。
        """
        if not text.startswith("---"):
            raise ValueError("SKILL.md 缺少 YAML frontmatter")
        lines = text.splitlines()
        if not lines or lines[0].strip() != "---":
            raise ValueError("SKILL.md frontmatter 起始分隔符无效")
        end_idx = None
        for idx in range(1, len(lines)):
            if lines[idx].strip() == "---":
                end_idx = idx
                break
        if end_idx is None:
            raise ValueError("SKILL.md frontmatter 缺少结束分隔符")
        frontmatter_text = "\n".join(lines[1:end_idx])
        body = "\n".join(lines[end_idx + 1 :])
        return frontmatter_text, body

    def _parse_frontmatter(self, frontmatter_text: str) -> dict[str, Any]:
        """功能：优先用 PyYAML 解析 frontmatter；失败时回退到内置解析器。
        参数：
        - frontmatter_text：frontmatter 原始文本。
        返回值：
        - dict[str, Any]：解析后的 frontmatter 键值映射。
        """
        try:
            import yaml  # type: ignore

            loaded = yaml.safe_load(frontmatter_text) or {}
            if not isinstance(loaded, dict):
                raise ValueError("frontmatter 顶层必须是对象")
            return dict(loaded)
        except Exception:
            return self._parse_frontmatter_fallback(frontmatter_text)

    def _parse_frontmatter_fallback(self, frontmatter_text: str) -> dict[str, Any]:
        """功能：在缺少 PyYAML 或解析失败时，使用内置规则解析 frontmatter 文本。
        参数：
        - frontmatter_text：frontmatter 原始文本。
        返回值：
        - dict[str, Any]：按简化 YAML 规则解析出的键值映射。
        """
        lines = frontmatter_text.splitlines()
        result: dict[str, Any] = {}
        i = 0
        while i < len(lines):
            raw = lines[i]
            stripped = raw.strip()
            i += 1
            if not stripped or stripped.startswith("#"):
                continue
            if ":" not in stripped:
                raise ValueError(f"无法解析 frontmatter 行：{raw}")
            key, remainder = stripped.split(":", 1)
            key = key.strip()
            remainder = remainder.strip()
            if remainder:
                result[key] = self._parse_scalar(remainder)
                continue

            nested_lines = []
            while i < len(lines):
                peek = lines[i]
                if not peek.strip():
                    nested_lines.append(peek)
                    i += 1
                    continue
                indent = len(peek) - len(peek.lstrip(" "))
                if indent <= 0:
                    break
                nested_lines.append(peek)
                i += 1
            result[key] = self._parse_nested_block(nested_lines)
        return result

    def _parse_nested_block(self, lines: list[str]) -> Any:
        """功能：解析 frontmatter 中缩进嵌套块（列表或字典）。
        参数：
        - lines：嵌套块原始行列表。
        返回值：
        - Any：解析后的 Python 结构（list/dict）。
        """
        effective = [line for line in lines if line.strip()]
        if not effective:
            return {}
        min_indent = min(len(line) - len(line.lstrip(" ")) for line in effective)
        normalized = [line[min_indent:] for line in effective]
        if all(line.lstrip().startswith("- ") for line in normalized):
            return [self._parse_scalar(line.lstrip()[2:].strip()) for line in normalized]
        nested: dict[str, Any] = {}
        for line in normalized:
            stripped = line.strip()
            if ":" not in stripped:
                raise ValueError(f"无法解析嵌套 frontmatter 行：{line}")
            key, remainder = stripped.split(":", 1)
            nested[key.strip()] = self._parse_scalar(remainder.strip())
        return nested

    def _parse_scalar(self, value: str) -> Any:
        """功能：把 frontmatter 标量文本转换为 Python 值（布尔、空、列表、数字或字符串）。
        参数：
        - value：待转换的输入值。
        返回值：
        - Any：转换后的 Python 值。
        """
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered in {"null", "none"}:
            return None
        if value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            if not inner:
                return []
            return [self._parse_scalar(part.strip()) for part in inner.split(",")]
        try:
            return ast.literal_eval(value)
        except Exception:
            return value.strip().strip('"').strip("'")

    @staticmethod
    def _build_resource_summary(skill_dir: Path) -> SkillResourceSummary:
        """功能：统计技能目录资源数量并构建摘要。
        参数：
        - skill_dir：技能目录路径。
        返回值：
        - SkillResourceSummary：引用、脚本、资产的存在性与数量统计。
        """
        references = SkillParser._list_relative_files(skill_dir / "references")
        scripts = SkillParser._list_relative_files(skill_dir / "scripts")
        assets = SkillParser._list_relative_files(skill_dir / "assets")
        return SkillResourceSummary(
            has_references=bool(references),
            reference_count=len(references),
            has_scripts=bool(scripts),
            script_count=len(scripts),
            has_assets=bool(assets),
            asset_count=len(assets),
        )

    @staticmethod
    def _list_relative_files(base_dir: Path) -> tuple[str, ...]:
        """功能：递归收集目录下全部文件并返回相对路径列表。
        参数：
        - base_dir：基础目录路径。
        返回值：
        - tuple[str, ...]：按字典序排序的相对路径元组；目录不存在时返回空元组。
        """
        if not base_dir.is_dir():
            return ()
        files = [
            str(path.relative_to(base_dir)).replace("\\", "/")
            for path in sorted(base_dir.rglob("*"))
            if path.is_file()
        ]
        return tuple(files)

    @staticmethod
    def _tuple_of_text(value: Any) -> tuple[str, ...]:
        """功能：把单值或序列值归一化为去空白的字符串元组。
        参数：
        - value：待转换的输入值。
        返回值：
        - tuple[str, ...]：清洗后的字符串元组。
        """
        if value is None:
            return ()
        if isinstance(value, str):
            return (value,) if value.strip() else ()
        if isinstance(value, (list, tuple)):
            return tuple(str(item).strip() for item in value if str(item).strip())
        return (str(value).strip(),) if str(value).strip() else ()

    @staticmethod
    def _coerce_mapping(value: Any) -> dict[str, Any]:
        """功能：将输入值安全转换为字典类型，非字典输入返回空字典。
        参数：
        - value：待转换的输入值。
        返回值：
        - dict[str, Any]：转换结果字典。
        """
        if isinstance(value, dict):
            return dict(value)
        return {}

    @staticmethod
    def _coerce_int(value: Any, *, default: int) -> int:
        """功能：将输入值转换为整数，转换失败时回退为默认值。
        参数：
        - value：待转换的输入值。
        - default：默认值。
        返回值：
        - int：转换后的整数结果或默认值。
        """
        if value is None:
            return default
        try:
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _require_text(frontmatter: dict[str, Any], key: str) -> str:
        """功能：读取 frontmatter 必填文本字段。
        参数：
        - frontmatter：frontmatter 字典对象。
        - key：必填字段名。
        返回值：
        - str：字段对应的非空文本。
        """
        value = str(frontmatter.get(key) or "").strip()
        if not value:
            raise ValueError(f"frontmatter 缺少必填字段：{key}")
        return value
