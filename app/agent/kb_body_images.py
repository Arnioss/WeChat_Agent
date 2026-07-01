"""知识库正文图片标记解析与流式拆分工具。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List, Tuple

KB_IMAGE_MARKER_RE = re.compile(r"\[KB_IMAGE:([^\]]+)\]")

_MARKER_PREFIX = "[KB_IMAGE:"


def strip_kb_image_markers(text: str) -> str:
    """功能：移除文本中所有 `[KB_IMAGE:...]` 标记。
    参数：
    - text：待处理文本。
    返回值：
    - str：移除标记后的文本。
    """
    return KB_IMAGE_MARKER_RE.sub("", text or "")


def normalize_kb_image_path(path: str, project_root: Path | None = None) -> Path | None:
    """功能：将 `[KB_IMAGE:...]` 中的路径解析为可读本地文件。
    参数：
    - path：标记内的原始路径字符串。
    - project_root：可选项目根目录，用于相对路径与 `.rag_store` 候选解析。
    返回值：
    - Optional[Path]：首个存在的本地文件绝对路径；无法解析时返回 None。
    """
    raw = (path or "").strip().strip('"').replace("\r", "")
    if not raw:
        return None

    fixed = raw.replace("Agent.rag_store", "Agent/.rag_store")
    fixed = fixed.replace("WeChat_Agent.rag_store", "WeChat_Agent/.rag_store")

    candidates: List[Path] = []
    direct = Path(fixed)
    candidates.append(direct)
    if not direct.is_absolute() and project_root is not None:
        candidates.append(project_root / fixed)

    assets_idx = fixed.replace("\\", "/").find("/assets/")
    if assets_idx == -1:
        assets_idx = fixed.replace("\\", "/").find("assets/")
    if assets_idx != -1 and project_root is not None:
        rel = fixed.replace("\\", "/")[assets_idx:].lstrip("/")
        candidates.append(project_root / ".rag_store" / rel)
        if rel.startswith("assets/"):
            candidates.append(project_root / ".rag_store" / rel)

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            resolved = candidate.resolve()
        except Exception:
            continue
        if resolved.is_file():
            return resolved
    return None


def split_kb_image_markers(text: str) -> List[Tuple[str, str]]:
    """功能：将完整文本拆成 ('text'|'image', payload) 片段列表。
    参数：
    - text：含 `[KB_IMAGE:...]` 标记的完整文本。
    返回值：
    - List[Tuple[str, str]]：按顺序排列的文本或图片路径片段。
    """
    if not text:
        return []
    parts: List[Tuple[str, str]] = []
    last = 0
    for match in KB_IMAGE_MARKER_RE.finditer(text):
        if match.start() > last:
            parts.append(("text", text[last : match.start()]))
        path = (match.group(1) or "").strip()
        if path:
            parts.append(("image", path))
        last = match.end()
    if last < len(text):
        parts.append(("text", text[last:]))
    return parts


def _partial_marker_suffix(text: str) -> str:
    """功能：检测文本末尾是否为未闭合的 `[KB_IMAGE:` 标记前缀。
    参数：
    - text：流式缓冲区文本。
    返回值：
    - str：需保留在缓冲区中的不完整标记后缀；无则返回空串。
    """
    if not text:
        return ""
    max_len = min(len(text), len(_MARKER_PREFIX))
    for size in range(max_len, 0, -1):
        suffix = text[-size:]
        if _MARKER_PREFIX.startswith(suffix) and suffix != _MARKER_PREFIX:
            return suffix
    if text.endswith("[") or text.endswith("[K") or text.endswith("[KB"):
        return text[text.rfind("[") :]
    return ""


class KbImageStreamSplitter:
    """功能：流式解析 final_answer 中的 `[KB_IMAGE:...]` 标记并产出文本/图片片段。
    参数：
    - 无。
    返回值：
    - 无。内部缓冲未完成标记，避免跨 chunk 误切分。
    """

    def __init__(self) -> None:
        """功能：初始化空缓冲区。
        参数：
        - 无。
        返回值：
        - 无。
        """
        self._buffer = ""

    def feed(self, chunk: str) -> Iterable[Tuple[str, str]]:
        """功能：喂入流式文本分片并产出已完成的 text/image 片段。
        参数：
        - chunk：新增的流式文本分片。
        返回值：
        - Iterable[Tuple[str, str]]：可立即输出的 ('text'|'image', payload) 片段。
        """
        if not chunk:
            return
        self._buffer += chunk
        while self._buffer:
            match = KB_IMAGE_MARKER_RE.search(self._buffer)
            if match:
                before = self._buffer[: match.start()]
                path = (match.group(1) or "").strip()
                self._buffer = self._buffer[match.end() :]
                if before:
                    yield ("text", before)
                if path:
                    yield ("image", path)
                continue

            marker_start = self._buffer.rfind(_MARKER_PREFIX)
            if marker_start != -1:
                if marker_start > 0:
                    yield ("text", self._buffer[:marker_start])
                    self._buffer = self._buffer[marker_start:]
                break

            partial = _partial_marker_suffix(self._buffer)
            safe_len = len(self._buffer) - len(partial)
            if safe_len <= 0:
                break
            yield ("text", self._buffer[:safe_len])
            self._buffer = self._buffer[safe_len:]

    def flush(self) -> Iterable[Tuple[str, str]]:
        """功能：冲刷缓冲区，输出剩余文本与未闭合标记中的完整片段。
        参数：
        - 无。
        返回值：
        - Iterable[Tuple[str, str]]：缓冲区中剩余的 text/image 片段。
        """
        if not self._buffer:
            return
        for kind, payload in split_kb_image_markers(self._buffer):
            if kind == "text" and payload:
                yield ("text", payload)
            elif kind == "image" and payload:
                yield ("image", payload)
        self._buffer = ""


def list_kb_image_paths(text: str) -> List[str]:
    """功能：提取文本中所有 `[KB_IMAGE:...]` 标记内的路径。
    参数：
    - text：待扫描文本。
    返回值：
    - List[str]：去空白后的路径字符串列表。
    """
    return [(m or "").strip() for m in KB_IMAGE_MARKER_RE.findall(text or "") if (m or "").strip()]


def build_kb_image_inline_hint(observation_text: str) -> str:
    """功能：当 RAG observation 含图示标记时，生成约束主模型内联插入标记的提示。
    参数：
    - observation_text：工具返回的 observation 文本。
    返回值：
    - str：插入要求说明与须原样保留的标记行；无图示时返回空串。
    """
    paths = list_kb_image_paths(observation_text)
    if not paths:
        return ""
    lines = [
        f"observation 中含 {len(paths)} 个知识库图示，标记格式为 [KB_IMAGE:绝对路径]。",
        "撰写最终回答时必须：",
        "1) 讲到对应步骤/图示说明后，**紧接下一行**插入与下列完全相同的标记（勿改路径）；",
        "2) 禁止改为「见图」等文字；禁止把标记全部挪到文末集中罗列。",
        "须分散插入的标记行：",
    ]
    for path in paths:
        lines.append(f"[KB_IMAGE:{path}]")
    return "\n".join(lines)
