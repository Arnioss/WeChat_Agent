"""从检索文档 metadata 中解析并解析图片资源路径。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rag.ingestion.asset_store import AssetStore


def parse_image_lists(doc: dict) -> Tuple[List[str], List[str]]:
    """从文档 metadata 解析图片资源路径与说明列表。

    功能：
        读取 doc 中的 ``image_assets`` 与 ``image_captions`` 字段（JSON 字符串或列表），
        解析为字符串列表；解析失败或类型不符时返回空列表。

    参数：
        doc: 检索结果文档字典，可含 ``image_assets``、``image_captions`` 键。

    返回值：
        (assets, captions) 元组，均为非空字符串列表；assets 与 captions 按索引对应。

    异常：
        无；JSON 解析或类型错误时静默返回 ``([], [])``。
    """
    assets_raw = doc.get("image_assets") or "[]"
    captions_raw = doc.get("image_captions") or "[]"
    try:
        assets = json.loads(assets_raw) if isinstance(assets_raw, str) else list(assets_raw or [])
        captions = json.loads(captions_raw) if isinstance(captions_raw, str) else list(captions_raw or [])
    except Exception:
        return [], []
    if not isinstance(assets, list):
        assets = []
    if not isinstance(captions, list):
        captions = []
    return [str(a) for a in assets if a], [str(c) for c in captions if c]


def resolve_asset_path(persist_dir: Path, relative_or_abs: str) -> Path | None:
    """将相对或绝对路径解析为本地存在的图片文件路径。

    功能：
        依次尝试：直接文件路径、AssetStore 解析、``.rag_store/`` 后缀、
        ``assets/`` 前缀等多种规则，定位 persist_dir 下的实际图片文件。

    参数：
        persist_dir: RAG 持久化根目录（``.rag_store``）。
        relative_or_abs: 资源相对路径、绝对路径或含 ``.rag_store/`` 标记的路径。

    返回值：
        解析成功且文件存在时返回 ``Path``；否则返回 ``None``。

    异常：
        无；AssetStore 解析失败时回退到其他候选路径。
    """
    raw = (relative_or_abs or "").strip()
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_file():
        return candidate.resolve()
    store = AssetStore(persist_dir)
    try:
        resolved = store.resolve_path(persist_dir, raw.replace("\\", "/"))
    except Exception:
        resolved = (persist_dir / raw).resolve()
    if resolved.is_file():
        return resolved
    normalized = raw.replace("\\", "/")
    marker = ".rag_store/"
    if marker in normalized:
        candidate = (persist_dir / normalized.split(marker, 1)[1]).resolve()
        if candidate.is_file():
            return candidate
    if normalized.startswith("assets/"):
        candidate = (persist_dir / normalized).resolve()
        if candidate.is_file():
            return candidate
    return None


def collect_doc_images(
    doc: dict,
    *,
    persist_dir: Path,
    seen: set[str],
) -> List[Dict[str, str]]:
    """收集单条文档中的去重图片信息。

    功能：
        解析 doc 的 image_assets/image_captions，解析本地路径并按 ``seen`` 去重，
        组装含 path、caption、source 的字典列表。

    参数：
        doc: 检索结果文档字典。
        persist_dir: RAG 持久化根目录。
        seen: 已处理过的绝对路径集合（会被原地更新）。

    返回值：
        图片信息字典列表，每项含 ``path``、``caption``、``source`` 键。

    异常：
        无；无法解析或重复的路径会被跳过。
    """
    assets, captions = parse_image_lists(doc)
    source = str(doc.get("source") or "")
    items: List[Dict[str, str]] = []
    for idx, rel in enumerate(assets):
        resolved = resolve_asset_path(persist_dir, rel)
        if resolved is None:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        cap = captions[idx] if idx < len(captions) else ""
        items.append(
            {
                "path": key,
                "caption": cap,
                "source": source,
            }
        )
    return items


def iter_retrieved_images(
    docs: Iterable[dict],
    *,
    persist_dir: Path,
) -> Iterable[Dict[str, str]]:
    """迭代多条检索文档中的去重图片。

    功能：
        遍历 docs，对每条调用 ``collect_doc_images``，跨文档按绝对路径去重后逐个 yield。

    参数：
        docs: 检索结果文档字典的可迭代对象。
        persist_dir: RAG 持久化根目录。

    返回值：
        生成器，逐项 yield 含 ``path``、``caption``、``source`` 的字典。

    异常：
        无。
    """
    seen: set[str] = set()
    for doc in docs:
        for item in collect_doc_images(doc, persist_dir=persist_dir, seen=seen):
            yield item
