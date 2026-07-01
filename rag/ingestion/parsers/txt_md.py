"""纯文本与 Markdown 文件的解析，支持内嵌图片引用。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from rag.ingestion.asset_store import AssetStore
from rag.ingestion.models import ImageBlock, ParsedDocument, TextBlock


_IMG_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


def parse_txt_md(
    file_path: Path,
    *,
    doc_id: str,
    display_path: str,
    data_dir: Path,
    asset_store: AssetStore,
) -> ParsedDocument:
    """解析 .txt/.md 文件为 TextBlock 与 ImageBlock 序列。

    功能:
        按 Markdown 图片语法 ``![alt](ref)`` 切分正文与图片；本地图片复制至资产目录，
        远程/data URL 写入占位路径；缺失引用写入 missing 标记文件。

    参数:
        file_path: 源文件路径。
        doc_id: 文档标识符。
        display_path: 展示用路径。
        data_dir: 数据根目录（解析相对图片路径的备选根）。
        asset_store: 图片资产存储。

    返回值:
        含有序 blocks 的 ``ParsedDocument``。

    异常:
        无；读取使用 UTF-8，非法字节被 ignore。
    """
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    blocks: List = []
    order = 0
    last_end = 0
    image_counter = 0

    for match in _IMG_PATTERN.finditer(raw):
        before = raw[last_end : match.start()].strip()
        if before:
            blocks.append(TextBlock(text=before, order=order))
            order += 1
        alt = match.group(1).strip()
        ref = match.group(2).strip()
        image_counter += 1
        asset_id = f"img_{image_counter:03d}"
        rel = _materialize_image_ref(
            ref,
            file_path=file_path,
            data_dir=data_dir,
            asset_store=asset_store,
            doc_id=doc_id,
            asset_id=asset_id,
        )
        blocks.append(
            ImageBlock(
                asset_id=asset_id,
                relative_path=rel,
                alt_text=alt or None,
                order=order,
            )
        )
        order += 1
        last_end = match.end()

    tail = raw[last_end:].strip()
    if tail:
        blocks.append(TextBlock(text=tail, order=order))

    if not blocks and raw.strip():
        blocks.append(TextBlock(text=raw.strip(), order=0))

    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(file_path),
        display_path=display_path,
        blocks=blocks,
    )


def _materialize_image_ref(
    ref: str,
    *,
    file_path: Path,
    data_dir: Path,
    asset_store: AssetStore,
    doc_id: str,
    asset_id: str,
) -> str:
    """将 Markdown 图片引用物化为资产目录中的路径。

    功能:
        远程 URL/data URI 返回占位 txt 路径；本地文件从 file_path 父目录或 data_dir
        查找并 save_bytes；找不到则写入 missing 标记文件。

    参数:
        ref: Markdown 图片 URL 或相对路径。
        file_path: 当前 Markdown 文件路径。
        data_dir: 数据根目录。
        asset_store: 资产存储。
        doc_id: 文档 ID。
        asset_id: 资产文件名标识。

    返回值:
        相对 persist 父目录的资源路径字符串。

    异常:
        无。
    """
    if ref.startswith(("http://", "https://", "data:")):
        return f"assets/{doc_id}/{asset_id}_remote.txt"

    candidate = (file_path.parent / ref).resolve()
    if not candidate.exists():
        candidate = (data_dir / ref).resolve()
    if candidate.exists() and candidate.is_file():
        suffix = candidate.suffix.lower() or ".png"
        data = candidate.read_bytes()
        return asset_store.save_bytes(doc_id, asset_id, data, suffix=suffix)

    placeholder = asset_store.doc_dir(doc_id)
    placeholder.mkdir(parents=True, exist_ok=True)
    note = placeholder / f"{asset_id}_missing.txt"
    note.write_text(f"missing image ref: {ref}", encoding="utf-8")
    return str(note.relative_to(asset_store.root.parent)).replace("\\", "/")
