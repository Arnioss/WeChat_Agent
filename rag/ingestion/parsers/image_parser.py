"""独立图片文件的解析，整文件作为单个 ImageBlock 入库。"""

from __future__ import annotations

from pathlib import Path

from rag.ingestion.asset_store import AssetStore
from rag.ingestion.models import ImageBlock, ParsedDocument


def parse_image_file(
    file_path: Path,
    *,
    doc_id: str,
    display_path: str,
    asset_store: AssetStore,
) -> ParsedDocument:
    """将单张图片文件复制到资产目录并包装为 ParsedDocument。

    功能:
        读取文件字节，以 ``img_001`` 为 asset_id 保存，文档仅含一个 ImageBlock。

    参数:
        file_path: 图片源文件路径。
        doc_id: 文档标识符。
        display_path: 展示用路径。
        asset_store: 图片资产存储。

    返回值:
        仅含一个 ImageBlock 的 ``ParsedDocument``。

    异常:
        透传 ``save_bytes`` 可能抛出的 ValueError（如文件过大）。
    """
    data = file_path.read_bytes()
    suffix = file_path.suffix.lower() or ".png"
    asset_id = "img_001"
    rel = asset_store.save_bytes(doc_id, asset_id, data, suffix=suffix)
    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(file_path),
        display_path=display_path,
        blocks=[
            ImageBlock(
                asset_id=asset_id,
                relative_path=rel,
                order=0,
            )
        ],
    )
