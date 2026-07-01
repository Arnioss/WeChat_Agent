"""按文件后缀将源文件路由至对应解析器。"""

from __future__ import annotations

from pathlib import Path

from rag.env_config import RagEnvConfig
from rag.ingestion.asset_store import AssetStore, stable_doc_id
from rag.ingestion.models import ParsedDocument
from rag.ingestion.parsers.docx_parser import parse_docx
from rag.ingestion.parsers.image_parser import parse_image_file
from rag.ingestion.parsers.pdf_parser import parse_pdf
from rag.ingestion.parsers.txt_md import parse_txt_md

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def parse_file(
    file_path: Path,
    *,
    display_path: str,
    data_dir: Path,
    asset_store: AssetStore,
    config: RagEnvConfig,
) -> ParsedDocument:
    """根据文件后缀选择解析器并返回 ParsedDocument。

    功能:
        计算 stable_doc_id，按后缀分发至 txt/md、pdf、docx、图片或抛出不支持错误。

    参数:
        file_path: 待解析文件的绝对路径。
        display_path: 展示用路径。
        data_dir: 数据根目录（txt/md 解析图片引用用）。
        asset_store: 图片资产存储实例。
        config: RAG 环境配置（PDF 页数上限、最小图面积比等）。

    返回值:
        含有序 TextBlock/ImageBlock 的 ``ParsedDocument``。

    异常:
        ValueError: 文件后缀不在支持列表中。
    """
    file_key = str(file_path.resolve())
    doc_id = stable_doc_id(file_key)
    suffix = file_path.suffix.lower()

    if suffix in (".txt", ".md"):
        return parse_txt_md(
            file_path,
            doc_id=doc_id,
            display_path=display_path,
            data_dir=data_dir,
            asset_store=asset_store,
        )
    if suffix == ".pdf":
        return parse_pdf(
            file_path,
            doc_id=doc_id,
            display_path=display_path,
            asset_store=asset_store,
            max_pages=config.max_pages,
            min_figure_area_ratio=config.min_figure_area_ratio,
        )
    if suffix == ".docx":
        return parse_docx(
            file_path,
            doc_id=doc_id,
            display_path=display_path,
            asset_store=asset_store,
        )
    if suffix in _IMAGE_SUFFIXES:
        return parse_image_file(
            file_path,
            doc_id=doc_id,
            display_path=display_path,
            asset_store=asset_store,
        )
    raise ValueError(f"不支持的文件类型: {suffix}")
