"""文档解析与分片过程中的核心数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Union


@dataclass
class TextBlock:
    """文档中的纯文本段落块。

    功能：
        表示解析后文档中一段连续文本及其在文档中的顺序位置。

    参数：
        text: 段落正文。
        order: 在文档中的顺序序号。

    返回值：
        无（数据类实例）。
    """

    text: str
    order: int


@dataclass
class ImageBlock:
    """文档中的嵌入或独立图片块。

    功能：
        表示文档内图片资产及其说明、页码与质量标记。

    参数：
        asset_id: 资产文件名标识（不含路径）。
        relative_path: 相对 persist_dir 的存储路径。
        caption: 视觉模型或人工生成的说明。
        alt_text: 原始 alt 文本（如 Markdown）。
        page_index: PDF 页码（0-based）；非 PDF 为 None。
        order: 在文档中的顺序序号。
        low_quality: 是否标记为低质量（无描述、识别失败等）。

    返回值：
        无（数据类实例）。
    """

    asset_id: str
    relative_path: str
    caption: str = ""
    alt_text: str | None = None
    page_index: int | None = None
    order: int = 0
    low_quality: bool = False


Block = Union[TextBlock, ImageBlock]


@dataclass
class ParsedDocument:
    """解析器输出的结构化文档表示。

    功能：
        聚合文档标识、源路径与按阅读顺序排列的内容块列表。

    参数：
        doc_id: 稳定文档标识符。
        source_path: 源文件绝对路径。
        display_path: 展示用路径（如相对 data_dir）。
        blocks: 按阅读顺序排列的 TextBlock/ImageBlock 列表。

    返回值：
        无（数据类实例）。
    """

    doc_id: str
    source_path: str
    display_path: str
    blocks: List[Block] = field(default_factory=list)


@dataclass
class ChunkRecord:
    """入库前的单个检索分片记录。

    功能：
        承载分片正文、块类型、覆盖范围及关联图片 metadata，供向量库写入。

    参数：
        text: 分片正文（含图片占位行）。
        chunk_index: 文档内分片序号。
        block_types: 块类型集合的逗号分隔字符串（如 ``text,image``）。
        doc_id: 所属文档 ID。
        block_start: 覆盖的起始 block 索引。
        block_end: 覆盖的结束 block 索引。
        image_assets: 分片内图片相对路径列表。
        image_captions: 与 image_assets 同索引的说明列表。

    返回值：
        无（数据类实例）。
    """

    text: str
    chunk_index: int
    block_types: str
    doc_id: str = ""
    block_start: int = 0
    block_end: int = 0
    image_assets: List[str] = field(default_factory=list)
    image_captions: List[str] = field(default_factory=list)
