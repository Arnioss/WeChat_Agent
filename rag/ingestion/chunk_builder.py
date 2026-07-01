"""将有序解析块构建为检索分片。"""

from __future__ import annotations

from typing import List

from rag.ingestion.models import Block, ChunkRecord, ImageBlock, ParsedDocument, TextBlock


def format_image_line(img: ImageBlock, index: int) -> str:
    """生成用于嵌入的低权重图片占位行。

    功能：
        输出 ``[image N] asset=...`` 占位符；caption 保留在 metadata/block store 中，
        不作为与正文等权的内容混入分片文本。

    参数：
        img: 图片块。
        index: 文档内图片序号（从 1 开始）。

    返回值：
        图片占位行字符串。

    异常：
        无。
    """

    return f"[image {index}] asset={img.relative_path}"


def block_to_text(block: Block, image_index: int) -> str:
    """将单个内容块转换为分片文本片段。

    功能：
        TextBlock 返回去空白后的正文；ImageBlock 返回占位行；未知类型返回空串。

    参数：
        block: 文本或图片块。
        image_index: 当前图片序号（仅 ImageBlock 使用）。

    返回值：
        块对应的文本片段；空块返回 ``""``。

    异常：
        无。
    """
    if isinstance(block, TextBlock):
        return (block.text or "").strip()
    if isinstance(block, ImageBlock):
        return format_image_line(block, image_index)
    return ""


def blocks_to_plain_text(blocks: List[Block]) -> str:
    """将多块列表拼接为纯文本。

    功能：
        按顺序提取各块文本片段，以双换行符连接。

    参数：
        blocks: 按阅读顺序排列的内容块列表。

    返回值：
        拼接后的完整文本；无有效内容时返回 ``""``。

    异常：
        无。
    """
    parts: List[str] = []
    image_index = 0
    for block in blocks:
        if isinstance(block, TextBlock):
            text = (block.text or "").strip()
            if text:
                parts.append(text)
        elif isinstance(block, ImageBlock):
            image_index += 1
            parts.append(format_image_line(block, image_index))
    return "\n\n".join(parts)


def build_chunks(
    doc: ParsedDocument,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[ChunkRecord]:
    """按字符上限将解析文档切分为检索分片。

    功能：
        顺序遍历 blocks，累积文本直至超过 chunk_size 时刷出分片；
        支持 chunk_overlap 尾部重叠；记录块类型与关联图片 metadata。

    参数：
        doc: 已解析文档。
        chunk_size: 单分片最大字符数。
        chunk_overlap: 相邻分片重叠字符数。

    返回值：
        ``ChunkRecord`` 列表。

    异常：
        无。
    """
    records: List[ChunkRecord] = []
    current_parts: List[str] = []
    current_images: List[ImageBlock] = []
    current_types: set[str] = set()
    current_start: int | None = None
    current_end = 0
    image_index = 0

    def flush() -> None:
        """将当前累积内容刷为一个分片并重置或保留重叠尾部。

        功能：
            拼接 current_parts 生成 ChunkRecord 并追加至 records；
            若 chunk_overlap > 0 则保留尾部文本作为下一片段起点。

        参数：
            无（闭包引用外层累积状态）。

        返回值：
            无。

        异常：
            无。
        """
        nonlocal current_parts, current_images, current_types, current_start, current_end
        text = "\n\n".join(part for part in current_parts if part.strip()).strip()
        if not text:
            current_parts = []
            current_images = []
            current_types = set()
            current_start = None
            return
        block_types = ",".join(sorted(current_types)) or "text"
        records.append(
            ChunkRecord(
                text=text,
                chunk_index=len(records),
                block_types=block_types,
                doc_id=doc.doc_id,
                block_start=current_start or 0,
                block_end=current_end,
                image_assets=[img.relative_path for img in current_images],
                image_captions=[img.caption or img.alt_text or "" for img in current_images],
            )
        )
        if chunk_overlap > 0 and current_parts:
            overlap_text = _tail_text(current_parts, chunk_overlap)
            current_parts = [overlap_text] if overlap_text else []
            current_images = []
            current_types = {"text"} if overlap_text else set()
            current_start = current_end if overlap_text else None
        else:
            current_parts = []
            current_images = []
            current_types = set()
            current_start = None

    for block_pos, block in enumerate(doc.blocks):
        if isinstance(block, ImageBlock):
            image_index += 1
        part = block_to_text(block, image_index)
        if not part:
            continue

        projected = "\n\n".join(current_parts + [part])
        if current_parts and len(projected) > chunk_size:
            flush()

        if current_start is None:
            current_start = block_pos
        current_end = block_pos
        if isinstance(block, ImageBlock):
            current_images.append(block)
            current_types.add("image")
        else:
            current_types.add("text")
        current_parts.append(part)

    flush()

    return records


def _tail_text(parts: List[str], max_chars: int) -> str:
    """取拼接文本的尾部重叠片段。

    功能：
        将 parts 拼接后，若长度超过 max_chars 则截取末尾 max_chars 个字符。

    参数：
        parts: 文本片段列表。
        max_chars: 最大保留字符数。

    返回值：
        尾部重叠文本；不足 max_chars 时返回全文。

    异常：
        无。
    """
    text = "\n\n".join(part for part in parts if part.strip()).strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:].lstrip()


def _split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> List[str]:
    """按固定窗口与步长切分纯文本。

    功能：
        以 chunk_size 为窗口、chunk_size - chunk_overlap 为步长滑动切分字符串。

    参数：
        text: 待切分文本。
        chunk_size: 单段最大字符数。
        chunk_overlap: 相邻段重叠字符数。

    返回值：
        非空文本段列表；输入为空时返回 ``[]``。

    异常：
        无。
    """
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - chunk_overlap)
    while start < len(cleaned):
        chunks.append(cleaned[start : start + chunk_size])
        start += step
    return [x for x in chunks if x.strip()]
