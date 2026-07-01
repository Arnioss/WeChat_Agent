"""PDF 文档解析：提取页面文本与按显示区域光栅化的图示。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Set, TYPE_CHECKING

from rag.ingestion.asset_store import AssetStore
from rag.ingestion.models import ImageBlock, ParsedDocument, TextBlock

if TYPE_CHECKING:
    import fitz

# 按页面显示区域光栅化（与阅读器所见一致）
_PDF_CLIP_DPI = 144
# 重叠区域合并（主图 + 蒙版）
_RECT_MERGE_OVERLAP_RATIO = 0.2
# 上下拼接：仅合并「同一张图被 PDF 切成上下两块」（间隙极小、几乎贴在一起）
_RECT_VERTICAL_GAP_TOUCH_MAX = 2.0  # pt，大于此值视为两张独立图
_RECT_VERTICAL_GAP_RELATIVE = 0.04  # 间隙不超过较短边高度的 4%
_HORIZ_ALIGN_RATIO = 0.72


def parse_pdf(
    file_path: Path,
    *,
    doc_id: str,
    display_path: str,
    asset_store: AssetStore,
    max_pages: int,
    min_figure_area_ratio: float,
) -> ParsedDocument:
    """解析 PDF 为页面文本块与提取的图示块。

    功能:
        逐页提取 get_text；对满足面积阈值的图片 bbox 合并后光栅化为 PNG 入库；
        无 bbox 的 xref 回退为 Pixmap 转 PNG。

    参数:
        file_path: PDF 文件路径。
        doc_id: 文档标识符。
        display_path: 展示用路径。
        asset_store: 图片资产存储。
        max_pages: 最多处理的页数。
        min_figure_area_ratio: 图片占页面面积的最小比例，低于则跳过。

    返回值:
        含 TextBlock 与 ImageBlock 的 ``ParsedDocument``。

    异常:
        RuntimeError: 未安装 pymupdf 依赖。
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("未安装 pymupdf，请执行 pip install pymupdf") from exc

    blocks: List = []
    order = 0
    doc = fitz.open(str(file_path))
    try:
        page_count = min(len(doc), max_pages)
        for page_index in range(page_count):
            page = doc[page_index]
            page_items = []

            for text_block in page.get_text("blocks") or []:
                if len(text_block) < 5:
                    continue
                if len(text_block) >= 7 and int(text_block[6] or 0) != 0:
                    continue
                x0, y0, x1, y1, text = text_block[:5]
                text = (text or "").strip()
                if text:
                    page_items.append(
                        (
                            float(y0),
                            float(x0),
                            len(page_items),
                            TextBlock(text=text, order=0),
                        )
                    )

            page_area = max(page.rect.width * page.rect.height, 1.0)
            image_list = page.get_images(full=True)
            candidate_rects: List["fitz.Rect"] = []
            no_rect_xrefs: List[int] = []
            seen_xref: Set[int] = set()

            for img in image_list:
                xref = int(img[0])
                if xref in seen_xref:
                    continue
                seen_xref.add(xref)

                rects = page.get_image_rects(xref)
                if rects:
                    union_rect = _union_rects(rects)
                    if _rect_area_ratio(union_rect, page_area) >= min_figure_area_ratio:
                        candidate_rects.append(union_rect)
                else:
                    no_rect_xrefs.append(xref)

            figure_groups = _merge_figure_rects(candidate_rects)
            for fig_idx, group_rect in enumerate(figure_groups, start=1):
                png_bytes = _clip_rect_to_png(page, group_rect, dpi=_PDF_CLIP_DPI)
                if not png_bytes:
                    continue
                asset_id = f"p{page_index + 1}_fig_{fig_idx:03d}"
                rel = asset_store.save_bytes(doc_id, asset_id, png_bytes, suffix=".png")
                page_items.append(
                    (
                        float(group_rect.y0),
                        float(group_rect.x0),
                        len(page_items),
                        ImageBlock(
                            asset_id=asset_id,
                            relative_path=rel,
                            page_index=page_index,
                            order=0,
                        ),
                    )
                )

            for xref in no_rect_xrefs:
                png_bytes = _xref_to_png(doc, xref)
                if not png_bytes:
                    continue
                asset_id = f"p{page_index + 1}_xref_{xref}"
                rel = asset_store.save_bytes(doc_id, asset_id, png_bytes, suffix=".png")
                page_items.append(
                    (
                        float(page.rect.y1) + len(page_items),
                        0.0,
                        len(page_items),
                        ImageBlock(
                            asset_id=asset_id,
                            relative_path=rel,
                            page_index=page_index,
                            order=0,
                        ),
                    )
                )

            for _, _, _, block in sorted(page_items, key=lambda item: (item[0], item[1], item[2])):
                block.order = order
                blocks.append(block)
                order += 1
    finally:
        doc.close()

    return ParsedDocument(
        doc_id=doc_id,
        source_path=str(file_path),
        display_path=display_path,
        blocks=blocks,
    )


def _union_rects(rects: List["fitz.Rect"]) -> "fitz.Rect":
    """合并多个矩形为包围盒。

    功能:
        对 rects 做逐元素 ``|=`` 并集运算。

    参数:
        rects: PyMuPDF Rect 列表，至少含一个元素。

    返回值:
        包含全部输入矩形的最小外接 Rect。

    异常:
        无。
    """
    import fitz

    union_rect = fitz.Rect(rects[0])
    for rect in rects[1:]:
        union_rect |= rect
    return union_rect


def _rect_area_ratio(rect: "fitz.Rect", page_area: float) -> float:
    """计算矩形面积占页面面积的比例。

    功能:
        ``(width * height) / page_area``。

    参数:
        rect: 页面上的矩形区域。
        page_area: 页面总面积（pt²）。

    返回值:
        0~1 之间的面积比。

    异常:
        无。
    """
    return (rect.width * rect.height) / max(page_area, 1.0)


def _overlap_ratio(a: "fitz.Rect", b: "fitz.Rect") -> float:
    """计算两矩形交集面积占较小矩形面积的比例。

    功能:
        用于判断重叠/蒙版关系是否达到合并阈值。

    参数:
        a: 第一个 Rect。
        b: 第二个 Rect。

    返回值:
        交集面积 / min(a面积, b面积)；无交集时为 0。

    异常:
        无。
    """
    inter = a & b
    if inter.is_empty or inter.width <= 0 or inter.height <= 0:
        return 0.0
    inter_area = inter.width * inter.height
    min_area = min(a.width * a.height, b.width * b.height)
    if min_area <= 0:
        return 0.0
    return inter_area / min_area


def _horiz_overlap_ratio(a: "fitz.Rect", b: "fitz.Rect") -> float:
    """计算两矩形在水平方向上的重叠比例。

    功能:
        重叠宽度除以较短矩形的宽度。

    参数:
        a: 第一个 Rect。
        b: 第二个 Rect。

    返回值:
        水平重叠比；无水平重叠时为 0。

    异常:
        无。
    """
    x0 = max(a.x0, b.x0)
    x1 = min(a.x1, b.x1)
    if x1 <= x0:
        return 0.0
    return (x1 - x0) / max(1.0, min(a.width, b.width))


def _vertical_gap(a: "fitz.Rect", b: "fitz.Rect") -> float:
    """计算两矩形在垂直方向上的间隙。

    功能:
        若 a 在 b 上方则返回 b.y0 - a.y1；若在下方则返回 a.y0 - b.y1；相交返回 -1。

    参数:
        a: 第一个 Rect。
        b: 第二个 Rect。

    返回值:
        垂直间隙（pt）；相交时 -1。

    异常:
        无。
    """
    if a.y1 <= b.y0:
        return b.y0 - a.y1
    if b.y1 <= a.y0:
        return a.y0 - b.y1
    return -1.0


def _should_merge_figure_rect(a: "fitz.Rect", b: "fitz.Rect") -> bool:
    """判断两个图片 bbox 是否应合并为同一逻辑图。

    功能:
        满足重叠/蒙版阈值，或水平对齐且垂直间隙极小（同图上下切块）时返回 True。

    参数:
        a: 第一个 Rect。
        b: 第二个 Rect。

    返回值:
        应合并时 True。

    异常:
        无。
    """
    if _overlap_ratio(a, b) >= _RECT_MERGE_OVERLAP_RATIO:
        return True
    if _horiz_overlap_ratio(a, b) < _HORIZ_ALIGN_RATIO:
        return False

    gap = _vertical_gap(a, b)
    if gap < 0:
        # 边界重合导致的轻微垂直重叠（如 xref27 底与 xref28 顶同一 y）
        inter = a & b
        return not inter.is_empty and inter.height <= 3.0

    if gap > _RECT_VERTICAL_GAP_TOUCH_MAX:
        return False
    min_height = min(a.height, b.height)
    return gap <= _RECT_VERTICAL_GAP_RELATIVE * max(min_height, 1.0)


def _merge_figure_rects(rects: List["fitz.Rect"]) -> List["fitz.Rect"]:
    """迭代合并应视为同一图的矩形组。

    功能:
        反复 pairwise 合并满足 ``_should_merge_figure_rect`` 的 Rect，直至稳定。

    参数:
        rects: 候选图片 bbox 列表。

    返回值:
        合并后的 Rect 列表。

    异常:
        无；空输入返回 ``[]``。
    """
    import fitz

    if not rects:
        return []
    groups = [fitz.Rect(r) for r in rects]
    while True:
        changed = False
        next_groups: List[fitz.Rect] = []
        used = [False] * len(groups)
        for i, base in enumerate(groups):
            if used[i]:
                continue
            merged = fitz.Rect(base)
            used[i] = True
            for j in range(i + 1, len(groups)):
                if used[j]:
                    continue
                if _should_merge_figure_rect(merged, groups[j]):
                    merged |= groups[j]
                    used[j] = True
                    changed = True
            next_groups.append(merged)
        groups = next_groups
        if not changed:
            break
    return groups


def _clip_rect_to_png(page, rect, *, dpi: int) -> Optional[bytes]:
    """按页面上显示区域光栅化为 PNG 字节。

    功能:
        使用指定 DPI 对 rect 区域 clip 渲染；宽高小于 8px 时视为无效。

    参数:
        page: PyMuPDF Page 对象。
        rect: 页面上的裁剪区域。
        dpi: 光栅化分辨率。

    返回值:
        PNG 字节；失败或过小返回 None。

    异常:
        无；渲染异常时返回 None。
    """
    try:
        import fitz

        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
        if pix.width < 8 or pix.height < 8:
            return None
        return pix.tobytes("png")
    except Exception:
        return None


def _xref_to_png(doc, xref: int) -> Optional[bytes]:
    """通过 xref 将 PDF 内嵌图片转为 PNG（无 bbox 时的回退路径）。

    功能:
        经 Pixmap 转 RGB，处理 CMYK 与传真反色；去除 alpha 通道。

    参数:
        doc: 已打开的 PyMuPDF Document。
        xref: 图片 xref 编号。

    返回值:
        PNG 字节；失败或过小返回 None。

    异常:
        无；转换异常时返回 None。
    """
    try:
        import fitz

        pix = fitz.Pixmap(doc, xref)
        if pix.width < 8 or pix.height < 8:
            return None

        cs = pix.colorspace
        if cs is not None and cs.n == 0:
            pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.invert_irect()
        elif pix.n - pix.alpha < 4:
            pix = fitz.Pixmap(fitz.csRGB, pix)
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        return pix.tobytes("png")
    except Exception:
        return None
