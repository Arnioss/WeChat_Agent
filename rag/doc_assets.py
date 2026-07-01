"""按 doc_id 列出文档全部入库图示（检索阶段补全，不依赖单 chunk 的 image_assets）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from rag.ingestion.asset_store import AssetStore
from rag.ingestion.caption_service import hashlib_name

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _load_caption_from_cache(image_path: Path, *, vision_model: str, cache_dir: Path) -> str:
    """从 caption 缓存文件读取图片描述。

    功能：
        根据图片绝对路径与 vision 模型名生成缓存键，读取 JSON 缓存中的 caption 字段。

    参数：
        image_path: 本地图片文件路径。
        vision_model: 视觉模型名称，参与缓存键计算。
        cache_dir: caption 缓存目录。

    返回值：
        缓存中的说明文本；无缓存或读取失败时返回空字符串。

    异常：
        无；读取或 JSON 解析失败时返回 ``""``。
    """
    key = hashlib_name(str(image_path.resolve()) + vision_model)
    cache_file = cache_dir / f"{key}.json"
    if not cache_file.is_file():
        return ""
    try:
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        return str(data.get("caption") or "").strip()
    except Exception:
        return ""


def list_document_image_assets(
    persist_dir: Path,
    doc_id: str,
    *,
    vision_model: str = "",
    cache_dir: Path | None = None,
) -> Tuple[List[str], List[str]]:
    """列出指定文档在资产目录中的全部图片及说明。

    功能：
        扫描 ``assets/{doc_id}/`` 下支持的图片后缀文件，返回相对 persist_dir 的路径列表；
        若提供 vision_model，则尝试从 caption 缓存补全说明。

    参数：
        persist_dir: RAG 持久化根目录。
        doc_id: 文档标识符。
        vision_model: 视觉模型名称，用于读取 caption 缓存；空则跳过。
        cache_dir: caption 缓存目录；默认 ``persist_dir / "captions"``。

    返回值：
        (assets, captions) 元组；assets 为相对路径列表（如 ``assets/doc_xxx/media_001.png``），
        captions 为同索引的说明列表（可能为空字符串）。

    异常：
        无；文档目录不存在时返回 ``([], [])``。
    """
    store = AssetStore(persist_dir)
    folder = store.doc_dir(doc_id)
    if not folder.is_dir():
        return [], []

    captions_dir = cache_dir or (persist_dir / "captions")
    assets: List[str] = []
    captions: List[str] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        try:
            rel = str(path.relative_to(persist_dir)).replace("\\", "/")
        except ValueError:
            rel = str(path)
        assets.append(rel)
        cap = ""
        if vision_model:
            cap = _load_caption_from_cache(path, vision_model=vision_model, cache_dir=captions_dir)
        captions.append(cap)
    return assets, captions


def enrich_retrieved_rows(
    rows: List[Dict[str, str]],
    *,
    persist_dir: Path,
    vision_model: str = "",
    fill_missing_only: bool = True,
) -> List[Dict[str, str]]:
    """为检索结果行补全文档级图片 metadata。

    功能：
        命中某文档任一分片时，按 doc_id 从资产目录加载该文档全部图示，
        写入 ``image_assets``、``image_captions``，并在 ``block_types`` 中追加 ``image``。

    参数：
        rows: 检索结果行列表，每行含 ``doc_id`` 等 metadata 字段。
        persist_dir: RAG 持久化根目录。
        vision_model: 视觉模型名称，传给 ``list_document_image_assets`` 读取 caption 缓存。
        fill_missing_only: 为 True 时，已有 ``image_assets`` 的行不再补全。

    返回值：
        补全后的新行列表（原行未被原地修改）。

    异常：
        无；无 doc_id 或无图片资产的行保持原样。
    """
    if not rows:
        return rows

    doc_cache: Dict[str, Tuple[List[str], List[str]]] = {}
    enriched: List[Dict[str, str]] = []
    for row in rows:
        doc_id = (row.get("doc_id") or "").strip()
        if not doc_id:
            enriched.append(row)
            continue
        if fill_missing_only and "image_assets" in row:
            enriched.append(row)
            continue
        if doc_id not in doc_cache:
            doc_cache[doc_id] = list_document_image_assets(
                persist_dir,
                doc_id,
                vision_model=vision_model,
            )
        assets, captions = doc_cache[doc_id]
        if not assets:
            enriched.append(row)
            continue
        new_row = dict(row)
        new_row["image_assets"] = json.dumps(assets, ensure_ascii=False)
        new_row["image_captions"] = json.dumps(captions, ensure_ascii=False)
        if "image" not in (new_row.get("block_types") or ""):
            types = (new_row.get("block_types") or "text").strip()
            new_row["block_types"] = types if "image" in types else f"{types},image"
        enriched.append(new_row)
    return enriched
