"""文档入库流水线：解析、图片说明补全与分片。"""

from __future__ import annotations

import os
from pathlib import Path

from rag.env_config import RagEnvConfig
from rag.ingestion.asset_store import AssetStore
from rag.block_store import DocumentBlockStore
from rag.ingestion.caption_service import CaptionService
from rag.ingestion.chunk_builder import build_chunks
from rag.ingestion.models import ChunkRecord, ImageBlock, ParsedDocument
from rag.ingestion.parsers.router import parse_file
from rag.logging_utils import rag_log


class DocumentIngestionPipeline:
    """协调文件解析、图片说明（caption）补全与分片构建的入库流水线。

    功能：
        封装单文件解析、图片 caption 补全、块存储与分片构建的完整入库流程。

    参数：
        无（实例属性由 ``__init__`` 设置）。

    返回值：
        无（类定义）。
    """

    def __init__(
        self,
        *,
        config: RagEnvConfig,
        persist_dir: Path,
        data_dir: Path,
        chunk_size: int,
        chunk_overlap: int,
    ):
        """初始化入库流水线。

        功能：
            创建 AssetStore、CaptionService，并保存分片与目录配置。

        参数：
            config: RAG 环境配置。
            persist_dir: 持久化根目录（``.rag_store``）。
            data_dir: 原始数据根目录（解析 Markdown 图片相对路径用）。
            chunk_size: 分片最大字符数。
            chunk_overlap: 分片重叠字符数。

        返回值：
            无。

        异常：
            无；资产子目录名取自环境变量 ``RAG_ASSETS_DIR``，默认 ``assets``。
        """
        assets_subdir = (os.getenv("RAG_ASSETS_DIR") or "assets").strip()
        self.config = config
        self.persist_dir = persist_dir
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.asset_store = AssetStore(persist_dir, assets_subdir=assets_subdir)
        self.block_store = DocumentBlockStore(persist_dir)
        cache_dir = persist_dir / "captions"
        self.caption_service = CaptionService(config, cache_dir=cache_dir)

    def ingest_file(self, file_path: Path, *, display_path: str) -> list[ChunkRecord]:
        """解析单个文件并返回分片列表。

        功能：
            按后缀路由解析 → 为图片块补全 caption → 按 chunk_size 切分。

        参数：
            file_path: 待入库文件的绝对路径。
            display_path: 写入 metadata 的展示路径。

        返回值：
            ``ChunkRecord`` 列表。

        异常：
            透传解析器或 ``build_chunks`` 可能抛出的异常（如不支持类型、缺少依赖）。
        """
        doc = parse_file(
            file_path,
            display_path=display_path,
            data_dir=self.data_dir,
            asset_store=self.asset_store,
            config=self.config,
        )
        self._enrich_captions(doc, display_path=display_path)
        self.block_store.upsert_document(doc)
        return build_chunks(
            doc,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def delete_file_assets(self, file_path: Path) -> None:
        """删除与源文件关联的全部图片资产。

        功能：
            根据 file_path 计算 stable_doc_id，调用 AssetStore.delete_doc。

        参数：
            file_path: 源文件路径。

        返回值：
            无。

        异常：
            无。
        """
        from rag.ingestion.asset_store import stable_doc_id

        doc_id = stable_doc_id(str(file_path.resolve()))
        self.asset_store.delete_doc(doc_id)
        self.block_store.delete_source(str(file_path.resolve()))

    def _enrich_captions(self, doc: ParsedDocument, *, display_path: str = "") -> None:
        """为文档中尚无 caption 的图片块调用视觉识别。

        功能：
            遍历 ImageBlock，跳过已有 caption；缺失文件或识别失败时写入占位符并标记 low_quality。

        参数：
            doc: 已解析文档（blocks 会被原地更新 caption/low_quality）。
            display_path: 日志展示用路径。

        返回值：
            无。

        异常：
            无；单张图片失败时记录日志并继续处理其余图片。
        """
        image_blocks = [b for b in doc.blocks if isinstance(b, ImageBlock)]
        total = len(image_blocks)
        if total == 0:
            return
        label = display_path or doc.display_path or doc.source_path
        rag_log(f"[RAG] 开始图片识别，共 {total} 张: {label}", flush=True)

        for index, block in enumerate(image_blocks, start=1):
            filename = Path(block.relative_path).name or block.asset_id
            if block.caption and block.caption.strip():
                rag_log(f"[RAG] 图片识别 {index}/{total}: {filename}（已有描述，跳过）", flush=True)
                continue
            rag_log(f"[RAG] 图片识别 {index}/{total}: {filename} ...", flush=True)
            try:
                image_path = self.persist_dir / block.relative_path
                if not image_path.exists():
                    block.caption = block.alt_text or "[图-资源缺失]"
                    block.low_quality = True
                    rag_log(f"[RAG] 图片识别 {index}/{total}: {filename}（资源缺失）", flush=True)
                    continue
                cached_before = self.caption_service.has_cached(image_path)
                block.caption = self.caption_service.caption_image_file(
                    image_path,
                    alt_text=block.alt_text,
                )
                if cached_before:
                    rag_log(
                        f"[RAG] 图片识别 {index}/{total}: {filename}（缓存命中，"
                        f"未调用 {self.caption_service.config.vision_model}；"
                        f"缓存目录 .rag_store/captions/）",
                        flush=True,
                    )
                else:
                    rag_log(f"[RAG] 图片识别 {index}/{total}: {filename} 完成", flush=True)
                if block.caption in ("", "[图-无描述]", "无法识别图示内容"):
                    block.low_quality = True
            except Exception as exc:
                rag_log(f"[RAG] 图片识别 {index}/{total}: {filename} 失败 — {exc}", flush=True)
                block.caption = block.alt_text or "[图-识别失败]"
                block.low_quality = True
