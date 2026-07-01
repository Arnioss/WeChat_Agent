"""SQLite store for ordered document blocks used to rebuild inline context."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from rag.ingestion.models import ParsedDocument


class DocumentBlockStore:
    """功能：以 SQLite 持久化文档块顺序，供检索后重建行内上下文。
    参数：
    - 无（实例字段由构造器初始化）。
    返回值：
    - 无。作为上下文管理器使用时在退出时关闭连接。
    """

    def __init__(self, persist_dir: Path):
        """功能：初始化块存储并创建或打开 SQLite 数据库。
        参数：
        - persist_dir：持久化目录，数据库文件为 `rag_blocks.sqlite3`。
        返回值：
        - 无。
        """
        self.persist_dir = persist_dir
        self.db_path = persist_dir / "rag_blocks.sqlite3"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rag_blocks (
                doc_id TEXT NOT NULL,
                source TEXT NOT NULL,
                block_id TEXT NOT NULL,
                block_order INTEGER NOT NULL,
                block_type TEXT NOT NULL,
                text TEXT NOT NULL DEFAULT '',
                image_asset TEXT NOT NULL DEFAULT '',
                image_caption TEXT NOT NULL DEFAULT '',
                section_title TEXT NOT NULL DEFAULT '',
                page_index INTEGER,
                PRIMARY KEY (doc_id, block_order)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_rag_blocks_source ON rag_blocks(source)"
        )
        self._conn.commit()

    def delete_source(self, source: str) -> None:
        """功能：按来源路径删除该文档的全部块记录。
        参数：
        - source：文档来源的绝对路径字符串。
        返回值：
        - 无。
        """
        self._conn.execute("DELETE FROM rag_blocks WHERE source = ?", (source,))
        self._conn.commit()

    def upsert_document(self, doc: "ParsedDocument") -> None:
        """功能：将解析后的文档块写入存储，先删除同来源旧记录再批量插入。
        参数：
        - doc：包含 doc_id、source_path 与 blocks 的 ParsedDocument。
        返回值：
        - 无。
        """
        source = str(Path(doc.source_path).resolve())
        self.delete_source(source)
        rows = []
        section_title = ""
        for index, block in enumerate(doc.blocks):
            block_order = int(getattr(block, "order", index))
            block_id = f"{doc.doc_id}:{block_order}"
            if _is_text_block(block):
                text = (block.text or "").strip()
                block_type = _text_block_type(text)
                if _looks_like_heading(text):
                    section_title = text
                rows.append(
                    (
                        doc.doc_id,
                        source,
                        block_id,
                        block_order,
                        block_type,
                        text,
                        "",
                        "",
                        section_title,
                        None,
                    )
                )
            elif _is_image_block(block):
                rows.append(
                    (
                        doc.doc_id,
                        source,
                        block_id,
                        block_order,
                        "image",
                        "",
                        block.relative_path or "",
                        block.caption or block.alt_text or "",
                        section_title,
                        block.page_index,
                    )
                )
        if rows:
            self._conn.executemany(
                """
                INSERT OR REPLACE INTO rag_blocks
                    (doc_id, source, block_id, block_order, block_type, text,
                     image_asset, image_caption, section_title, page_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            self._conn.commit()

    def get_range(
        self,
        *,
        doc_id: str = "",
        source: str = "",
        block_start: int,
        block_end: int,
        window_before: int = 0,
        window_after: int = 0,
    ) -> List[Dict[str, str]]:
        """功能：按 doc_id 或 source 查询指定块序区间，可扩展前后窗口。
        参数：
        - doc_id：文档 ID；与 source 二选一。
        - source：文档来源绝对路径；与 doc_id 二选一。
        - block_start：起始块序（含）。
        - block_end：结束块序（含）。
        - window_before：向前扩展的块数。
        - window_after：向后扩展的块数。
        返回值：
        - List[Dict[str, str]]：按 block_order 升序的块字典列表；doc_id 与 source 均未提供时返回空列表。
        """
        start = max(0, int(block_start) - max(0, window_before))
        end = int(block_end) + max(0, window_after)
        if doc_id:
            cursor = self._conn.execute(
                """
                SELECT doc_id, source, block_id, block_order, block_type, text,
                       image_asset, image_caption, section_title, page_index
                FROM rag_blocks
                WHERE doc_id = ? AND block_order BETWEEN ? AND ?
                ORDER BY block_order ASC
                """,
                (doc_id, start, end),
            )
        elif source:
            cursor = self._conn.execute(
                """
                SELECT doc_id, source, block_id, block_order, block_type, text,
                       image_asset, image_caption, section_title, page_index
                FROM rag_blocks
                WHERE source = ? AND block_order BETWEEN ? AND ?
                ORDER BY block_order ASC
                """,
                (source, start, end),
            )
        else:
            return []
        return [_row_to_dict(row) for row in cursor]

    def close(self) -> None:
        """功能：关闭底层 SQLite 连接。
        参数：
        - 无。
        返回值：
        - 无。
        """
        self._conn.close()

    def __enter__(self) -> "DocumentBlockStore":
        """功能：进入上下文管理器，返回自身。
        参数：
        - 无。
        返回值：
        - DocumentBlockStore：当前实例。
        """
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """功能：退出上下文管理器时关闭数据库连接。
        参数：
        - exc_type、exc、tb：异常类型、实例与回溯（未使用）。
        返回值：
        - 无。
        """
        self.close()


def render_blocks_inline(blocks: List[Dict[str, str]], *, persist_dir: Path) -> str:
    """功能：将块列表渲染为行内文本，图片块输出说明与 `[KB_IMAGE:...]` 占位符。
    参数：
    - blocks：块字典列表（含 block_type、text、image_asset 等字段）。
    - persist_dir：图片资源解析所用的持久化根目录。
    返回值：
    - str：段落以双换行拼接的文本；无有效内容时返回空字符串。
    """
    from rag.rag_images import resolve_asset_path

    lines: List[str] = []
    for block in blocks:
        block_type = block.get("block_type") or ""
        if block_type == "image":
            caption = (block.get("image_caption") or "").strip()
            if caption:
                lines.append(caption)
            resolved = resolve_asset_path(persist_dir, block.get("image_asset") or "")
            if resolved is not None:
                lines.append(f"[KB_IMAGE:{resolved.as_posix()}]")
            continue
        text = (block.get("text") or "").strip()
        if text:
            lines.append(text)
    return "\n\n".join(lines).strip()


def _row_to_dict(row) -> Dict[str, str]:
    """功能：将 SQLite 查询行转换为块字段字典。
    参数：
    - row：与 rag_blocks 表列顺序一致的元组或序列。
    返回值：
    - Dict[str, str]：非 None 字段转为字符串的字典。
    """
    keys = (
        "doc_id",
        "source",
        "block_id",
        "block_order",
        "block_type",
        "text",
        "image_asset",
        "image_caption",
        "section_title",
        "page_index",
    )
    result: Dict[str, str] = {}
    for key, value in zip(keys, row):
        if value is None:
            continue
        result[key] = str(value)
    return result


def _is_text_block(block: object) -> bool:
    """功能：判断解析块是否为文本块（有 text 且无 relative_path）。
    参数：
    - block：解析管线中的块对象。
    返回值：
    - bool：满足文本块特征时返回 True。
    """
    return hasattr(block, "text") and not hasattr(block, "relative_path")


def _is_image_block(block: object) -> bool:
    """功能：判断解析块是否为图片块（含 relative_path 属性）。
    参数：
    - block：解析管线中的块对象。
    返回值：
    - bool：满足图片块特征时返回 True。
    """
    return hasattr(block, "relative_path")


def _text_block_type(text: str) -> str:
    """功能：根据文本内容推断块类型（table、heading 或 text）。
    参数：
    - text：块文本内容。
    返回值：
    - str：`"table"`、`"heading"` 或 `"text"`。
    """
    stripped = (text or "").lstrip()
    lowered = stripped.lower()
    if lowered.startswith(("table:", "表格:", "表格：", "|")):
        return "table"
    if stripped.startswith("#") or _looks_like_heading(stripped):
        return "heading"
    return "text"


def _looks_like_heading(text: str) -> bool:
    """功能：启发式判断单行短文本是否像标题（Markdown # 或编号/中文序号开头）。
    参数：
    - text：待检测文本。
    返回值：
    - bool：像标题时返回 True。
    """
    stripped = (text or "").strip()
    if not stripped or "\n" in stripped or len(stripped) > 80:
        return False
    if stripped.startswith("#"):
        return True
    return bool(re.match(r"^(\d+(\.\d+){0,4}|[一二三四五六七八九十]+[、.．])\s*\S+", stripped))
