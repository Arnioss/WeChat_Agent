"""文档图片资产的本地存储与路径解析。"""

from __future__ import annotations

import hashlib
import re
import shutil
from pathlib import Path


def stable_doc_id(source_path: str) -> str:
    """根据源文件路径生成稳定的文档 ID。

    功能：
        对 source_path 做 UTF-8 MD5 哈希，取前 16 位十六进制，前缀 ``doc_``。

    参数：
        source_path: 源文件的绝对或规范化路径字符串。

    返回值：
        形如 ``doc_{16位hex}`` 的文档标识符。

    异常：
        无。
    """
    digest = hashlib.md5(source_path.encode("utf-8")).hexdigest()[:16]
    return f"doc_{digest}"


class AssetStore:
    """管理 RAG 持久化目录下的文档图片资产。

    功能：
        提供文档资产目录管理、二进制保存、删除与路径安全解析。

    参数：
        无（实例属性由 ``__init__`` 设置）。

    返回值：
        无（类定义）。
    """

    def __init__(self, persist_dir: Path, assets_subdir: str = "assets"):
        """初始化资产存储。

        功能：
            在 persist_dir 下创建 assets 子目录作为资产根目录。

        参数：
            persist_dir: RAG 持久化根目录。
            assets_subdir: 资产相对子目录名，默认 ``"assets"``。

        返回值：
            无。

        异常：
            无；目录不存在时会自动创建。
        """
        self.root = (persist_dir / assets_subdir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def doc_dir(self, doc_id: str) -> Path:
        """返回指定文档的资产目录路径。

        功能：
            将 doc_id 净化为安全文件名，返回 ``{root}/{safe_doc_id}`` 目录。

        参数：
            doc_id: 文档标识符。

        返回值：
            该文档专属的资产目录 ``Path``。

        异常：
            ValueError: doc_id 净化后路径逃逸出 assets 根目录。
        """
        safe = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)
        path = (self.root / safe).resolve()
        if not str(path).startswith(str(self.root)):
            raise ValueError("非法 doc_id")
        return path

    def save_bytes(self, doc_id: str, asset_id: str, data: bytes, suffix: str = ".png") -> str:
        """将二进制图片数据写入文档资产目录。

        功能：
            在 doc_dir 下以 asset_id 为文件名保存 data，返回相对 persist 父目录的路径。

        参数：
            doc_id: 文档标识符。
            asset_id: 资产文件名（不含后缀，会被净化）。
            data: 图片二进制内容。
            suffix: 文件后缀，默认 ``".png"``。

        返回值：
            相对 persist_dir 的正斜杠路径字符串（如 ``assets/doc_xxx/media_001.png``）。

        异常：
            ValueError: 数据超过 50MB，或 doc_id 非法。
        """
        if len(data) > 50 * 1024 * 1024:
            raise ValueError("图片过大")
        folder = self.doc_dir(doc_id)
        folder.mkdir(parents=True, exist_ok=True)
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", asset_id)
        dest = folder / f"{safe_id}{suffix}"
        dest.write_bytes(data)
        rel = dest.relative_to(self.root.parent)
        return str(rel).replace("\\", "/")

    def delete_doc(self, doc_id: str) -> None:
        """删除指定文档的全部资产文件。

        功能：
            递归删除 doc_dir 对应目录（若存在）。

        参数：
            doc_id: 文档标识符。

        返回值：
            无。

        异常：
            无；删除过程中的错误被忽略。
        """
        folder = self.doc_dir(doc_id)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    def resolve_path(self, persist_dir: Path, relative_path: str) -> Path:
        """将相对资源路径解析为绝对文件路径并校验安全性。

        功能：
            基于 persist_dir 拼接 relative_path，确保结果位于 persist 目录树内。

        参数：
            persist_dir: RAG 持久化根目录。
            relative_path: 相对路径字符串。

        返回值：
            解析后的绝对 ``Path``。

        异常：
            ValueError: 解析结果逃逸出允许的 persist 目录范围。
        """
        candidate = (persist_dir / relative_path).resolve()
        assets_root = self.root.resolve()
        if not str(candidate).startswith(str(assets_root.parent.resolve())):
            raise ValueError("非法资源路径")
        return candidate
