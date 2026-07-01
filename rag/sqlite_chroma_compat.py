"""在导入 ChromaDB 前确保 SQLite >= 3.35（部分 Linux 自带 Python 版本过旧）。"""
from __future__ import annotations

import sys


def ensure_chroma_sqlite() -> None:
    """功能：在导入 Chroma 之前确保运行时 SQLite 版本满足最低要求。
    参数：
    - 无。
    返回值：
    - 无。版本不足时尝试用 pysqlite3 替换 `sys.modules['sqlite3']`。
    异常：
    - RuntimeError：SQLite 过旧且未安装 pysqlite3-binary 时抛出。
    """
    import sqlite3

    if sqlite3.sqlite_version_info >= (3, 35, 0):
        return
    try:
        import pysqlite3 as _pysqlite3
    except ImportError as e:
        raise RuntimeError(
            "ChromaDB 需要 SQLite >= 3.35.0，当前 Python 使用 "
            f"{sqlite3.sqlite_version}。请安装: pip install pysqlite3-binary"
        ) from e
    sys.modules["sqlite3"] = _pysqlite3
