"""Ensure SQLite >= 3.35 before ChromaDB imports (old Linux glibc Python builds)."""
from __future__ import annotations

import sys


def ensure_chroma_sqlite() -> None:
    """功能：在导入 Chroma 之前确保运行时 SQLite 版本满足最低要求。
    参数：
    - 无。
    返回值：
    - 无。
    """
    import sqlite3

    if sqlite3.sqlite_version_info >= (3, 35, 0):
        return
    try:
        import pysqlite3 as _pysqlite3
    except ImportError as e:
        raise RuntimeError(
            "ChromaDB needs SQLite >= 3.35.0; this Python is using "
            f"{sqlite3.sqlite_version}. Install: pip install pysqlite3-binary"
        ) from e
    sys.modules["sqlite3"] = _pysqlite3
