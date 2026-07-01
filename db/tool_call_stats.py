"""MySQL：工具调用次数统计（按工具名聚合）。"""
from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List

from db.client import get_mysql_pool, load_mysql_config

logger = logging.getLogger(__name__)

_TABLE_READY = False
_TABLE_LOCK = threading.Lock()


def ensure_tool_call_stats_table() -> None:
    """功能：初始化 tool_call_stats 表，用于按工具名聚合调用次数与耗时。
    参数：
    - 无。
    返回值：
    - 无。
    异常：
    - RuntimeError：MySQL 未启用或连接池不可用时由底层抛出。
    """
    with get_mysql_pool().connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS tool_call_stats (
                    tool_name VARCHAR(128) NOT NULL PRIMARY KEY,
                    call_count BIGINT NOT NULL DEFAULT 0,
                    success_count BIGINT NOT NULL DEFAULT 0,
                    failure_count BIGINT NOT NULL DEFAULT 0,
                    total_duration_ms DOUBLE NOT NULL DEFAULT 0,
                    last_called_at TIMESTAMP NULL DEFAULT NULL,
                    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                    KEY idx_call_count (call_count),
                    KEY idx_last_called_at (last_called_at)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                """
            )


def _ensure_table_once() -> bool:
    """功能：线程安全地确保 tool_call_stats 表已创建（仅执行一次）。
    参数：
    - 无。
    返回值：
    - bool：MySQL 已启用且表就绪时返回 True，否则 False。
    """
    global _TABLE_READY
    if _TABLE_READY:
        return True
    cfg = load_mysql_config()
    if not cfg.get("enabled"):
        return False
    with _TABLE_LOCK:
        if _TABLE_READY:
            return True
        ensure_tool_call_stats_table()
        _TABLE_READY = True
    return True


def record_tool_call(tool_name: str, *, ok: bool = True, duration_ms: float = 0.0) -> None:
    """功能：记录一次工具调用，按工具名累加次数与耗时。
    参数：
    - tool_name：工具名称。
    - ok：是否执行成功。
    - duration_ms：耗时毫秒数。
    返回值：
    - 无。MySQL 未启用或写入失败时静默跳过，不影响工具执行。
    """
    name = str(tool_name or "").strip()
    if not name:
        return
    try:
        if not _ensure_table_once():
            return
        success_delta = 1 if ok else 0
        failure_delta = 0 if ok else 1
        duration = max(0.0, float(duration_ms or 0.0))
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO tool_call_stats (
                        tool_name,
                        call_count,
                        success_count,
                        failure_count,
                        total_duration_ms,
                        last_called_at
                    )
                    VALUES (%s, 1, %s, %s, %s, NOW())
                    ON DUPLICATE KEY UPDATE
                        call_count = call_count + 1,
                        success_count = success_count + VALUES(success_count),
                        failure_count = failure_count + VALUES(failure_count),
                        total_duration_ms = total_duration_ms + VALUES(total_duration_ms),
                        last_called_at = NOW()
                    """,
                    (name, success_delta, failure_delta, duration),
                )
    except Exception as exc:
        logger.warning("记录工具调用统计失败 tool=%s: %s", name, exc)


def list_tool_call_stats(*, order_by: str = "call_count") -> List[Dict[str, Any]]:
    """功能：查询各工具的调用统计汇总。
    参数：
    - order_by：排序字段，支持 call_count / last_called_at / tool_name。
    返回值：
    - List[Dict[str, Any]]：每行包含 tool_name、call_count、success_count、failure_count、
      total_duration_ms、avg_duration_ms、last_called_at、updated_at。
    """
    allowed = {
        "call_count": "call_count DESC",
        "last_called_at": "last_called_at DESC",
        "tool_name": "tool_name ASC",
    }
    order_clause = allowed.get(order_by, allowed["call_count"])
    if not _ensure_table_once():
        return []
    with get_mysql_pool().connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"""
                SELECT
                    tool_name,
                    call_count,
                    success_count,
                    failure_count,
                    total_duration_ms,
                    CASE
                        WHEN call_count > 0 THEN total_duration_ms / call_count
                        ELSE 0
                    END AS avg_duration_ms,
                    last_called_at,
                    updated_at
                FROM tool_call_stats
                ORDER BY {order_clause}
                """
            )
            return list(cursor.fetchall() or [])
