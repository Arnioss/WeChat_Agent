"""工具调用观测：写入 MySQL 统计表，并支持链式组合多个 observer。"""
from __future__ import annotations

import logging
from typing import Callable, Optional, Sequence

logger = logging.getLogger(__name__)

ToolObserver = Callable[[str, bool, float], None]


def tool_call_observer(tool_name: str, ok: bool, duration_ms: float) -> None:
    """功能：将单次工具调用写入 tool_call_stats 表。
    参数：
    - tool_name：工具名称。
    - ok：是否执行成功。
    - duration_ms：耗时毫秒数。
    返回值：
    - 无。
    """
    from db.tool_call_stats import record_tool_call

    record_tool_call(tool_name, ok=ok, duration_ms=duration_ms)


def chain_tool_observers(*callbacks: Optional[ToolObserver]) -> ToolObserver:
    """功能：组合多个 observer，依次调用且单个失败不影响后续与工具执行。
    参数：
    - callbacks：observer 回调序列，None 会被跳过。
    返回值：
    - ToolObserver：可用于 ToolRegistry.observer 的组合回调。
    """

    observers: Sequence[ToolObserver] = tuple(cb for cb in callbacks if cb is not None)

    def _observer(tool_name: str, ok: bool, duration_ms: float) -> None:
        """功能：依次调用组合 observer，单个失败仅记录警告不中断后续回调。
        参数：
        - tool_name：工具名称。
        - ok：是否执行成功。
        - duration_ms：耗时毫秒数。
        返回值：
        - 无。
        """
        for callback in observers:
            try:
                callback(tool_name, ok, duration_ms)
            except Exception as exc:
                logger.warning("工具 observer 执行失败 tool=%s: %s", tool_name, exc)

    return _observer
