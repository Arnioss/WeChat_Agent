"""RAG 模块诊断日志输出。"""
from __future__ import annotations

import errno
import os
import sys
from datetime import datetime


def rag_verbose_enabled() -> bool:
    """功能：判断 RAG 详细日志是否开启。
    参数：
    - 无。
    返回值：
    - bool：`RAG_VERBOSE` 为 1/true/yes/on 时返回 True。
    """
    return (os.getenv("RAG_VERBOSE") or "").strip().lower() in ("1", "true", "yes", "on")


def rag_log(*args, **kwargs) -> None:
    """功能：在启用详细模式时向 stderr 打印 RAG 诊断信息。
    参数：
    - args、kwargs：与内置 `print` 相同。
    返回值：
    - 无。未开启 `RAG_VERBOSE` 时不输出。
    """
    if not rag_verbose_enabled():
        return
    kwargs.setdefault("flush", True)
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def rag_trace_enabled() -> bool:
    """功能：判断 RAG 检索链路简要日志是否开启（与 RAG_VERBOSE 无关）。
    参数：
    - 无。
    返回值：
    - bool：`RAG_TRACE_LOG` 未设置时默认 True；为 1/true/yes/on 时返回 True。
    """
    value = (os.getenv("RAG_TRACE_LOG") or "").strip().lower()
    if not value:
        return True
    return value in ("1", "true", "yes", "on")


def _console_timestamp() -> str:
    """功能：生成控制台日志用的时间戳字符串（毫秒精度）。
    参数：
    - 无。
    返回值：
    - str：格式为 `YYYY-MM-DD HH:MM:SS.mmm` 的本地时间。
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def rag_trace_log(message: str) -> None:
    """功能：输出 RAG 检索链路日志，优先写入 agent.console 日志，否则打印到 stdout。
    参数：
    - message：日志正文；空白时忽略。
    返回值：
    - 无。未开启 `RAG_TRACE_LOG` 时不输出。
    """
    if not rag_trace_enabled():
        return
    text = (message or "").strip()
    if not text:
        return
    line = f"[{_console_timestamp()}] - {text}"
    try:
        import logging

        console_logger = logging.getLogger("agent.console")
        if console_logger.handlers or logging.getLogger().handlers:
            console_logger.info("%s", line)
            return
    except Exception:
        pass
    try:
        print(line, flush=True)
    except OSError as exc:
        if getattr(exc, "errno", None) == errno.EINVAL:
            return
        if getattr(exc, "winerror", None) == 87:
            return
        raise
