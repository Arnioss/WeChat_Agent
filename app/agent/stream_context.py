"""线程本地 stream emitter 上下文绑定。"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Callable, Iterator, Optional

StreamCallback = Callable[[str, str], None]

_emitter_var: ContextVar[Optional[StreamCallback]] = ContextVar("stream_emitter", default=None)


def get_stream_emitter() -> Optional[StreamCallback]:
    """功能：获取当前上下文绑定的流式事件回调。
    参数：
    - 无。
    返回值：
    - Optional[StreamCallback]：已绑定的 `(event_kind, text)` 回调；未绑定时返回 None。
    """
    return _emitter_var.get()


@contextmanager
def bind_stream_emitter(emitter: Optional[StreamCallback]) -> Iterator[None]:
    """功能：在上下文管理器作用域内绑定流式事件回调。
    参数：
    - emitter：要绑定的流式回调；可为 None 表示清除绑定。
    返回值：
    - Iterator[None]：上下文管理器，退出时自动恢复先前绑定。
    """
    token = _emitter_var.set(emitter)
    try:
        yield
    finally:
        _emitter_var.reset(token)
