"""API Key 分 channel 池、round-robin 与 LLM 全局并发控制。"""
from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Callable, Dict, List, Literal, Optional, Sequence

from dotenv import load_dotenv

KeyChannel = Literal["wechat", "web", "default"]

_LLM_LOGGER = logging.getLogger("app.llm")
_SCHEDULER_LOCK = threading.Lock()
_SCHEDULERS: Dict[str, "ChannelKeyScheduler"] = {}

_LLM_ASYNC_SEMAPHORE: Optional[asyncio.Semaphore] = None
_LLM_SYNC_SEMAPHORE: Optional[threading.Semaphore] = None
_LLM_INFLIGHT = 0
_LLM_INFLIGHT_LOCK = threading.Lock()

_llm_queue_wait_callback: Optional[Callable[[float], None]] = None
_llm_queue_wait_var: contextvars.ContextVar[Optional[Callable[[float], None]]] = contextvars.ContextVar(
    "llm_queue_wait_cb",
    default=None,
)
_metrics_hook: Optional[Callable[[str, float], None]] = None

_FALLBACK_MAP: Dict[str, str] = {
    "wechat": "web",
    "web": "wechat",
    "default": "default",
}


def set_llm_queue_wait_callback(callback: Optional[Callable[[float], None]]) -> None:
    """功能：注册全局 LLM 排队等待回调（等待 ≥0.5s 时触发）。
    参数：
    - callback：接收 wait_s 的回调；None 表示清除。
    返回值：
    - 无。
    """
    global _llm_queue_wait_callback
    _llm_queue_wait_callback = callback


def llm_queue_wait_context(callback: Optional[Callable[[float], None]]):
    """功能：在 async 任务级设置排队等待回调（contextvar）。
    参数：
    - callback：任务内优先于全局回调使用的 wait 通知函数。
    返回值：
    - contextvars.Token：用于 reset_llm_queue_wait_context 还原。
    """
    return _llm_queue_wait_var.set(callback)


def reset_llm_queue_wait_context(token: contextvars.Token) -> None:
    """功能：还原 llm_queue_wait_context 设置的 contextvar。
    参数：
    - token：llm_queue_wait_context 返回的 Token。
    返回值：
    - 无。
    """
    _llm_queue_wait_var.reset(token)


def _fire_queue_wait_callback(wait_s: float) -> None:
    """功能：排队等待 ≥0.5s 时触发 contextvar 或全局 LLM 排队回调。
    参数：
    - wait_s：获取并发槽位的等待秒数。
    返回值：
    - 无。
    """
    if wait_s < 0.5:
        return
    cb = _llm_queue_wait_var.get()
    if cb is None:
        cb = _llm_queue_wait_callback
    if cb:
        cb(wait_s)


def set_llm_metrics_hook(hook: Optional[Callable[[str, float], None]]) -> None:
    """功能：注册 LLM 指标钩子，用于 inc/gauge/observe 等观测。
    参数：
    - hook：接收 (event_name, value) 的回调；None 表示清除。
    返回值：
    - 无。
    """
    global _metrics_hook
    _metrics_hook = hook


def _metric_inc(name: str, value: float = 1.0) -> None:
    """功能：通过已注册指标钩子递增计数器。
    参数：
    - name：指标名称。
    - value：递增值，默认 1.0。
    返回值：
    - 无。
    """
    if _metrics_hook:
        _metrics_hook(f"inc:{name}", value)


def _metric_gauge(name: str, value: float) -> None:
    """功能：通过已注册指标钩子设置仪表盘数值。
    参数：
    - name：指标名称。
    - value：最新指标值。
    返回值：
    - 无。
    """
    if _metrics_hook:
        _metrics_hook(f"gauge:{name}", value)


def _metric_observe_ms(name: str, value_ms: float) -> None:
    """功能：通过已注册指标钩子记录毫秒级耗时观测值。
    参数：
    - name：指标前缀名称。
    - value_ms：本次耗时（毫秒）。
    返回值：
    - 无。
    """
    if _metrics_hook:
        _metrics_hook(f"observe_ms:{name}", value_ms)


def _parse_key_list(raw: str) -> List[str]:
    """功能：从逗号/换行/分号分隔的字符串解析 API Key 列表并去重。
    参数：
    - raw：原始 Key 列表字符串。
    返回值：
    - List[str]：去重后的非空 Key 列表。
    """
    keys: List[str] = []
    for part in re.split(r"[,\n;]+", raw or ""):
        key = (part or "").strip()
        if key and key not in keys:
            keys.append(key)
    return keys


def load_default_api_keys() -> List[str]:
    """功能：从 OPENROUTER_API_KEY / OPENROUTER_API_KEYS 加载默认 API Key 列表。
    参数：
    - 无。
    返回值：
    - List[str]：去重后的 Key 列表。
    异常：
    - ValueError：未配置任何 Key 时抛出。
    """
    load_dotenv()
    keys: List[str] = []
    primary = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if primary:
        keys.append(primary)
    keys.extend(_parse_key_list(os.getenv("OPENROUTER_API_KEYS") or ""))
    deduped: List[str] = []
    for key in keys:
        if key not in deduped:
            deduped.append(key)
    if not deduped:
        raise ValueError("未找到 OPENROUTER_API_KEY / OPENROUTER_API_KEYS，请在 .env 文件中设置。")
    return deduped


def load_api_keys_for_channel(channel: str) -> List[str]:
    """功能：按 channel 加载 API Key 池；未配置 channel 专用 Key 时回退默认池。
    参数：
    - channel：wechat / web / default。
    返回值：
    - List[str]：该 channel 可用的 Key 列表。
    """
    load_dotenv()
    channel = (channel or "default").strip().lower() or "default"
    if channel == "wechat":
        keys = _parse_key_list(os.getenv("OPENROUTER_API_KEYS_WECHAT") or "")
    elif channel == "web":
        keys = _parse_key_list(os.getenv("OPENROUTER_API_KEYS_WEB") or "")
    else:
        keys = []
    if keys:
        return keys
    return load_default_api_keys()


def channel_fallback_enabled() -> bool:
    """功能：读取 LLM_KEY_CHANNEL_FALLBACK 是否启用跨 channel Key 回退。
    参数：
    - 无。
    返回值：
    - bool：启用为 True。
    """
    raw = (os.getenv("LLM_KEY_CHANNEL_FALLBACK") or "1").strip().lower()
    return raw in ("1", "true", "yes", "on")


def channels_for_attempt(primary: str) -> List[str]:
    """功能：生成 Key 尝试顺序（主 channel，必要时追加 fallback channel）。
    参数：
    - primary：主 channel 名。
    返回值：
    - List[str]：按尝试顺序排列的 channel 列表。
    """
    primary = (primary or "default").strip().lower() or "default"
    if not channel_fallback_enabled():
        return [primary]
    fallback = _FALLBACK_MAP.get(primary, "default")
    if fallback == primary:
        return [primary]
    return [primary, fallback]


def key_cooldown_seconds() -> float:
    """功能：读取 429 后 Key 冷却秒数（LLM_KEY_COOLDOWN_SECONDS，最小 1 秒）。
    参数：
    - 无。
    返回值：
    - float：冷却时长秒数。
    """
    return max(1.0, float(os.getenv("LLM_KEY_COOLDOWN_SECONDS", "45")))


def llm_max_inflight() -> int:
    """功能：读取 LLM 全局最大并发数（LLM_MAX_INFLIGHT，最小 1）。
    参数：
    - 无。
    返回值：
    - int：并发上限。
    """
    return max(1, int(os.getenv("LLM_MAX_INFLIGHT", "12")))


class ChannelKeyScheduler:
    """功能：单 channel 内 API Key 的 round-robin 调度与 429 冷却管理。
    参数：
    - 无（通过 __init__ 注入 channel 与 api_keys）。
    返回值：
    - 无。
    """

    def __init__(self, channel: str, api_keys: Sequence[str]):
        """功能：初始化指定 channel 的 Key 调度器。
        参数：
        - channel：channel 名称。
        - api_keys：该 channel 的 Key 列表。
        返回值：
        - 无。
        """
        self.channel = channel
        self.api_keys = list(api_keys)
        self._lock = threading.Lock()
        self._next_idx = 0
        self._cooldown_until: Dict[int, float] = {}

    def iter_indices(self) -> List[int]:
        """功能：按 round-robin 返回 Key 索引尝试顺序，优先未冷却 Key。
        参数：
        - 无。
        返回值：
        - List[int]：Key 索引列表；无 Key 时返回 []。
        """
        n = len(self.api_keys)
        if n == 0:
            return []
        with self._lock:
            start = self._next_idx
            self._next_idx = (start + 1) % n
        now = time.time()
        ordered = [(start + i) % n for i in range(n)]
        available = [i for i in ordered if self._cooldown_until.get(i, 0.0) <= now]
        cooled = [i for i in ordered if i not in available]
        return available + cooled

    def mark_rate_limited(self, idx: int) -> None:
        """功能：将指定 Key 标记为 429 限流，进入冷却期。
        参数：
        - idx：Key 在 api_keys 中的索引。
        返回值：
        - 无。
        """
        with self._lock:
            self._cooldown_until[idx] = time.time() + key_cooldown_seconds()
        _metric_inc("llm_key_429_total")


def get_scheduler(channel: str) -> ChannelKeyScheduler:
    """功能：获取或创建指定 channel 的全局 Key 调度器单例。
    参数：
    - channel：channel 名称。
    返回值：
    - ChannelKeyScheduler：该 channel 的调度器实例。
    """
    channel = (channel or "default").strip().lower() or "default"
    with _SCHEDULER_LOCK:
        scheduler = _SCHEDULERS.get(channel)
        if scheduler is None:
            keys = load_api_keys_for_channel(channel)
            scheduler = ChannelKeyScheduler(channel, keys)
            _SCHEDULERS[channel] = scheduler
        return scheduler


def iter_client_attempts(key_channel: str) -> List[tuple[str, int]]:
    """功能：生成 LLM 请求的 (channel, key_index) 尝试顺序。
    参数：
    - key_channel：主 channel；启用 fallback 时可能追加备用 channel。
    返回值：
    - List[tuple[str, int]]：去重后的尝试序列。
    """
    attempts: List[tuple[str, int]] = []
    seen: set[tuple[str, int]] = set()
    for channel in channels_for_attempt(key_channel):
        scheduler = get_scheduler(channel)
        for idx in scheduler.iter_indices():
            item = (channel, idx)
            if item not in seen:
                seen.add(item)
                attempts.append(item)
    return attempts


def mark_key_rate_limited(channel: str, idx: int) -> None:
    """功能：将指定 channel 的 Key 标记为 429 限流。
    参数：
    - channel：channel 名称。
    - idx：Key 索引。
    返回值：
    - 无。
    """
    get_scheduler(channel).mark_rate_limited(idx)


def log_llm_key_use(*, channel: str, key_idx: int, pool_size: int, purpose: str) -> None:
    """功能：记录本次 LLM 请求使用的 channel 与 Key 序号。
    参数：
    - channel：channel 名称。
    - key_idx：Key 索引（0-based）。
    - pool_size：该 channel Key 池大小。
    - purpose：调用用途标识（如 context_selector、agent）。
    返回值：
    - 无。
    """
    _LLM_LOGGER.info(
        "LLM 请求 channel=%s key=%s/%s purpose=%s",
        channel,
        key_idx + 1,
        pool_size,
        purpose,
    )


def _adjust_inflight(delta: int) -> None:
    """功能：调整 LLM 全局在途请求计数并上报 gauge 指标。
    参数：
    - delta：在途计数增量（正数增加、负数减少）。
    返回值：
    - 无。
    """
    global _LLM_INFLIGHT
    with _LLM_INFLIGHT_LOCK:
        _LLM_INFLIGHT = max(0, _LLM_INFLIGHT + delta)
        _metric_gauge("llm_inflight", float(_LLM_INFLIGHT))


def _get_sync_semaphore() -> threading.Semaphore:
    """功能：懒创建并返回同步 LLM 全局并发信号量。
    参数：
    - 无。
    返回值：
    - threading.Semaphore：并发上限由 llm_max_inflight() 决定。
    """
    global _LLM_SYNC_SEMAPHORE
    if _LLM_SYNC_SEMAPHORE is None:
        _LLM_SYNC_SEMAPHORE = threading.Semaphore(llm_max_inflight())
    return _LLM_SYNC_SEMAPHORE


def _get_async_semaphore() -> asyncio.Semaphore:
    """功能：懒创建并返回异步 LLM 全局并发信号量。
    参数：
    - 无。
    返回值：
    - asyncio.Semaphore：并发上限由 llm_max_inflight() 决定。
    """
    global _LLM_ASYNC_SEMAPHORE
    if _LLM_ASYNC_SEMAPHORE is None:
        _LLM_ASYNC_SEMAPHORE = asyncio.Semaphore(llm_max_inflight())
    return _LLM_ASYNC_SEMAPHORE


@contextmanager
def llm_sync_slot():
    """功能：同步 LLM 请求的全局并发槽位上下文管理器；yield 排队等待毫秒数。
    参数：
    - 无。
    返回值：
    - 上下文管理器，yield wait_ms。
    """
    sem = _get_sync_semaphore()
    wait_started = time.monotonic()
    sem.acquire()
    wait_s = time.monotonic() - wait_started
    wait_ms = wait_s * 1000.0
    if wait_ms > 0:
        _metric_observe_ms("llm_queue_wait", wait_ms)
    if wait_s >= 0.5:
        _fire_queue_wait_callback(wait_s)
    _adjust_inflight(1)
    try:
        yield wait_ms
    finally:
        _adjust_inflight(-1)
        sem.release()


@asynccontextmanager
async def llm_async_slot():
    """功能：异步 LLM 请求的全局并发槽位上下文管理器；yield 排队等待毫秒数。
    参数：
    - 无。
    返回值：
    - 异步上下文管理器，yield wait_ms。
    """
    sem = _get_async_semaphore()
    wait_started = time.monotonic()
    await sem.acquire()
    wait_s = time.monotonic() - wait_started
    wait_ms = wait_s * 1000.0
    if wait_ms > 0:
        _metric_observe_ms("llm_queue_wait", wait_ms)
    if wait_s >= 0.5:
        _fire_queue_wait_callback(wait_s)
    _adjust_inflight(1)
    try:
        yield wait_ms
    finally:
        _adjust_inflight(-1)
        sem.release()
