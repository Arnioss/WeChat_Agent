"""RAG HTTP 网关 API Key 池：加载 OPENROUTER_API_KEYS，429 时跨嵌入/Rerank/识图共享冷却。"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Dict, List

from rag.logging_utils import rag_log

_COOLDOWN_UNTIL: Dict[str, float] = {}
_ROUND_ROBIN_LOCK = threading.Lock()
_ROUND_ROBIN_START = 0


def is_rate_limit_error(err: Exception) -> bool:
    """功能：根据异常消息判断是否为 HTTP 429 限流错误。
    参数：
    - err：捕获的异常实例。
    返回值：
    - bool：消息含 RateLimitError 或 429 相关标记时返回 True。
    """
    text = str(err or "")
    return any(
        marker in text
        for marker in (
            "RateLimitError",
            "Error code: 429",
            "429 Too Many Requests",
            "429 Client Error",
            "TOO MANY REQUESTS",
        )
    )


def load_rag_api_keys(*, fallback_key: str = "") -> List[str]:
    """功能：从环境变量加载 OpenRouter API Key 列表并去重。
    参数：
    - fallback_key：未配置环境变量时使用的备用 Key。
    返回值：
    - List[str]：优先 `OPENROUTER_API_KEY`，再合并 `OPENROUTER_API_KEYS` 中各项。
    """
    keys: List[str] = []
    primary = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if primary:
        keys.append(primary)
    raw = (os.getenv("OPENROUTER_API_KEYS") or "").strip()
    if raw:
        for part in re.split(r"[,\n;]+", raw):
            key = (part or "").strip()
            if key and key not in keys:
                keys.append(key)
    if not keys and fallback_key:
        fallback = fallback_key.strip()
        if fallback:
            keys.append(fallback)
    return keys


def key_cooldown_seconds() -> float:
    """功能：读取 Key 触发 429 后的冷却时长（秒）。
    参数：
    - 无。
    返回值：
    - float：`LLM_KEY_COOLDOWN_SECONDS` 解析值，至少为 1.0 秒。
    """
    return max(1.0, float(os.getenv("LLM_KEY_COOLDOWN_SECONDS", "45")))


class RagHttpKeyPool:
    """功能：RAG 侧 HTTP 请求共用的 Key 列表与 429 冷却（按 Key 字符串全局共享）。
    参数：
    - 无（实例字段由构造器或 `from_env` 填充）。
    返回值：
    - 无。提供轮询索引、Bearer 令牌与限流标记能力。
    """

    def __init__(self, api_keys: List[str], *, service: str = "RAG"):
        """功能：构造 Key 池实例。
        参数：
        - api_keys：可用 API Key 列表。
        - service：服务名称，用于日志前缀。
        返回值：
        - 无。
        """
        self.api_keys = list(api_keys)
        self.service = service

    @classmethod
    def from_env(cls, *, fallback_key: str = "", service: str = "RAG") -> "RagHttpKeyPool":
        """功能：从环境变量加载 Key 并构造池；多 Key 时输出轮换提示。
        参数：
        - fallback_key：环境变量未配置时的备用 Key。
        - service：服务名称，用于日志前缀。
        返回值：
        - RagHttpKeyPool：已加载 Key 的池实例。
        """
        keys = load_rag_api_keys(fallback_key=fallback_key)
        pool = cls(keys, service=service)
        if len(keys) > 1:
            rag_log(f"[RAG] {service} API Key 池：共 {len(keys)} 把，429 时自动轮换。", flush=True)
        return pool

    def __len__(self) -> int:
        """功能：返回池中 Key 的数量。
        参数：
        - 无。
        返回值：
        - int：Key 个数。
        """
        return len(self.api_keys)

    def bearer(self, idx: int) -> str:
        """功能：按索引返回 Bearer 令牌字符串。
        参数：
        - idx：Key 在池中的下标。
        返回值：
        - str：对应 API Key。
        """
        return self.api_keys[idx]

    def iter_indices(self) -> List[int]:
        """功能：按全局轮询起点返回 Key 索引，未冷却的优先。
        参数：
        - 无。
        返回值：
        - List[int]：可用索引在前、仍在冷却的索引在后的列表；无 Key 时为空列表。
        """
        global _ROUND_ROBIN_START
        n = len(self.api_keys)
        if n == 0:
            return []
        with _ROUND_ROBIN_LOCK:
            start = _ROUND_ROBIN_START
            _ROUND_ROBIN_START = (start + 1) % n
        now = time.time()
        ordered = [(start + i) % n for i in range(n)]
        available = [
            i for i in ordered if _COOLDOWN_UNTIL.get(self.api_keys[i], 0.0) <= now
        ]
        cooled = [i for i in ordered if i not in available]
        return available + cooled

    def mark_rate_limited(self, idx: int) -> None:
        """功能：将指定 Key 标记为限流，进入全局冷却窗口。
        参数：
        - idx：触发 429 的 Key 下标。
        返回值：
        - 无。
        """
        _COOLDOWN_UNTIL[self.api_keys[idx]] = time.time() + key_cooldown_seconds()

    def log_switch(self, idx: int) -> None:
        """功能：输出 Key 因 429 切换的诊断日志。
        参数：
        - idx：当前触发限流的 Key 下标。
        返回值：
        - 无。
        """
        rag_log(
            f"[RAG] {self.service} key {idx + 1}/{len(self.api_keys)} 触发 429，切换 key...",
            flush=True,
        )
