import asyncio
import errno
import io
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict
from typing import Callable, List, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from app.agent.llm_key_pool import (
    get_scheduler,
    iter_client_attempts,
    llm_async_slot,
    llm_sync_slot,
    load_api_keys_for_channel,
    log_llm_key_use,
    mark_key_rate_limited,
)


_ASYNC_OPENAI_CLIENTS: Dict[tuple[str, str], AsyncOpenAI] = {}
_SYNC_OPENAI_CLIENTS: Dict[tuple[str, str], OpenAI] = {}


def _shared_async_openai_client(*, base_url: str, api_key: str) -> AsyncOpenAI:
    """功能：按 base_url 与 api_key 复用 AsyncOpenAI 客户端，避免重复初始化 HTTP 栈。
    参数：
    - base_url：OpenAI 兼容 API 基址。
    - api_key：API 密钥。
    返回值：
    - AsyncOpenAI：缓存或新创建的异步客户端实例。
    """
    cache_key = (base_url, api_key)
    client = _ASYNC_OPENAI_CLIENTS.get(cache_key)
    if client is None:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        _ASYNC_OPENAI_CLIENTS[cache_key] = client
    return client


def _shared_sync_openai_client(*, base_url: str, api_key: str) -> OpenAI:
    """功能：按 base_url 与 api_key 复用 OpenAI 客户端，避免重复初始化 HTTP 栈。
    参数：
    - base_url：OpenAI 兼容 API 基址。
    - api_key：API 密钥。
    返回值：
    - OpenAI：缓存或新创建的同步客户端实例。
    """
    cache_key = (base_url, api_key)
    client = _SYNC_OPENAI_CLIENTS.get(cache_key)
    if client is None:
        client = OpenAI(base_url=base_url, api_key=api_key)
        _SYNC_OPENAI_CLIENTS[cache_key] = client
    return client


@dataclass(frozen=True)
class ToolCallModelResult:
    """功能：封装一次 tool_calls 模型调用的响应内容。
    参数：
    - 无。
    返回值：
    - 无。包含 assistant 文本、tool_calls 列表与 finish_reason。
    """
    content: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""


def safe_print(*args, **kwargs) -> None:
    """功能：安全打印输出，兼容 Windows 控制台编码与句柄异常。
    参数：
    - 无。
    返回值：
    - 无。
    """
    if kwargs.get("file") in (None, sys.stdout):
        kwargs = dict(kwargs)
        kwargs["file"] = getattr(sys, "__stdout__", None) or sys.stdout
    try:
        print(*args, **kwargs)
    except OSError as e:
        if getattr(e, "errno", None) == errno.EINVAL:
            return
        if getattr(e, "winerror", None) == 87:
            return
        raise
    except UnicodeEncodeError:
        try:
            enc = getattr(sys.stdout, "encoding", None) or "utf-8"
            safe_args = tuple(str(a).encode(enc, errors="replace").decode(enc) for a in args)
            print(*safe_args, **kwargs)
        except Exception:
            pass


_CONSOLE_LOGGER_NAME = "agent.console"


def _console_logger() -> logging.Logger:
    """功能：获取 Agent 控制台专用 logger 实例。
    参数：
    - 无。
    返回值：
    - logging.Logger：名为 agent.console 的 logger。
    """
    return logging.getLogger(_CONSOLE_LOGGER_NAME)


def log_print(*args, **kwargs) -> None:
    """功能：将 Agent 执行轨迹写入项目统一日志格式（与 logging_setup 一致）。
    参数：
    - 与 `print` 相同；`end`/`flush` 非默认时仍走 safe_print（流式片段）。
    返回值：
    - 无。
    """
    end = kwargs.get("end", "\n")
    flush = kwargs.get("flush", False)
    if end != "\n" or flush:
        safe_print(*args, **kwargs)
        return
    if not args:
        return
    buf = io.StringIO()
    print(*args, file=buf, end="")
    message = buf.getvalue().strip("\n")
    if not message:
        return
    for line in message.splitlines():
        text = line.rstrip()
        if text:
            _console_logger().info("%s", text)


def log_answer(message: str) -> None:
    """功能：记录最终回答，保留换行便于阅读。
    参数：
    - message：最终回答文本。
    返回值：
    - 无。空文本时不输出。
    """
    text = str(message or "").strip("\n")
    if not text:
        return
    log_print("思考结果:")
    for line in text.splitlines():
        body = line.rstrip()
        if body:
            log_print(body)


class ModelClient:
    """功能：封装模型请求、tool_calls 调用与多 Key 重试策略。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(
        self,
        *,
        model: str,
        observer: Optional[Callable[[str, float], None]] = None,
        key_channel: str = "default",
    ):
        """功能：加载模型配置与 API Key 列表，创建可轮换的 OpenAI 客户端集合。
        参数：
        - model：要调用的模型名称。
        - observer：可选观测回调，接收状态和耗时毫秒。
        - key_channel：API Key 分池渠道（wechat / web / default）。
        返回值：
        - 无。初始化阶段若缺少 `OPENAI_BASE_URL` 或 API Key 会直接抛出配置错误。
        """
        self.model = model
        self.observer = observer
        self.key_channel = (key_channel or "default").strip().lower() or "default"
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("缺少环境变量 OPENAI_BASE_URL，请在 .env 文件中设置。")
        self._base_url = base_url
        self.api_keys = load_api_keys_for_channel(self.key_channel)
        self.clients = [
            _shared_sync_openai_client(base_url=base_url, api_key=key) for key in self.api_keys
        ]

    @staticmethod
    def is_rate_limit_error(err: Exception) -> bool:
        """功能：判断异常是否属于限流错误（429）。
        参数：
        - err：捕获到的异常对象。
        返回值：
        - bool：命中限流特征时返回 True。
        """
        text = str(err or "")
        return any(
            marker in text
            for marker in ("RateLimitError", "Error code: 429", "429 Too Many Requests")
        )

    @staticmethod
    def is_connection_error(err: Exception) -> bool:
        """功能：判断异常是否属于网络连接类错误。
        参数：
        - err：捕获到的异常对象。
        返回值：
        - bool：命中连接异常特征时返回 True。
        """
        text = str(err or "")
        markers = (
            "APIConnectionError",
            "Connection error.",
            "ConnectError",
            "ReadTimeout",
            "UNEXPECTED_EOF_WHILE_READING",
            "EOF occurred in violation of protocol",
            "timed out",
        )
        return any(marker in text for marker in markers)

    def call_tool_call_model(self, messages, *, tools, stop_event=None) -> ToolCallModelResult:
        """功能：发起带 tools 的非流式模型请求并返回 tool_calls 结果。
        参数：
        - messages：与模型交互的消息列表。
        - tools：OpenAI tools schema 列表。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - ToolCallModelResult：包含 content、tool_calls 与 finish_reason；超时中断时 content 为超时提示。
        """
        log_print("正在请求模型响应，请稍等...")
        connect_retries = max(0, int(os.getenv("REACT_MODEL_CONNECT_RETRIES") or "0"))
        last_err: Optional[Exception] = None

        with llm_sync_slot():
            for channel, client_idx in iter_client_attempts(self.key_channel):
                scheduler = get_scheduler(channel)
                if client_idx >= len(scheduler.api_keys):
                    continue
                api_key = scheduler.api_keys[client_idx]
                client = _shared_sync_openai_client(base_url=self._base_url, api_key=api_key)
                pool_size = len(scheduler.api_keys)
                for connect_attempt in range(connect_retries + 1):
                    try:
                        started_at = time.time()
                        log_llm_key_use(
                            channel=channel,
                            key_idx=client_idx,
                            pool_size=pool_size,
                            purpose="tool_call",
                        )
                        log_print(f"模型响应... ({channel} key {client_idx + 1}/{pool_size})")
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=tools,
                            tool_choice="auto",
                            stream=False,
                        )
                        if stop_event is not None and stop_event.is_set():
                            return ToolCallModelResult(content="请求处理超时，请稍后再试。")
                        try:
                            choice = response.choices[0]
                            message = choice.message
                        except (AttributeError, IndexError):
                            return ToolCallModelResult()
                        result = ToolCallModelResult(
                            content=(getattr(message, "content", None) or ""),
                            tool_calls=self._normalize_tool_calls(getattr(message, "tool_calls", None)),
                            finish_reason=(getattr(choice, "finish_reason", None) or ""),
                        )
                        if self.observer:
                            self.observer("success", (time.time() - started_at) * 1000.0)
                        return result
                    except Exception as e:
                        last_err = e
                        if self.observer:
                            if self.is_rate_limit_error(e):
                                self.observer("rate_limit", 0.0)
                            elif self.is_connection_error(e):
                                self.observer("connection_error", 0.0)
                            else:
                                self.observer("failure", 0.0)
                        if self.is_connection_error(e) and connect_attempt < connect_retries:
                            log_print(f"\n模型连接异常，准备重试... ({connect_attempt + 1}/{connect_retries})")
                            time.sleep(0.8 * (connect_attempt + 1))
                            continue
                        if self.is_rate_limit_error(e):
                            mark_key_rate_limited(channel, client_idx)
                            log_print(
                                f"\n当前 key 触发 429，切换 key... ({channel} {client_idx + 1}/{pool_size})"
                            )
                            break
                        if self.is_connection_error(e):
                            log_print(
                                f"\n当前 key 连接异常，切换 key... ({channel} {client_idx + 1}/{pool_size})"
                            )
                            break
                        if self.is_connection_error(e):
                            raise RuntimeError("模型服务连接异常，请稍后重试。") from e
                        raise

        if last_err:
            raise last_err
        raise RuntimeError("模型调用失败：无可用 API key")

    def call_text_model(
        self,
        messages,
        *,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        purpose: str = "text",
    ) -> str:
        """功能：发起不带 tools 的纯文本模型请求并返回 content。
        参数：
        - messages：与模型交互的消息列表。
        - model：可选覆盖默认模型名。
        - timeout_seconds：可选请求超时秒数。
        - purpose：观测回调用途标识，默认 text。
        返回值：
        - str：模型返回的文本内容；响应异常时可能为空串。
        """
        selected_model = model or self.model
        connect_retries = max(0, int(os.getenv("REACT_MODEL_CONNECT_RETRIES") or "0"))
        last_err: Optional[Exception] = None

        with llm_sync_slot():
            for channel, client_idx in iter_client_attempts(self.key_channel):
                scheduler = get_scheduler(channel)
                if client_idx >= len(scheduler.api_keys):
                    continue
                api_key = scheduler.api_keys[client_idx]
                client = _shared_sync_openai_client(base_url=self._base_url, api_key=api_key)
                pool_size = len(scheduler.api_keys)
                for connect_attempt in range(connect_retries + 1):
                    started_at = time.time()
                    try:
                        log_llm_key_use(
                            channel=channel,
                            key_idx=client_idx,
                            pool_size=pool_size,
                            purpose=purpose,
                        )
                        kwargs: Dict[str, Any] = {
                            "model": selected_model,
                            "messages": messages,
                            "stream": False,
                        }
                        if timeout_seconds is not None and timeout_seconds > 0:
                            kwargs["timeout"] = timeout_seconds
                        response = client.chat.completions.create(**kwargs)
                        try:
                            choice = response.choices[0]
                            message = choice.message
                        except (AttributeError, IndexError):
                            return ""
                        if self.observer:
                            self.observer(f"{purpose}_success", (time.time() - started_at) * 1000.0)
                        return str(getattr(message, "content", None) or "")
                    except Exception as e:
                        last_err = e
                        if self.observer:
                            if self.is_rate_limit_error(e):
                                self.observer("rate_limit", 0.0)
                            elif self.is_connection_error(e):
                                self.observer("connection_error", 0.0)
                            else:
                                self.observer(f"{purpose}_failure", 0.0)
                        if self.is_connection_error(e) and connect_attempt < connect_retries:
                            time.sleep(0.8 * (connect_attempt + 1))
                            continue
                        if self.is_rate_limit_error(e):
                            mark_key_rate_limited(channel, client_idx)
                            break
                        if self.is_connection_error(e):
                            break
                        if self.is_connection_error(e):
                            raise RuntimeError("模型服务连接异常，请稍后重试。") from e
                        raise

        if last_err:
            raise last_err
        raise RuntimeError("模型调用失败：无可用 API key")

    def call_final_answer(self, messages, *, stop_event=None) -> str:
        """功能：发起最终回答生成请求（不带 tools），返回完整文本。
        参数：
        - messages：与模型交互的消息列表。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - str：模型最终回答文本；超时中断时返回超时提示，响应异常时可能为空串。
        """
        log_print("正在生成最终回答...")
        connect_retries = max(0, int(os.getenv("REACT_MODEL_CONNECT_RETRIES") or "0"))
        last_err: Optional[Exception] = None

        with llm_sync_slot():
            for channel, client_idx in iter_client_attempts(self.key_channel):
                scheduler = get_scheduler(channel)
                if client_idx >= len(scheduler.api_keys):
                    continue
                api_key = scheduler.api_keys[client_idx]
                client = _shared_sync_openai_client(base_url=self._base_url, api_key=api_key)
                pool_size = len(scheduler.api_keys)
                for connect_attempt in range(connect_retries + 1):
                    started_at = time.time()
                    try:
                        log_llm_key_use(
                            channel=channel,
                            key_idx=client_idx,
                            pool_size=pool_size,
                            purpose="final_answer",
                        )
                        log_print(f"最终回答生成... ({channel} key {client_idx + 1}/{pool_size})")
                        response = client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            stream=False,
                        )
                        if stop_event is not None and stop_event.is_set():
                            elapsed_ms = (time.time() - started_at) * 1000.0
                            log_print(
                                f"最终回答生成中止，耗时 {elapsed_ms:.0f} ms"
                                f" ({channel} key {client_idx + 1}/{pool_size}, status=stopped)"
                            )
                            return "请求处理超时，请稍后再试。"
                        try:
                            choice = response.choices[0]
                            message = choice.message
                        except (AttributeError, IndexError):
                            elapsed_ms = (time.time() - started_at) * 1000.0
                            log_print(
                                f"最终回答生成完成，耗时 {elapsed_ms:.0f} ms"
                                f" ({channel} key {client_idx + 1}/{pool_size}, status=empty_response)"
                            )
                            return ""
                        content = getattr(message, "content", None) or ""
                        elapsed_ms = (time.time() - started_at) * 1000.0
                        log_print(
                            f"最终回答生成完成，耗时 {elapsed_ms:.0f} ms"
                            f" ({channel} key {client_idx + 1}/{pool_size}, status=success)"
                        )
                        if self.observer:
                            self.observer("final_success", elapsed_ms)
                        return str(content)
                    except Exception as e:
                        last_err = e
                        elapsed_ms = (time.time() - started_at) * 1000.0
                        if self.observer:
                            if self.is_rate_limit_error(e):
                                self.observer("rate_limit", 0.0)
                            elif self.is_connection_error(e):
                                self.observer("connection_error", 0.0)
                            else:
                                self.observer("final_failure", 0.0)
                        if self.is_connection_error(e) and connect_attempt < connect_retries:
                            log_print(
                                f"\n最终回答连接异常，耗时 {elapsed_ms:.0f} ms，准备重试..."
                                f" ({channel} key {client_idx + 1}/{pool_size}, retry {connect_attempt + 1}/{connect_retries})"
                            )
                            time.sleep(0.8 * (connect_attempt + 1))
                            continue
                        if self.is_rate_limit_error(e):
                            mark_key_rate_limited(channel, client_idx)
                            log_print(
                                f"\n最终回答当前 key 触发 429，耗时 {elapsed_ms:.0f} ms，切换 key..."
                                f" ({channel} key {client_idx + 1}/{pool_size})"
                            )
                            break
                        if self.is_connection_error(e):
                            log_print(
                                f"\n最终回答当前 key 连接异常，耗时 {elapsed_ms:.0f} ms，切换 key..."
                                f" ({channel} key {client_idx + 1}/{pool_size})"
                            )
                            break
                        if self.is_connection_error(e):
                            log_print(
                                f"\n最终回答生成失败，耗时 {elapsed_ms:.0f} ms"
                                f" ({channel} key {client_idx + 1}/{pool_size}, status=connection_error)：{e}"
                            )
                            raise RuntimeError("模型服务连接异常，请稍后重试。") from e
                        log_print(
                            f"\n最终回答生成失败，耗时 {elapsed_ms:.0f} ms"
                            f" ({channel} key {client_idx + 1}/{pool_size}, status=failure)：{e}"
                        )
                        raise

        if last_err:
            raise last_err
        raise RuntimeError("模型最终回答调用失败：无可用 API key")

    @staticmethod
    def _normalize_tool_calls(raw_tool_calls) -> List[Dict[str, Any]]:
        """功能：将 SDK 返回的 tool_calls 统一规范化为字典列表。
        参数：
        - raw_tool_calls：模型响应中的原始 tool_calls。
        返回值：
        - List[Dict[str, Any]]：每项含 id、type、function.name 与 function.arguments。
        """
        if not raw_tool_calls:
            return []
        normalized: List[Dict[str, Any]] = []
        for call in raw_tool_calls:
            if isinstance(call, dict):
                item = dict(call)
                function = dict(item.get("function") or {})
                normalized.append(
                    {
                        "id": str(item.get("id") or ""),
                        "type": item.get("type") or "function",
                        "function": {
                            "name": str(function.get("name") or ""),
                            "arguments": function.get("arguments") or "{}",
                        },
                    }
                )
                continue
            model_dump = getattr(call, "model_dump", None)
            if callable(model_dump):
                item = model_dump()
                function = dict(item.get("function") or {})
                normalized.append(
                    {
                        "id": str(item.get("id") or ""),
                        "type": item.get("type") or "function",
                        "function": {
                            "name": str(function.get("name") or ""),
                            "arguments": function.get("arguments") or "{}",
                        },
                    }
                )
                continue
            function = getattr(call, "function", None)
            normalized.append(
                {
                    "id": str(getattr(call, "id", "") or ""),
                    "type": getattr(call, "type", None) or "function",
                    "function": {
                        "name": str(getattr(function, "name", "") or ""),
                        "arguments": getattr(function, "arguments", None) or "{}",
                    },
                }
            )
        return normalized


class AsyncModelClient:
    """功能：ModelClient 的全异步版本，基于 AsyncOpenAI，支持多用户并发。
    参数：
    - 无。
    返回值：
    - 无。所有模型调用均为 async def，重试时使用 await asyncio.sleep。
    """

    def __init__(
        self,
        *,
        model: str,
        observer: Optional[Callable[[str, float], None]] = None,
        key_channel: str = "default",
    ):
        """功能：加载模型配置与 API Key 列表，创建可轮换的 AsyncOpenAI 客户端集合。
        参数：
        - model：要调用的模型名称。
        - observer：可选观测回调，接收状态和耗时毫秒。
        - key_channel：API Key 分池渠道（wechat / web / default）。
        返回值：
        - 无。缺少 OPENAI_BASE_URL 或 API Key 时会抛出配置错误。
        """
        self.model = model
        self.observer = observer
        self.key_channel = (key_channel or "default").strip().lower() or "default"
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("缺少环境变量 OPENAI_BASE_URL，请在 .env 文件中设置。")
        self._base_url = base_url
        self.api_keys = load_api_keys_for_channel(self.key_channel)
        self.clients = [
            _shared_async_openai_client(base_url=base_url, api_key=key) for key in self.api_keys
        ]

    @staticmethod
    def is_rate_limit_error(err: Exception) -> bool:
        """功能：判断异常是否属于限流错误（429）。
        参数：
        - err：捕获到的异常对象。
        返回值：
        - bool：命中限流特征时返回 True。
        """
        return ModelClient.is_rate_limit_error(err)

    @staticmethod
    def is_connection_error(err: Exception) -> bool:
        """功能：判断异常是否属于网络连接类错误。
        参数：
        - err：捕获到的异常对象。
        返回值：
        - bool：命中连接异常特征时返回 True。
        """
        return ModelClient.is_connection_error(err)

    async def call_tool_call_model(self, messages, *, tools, stop_event=None) -> ToolCallModelResult:
        """功能：异步发起带 tools 的模型请求并返回 tool_calls 结果。
        参数：
        - messages：与模型交互的消息列表。
        - tools：OpenAI tools schema 列表。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - ToolCallModelResult：包含 content、tool_calls 与 finish_reason。
        """
        log_print("正在请求模型响应，请稍等...")
        connect_retries = max(0, int(os.getenv("REACT_MODEL_CONNECT_RETRIES") or "0"))
        last_err: Optional[Exception] = None

        async with llm_async_slot():
            for channel, client_idx in iter_client_attempts(self.key_channel):
                scheduler = get_scheduler(channel)
                if client_idx >= len(scheduler.api_keys):
                    continue
                api_key = scheduler.api_keys[client_idx]
                client = _shared_async_openai_client(base_url=self._base_url, api_key=api_key)
                pool_size = len(scheduler.api_keys)
                for connect_attempt in range(connect_retries + 1):
                    try:
                        started_at = time.time()
                        log_llm_key_use(
                            channel=channel,
                            key_idx=client_idx,
                            pool_size=pool_size,
                            purpose="tool_call",
                        )
                        log_print(f"模型响应... ({channel} key {client_idx + 1}/{pool_size})")
                        response = await client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            tools=tools,
                            tool_choice="auto",
                            stream=False,
                        )
                        if stop_event is not None and stop_event.is_set():
                            return ToolCallModelResult(content="请求处理超时，请稍后再试。")
                        try:
                            choice = response.choices[0]
                            message = choice.message
                        except (AttributeError, IndexError):
                            return ToolCallModelResult()
                        result = ToolCallModelResult(
                            content=(getattr(message, "content", None) or ""),
                            tool_calls=ModelClient._normalize_tool_calls(getattr(message, "tool_calls", None)),
                            finish_reason=(getattr(choice, "finish_reason", None) or ""),
                        )
                        if self.observer:
                            self.observer("success", (time.time() - started_at) * 1000.0)
                        return result
                    except Exception as e:
                        last_err = e
                        if self.observer:
                            if ModelClient.is_rate_limit_error(e):
                                self.observer("rate_limit", 0.0)
                            elif ModelClient.is_connection_error(e):
                                self.observer("connection_error", 0.0)
                            else:
                                self.observer("failure", 0.0)
                        if ModelClient.is_connection_error(e) and connect_attempt < connect_retries:
                            log_print(f"\n模型连接异常，准备重试... ({connect_attempt + 1}/{connect_retries})")
                            await asyncio.sleep(0.8 * (connect_attempt + 1))
                            continue
                        if ModelClient.is_rate_limit_error(e):
                            mark_key_rate_limited(channel, client_idx)
                            log_print(
                                f"\n当前 key 触发 429，切换 key... ({channel} {client_idx + 1}/{pool_size})"
                            )
                            break
                        if ModelClient.is_connection_error(e):
                            log_print(
                                f"\n当前 key 连接异常，切换 key... ({channel} {client_idx + 1}/{pool_size})"
                            )
                            break
                        if ModelClient.is_connection_error(e):
                            raise RuntimeError("模型服务连接异常，请稍后重试。") from e
                        raise

        if last_err:
            raise last_err
        raise RuntimeError("模型调用失败：无可用 API key")

    async def call_text_model(
        self,
        messages,
        *,
        model: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        purpose: str = "text",
    ) -> str:
        """功能：异步发起不带 tools 的纯文本模型请求并返回 content。
        参数：
        - messages：与模型交互的消息列表。
        - model：可选覆盖默认模型名。
        - timeout_seconds：可选请求超时秒数。
        - purpose：观测回调用途标识，默认 text。
        返回值：
        - str：模型返回的文本内容；响应异常时可能为空串。
        """
        selected_model = model or self.model
        connect_retries = max(0, int(os.getenv("REACT_MODEL_CONNECT_RETRIES") or "0"))
        last_err: Optional[Exception] = None

        async with llm_async_slot():
            for channel, client_idx in iter_client_attempts(self.key_channel):
                scheduler = get_scheduler(channel)
                if client_idx >= len(scheduler.api_keys):
                    continue
                api_key = scheduler.api_keys[client_idx]
                client = _shared_async_openai_client(base_url=self._base_url, api_key=api_key)
                pool_size = len(scheduler.api_keys)
                for connect_attempt in range(connect_retries + 1):
                    started_at = time.time()
                    try:
                        log_llm_key_use(
                            channel=channel,
                            key_idx=client_idx,
                            pool_size=pool_size,
                            purpose=purpose,
                        )
                        kwargs: Dict[str, Any] = {
                            "model": selected_model,
                            "messages": messages,
                            "stream": False,
                        }
                        if timeout_seconds is not None and timeout_seconds > 0:
                            kwargs["timeout"] = timeout_seconds
                        response = await client.chat.completions.create(**kwargs)
                        try:
                            choice = response.choices[0]
                            message = choice.message
                        except (AttributeError, IndexError):
                            return ""
                        if self.observer:
                            self.observer(f"{purpose}_success", (time.time() - started_at) * 1000.0)
                        return str(getattr(message, "content", None) or "")
                    except Exception as e:
                        last_err = e
                        if self.observer:
                            if ModelClient.is_rate_limit_error(e):
                                self.observer("rate_limit", 0.0)
                            elif ModelClient.is_connection_error(e):
                                self.observer("connection_error", 0.0)
                            else:
                                self.observer(f"{purpose}_failure", 0.0)
                        if ModelClient.is_connection_error(e) and connect_attempt < connect_retries:
                            await asyncio.sleep(0.8 * (connect_attempt + 1))
                            continue
                        if ModelClient.is_rate_limit_error(e):
                            mark_key_rate_limited(channel, client_idx)
                            break
                        if ModelClient.is_connection_error(e):
                            break
                        if ModelClient.is_connection_error(e):
                            raise RuntimeError("模型服务连接异常，请稍后重试。") from e
                        raise

        if last_err:
            raise last_err
        raise RuntimeError("模型调用失败：无可用 API key")

    async def call_final_answer(self, messages, *, stop_event=None) -> str:
        """功能：异步发起最终回答生成请求（不带 tools），返回完整文本。
        参数：
        - messages：与模型交互的消息列表。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - str：模型最终回答文本；超时中断时返回超时提示。
        """
        log_print("正在生成最终回答...")
        connect_retries = max(0, int(os.getenv("REACT_MODEL_CONNECT_RETRIES") or "0"))
        last_err: Optional[Exception] = None

        async with llm_async_slot():
            for channel, client_idx in iter_client_attempts(self.key_channel):
                scheduler = get_scheduler(channel)
                if client_idx >= len(scheduler.api_keys):
                    continue
                api_key = scheduler.api_keys[client_idx]
                client = _shared_async_openai_client(base_url=self._base_url, api_key=api_key)
                pool_size = len(scheduler.api_keys)
                for connect_attempt in range(connect_retries + 1):
                    started_at = time.time()
                    try:
                        log_llm_key_use(
                            channel=channel,
                            key_idx=client_idx,
                            pool_size=pool_size,
                            purpose="final_answer",
                        )
                        log_print(f"最终回答生成... ({channel} key {client_idx + 1}/{pool_size})")
                        response = await client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            stream=False,
                        )
                        if stop_event is not None and stop_event.is_set():
                            elapsed_ms = (time.time() - started_at) * 1000.0
                            log_print(
                                f"最终回答生成中止，耗时 {elapsed_ms:.0f} ms"
                                f" ({channel} key {client_idx + 1}/{pool_size}, status=stopped)"
                            )
                            return "请求处理超时，请稍后再试。"
                        try:
                            choice = response.choices[0]
                            message = choice.message
                        except (AttributeError, IndexError):
                            elapsed_ms = (time.time() - started_at) * 1000.0
                            log_print(
                                f"最终回答生成完成，耗时 {elapsed_ms:.0f} ms"
                                f" ({channel} key {client_idx + 1}/{pool_size}, status=empty_response)"
                            )
                            return ""
                        content = getattr(message, "content", None) or ""
                        elapsed_ms = (time.time() - started_at) * 1000.0
                        log_print(
                            f"最终回答生成完成，耗时 {elapsed_ms:.0f} ms"
                            f" ({channel} key {client_idx + 1}/{pool_size}, status=success)"
                        )
                        if self.observer:
                            self.observer("final_success", elapsed_ms)
                        return str(content)
                    except Exception as e:
                        last_err = e
                        elapsed_ms = (time.time() - started_at) * 1000.0
                        if self.observer:
                            if ModelClient.is_rate_limit_error(e):
                                self.observer("rate_limit", 0.0)
                            elif ModelClient.is_connection_error(e):
                                self.observer("connection_error", 0.0)
                            else:
                                self.observer("final_failure", 0.0)
                        if ModelClient.is_connection_error(e) and connect_attempt < connect_retries:
                            log_print(
                                f"\n最终回答连接异常，耗时 {elapsed_ms:.0f} ms，准备重试..."
                                f" ({channel} key {client_idx + 1}/{pool_size}, retry {connect_attempt + 1}/{connect_retries})"
                            )
                            await asyncio.sleep(0.8 * (connect_attempt + 1))
                            continue
                        if ModelClient.is_rate_limit_error(e):
                            mark_key_rate_limited(channel, client_idx)
                            log_print(
                                f"\n最终回答当前 key 触发 429，耗时 {elapsed_ms:.0f} ms，切换 key..."
                                f" ({channel} key {client_idx + 1}/{pool_size})"
                            )
                            break
                        if ModelClient.is_connection_error(e):
                            log_print(
                                f"\n最终回答当前 key 连接异常，耗时 {elapsed_ms:.0f} ms，切换 key..."
                                f" ({channel} key {client_idx + 1}/{pool_size})"
                            )
                            break
                        if ModelClient.is_connection_error(e):
                            log_print(
                                f"\n最终回答生成失败，耗时 {elapsed_ms:.0f} ms"
                                f" ({channel} key {client_idx + 1}/{pool_size}, status=connection_error)：{e}"
                            )
                            raise RuntimeError("模型服务连接异常，请稍后重试。") from e
                        log_print(
                            f"\n最终回答生成失败，耗时 {elapsed_ms:.0f} ms"
                            f" ({channel} key {client_idx + 1}/{pool_size}, status=failure)：{e}"
                        )
                        raise

        if last_err:
            raise last_err
        raise RuntimeError("模型最终回答调用失败：无可用 API key")
