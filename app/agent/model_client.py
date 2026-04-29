import errno
import os
import re
import sys
import time
from typing import Callable, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


def safe_print(*args, **kwargs) -> None:
    """功能：安全打印输出，兼容 Windows 控制台编码与句柄异常。
    参数：
    - 无。
    返回值：
    - 无。
    """
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


class ModelClient:
    """功能：封装模型请求、流式输出与多 Key 重试策略。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self, *, model: str, observer: Optional[Callable[[str, float], None]] = None):
        """功能：加载模型配置与 API Key 列表，创建可轮换的 OpenAI 客户端集合。
        参数：
        - model：要调用的模型名称。
        - observer：可选观测回调，接收状态和耗时毫秒。
        返回值：
        - 无。初始化阶段若缺少 `OPENAI_BASE_URL` 或 API Key 会直接抛出配置错误。
        """
        self.model = model
        self.observer = observer
        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("缺少环境变量 OPENAI_BASE_URL，请在 .env 文件中设置。")
        self.api_keys = self._load_api_keys()
        self.clients = [OpenAI(base_url=base_url, api_key=key) for key in self.api_keys]
        self.active_client_idx = 0

    def _load_api_keys(self) -> List[str]:
        """功能：从环境变量加载并去重可用 API Key 列表。
        参数：
        - 无。
        返回值：
        - List[str]：按优先级排列的 API Key 列表。
        异常：
        - ValueError：未配置任何 API Key 时抛出。
        """
        load_dotenv()
        keys: List[str] = []
        primary = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        if primary:
            keys.append(primary)

        raw = (os.getenv("OPENROUTER_API_KEYS") or "").strip()
        if raw:
            parts = re.split(r"[,\n;]+", raw)
            for part in parts:
                key = (part or "").strip()
                if key and key not in keys:
                    keys.append(key)

        if not keys:
            raise ValueError("未找到 OPENROUTER_API_KEY / OPENROUTER_API_KEYS，请在 .env 文件中设置。")
        return keys

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

    def call_model(
        self,
        messages,
        *,
        stream_callback: Optional[Callable[[str, str], None]] = None,
        stop_event=None,
    ) -> str:
        """功能：发起模型流式请求并返回完整输出文本。
        参数：
        - messages：与模型交互的消息列表。
        - stream_callback：模型流式输出回调函数。
        - stop_event：用于中断模型调用的事件对象。
        返回值：
        - str：模型完整响应文本；超时中断时返回兜底 `<final_answer>`。
        """
        safe_print("\n\n正在请求模型，请稍等...")
        max_clients = max(1, len(self.clients))
        connect_retries = max(0, int(os.getenv("REACT_MODEL_CONNECT_RETRIES")))
        last_err: Optional[Exception] = None

        for attempt in range(max_clients):
            client_idx = (self.active_client_idx + attempt) % max_clients
            client = self.clients[client_idx]
            for connect_attempt in range(connect_retries + 1):
                try:
                    started_at = time.time()
                    safe_print(f"模型思考... (key {client_idx + 1}/{max_clients})")
                    stream = client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        stream=True,
                    )

                    content_parts = []
                    tags = {
                        "thought": ("<thought>", "</thought>"),
                        "final_answer": ("<final_answer>", "</final_answer>"),
                    }
                    display_buffer = ""
                    current_section = None
                    final_header_printed = False

                    for chunk in stream:
                        if stop_event is not None and stop_event.is_set():
                            break
                        try:
                            delta = chunk.choices[0].delta.content
                        except (AttributeError, IndexError):
                            delta = None
                        if delta:
                            content_parts.append(delta)
                            display_buffer += delta
                            while display_buffer:
                                if current_section is None:
                                    candidates = []
                                    for section, (open_tag, _) in tags.items():
                                        idx = display_buffer.find(open_tag)
                                        if idx != -1:
                                            candidates.append((idx, section))

                                    if not candidates:
                                        max_open_len = max(len(v[0]) for v in tags.values())
                                        keep_len = max(0, max_open_len - 1)
                                        if len(display_buffer) > keep_len:
                                            prefix = display_buffer[:-keep_len]
                                            display_buffer = display_buffer[-keep_len:]
                                            if prefix:
                                                safe_print(prefix, end="", flush=True)
                                                if stream_callback:
                                                    stream_callback("preamble", prefix)
                                        break

                                    start_idx, section = min(candidates, key=lambda x: x[0])
                                    open_tag = tags[section][0]
                                    if start_idx > 0:
                                        prefix = display_buffer[:start_idx]
                                        safe_print(prefix, end="", flush=True)
                                        if stream_callback:
                                            stream_callback("preamble", prefix)
                                    display_buffer = display_buffer[start_idx + len(open_tag):]
                                    current_section = section

                                    if current_section == "final_answer" and not final_header_printed:
                                        safe_print("\n思考结果:")
                                        final_header_printed = True
                                else:
                                    close_tag = tags[current_section][1]
                                    end_idx = display_buffer.find(close_tag)
                                    if end_idx == -1:
                                        keep_len = max(0, len(close_tag) - 1)
                                        safe_len = max(0, len(display_buffer) - keep_len)
                                        if safe_len > 0:
                                            safe_text = display_buffer[:safe_len]
                                            safe_print(safe_text, end="", flush=True)
                                            if stream_callback:
                                                stream_callback(current_section, safe_text)
                                            display_buffer = display_buffer[safe_len:]
                                        break

                                    if end_idx > 0:
                                        section_text = display_buffer[:end_idx]
                                        safe_print(section_text, end="", flush=True)
                                        if stream_callback:
                                            stream_callback(current_section, section_text)
                                    display_buffer = display_buffer[end_idx + len(close_tag):]
                                    current_section = None

                    safe_print("\n")
                    content = "".join(content_parts)
                    if stop_event is not None and stop_event.is_set():
                        if "<final_answer>" not in content or "</final_answer>" not in content:
                            content = "<final_answer>请求处理超时，请稍后再试。</final_answer>"
                    self.active_client_idx = client_idx
                    if self.observer:
                        self.observer("success", (time.time() - started_at) * 1000.0)
                    return content
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
                        safe_print(f"\n模型连接异常，准备重试... ({connect_attempt + 1}/{connect_retries})")
                        time.sleep(0.8 * (connect_attempt + 1))
                        continue
                    if self.is_rate_limit_error(e) and attempt < max_clients - 1:
                        safe_print(f"\n当前 key 触发 429，切换到下一个 key 重试... ({client_idx + 1}/{max_clients})")
                        break
                    if self.is_connection_error(e) and attempt < max_clients - 1:
                        safe_print(f"\n当前 key 连接异常，切换到下一个 key 重试... ({client_idx + 1}/{max_clients})")
                        break
                    if self.is_connection_error(e):
                        raise RuntimeError("模型服务连接异常，请稍后重试。") from e
                    raise

        if last_err:
            raise last_err
        raise RuntimeError("模型调用失败：无可用 API key")
