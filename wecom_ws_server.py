"""企业微信长连接机器人服务入口。"""

import asyncio
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

from aibot import WSClient, WSClientOptions

from agent import ReActAgent, warm_agent
from tools import warm_mcp_tools
from app.application.conversation.services import ChatStore, DedupService, SessionManager, StreamManager
from app.agent.model_client import log_answer, log_print, safe_print
from app.infrastructure.cache import RedisCache
from app.infrastructure.logging_setup import configure_project_logging
from app.infrastructure.observability import AppMetrics, bind_llm_metrics, log_event, log_timing
from app.infrastructure.tool_call_recorder import chain_tool_observers, tool_call_observer
from app.skills.system import build_skill_system
from app.agent.kb_body_images import KbImageStreamSplitter, split_kb_image_markers, strip_kb_image_markers
from wecom.aibot_logging import build_aibot_logger
from wecom.image_delivery import WecomImageDelivery
from wecom.message_parsers import extract_session_id, extract_user_id


ROOT_DIR = Path(__file__).resolve().parent
logger = configure_project_logging(ROOT_DIR, logger_name=__name__, log_basename="wechat_server")
metrics = AppMetrics()
bind_llm_metrics(metrics)
WECOM_EMPTY_THINK = "<think></think>"

MODEL_NAME = os.getenv("OPENROUTER_MODEL")
PROJECT_DIR = str(ROOT_DIR)
SKILL_SYSTEM = build_skill_system(project_directory=PROJECT_DIR)

SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "1800"))
STREAM_TTL_SECONDS = int(os.getenv("STREAM_TTL_SECONDS", "600"))
WECHAT_MAX_PROCESS_SECONDS = float(os.getenv("WECHAT_MAX_PROCESS_SECONDS", "180"))
WECHAT_LOG_MESSAGE_CONTENT = (os.getenv("WECHAT_LOG_MESSAGE_CONTENT") or "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
WELCOME_TEXT = os.getenv("WECHAT_ROBOT_WELCOME_TEXT", "欢迎进入智能助手。你可以直接提问，我会用流式消息回复你。")

WECHAT_BOT_ID = os.getenv("WECHAT_BOT_ID", "").strip()
WECHAT_BOT_SECRET = os.getenv("WECHAT_BOT_SECRET", "").strip()
WECHAT_WS_URL = os.getenv("WECHAT_WS_URL", "wss://openws.work.weixin.qq.com").strip()
WECHAT_WS_HEARTBEAT_MS = int(os.getenv("WECHAT_WS_HEARTBEAT_MS", "30000"))
WECHAT_WS_RECONNECT_BASE_MS = int(os.getenv("WECHAT_WS_RECONNECT_BASE_MS", "1000"))
WECHAT_WS_MAX_RECONNECT_ATTEMPTS = int(os.getenv("WECHAT_WS_MAX_RECONNECT_ATTEMPTS", "-1"))
WECOM_STREAM_PUSH_INTERVAL = float(os.getenv("WECOM_STREAM_PUSH_INTERVAL", "0.2"))
WECHAT_MCP_WARMUP_ENABLED = (os.getenv("WECHAT_MCP_WARMUP_ENABLED", "1") or "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)
WECHAT_AGENT_WARMUP_ENABLED = (os.getenv("WECHAT_AGENT_WARMUP_ENABLED", "1") or "").lower() in (
    "1",
    "true",
    "yes",
    "on",
)

state_cache = RedisCache()
chat_store = ChatStore(bot_id=WECHAT_BOT_ID)
dedup_service = DedupService(chat_store, cache=state_cache, bot_id=WECHAT_BOT_ID)
stream_manager = StreamManager(ttl_seconds=STREAM_TTL_SECONDS, cache=state_cache, bot_id=WECHAT_BOT_ID)

# per-session asyncio.Lock：防止同一会话并发处理，替代 state.lock（threading.Lock）
_session_async_locks: dict[str, asyncio.Lock] = {}
_session_locks_mu: asyncio.Lock  # 在事件循环启动后初始化


async def _get_session_async_lock(key: str) -> asyncio.Lock:
    """功能：获取或创建指定会话键对应的 asyncio 锁，防止同一会话并发处理。
    参数：
    - key：会话锁键，通常为 user_id:session_id。
    返回值：
    - asyncio.Lock：该会话专用的异步锁。
    """
    global _session_locks_mu
    if "_session_locks_mu" not in globals() or not isinstance(globals().get("_session_locks_mu"), asyncio.Lock):
        _session_locks_mu = asyncio.Lock()
    async with _session_locks_mu:
        if key not in _session_async_locks:
            _session_async_locks[key] = asyncio.Lock()
        return _session_async_locks[key]


def _build_agent() -> ReActAgent:
    """功能：创建带观测回调的 ReActAgent 实例。
    参数：
    - 无。
    返回值：
    - ReActAgent：绑定工具与模型指标回调的智能体。
    """
    agent = ReActAgent(
        model=MODEL_NAME,
        project_directory=PROJECT_DIR,
        skill_system=SKILL_SYSTEM,
        key_channel="wechat",
    )
    if hasattr(agent, "tool_registry"):
        agent.tool_registry.observer = chain_tool_observers(
            _record_tool_metric,
            tool_call_observer,
        )
    if hasattr(agent, "model_client"):
        agent.model_client.observer = _record_model_metric
    return agent


session_manager = SessionManager(
    agent_factory=_build_agent,
    chat_store=chat_store,
    ttl_seconds=SESSION_TTL_SECONDS,
    cache=state_cache,
    bot_id=WECHAT_BOT_ID,
)


def _record_tool_metric(tool_name: str, ok: bool, duration_ms: float) -> None:
    """功能：记录工具调用指标。
    参数：
    - tool_name：工具名称。
    - ok：是否执行成功。
    - duration_ms：耗时毫秒数。
    返回值：
    - 无。
    """
    metrics.inc("tool_exec_total")
    metrics.observe_ms("tool_exec_duration", duration_ms)
    metrics.inc(f"tool_exec_{tool_name}_total")
    if ok:
        metrics.inc("tool_exec_success_total")
    else:
        metrics.inc("tool_exec_failure_total")


def _record_model_metric(event: str, duration_ms: float) -> None:
    """功能：记录模型调用指标。
    参数：
    - event：模型事件名。
    - duration_ms：耗时毫秒数。
    返回值：
    - 无。
    """
    metrics.inc(f"model_{event}_total")
    if duration_ms > 0:
        metrics.observe_ms("model_call_duration", duration_ms)
        if event.endswith("_success"):
            purpose = event[: -len("_success")] if event != "success" else "tool_call"
            metrics.observe_ms(f"model_{purpose}_duration", duration_ms)


def _selector_elapsed_ms(agent: Any) -> float:
    """功能：读取最近一次 Context Selector 耗时毫秒数。
    参数：
    - agent：ReActAgent 实例。
    返回值：
    - float：耗时毫秒；无法读取时返回 0。
    """
    try:
        runtime = getattr(agent, "runtime", None)
        selector = getattr(runtime, "context_selector", None)
        return float(getattr(selector, "last_elapsed_ms", 0.0) or 0.0)
    except Exception:
        return 0.0


def _clip_text(text: str, limit: int = 200) -> str:
    """功能：截断日志展示用文本，避免过长输出。
    参数：
    - text：原始文本。
    - limit：最大保留字符数。
    返回值：
    - str：截断后的文本。
    """
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated)"


def _sanitize_for_display(text: str) -> str:
    """功能：清理模型输出中的 ReAct 协议标签，保留可展示正文。
    参数：
    - text：模型原始输出。
    返回值：
    - str：适合企业微信展示的正文。
    """
    if not text:
        return ""
    s = str(text)
    s = re.sub(r"<action>.*?</action>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<observation>.*?</observation>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(r"<final_answer>.*?</final_answer>", "", s, flags=re.DOTALL | re.IGNORECASE)
    s = re.sub(
        r"</?(?:thought|thinking|think|action|observation|final_answer|question)\b[^>]*>",
        "",
        s,
        flags=re.IGNORECASE,
    )
    return s.strip()


def _print_console_kb_image(path: str) -> None:
    """功能：在控制台打印知识库图片本地路径与 file:// 链接。
    参数：
    - path：图片路径。
    返回值：
    - 无。
    """
    abs_path = os.path.abspath(path.strip())
    file_url = "file:///" + abs_path.replace("\\", "/")
    safe_print(f"\n[知识库图示] {abs_path}\n  打开: {file_url}\n", end="", flush=True)


def _compose_stream_content(final_answer: str) -> str:
    """功能：组装企业微信流式消息正文，附带空思考占位符。
    参数：
    - final_answer：最终答案文本。
    返回值：
    - str：符合企微 stream 展示要求的正文。
    """
    fin_raw = (final_answer or "").strip()
    fin = _sanitize_for_display(fin_raw) if fin_raw and "<" in fin_raw else fin_raw
    if fin:
        return f"{WECOM_EMPTY_THINK}\n{fin}"
    return WECOM_EMPTY_THINK


async def _run_agent_async(
    *,
    user_id: str,
    session_id: str,
    msgid: str,
    user_text: str,
    stream_id: str,
    message_received_at: float,
    accept_phases_ms: Optional[dict[str, float]] = None,
    image_delivery: Optional[WecomImageDelivery] = None,
) -> None:
    """功能：异步运行智能体并持续更新流式状态，不阻塞事件循环。
    参数：
    - user_id：用户标识。
    - session_id：会话标识。
    - msgid：消息 ID，用于去重与租约。
    - user_text：用户输入文本。
    - stream_id：流式消息 ID。
    - message_received_at：消息接收时间戳，用于端到端耗时统计。
    - accept_phases_ms：消息受理阶段耗时字典，可选。
    - image_delivery：可选知识库图片异步投递器。
    返回值：
    - 无。
    """
    # session_manager.get() 含 DB 访问，放线程池执行
    agent_task_started = time.time()
    session_get_started = time.time()
    state, session_created = await asyncio.to_thread(
        session_manager.get_with_meta,
        user_id=user_id,
        session_id=session_id,
    )
    session_get_ms = (time.time() - session_get_started) * 1000.0
    final_accumulator = ""
    kb_splitter = KbImageStreamSplitter()
    # update_lock 保留 threading.Lock：callback 可能从 asyncio.to_thread 工具线程调用
    update_lock = threading.Lock()
    trace_header_printed = False
    max_seconds = max(0.0, WECHAT_MAX_PROCESS_SECONDS)
    # Risk1 解决：asyncio.Event，timer 改为 async task
    stop_event = asyncio.Event()
    lease_owner = msgid or stream_id

    metrics.inc("agent_runs_total")
    metrics.set_gauge("session_manager_local_size", session_manager.local_size())
    metrics.set_gauge("stream_manager_local_size", stream_manager.local_size())

    if max_seconds > 0:
        async def _timeout_setter() -> None:
            """功能：在软超时到达后设置 stop_event，通知智能体停止生成。
            参数：
            - 无（闭包捕获 max_seconds 与 stop_event）。
            返回值：
            - 无。
            """
            await asyncio.sleep(max_seconds)
            stop_event.set()
        asyncio.create_task(_timeout_setter(), name="agent-soft-timeout")

    def stream_callback(section: str, chunk: str) -> None:
        """功能：智能体流式回调，累积最终答案、投递知识库图片并更新流式状态。
        参数：
        - section：流式片段类型（如 final_answer、tool_start 等）。
        - chunk：该片段对应的文本内容。
        返回值：
        - 无。
        """
        nonlocal final_accumulator, trace_header_printed
        if stop_event.is_set():
            return
        with update_lock:
            if section == "final_answer_flush":
                for kind, payload in kb_splitter.flush():
                    if kind == "image" and payload:
                        if image_delivery is not None:
                            image_delivery.schedule_image(payload)
                        _print_console_kb_image(payload)
                    elif kind == "text" and payload:
                        final_accumulator += payload
                return
            elif section in ("agent_start", "model_decision", "tool_start", "tool_end", "agent_finish", "error"):
                if chunk:
                    if not trace_header_printed:
                        log_print("执行过程：")
                        trace_header_printed = True
                    log_print(chunk)
                return
            elif section == "final_answer":
                if not chunk:
                    return
                for kind, payload in kb_splitter.feed(chunk):
                    if kind == "image" and payload:
                        if image_delivery is not None:
                            image_delivery.schedule_image(payload)
                        _print_console_kb_image(payload)
                    elif kind == "text" and payload:
                        final_accumulator += payload
            else:
                return

            # Phase1：stream_manager.update() 同步调用，Phase2 会改成 async
            stream_manager.update(
                stream_id=stream_id,
                content=_compose_stream_content(final_accumulator),
                finish=False,
                status="processing",
            )

    answer = ""
    session_lock = await _get_session_async_lock(f"{user_id}:{session_id}")
    lease_acquire_ms = 0.0
    arun_ms = 0.0
    selector_ms = 0.0
    run_success = True
    run_error = ""
    try:
        lease_started = time.time()
        acquired = await asyncio.to_thread(session_manager.acquire_lease, user_id, session_id, lease_owner)
        lease_acquire_ms = (time.time() - lease_started) * 1000.0
        if not acquired:
            answer = "当前会话正在处理中，请稍后重试。"
            metrics.inc("session_lease_conflicts_total")
            await asyncio.to_thread(stream_manager.mark_failed, stream_id=stream_id, content=answer)
            if msgid:
                await asyncio.to_thread(dedup_service.mark_done, msgid, answer)
            log_timing(
                logger,
                metrics,
                "ws_request_timing",
                phases={
                    **(accept_phases_ms or {}),
                    "session_get": session_get_ms,
                    "lease_acquire": lease_acquire_ms,
                },
                total_ms=(time.time() - message_received_at) * 1000.0,
                user_id=user_id,
                session_id=session_id,
                msgid=msgid,
                stream_id=stream_id,
                success=False,
                session_created=session_created,
                lease_conflict=True,
            )
            return

        async with session_lock:
            prep_ms = (time.time() - agent_task_started) * 1000.0
            log_event(
                logger,
                "ws_agent_run_start",
                user_id=user_id,
                session_id=session_id,
                msgid=msgid,
                stream_id=stream_id,
                session_created=session_created,
                session_get_ms=round(session_get_ms, 2),
                lease_acquire_ms=round(lease_acquire_ms, 2),
                prep_ms=round(prep_ms, 2),
            )
            arun_started = time.time()
            answer = await state.agent.arun(user_text, stream_callback=stream_callback, stop_event=stop_event)
            answer_for_log = strip_kb_image_markers(str(answer or final_accumulator or ""))
            log_answer(answer_for_log)
            arun_ms = (time.time() - arun_started) * 1000.0
            selector_ms = _selector_elapsed_ms(state.agent)
            state.last_active = time.time()
            metrics.inc("agent_runs_success_total")
            metrics.observe_ms("agent_run_duration", arun_ms)
            log_event(
                logger,
                "ws_agent_run_end",
                user_id=user_id,
                session_id=session_id,
                msgid=msgid,
                stream_id=stream_id,
                success=True,
                answer_length=len(str(answer or "")),
                duration_ms=round(arun_ms, 2),
                session_created=session_created,
                session_get_ms=round(session_get_ms, 2),
                lease_acquire_ms=round(lease_acquire_ms, 2),
                prep_ms=round(prep_ms, 2),
                selector_ms=round(selector_ms, 2),
            )
            logger.info(
                "Agent 运行结束 user_id=%s session_id=%s msgid=%s stream_id=%s 回答长度=%s",
                user_id,
                session_id,
                msgid,
                stream_id,
                len(str(answer or "")),
            )
    except Exception as exc:
        run_success = False
        run_error = str(exc)
        metrics.inc("agent_runs_failure_total")
        if arun_ms <= 0:
            arun_ms = (time.time() - agent_task_started) * 1000.0
        metrics.observe_ms("agent_run_duration", arun_ms)
        logger.exception("Agent 运行异常 user_id=%s session_id=%s msgid=%s", user_id, session_id, msgid)
        log_event(
            logger,
            "ws_agent_run_error",
            user_id=user_id,
            session_id=session_id,
            msgid=msgid,
            stream_id=stream_id,
            success=False,
            error=str(exc),
            duration_ms=round(arun_ms, 2),
            session_created=session_created,
            session_get_ms=round(session_get_ms, 2),
            lease_acquire_ms=round(lease_acquire_ms, 2),
        )
        answer = f"抱歉，处理你的消息时出现异常：{exc}"
    finally:
        for kind, payload in kb_splitter.flush():
            if kind == "image" and payload:
                if image_delivery is not None:
                    image_delivery.schedule_image(payload)
            elif kind == "text" and payload:
                final_accumulator += payload
        await asyncio.to_thread(session_manager.release_lease, user_id, session_id, lease_owner)

    full_answer = str(answer or "")
    display_answer = strip_kb_image_markers(full_answer)
    if image_delivery is not None:
        for kind, payload in split_kb_image_markers(full_answer):
            if kind == "image" and payload:
                image_delivery.schedule_image(payload)

    stream_body = _compose_stream_content(display_answer or final_accumulator)
    if not stream_body.strip() and display_answer:
        stream_body = display_answer

    persist_started = time.time()
    try:
        await asyncio.to_thread(
            chat_store.append_turn,
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            answer=display_answer,
        )
    except Exception:
        logger.exception("会话消息写入失败 user_id=%s session_id=%s msgid=%s", user_id, session_id, msgid)

    await asyncio.to_thread(
        stream_manager.update,
        stream_id=stream_id,
        content=stream_body,
        finish=True,
        status="done",
    )
    if msgid:
        await asyncio.to_thread(dedup_service.mark_done, msgid, display_answer)
    post_process_ms = (time.time() - persist_started) * 1000.0

    total_e2e_ms = (time.time() - message_received_at) * 1000.0
    timing_fields = {
        "user_id": user_id,
        "session_id": session_id,
        "msgid": msgid,
        "stream_id": stream_id,
        "success": run_success,
        "session_created": session_created,
        "answer_length": len(display_answer),
    }
    if run_error:
        timing_fields["error"] = run_error
    log_timing(
        logger,
        metrics,
        "ws_request_timing",
        phases={
            **(accept_phases_ms or {}),
            "session_get": session_get_ms,
            "lease_acquire": lease_acquire_ms,
            "arun": arun_ms,
            "selector": selector_ms,
            "post_process": post_process_ms,
        },
        total_ms=total_e2e_ms,
        **timing_fields,
    )


async def _push_stream_updates(ws_client: WSClient, frame: dict, stream_id: str) -> None:
    """功能：轮询流式状态并向企业微信推送增量内容。
    参数：
    - ws_client：WebSocket 客户端。
    - frame：原始消息帧。
    - stream_id：流式消息 ID。
    返回值：
    - 无。
    """
    last_content = None
    last_finish = False
    started_at = time.time()
    max_stream_seconds = max(1, int(os.getenv("WECHAT_WS_STREAM_MAX_SECONDS", "590")))

    while True:
        stream_state = stream_manager.get(stream_id)
        if not stream_state:
            await asyncio.sleep(0.4)
            if (time.time() - started_at) > 10:
                return
            continue

        content = stream_state.content or WECOM_EMPTY_THINK
        finish = bool(stream_state.finish)
        if content != last_content or finish != last_finish:
            await ws_client.reply_stream(frame, stream_id, content, finish)
            last_content = content
            last_finish = finish
            metrics.inc("wecom_ws_stream_push_total")

        if finish:
            return

        if (time.time() - started_at) > max_stream_seconds:
            timeout_text = "本次处理超时，请重试。若多次失败，请联系管理员查看服务日志。"
            stream_manager.mark_timeout(stream_id=stream_id, content=timeout_text)
            await ws_client.reply_stream(frame, stream_id, timeout_text, True)
            metrics.inc("stream_timeout_total")
            return

        await asyncio.sleep(WECOM_STREAM_PUSH_INTERVAL)


async def _handle_text_message(ws_client: WSClient, frame: dict) -> None:
    """功能：处理用户文本消息：去重、创建流式会话并异步运行智能体。
    参数：
    - ws_client：WebSocket 客户端。
    - frame：上行消息帧。
    返回值：
    - 无。
    """
    body = frame.get("body", {}) if isinstance(frame, dict) else {}
    msgid = str(body.get("msgid") or frame.get("msgid") or "")
    user_id = extract_user_id(body)
    session_id = extract_session_id(body)
    text_obj = body.get("text") if isinstance(body.get("text"), dict) else {}
    user_text = str(text_obj.get("content") or "").strip()
    message_received_at = time.time()
    accept_phases_ms: dict[str, float] = {}

    logger.info(
        "收到文本消息 msgid=%s user_id=%s session_id=%s 内容=%s",
        msgid,
        user_id,
        session_id,
        _clip_text(user_text, limit=200) if WECHAT_LOG_MESSAGE_CONTENT else "<已隐藏>",
    )
    metrics.inc("wecom_ws_text_message_total")

    if not user_text:
        stream_id = uuid.uuid4().hex
        await ws_client.reply_stream(frame, stream_id, "收到空消息，请重新输入。", True)
        return

    stream_id = uuid.uuid4().hex
    dedup_started = time.time()
    dedup_ok = True
    if msgid:
        dedup_ok = await asyncio.to_thread(dedup_service.mark_processing, msgid, stream_id)
    accept_phases_ms["dedup"] = (time.time() - dedup_started) * 1000.0
    if msgid and not dedup_ok:
        metrics.inc("dedup_hits_total")
        existed = await asyncio.to_thread(dedup_service.get, msgid) or {}
        existed_stream_id = str(existed.get("stream_id") or stream_id)
        if existed.get("status") == "done" and existed.get("answer"):
            await ws_client.reply_stream(frame, existed_stream_id, str(existed["answer"]), True)
        else:
            stream_state = await asyncio.to_thread(stream_manager.get, existed_stream_id)
            await ws_client.reply_stream(
                frame,
                existed_stream_id,
                stream_state.content if stream_state and stream_state.content else WECOM_EMPTY_THINK,
                bool(stream_state.finish) if stream_state else False,
            )
        return

    initial_content = WECOM_EMPTY_THINK
    stream_setup_started = time.time()
    await asyncio.to_thread(
        stream_manager.create,
        stream_id=stream_id,
        content=initial_content,
        finish=False,
        session_id=session_id,
        status="processing",
    )
    await ws_client.reply_stream(frame, stream_id, initial_content, False)
    accept_phases_ms["stream_setup"] = (time.time() - stream_setup_started) * 1000.0
    log_timing(
        logger,
        metrics,
        "ws_message_accepted",
        phases=accept_phases_ms,
        total_ms=(time.time() - message_received_at) * 1000.0,
        user_id=user_id,
        session_id=session_id,
        msgid=msgid,
        stream_id=stream_id,
    )

    loop = asyncio.get_running_loop()
    image_delivery = WecomImageDelivery(loop=loop, ws_client=ws_client, frame=frame)

    agent_task = asyncio.create_task(
        _run_agent_async(
            user_id=user_id,
            session_id=session_id,
            msgid=msgid,
            user_text=user_text,
            stream_id=stream_id,
            message_received_at=message_received_at,
            accept_phases_ms=dict(accept_phases_ms),
            image_delivery=image_delivery,
        ),
        name=f"agent-{stream_id}",
    )
    await _push_stream_updates(ws_client, frame, stream_id)
    await agent_task


async def _handle_enter_chat(ws_client: WSClient, frame: dict) -> None:
    """功能：响应用户进入会话事件并发送欢迎语。
    参数：
    - ws_client：WebSocket 客户端。
    - frame：进入会话事件帧。
    返回值：
    - 无。
    """
    await ws_client.reply_welcome(
        frame,
        {
            "msgtype": "text",
            "text": {"content": WELCOME_TEXT},
        },
    )
    metrics.inc("wecom_ws_enter_chat_total")


async def _run() -> None:
    """功能：启动企业微信长连接服务，注册事件处理器并保持运行。
    参数：
    - 无。
    返回值：
    - 无。
    异常：
    - ValueError：缺少 WECHAT_BOT_ID 或 WECHAT_BOT_SECRET 时抛出。
    """
    if not WECHAT_BOT_ID or not WECHAT_BOT_SECRET:
        raise ValueError("缺少 WECHAT_BOT_ID / WECHAT_BOT_SECRET，请先在 .env 中配置长连接凭据。")

    # 启动预热必须严格串行：先 MCP（拉远端/落盘缓存 + 构建 wrapper），再 Agent（依赖 MCP 工具列表）。
    if WECHAT_MCP_WARMUP_ENABLED:
        try:
            logger.info("企微连接前开始 MCP 预热…")
            warmup = await asyncio.to_thread(
                warm_mcp_tools,
                PROJECT_DIR,
                force_refresh=True,
            )
            logger.info(
                "MCP 预热完成: 服务数=%s 工具数=%s 包装数=%s 耗时毫秒=%s",
                warmup.get("server_count"),
                warmup.get("tool_count"),
                warmup.get("wrapper_count"),
                warmup.get("duration_ms"),
            )
        except Exception:
            logger.exception("MCP 预热失败，服务将继续使用缓存或不可用的 MCP 工具")
            metrics.inc("mcp_warmup_failure_total")

    if WECHAT_AGENT_WARMUP_ENABLED:
        try:
            logger.info("企微连接前开始 Agent 预热…")
            agent_warmup = await asyncio.to_thread(
                warm_agent,
                model=MODEL_NAME,
                project_directory=PROJECT_DIR,
                skill_system=SKILL_SYSTEM,
                key_channel="wechat",
            )
            metrics.observe_ms("agent_warmup_duration", float(agent_warmup.get("duration_ms") or 0))
            log_event(
                logger,
                "agent_warmup_complete",
                duration_ms=agent_warmup.get("duration_ms"),
                tool_count=agent_warmup.get("tool_count"),
                prompt_chars=agent_warmup.get("prompt_chars"),
            )
            logger.info(
                "Agent 预热完成: 工具数=%s 提示词字符=%s 耗时毫秒=%s",
                agent_warmup.get("tool_count"),
                agent_warmup.get("prompt_chars"),
                agent_warmup.get("duration_ms"),
            )
        except Exception:
            logger.exception("Agent 预热失败，首条消息可能仍有冷启动延迟")
            metrics.inc("agent_warmup_failure_total")

    ws_client = WSClient(
        WSClientOptions(
            bot_id=WECHAT_BOT_ID,
            secret=WECHAT_BOT_SECRET,
            ws_url=WECHAT_WS_URL,
            heartbeat_interval=WECHAT_WS_HEARTBEAT_MS,
            reconnect_interval=WECHAT_WS_RECONNECT_BASE_MS,
            max_reconnect_attempts=WECHAT_WS_MAX_RECONNECT_ATTEMPTS,
            logger=build_aibot_logger(logger),
        )
    )

    @ws_client.on("connected")
    def _on_connected():
        """功能：WebSocket 连接成功时的回调，记录日志并递增连接指标。
        参数：
        - 无。
        返回值：
        - 无。
        """
        logger.info("企微 WebSocket 已连接: %s", WECHAT_WS_URL)
        metrics.inc("wecom_ws_connected_total")

    @ws_client.on("authenticated")
    def _on_authenticated():
        """功能：WebSocket 认证成功时的回调，记录日志并递增认证指标。
        参数：
        - 无。
        返回值：
        - 无。
        """
        logger.info("企微 WebSocket 已认证")
        metrics.inc("wecom_ws_authenticated_total")

    @ws_client.on("reconnecting")
    def _on_reconnecting(attempt: int):
        """功能：WebSocket 重连过程中的回调，记录重连次数并递增指标。
        参数：
        - attempt：当前重连尝试次数。
        返回值：
        - 无。
        """
        logger.warning("企微 WebSocket 重连中，第 %s 次", attempt)
        metrics.inc("wecom_ws_reconnecting_total")

    @ws_client.on("disconnected")
    def _on_disconnected(reason: Optional[str] = None):
        """功能：WebSocket 断开连接时的回调，记录断开原因并递增指标。
        参数：
        - reason：断开原因描述，可选。
        返回值：
        - 无。
        """
        logger.warning("企微 WebSocket 已断开，原因=%s", reason)
        metrics.inc("wecom_ws_disconnected_total")

    @ws_client.on("error")
    def _on_error(error: Exception):
        """功能：WebSocket 发生错误时的回调，记录异常并递增错误指标。
        参数：
        - error：捕获到的异常对象。
        返回值：
        - 无。
        """
        logger.exception("企微 WebSocket 错误: %s", error)
        metrics.inc("wecom_ws_error_total")

    @ws_client.on("event.enter_chat")
    async def _on_enter_chat(frame):
        """功能：用户进入会话事件处理器，委托发送欢迎语。
        参数：
        - frame：进入会话事件消息帧。
        返回值：
        - 无。
        """
        try:
            await _handle_enter_chat(ws_client, frame)
        except Exception:
            logger.exception("进入会话事件处理异常")
            metrics.inc("wecom_ws_enter_chat_failure_total")

    @ws_client.on("message.text")
    async def _on_text_message(frame):
        """功能：用户文本消息事件处理器，委托去重与智能体异步处理。
        参数：
        - frame：文本消息上行帧。
        返回值：
        - 无。
        """
        try:
            await _handle_text_message(ws_client, frame)
        except Exception:
            logger.exception("文本消息处理异常")
            metrics.inc("wecom_ws_text_message_failure_total")

    @ws_client.on("event.template_card_event")
    async def _on_template_card_event(frame):
        """功能：模板卡片事件处理器，记录指标并忽略业务处理。
        参数：
        - frame：模板卡片事件消息帧。
        返回值：
        - 无。
        """
        metrics.inc("wecom_ws_template_card_event_total")
        logger.info("收到模板卡片事件，已忽略")

    @ws_client.on("event.feedback_event")
    async def _on_feedback_event(frame):
        """功能：用户反馈事件处理器，记录指标并记录日志。
        参数：
        - frame：反馈事件消息帧。
        返回值：
        - 无。
        """
        metrics.inc("wecom_ws_feedback_event_total")
        logger.info("收到反馈事件")

    await ws_client.connect()
    await asyncio.Event().wait()


if __name__ == "__main__":
    logger.info("正在启动企微 WebSocket 机器人…")
    asyncio.run(_run())
