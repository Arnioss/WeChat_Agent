import json
import os
import re
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
load_dotenv()

for _stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
    try:
        if _stream and hasattr(_stream, "reconfigure"):
            _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent

from agent import ReActAgent
from app.application.conversation.services import ChatStore, DedupService, SessionManager, StreamManager
from app.channel.wecom.adapter import to_wecom_payload
from app.channel.wecom.crypto_service import WeComCryptoService
from app.channel.wecom.router import RouteContext, WeComMessageRouter
from app.contracts.protocols import OutboundMessage
from app.infrastructure.cache import RedisCache
from app.infrastructure.logging_setup import configure_project_logging
from app.infrastructure.observability import AppMetrics, log_event
from app.skills.system import build_skill_system
from wecom.reply_builders import (
    build_stream_reply as _build_stream_reply,
    build_stream_with_template_card_reply as _build_stream_with_template_card_reply,
    build_text_reply as _build_text_reply,
    build_update_template_card_text_notice as _build_update_template_card_text_notice,
)
from wxwork.crypto import WXBizJsonMsgCrypt
def _build_agent() -> ReActAgent:
    """功能：创建并配置单个 ReActAgent 实例，同时挂载指标回调。
    参数：
    - 无。
    返回值：
    - ReActAgent：可用于处理会话请求的智能体实例。
    """
    agent = ReActAgent(
        model=MODEL_NAME,
        project_directory=PROJECT_DIR,
        skill_system=SKILL_SYSTEM,
    )
    if hasattr(agent, "tool_registry"):
        agent.tool_registry.observer = _record_tool_metric
    if hasattr(agent, "model_client"):
        agent.model_client.observer = _record_model_metric
    return agent


app = Flask(__name__)

logger = configure_project_logging(ROOT_DIR, logger_name=__name__)
metrics = AppMetrics()

WECHAT_TOKEN = os.getenv("WECHAT_ROBOT_TOKEN")
WECHAT_ENCODING_AES_KEY = os.getenv("WECHAT_ROBOT_ENCODING_AES_KEY")
WECHAT_RECEIVE_ID = os.getenv("WECHAT_ROBOT_RECEIVE_ID")
WECHAT_REPLY_DEBUG_LOG = (os.getenv("WECHAT_REPLY_DEBUG_LOG") or "").lower() in (
    "1", "true", "yes", "on"
)
WELCOME_TEXT = os.getenv("WECHAT_ROBOT_WELCOME_TEXT")

MODEL_NAME = os.getenv("OPENROUTER_MODEL")
PROJECT_DIR = str(ROOT_DIR)
SKILL_SYSTEM = build_skill_system(project_directory=PROJECT_DIR)
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS"))
WECHAT_LOG_MESSAGE_CONTENT = (os.getenv("WECHAT_LOG_MESSAGE_CONTENT") or "").lower() in (
    "1", "true", "yes", "on"
)
WECHAT_WORKER_MAX = int(os.getenv("WECHAT_WORKER_MAX"))
WECHAT_MAX_PROCESS_SECONDS = float(os.getenv("WECHAT_MAX_PROCESS_SECONDS"))
WECHAT_USE_WAITRESS = (os.getenv("WECHAT_USE_WAITRESS") or "").lower() in (
    "1", "true", "yes", "on"
)

state_cache = RedisCache()
chat_store = ChatStore()
dedup_service = DedupService(chat_store, cache=state_cache)
session_manager = SessionManager(
    agent_factory=_build_agent,
    chat_store=chat_store,
    ttl_seconds=SESSION_TTL_SECONDS,
    cache=state_cache,
)
stream_manager = StreamManager(ttl_seconds=int(os.getenv("STREAM_TTL_SECONDS")), cache=state_cache)
worker_pool = ThreadPoolExecutor(max_workers=max(1, WECHAT_WORKER_MAX))


def _safe_json(data) -> str:
    """功能：安全序列化对象为 JSON 文本，失败时降级为字符串。
    参数：
    - data：待序列化对象。
    返回值：
    - str：格式化后的 JSON 文本或对象字符串表示。
    """
    try:
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return str(data)


crypto_service = WeComCryptoService(
    token=WECHAT_TOKEN,
    encoding_aes_key=WECHAT_ENCODING_AES_KEY,
    receive_id=WECHAT_RECEIVE_ID,
    reply_debug_log=WECHAT_REPLY_DEBUG_LOG,
    build_stream_reply=_build_stream_reply,
    safe_json=_safe_json,
    logger=logger,
)


def _mask_request_args(args_dict: dict) -> dict:
    """功能：脱敏请求查询参数中的敏感字段。
    参数：
    - args_dict：原始请求参数字典。
    返回值：
    - dict：敏感字段已替换为掩码的参数字典。
    """
    masked = dict(args_dict or {})
    for key in ("msg_signature", "echostr"):
        if key in masked and masked[key]:
            masked[key] = "***MASKED***"
    return masked


def _clip_text(text: str, limit: int = 200) -> str:
    """功能：按长度上限截断日志展示文本，避免输出过长内容。
    参数：
    - text：待处理文本内容。
    - limit：返回结果数量上限。
    返回值：
    - str：未超限时返回原文，超限时返回截断文本并追加标记。
    """
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...(truncated)"


def _record_tool_metric(tool_name: str, ok: bool, duration_ms: float) -> None:
    """功能：记录工具调用计数与耗时指标。
    参数：
    - tool_name：工具名称。
    - ok：调用是否成功。
    - duration_ms：调用耗时（毫秒）。
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
    """功能：记录模型调用事件指标。
    参数：
    - event：模型事件类型（如 success/rate_limit）。
    - duration_ms：调用耗时（毫秒）。
    返回值：
    - 无。
    """
    metrics.inc(f"model_{event}_total")
    if duration_ms > 0:
        metrics.observe_ms("model_call_duration", duration_ms)


def _new_crypt() -> WXBizJsonMsgCrypt:
    """功能：根据当前环境变量创建企业微信加解密对象。
    参数：
    - 无。
    返回值：
    - WXBizJsonMsgCrypt：用于请求解密和响应加密的对象实例。
    """
    return crypto_service.new_crypt()


def _encrypt_reply(
    crypt: WXBizJsonMsgCrypt,
    plaintext_json: Any,
    nonce: str,
    timestamp: str,
) -> Response:
    """功能：将出站消息统一转换并加密为企业微信回包。
    参数：
    - crypt：企业微信加解密对象。
    - plaintext_json：出站消息对象或明文字典。
    - nonce：请求随机串。
    - timestamp：请求时间戳。
    返回值：
    - Response：可直接返回给企业微信的加密响应。
    """
    if isinstance(plaintext_json, OutboundMessage):
        plaintext_json = to_wecom_payload(
            plaintext_json,
            build_text_reply=_build_text_reply,
            build_stream_reply=_build_stream_reply,
            build_stream_with_template_card_reply=_build_stream_with_template_card_reply,
            build_update_template_card_text_notice=_build_update_template_card_text_notice,
        )
    return crypto_service.encrypt_reply(
        crypt=crypt,
        plaintext_json=plaintext_json,
        nonce=nonce,
        timestamp=timestamp,
    )


def _run_agent_and_update_stream(
    user_id: str,
    session_id: str,
    msgid: str,
    user_text: str,
    stream_id: str,
):
    """功能：异步运行智能体并持续更新流式输出状态。
    参数：
    - user_id：用户标识。
    - session_id：会话标识。
    - msgid：消息唯一 ID。
    - user_text：用户输入文本。
    - stream_id：流式响应 ID。
    返回值：
    - 无。
    """
    state = session_manager.get(user_id=user_id, session_id=session_id)
    think_accumulator = ""
    final_accumulator = ""
    preamble_accumulator = ""
    update_lock = threading.Lock()
    max_seconds = max(0.0, WECHAT_MAX_PROCESS_SECONDS)
    stop_event = threading.Event()
    lease_owner = msgid or stream_id
    started_at = time.time()
    metrics.inc("agent_runs_total")
    metrics.set_gauge("session_manager_local_size", session_manager.local_size())
    metrics.set_gauge("stream_manager_local_size", stream_manager.local_size())

    if max_seconds > 0:
        timer_thread = threading.Thread(
            target=lambda: (time.sleep(max_seconds), stop_event.set()),
            name="agent-soft-timeout",
            daemon=True,
        )
        timer_thread.start()

    def _sanitize_stream_fragment(text: str) -> str:
        """功能：清洗流式片段中的 action/observation 标签和 HTML 标签。
        参数：
        - text：待处理文本内容。
        返回值：
        - str：去除控制标签后的可展示文本。
        """
        if not text:
            return ""
        s = str(text)
        s = re.sub(r"<action>.*?</action>", "", s, flags=re.DOTALL)
        s = re.sub(r"<observation>.*?</observation>", "", s, flags=re.DOTALL)
        s = re.sub(r"<final_answer>.*?</final_answer>", "", s, flags=re.DOTALL)
        s = re.sub(r"</?[^>]+>", "", s)
        return s.strip()

    def _compose_stream_content() -> str:
        """功能：合并前言、思考和最终答案片段，生成当前流式展示文本。
        参数：
        - 无。
        返回值：
        - str：优先展示最终答案，未完成时返回思考态占位文本。
        """
        pre = _sanitize_stream_fragment(preamble_accumulator)
        th = _sanitize_stream_fragment(think_accumulator)
        fin = _sanitize_stream_fragment(final_accumulator)

        parts: List[str] = []
        if pre:
            parts.append(pre)
        if th:
            parts.append(f"<think>{th}</think>")
        body = "\n\n".join(parts)
        if fin:
            return fin if not body else (body + "\n\n" + fin)
        return body if body else "<think>思考中，请稍候...</think>"

    def stream_callback(section: str, chunk: str):
        """功能：接收模型流式分片并同步写入流状态。
        参数：
        - section：分片所属区段（preamble/thought/final_answer）。
        - chunk：本次新增文本片段。
        返回值：
        - 无。
        """
        nonlocal think_accumulator, final_accumulator, preamble_accumulator
        if stop_event.is_set() or not chunk:
            return

        with update_lock:
            if section == "preamble":
                preamble_accumulator += chunk
            elif section == "thought":
                think_accumulator += chunk
            elif section == "final_answer":
                final_accumulator += chunk
            else:
                return
            stream_manager.update(
                stream_id=stream_id,
                content=_compose_stream_content(),
                finish=False,
                status="processing",
            )

    try:
        if not session_manager.acquire_lease(user_id, session_id, lease_owner):
            answer = "当前会话正在处理中，请稍后重试。"
            metrics.inc("session_lease_conflicts_total")
            stream_manager.mark_failed(stream_id=stream_id, content=answer)
            if msgid:
                dedup_service.mark_done(msgid, answer)
            return

        with state.lock:
            log_event(
                logger,
                "agent_run_start",
                user_id=user_id,
                session_id=session_id,
                msgid=msgid,
                stream_id=stream_id,
            )
            answer = state.agent.run(user_text, stream_callback=stream_callback, stop_event=stop_event)
            state.last_active = time.time()
            metrics.inc("agent_runs_success_total")
            metrics.observe_ms("agent_run_duration", (time.time() - started_at) * 1000.0)
    except Exception as e:
        metrics.inc("agent_runs_failure_total")
        metrics.observe_ms("agent_run_duration", (time.time() - started_at) * 1000.0)
        logger.exception("AGENT_RUN_EXCEPTION user_id=%s session_id=%s msgid=%s", user_id, session_id, msgid)
        answer = f"抱歉，处理你的消息时出现异常：{e}"
    finally:
        session_manager.release_lease(user_id, session_id, lease_owner)

    try:
        chat_store.append_turn(
            user_id=user_id,
            session_id=session_id,
            user_text=user_text,
            answer=answer,
        )
    except Exception:
        logger.exception("CHAT_STORE_APPEND_FAILED user_id=%s session_id=%s msgid=%s", user_id, session_id, msgid)

    stream_manager.update(
        stream_id=stream_id,
        content=str(answer or ""),
        finish=True,
        status="done",
    )
    if msgid:
        dedup_service.mark_done(msgid, answer)


def _handle_enter_chat(context: RouteContext):
    """功能：处理进入会话事件并返回欢迎消息。
    参数：
    - context：路由上下文对象。
    返回值：
    - Response：加密后的欢迎消息响应。
    """
    reply = OutboundMessage(type="text", content=WELCOME_TEXT)
    return _encrypt_reply(context.crypt, reply, context.nonce, context.timestamp)


def _handle_template_card_event(context: RouteContext):
    # 卡片事件不再处理，直接返回 success。
    """功能：处理模板卡片事件（当前策略为直接忽略）。
    参数：
    - context：路由上下文对象。
    返回值：
    - Response：固定 `success` 响应。
    """
    return Response("success", mimetype="text/plain")


def _handle_other_event(context: RouteContext):
    """功能：处理未单独支持的事件消息。
    参数：
    - context：路由上下文对象。
    返回值：
    - Response：固定 `success` 响应。
    """
    return Response("success", mimetype="text/plain")


def _handle_stream_message(context: RouteContext):
    """功能：处理客户端轮询的 stream 消息并返回当前流状态。
    参数：
    - context：路由上下文对象。
    返回值：
    - Response：加密后的流式内容响应。
    """
    inbound = context.inbound
    crypt = context.crypt
    nonce = context.nonce
    timestamp = context.timestamp
    stream_obj = inbound.raw_payload.get("stream") if isinstance(inbound.raw_payload.get("stream"), dict) else {}
    stream_id = str(stream_obj.get("id") or "").strip()
    metrics.inc("wecom_stream_pull_total")

    if not stream_id:
        return Response("success", mimetype="text/plain")

    stream_state = stream_manager.get(stream_id)
    if not stream_state:
        reply = OutboundMessage(
            type="stream",
            content="<think>处理中，请稍候...</think>",
            stream_id=stream_id,
            finish=False,
        )
        return _encrypt_reply(crypt, reply, nonce, timestamp)

    if (not stream_state.finish) and (time.time() - float(stream_state.created_at or 0) > 180):
        timeout_text = "本次处理超时，请重试。若多次失败，请联系管理员查看服务日志。"
        metrics.inc("stream_timeout_total")
        stream_manager.mark_timeout(
            stream_id=stream_id,
            content=timeout_text,
        )
        stream_state = stream_manager.get(stream_id) or stream_state

    reply = OutboundMessage(
        type="stream",
        content=stream_state.content or "<think>处理中，请稍候...</think>",
        stream_id=stream_id,
        finish=stream_state.finish,
    )
    return _encrypt_reply(crypt, reply, nonce, timestamp)


def _handle_text_message(context: RouteContext):
    """功能：处理文本消息并触发异步智能体执行。
    参数：
    - context：路由上下文对象。
    返回值：
    - Response：即时返回的流式占位响应或去重命中响应。
    """
    inbound = context.inbound
    crypt = context.crypt
    nonce = context.nonce
    timestamp = context.timestamp
    user_text = str(inbound.text or "").strip()
    msgid = inbound.message_id
    user_id = inbound.user_id
    session_id = inbound.session_id

    logger.info(
        "WECHAT_TEXT msgid=%s user_id=%s session_id=%s text=%s",
        msgid,
        user_id,
        session_id,
        _clip_text(user_text, limit=200) if WECHAT_LOG_MESSAGE_CONTENT else "<hidden>",
    )
    metrics.inc("wecom_text_message_total")

    if not user_text:
        reply = OutboundMessage(
            type="stream",
            content="收到空消息，请重新输入。",
            stream_id=uuid.uuid4().hex,
            finish=True,
        )
        return _encrypt_reply(crypt, reply, nonce, timestamp)

    stream_id = uuid.uuid4().hex
    if msgid and not dedup_service.mark_processing(msgid, stream_id):
        metrics.inc("dedup_hits_total")
        existed = dedup_service.get(msgid) or {}
        existed_stream_id = str(existed.get("stream_id") or stream_id)
        if existed.get("status") == "done" and existed.get("answer"):
            reply = _build_stream_reply(
                content=str(existed["answer"]),
                stream_id=existed_stream_id,
                finish=True,
            )
        else:
            stream_state = stream_manager.get(existed_stream_id)
            reply = _build_stream_reply(
                content=(
                    stream_state.content
                    if stream_state and stream_state.content
                    else "<think>思考中，请稍候...</think>"
                ),
                stream_id=existed_stream_id,
                finish=stream_state.finish if stream_state else False,
            )
        return _encrypt_reply(crypt, reply, nonce, timestamp)

    initial_content = "<think>思考中，请稍候...</think>"
    stream_manager.create(
        stream_id=stream_id,
        content=initial_content,
        finish=False,
        session_id=session_id,
        status="processing",
    )
    worker_pool.submit(
        _run_agent_and_update_stream,
        user_id=user_id,
        session_id=session_id,
        msgid=msgid,
        user_text=user_text,
        stream_id=stream_id,
    )
    immediate_reply = _build_stream_reply(
        content=initial_content,
        stream_id=stream_id,
        finish=False,
    )
    return _encrypt_reply(crypt, immediate_reply, nonce, timestamp)


message_router = WeComMessageRouter(
    handle_enter_chat=_handle_enter_chat,
    handle_template_card_event=_handle_template_card_event,
    handle_other_event=_handle_other_event,
    handle_stream=_handle_stream_message,
    handle_text=_handle_text_message,
    handle_unsupported=_handle_other_event,
)


@app.get("/api/wechat/robot")
def wechat_verify():
    """功能：处理企业微信 URL 验证请求，验证签名并回显解密后的 echostr。
    参数：
    - 无。
    返回值：
    - Response：验证成功返回明文 echostr，失败返回错误状态响应。
    """
    msg_signature = request.args.get("msg_signature", "")
    timestamp = request.args.get("timestamp", "")
    nonce = request.args.get("nonce", "")
    echostr = request.args.get("echostr", "")
    metrics.inc("wecom_verify_total")
    try:
        crypt = _new_crypt()
        ret, plain_echo = crypt.VerifyURL(msg_signature, timestamp, nonce, echostr)
        if ret == 0 and plain_echo:
            metrics.inc("wecom_verify_success_total")
            return Response(plain_echo, mimetype="text/plain")
        metrics.inc("wecom_verify_failure_total")
        return Response("error", status=400, mimetype="text/plain")
    except Exception:
        metrics.inc("wecom_verify_exception_total")
        logger.exception("WECHAT_VERIFY_EXCEPTION")
        return Response("error", status=500, mimetype="text/plain")


@app.post("/api/wechat/robot")
def wechat_callback():
    """功能：处理企业微信主回调，解密消息后按类型路由并返回加密响应。
    参数：
    - 无。
    返回值：
    - Response：路由处理后的企业微信响应；异常时返回 `success` 避免重试风暴。
    """
    msg_signature = request.args.get("msg_signature", "")
    timestamp = request.args.get("timestamp", str(int(time.time())))
    nonce = request.args.get("nonce", "")
    body = request.get_json(silent=True) or {}
    request_started_at = time.time()
    metrics.inc("wecom_callback_total")

    if WECHAT_REPLY_DEBUG_LOG:
        logger.info("WECHAT_CALLBACK args=%s", _mask_request_args(dict(request.args)))
        logger.info("WECHAT_CALLBACK body=\n%s", _safe_json(body))

    try:
        crypt = _new_crypt()
        ret, plain_msg = crypt.DecryptMsg(
            json.dumps(body, ensure_ascii=False),
            msg_signature,
            timestamp,
            nonce,
        )
        if ret != 0 or not plain_msg:
            metrics.inc("wecom_callback_decrypt_failure_total")
            metrics.observe_ms("wecom_callback_duration", (time.time() - request_started_at) * 1000.0)
            logger.warning("消息解密失败 ret=%s", ret)
            return Response("success", mimetype="text/plain")
        if WECHAT_REPLY_DEBUG_LOG:
            logger.info("WECHAT_DECRYPTED_MESSAGE=\n%s", _clip_text(plain_msg, limit=2000))
        msg = json.loads(plain_msg)
        log_event(
            logger,
            "wecom_callback",
            msgid=msg.get("MsgId") or msg.get("msgid"),
            msg_type=msg.get("MsgType"),
            wecom_event=msg.get("Event"),
            from_user=msg.get("FromUserName"),
        )
        response = message_router.route(msg, crypt=crypt, nonce=nonce, timestamp=timestamp)
        metrics.inc("wecom_callback_success_total")
        metrics.observe_ms("wecom_callback_duration", (time.time() - request_started_at) * 1000.0)
        return response
    except Exception:
        metrics.inc("wecom_callback_exception_total")
        metrics.observe_ms("wecom_callback_duration", (time.time() - request_started_at) * 1000.0)
        logger.exception("WECHAT_CALLBACK_EXCEPTION")
        return Response("success", mimetype="text/plain")


@app.get("/healthz")
def healthz():
    """功能：返回服务健康状态及核心运行指标快照。
    参数：
    - 无。
    返回值：
    - Response：包含健康状态、缓存状态和本地会话规模的 JSON 响应。
    """
    metrics.set_gauge("session_manager_local_size", session_manager.local_size())
    metrics.set_gauge("stream_manager_local_size", stream_manager.local_size())
    return jsonify(
        {
            "success": True,
            "status": "ok",
            "redis_enabled": state_cache.available,
            "mysql_enabled": chat_store.enabled,
            "worker_max": WECHAT_WORKER_MAX,
            "session_manager_local_size": session_manager.local_size(),
            "stream_manager_local_size": stream_manager.local_size(),
        }
    )


@app.get("/metrics")
def metrics_endpoint():
    """功能：导出 Prometheus 指标文本。
    参数：
    - 无。
    返回值：
    - Response：Prometheus exposition 格式的指标响应。
    """
    metrics.set_gauge("session_manager_local_size", session_manager.local_size())
    metrics.set_gauge("stream_manager_local_size", stream_manager.local_size())
    return Response(metrics.render_prometheus(), mimetype="text/plain")


@app.post("/api/wechat/robot/chat")
def debug_chat():
    """功能：提供本地调试问答接口，执行一次完整的会话处理流程。
    参数：
    - 无。
    返回值：
    - Response：包含用户标识、会话标识和答案文本的 JSON 响应。
    """
    payload = request.get_json(silent=True) or {}
    user_id = str(payload.get("user_id") or "unknown_user")
    session_id = str(payload.get("session_id") or "default")
    message = str(payload.get("message") or "").strip()

    if not message:
        return jsonify({"success": False, "message": "message 不能为空"}), 400

    state = session_manager.get(user_id=user_id, session_id=session_id)
    started_at = time.time()
    metrics.inc("debug_chat_total")
    with state.lock:
        answer = state.agent.run(message)
        state.last_active = time.time()
        try:
            chat_store.append_turn(
                user_id=user_id,
                session_id=session_id,
                user_text=message,
                answer=answer,
            )
        except Exception:
            logger.exception(
                "CHAT_STORE_APPEND_FAILED user_id=%s session_id=%s debug_chat=true",
                user_id,
                session_id,
            )
    metrics.inc("debug_chat_success_total")
    metrics.observe_ms("debug_chat_duration", (time.time() - started_at) * 1000.0)

    return jsonify(
        {
            "success": True,
            "user_id": user_id,
            "session_id": session_id,
            "answer": answer,
            "ttl_seconds": SESSION_TTL_SECONDS,
        }
    )


if __name__ == "__main__":
    host = os.getenv("WECHAT_SERVER_HOST")
    port = int(os.getenv("WECHAT_SERVER_PORT"))
    logger.info("Server starting at http://%s:%s", host, port)
    if WECHAT_USE_WAITRESS:
        try:
            from waitress import serve

            serve(app, host=host, port=port)
        except Exception as e:
            logger.warning("waitress 启动失败，回退到 Flask 内置服务器: %s", e)
            app.run(host=host, port=port, debug=False, threaded=True)
    else:
        app.run(host=host, port=port, debug=False, threaded=True)
