from typing import Any, Callable, Dict

from app.contracts.protocols import InboundMessage, OutboundMessage
from wecom.message_parsers import extract_session_id, extract_user_id


def to_inbound_message(raw_message: Dict[str, Any], *, timestamp: str, channel: str = "wecom") -> InboundMessage:
    """功能：把企业微信原始消息转换为统一入站消息对象。
    参数：
    - raw_message：企业微信原始消息字典。
    - timestamp：消息接收时间戳字符串。
    - channel：渠道标识，默认 `wecom`。
    返回值：
    - InboundMessage：标准化后的入站消息对象。
    """
    msgtype = str(raw_message.get("msgtype") or "")
    event = raw_message.get("event") if isinstance(raw_message.get("event"), dict) else {}
    eventtype = str(event.get("eventtype") or "")
    if msgtype == "event" and eventtype:
        message_type = f"event.{eventtype}"
    else:
        message_type = msgtype or "unknown"

    text_obj = raw_message.get("text") if isinstance(raw_message.get("text"), dict) else {}
    text = str(text_obj.get("content") or "").strip()

    return InboundMessage(
        message_id=str(raw_message.get("msgid") or ""),
        channel=channel,
        message_type=message_type,
        user_id=extract_user_id(raw_message),
        session_id=extract_session_id(raw_message),
        text=text,
        raw_payload=raw_message,
        timestamp=str(timestamp or ""),
    )


def to_wecom_payload(
    outbound_message: OutboundMessage,
    *,
    build_text_reply: Callable[[str], dict],
    build_stream_reply: Callable[..., dict],
    build_stream_with_template_card_reply: Callable[..., dict],
    build_update_template_card_text_notice: Callable[..., dict],
) -> dict:
    """功能：把统一出站消息对象转换为企业微信发送负载。
    参数：
    - outbound_message：统一出站消息对象。
    - build_text_reply：构建文本消息的函数。
    - build_stream_reply：构建流式消息的函数。
    - build_stream_with_template_card_reply：构建流式+卡片消息的函数。
    - build_update_template_card_text_notice：构建更新卡片通知消息的函数。
    返回值：
    - dict：可直接发送到企业微信接口的消息负载。
    异常：
    - ValueError：不支持的出站消息类型会抛出异常。
    """
    if outbound_message.type == "text":
        return build_text_reply(outbound_message.content)

    if outbound_message.type == "stream":
        return build_stream_reply(
            content=outbound_message.content,
            stream_id=outbound_message.stream_id,
            finish=outbound_message.finish,
        )

    if outbound_message.type == "stream_with_card":
        return build_stream_with_template_card_reply(
            content=outbound_message.content,
            stream_id=outbound_message.stream_id,
            finish=outbound_message.finish,
            template_card=outbound_message.template_card or {},
        )

    if outbound_message.type == "template_card":
        if outbound_message.metadata.get("response_type") == "update_template_card":
            return build_update_template_card_text_notice(
                task_id=outbound_message.task_id,
                userids=outbound_message.userids,
                content=outbound_message.content,
            )
        return {
            "msgtype": "template_card",
            "template_card": outbound_message.template_card or {},
        }

    raise ValueError(f"Unsupported outbound message type: {outbound_message.type}")
