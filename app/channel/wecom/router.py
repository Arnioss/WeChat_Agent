from dataclasses import dataclass
from typing import Any, Callable, Dict

from app.channel.wecom.adapter import to_inbound_message
from app.contracts.protocols import InboundMessage


@dataclass(frozen=True)
class RouteContext:
    """功能：封装路由处理阶段的上下文信息。
    参数：
    - 无。
    返回值：
    - 无。
    """
    raw_message: Dict[str, Any]
    inbound: InboundMessage
    crypt: Any
    nonce: str
    timestamp: str


class WeComMessageRouter:
    """功能：根据消息类型将企业微信消息分发到对应处理器。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(
        self,
        *,
        handle_enter_chat: Callable[[RouteContext], Any],
        handle_template_card_event: Callable[[RouteContext], Any],
        handle_other_event: Callable[[RouteContext], Any],
        handle_stream: Callable[[RouteContext], Any],
        handle_text: Callable[[RouteContext], Any],
        handle_unsupported: Callable[[RouteContext], Any],
    ):
        """功能：注册各类企业微信消息处理回调，构建可复用路由器实例。
        参数：
        - handle_enter_chat：处理进入会话事件的回调。
        - handle_template_card_event：处理模板卡片事件的回调。
        - handle_other_event：处理其他事件消息的回调。
        - handle_stream：处理流式消息的回调。
        - handle_text：处理文本消息的回调。
        - handle_unsupported：处理未支持消息类型的回调。
        返回值：
        - 无。路由器仅做类型分发，不处理业务副作用与响应拼装。
        """
        self.handle_enter_chat = handle_enter_chat
        self.handle_template_card_event = handle_template_card_event
        self.handle_other_event = handle_other_event
        self.handle_stream = handle_stream
        self.handle_text = handle_text
        self.handle_unsupported = handle_unsupported

    def route(self, msg: Dict[str, Any], *, crypt: Any, nonce: str, timestamp: str):
        """功能：解析入站消息并路由到对应处理函数。
        参数：
        - msg：企业微信原始消息字典。
        - crypt：当前请求使用的加解密对象。
        - nonce：请求随机串。
        - timestamp：请求时间戳。
        返回值：
        - Any：目标处理函数返回结果。
        """
        inbound = to_inbound_message(msg, timestamp=timestamp)
        context = RouteContext(
            raw_message=msg,
            inbound=inbound,
            crypt=crypt,
            nonce=nonce,
            timestamp=timestamp,
        )
        if inbound.message_type == "event.enter_chat":
            return self.handle_enter_chat(context)
        if inbound.message_type == "event.template_card_event":
            return self.handle_template_card_event(context)
        if inbound.message_type.startswith("event."):
            return self.handle_other_event(context)
        if inbound.message_type == "stream":
            return self.handle_stream(context)
        if inbound.message_type == "text":
            return self.handle_text(context)
        return self.handle_unsupported(context)
