from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class InboundMessage:
    """功能：统一描述渠道适配后的入站消息协议对象。
    参数：
    - 无。
    返回值：
    - 无。字段约束了后续路由所需最小信息集合（类型、用户、会话、原始载荷等）。
    """
    message_id: str
    channel: str
    message_type: str
    user_id: str
    session_id: str
    text: str
    raw_payload: Dict[str, Any]
    timestamp: str


@dataclass(frozen=True)
class CardAction:
    """功能：描述模板卡片回调动作及操作者信息。
    参数：
    - 无。
    返回值：
    - 无。`payload` 保留渠道原始动作字段，便于业务层二次解析。
    """
    action_type: str
    operator_user_id: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MessagePlan:
    """功能：定义消息处理后的发送计划（回复模式、内容与附加元数据）。
    参数：
    - 无。
    返回值：
    - 无。用于把决策层输出与发送层解耦，减少渠道分支逻辑耦合。
    """
    reply_mode: str
    final_text: str = ""
    stream_enabled: bool = False
    card_action: Optional[CardAction] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutboundMessage:
    """功能：统一描述待发送到渠道侧的出站消息结构。
    参数：
    - 无。
    返回值：
    - 无。兼容普通文本、流式消息、模板卡片等多种发送类型。
    """
    type: str
    content: str = ""
    stream_id: str = ""
    finish: bool = False
    template_card: Optional[Dict[str, Any]] = None
    task_id: str = ""
    userids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamState:
    """功能：表示对外查询时可见的流式任务状态快照。
    参数：
    - 无。
    返回值：
    - 无。作为协议层只读对象，不负责状态更新行为。
    """
    stream_id: str
    session_id: str
    status: str
    content: str
    finish: bool
    card: Optional[Dict[str, Any]]
    updated_at: float
