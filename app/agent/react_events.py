"""Agent 运行期流式事件定义与发射工具。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional


StreamCallback = Callable[[str, str], None]


CONTROL_TAG_RE = re.compile(
    r"</?(?:thought|thinking|think|action|observation|final_answer|question)\b[^>]*>",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ReActEvent:
    """功能：描述一次可渲染的 Agent 流式事件。
    参数：
    - 无。
    返回值：
    - 无。`kind` 决定渲染模板，`payload` 携带工具名与参数等附加信息。
    """
    kind: str
    text: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


def sanitize_display_text(text: Any) -> str:
    """功能：移除展示文本中的 XML 控制标签并 strip。
    参数：
    - text：原始展示文本。
    返回值：
    - str：清理后的文本。
    """
    value = "" if text is None else str(text)
    value = CONTROL_TAG_RE.sub("", value)
    return value.strip()


def clip_text(text: Any, limit: int = 1200) -> str:
    """功能：清理并截断展示文本到指定长度。
    参数：
    - text：原始文本。
    - limit：最大字符数，默认 1200。
    返回值：
    - str：截断后以 `...` 结尾的文本。
    """
    value = sanitize_display_text(text)
    if len(value) <= limit:
        return value
    return value[:limit].rstrip() + "..."


def emit_event(callback: Optional[StreamCallback], event: ReActEvent) -> None:
    """功能：将 ReActEvent 渲染为文本并通过回调发送。
    参数：
    - callback：流式事件回调；为 None 时不发送。
    - event：待发送的事件对象。
    返回值：
    - 无。
    """
    if callback is None:
        return
    text = render_event_text(event)
    if event.kind == "final_answer_flush":
        callback("final_answer_flush", "")
        return
    if not text and event.kind != "error":
        return
    callback(event.kind, text)


def render_event_text(event: ReActEvent) -> str:
    """功能：按事件类型将 ReActEvent 渲染为面向用户的展示文本。
    参数：
    - event：待渲染的事件对象。
    返回值：
    - str：渲染后的单行或多行展示文本。
    """
    if event.kind == "agent_start":
        return sanitize_display_text(event.text or "Agent 开始处理请求。")
    if event.kind == "model_decision":
        return sanitize_display_text(event.text)
    if event.kind == "tool_start":
        name = sanitize_display_text(event.payload.get("tool_name") or "")
        args = event.payload.get("tool_arguments") or {}
        if isinstance(args, Mapping) and not args:
            return f"调用工具：{name}()"
        args_text = _format_arguments(args)
        return f"调用工具：{name}({args_text})"
    if event.kind == "tool_end":
        return "观察结果：" + clip_text(event.text, limit=1200)
    if event.kind == "error":
        return "错误：" + clip_text(event.text, limit=800)
    if event.kind == "final_answer":
        return "" if event.text is None else str(event.text)
    if event.kind == "final_answer_flush":
        return ""
    return sanitize_display_text(event.text)


def emit_reasoning(callback: Optional[StreamCallback], text: str) -> None:
    """功能：发送模型推理/决策类流式事件。
    参数：
    - callback：流式事件回调。
    - text：决策说明文本。
    返回值：
    - 无。
    """
    emit_event(callback, ReActEvent(kind="model_decision", text=text))


def emit_agent_start(callback: Optional[StreamCallback], text: str = "Agent 开始处理请求。") -> None:
    """功能：发送 Agent 开始处理请求的流式事件。
    参数：
    - callback：流式事件回调。
    - text：可选自定义开始提示文本。
    返回值：
    - 无。
    """
    emit_event(callback, ReActEvent(kind="agent_start", text=text))


def emit_model_decision(callback: Optional[StreamCallback], text: str) -> None:
    """功能：发送模型决策说明的流式事件。
    参数：
    - callback：流式事件回调。
    - text：决策说明文本。
    返回值：
    - 无。
    """
    emit_event(callback, ReActEvent(kind="model_decision", text=text))


def emit_tool_call(callback: Optional[StreamCallback], tool_name: str, tool_arguments: Mapping[str, Any] | None = None) -> None:
    """功能：发送工具调用开始的流式事件。
    参数：
    - callback：流式事件回调。
    - tool_name：工具名称。
    - tool_arguments：工具参数字典，可为 None。
    返回值：
    - 无。
    """
    emit_event(
        callback,
        ReActEvent(
            kind="tool_start",
            payload={"tool_name": tool_name, "tool_arguments": dict(tool_arguments or {})},
        ),
    )


def emit_observation(callback: Optional[StreamCallback], observation: Any) -> None:
    """功能：发送工具观察结果的流式事件。
    参数：
    - callback：流式事件回调。
    - observation：工具返回的观察结果。
    返回值：
    - 无。
    """
    emit_event(callback, ReActEvent(kind="tool_end", text=clip_text(observation, limit=1200)))


def emit_tool_start(callback: Optional[StreamCallback], tool_name: str, tool_arguments: Mapping[str, Any] | None = None) -> None:
    """功能：发送工具调用开始事件（`emit_tool_call` 别名）。
    参数：
    - callback：流式事件回调。
    - tool_name：工具名称。
    - tool_arguments：工具参数字典，可为 None。
    返回值：
    - 无。
    """
    emit_tool_call(callback, tool_name, tool_arguments)


def emit_tool_end(callback: Optional[StreamCallback], observation: Any) -> None:
    """功能：发送工具观察结束事件（`emit_observation` 别名）。
    参数：
    - callback：流式事件回调。
    - observation：工具返回的观察结果。
    返回值：
    - 无。
    """
    emit_observation(callback, observation)


def emit_agent_finish(callback: Optional[StreamCallback], text: str) -> None:
    """功能：发送 Agent 完成处理的流式事件。
    参数：
    - callback：流式事件回调。
    - text：完成说明文本。
    返回值：
    - 无。
    """
    emit_event(callback, ReActEvent(kind="agent_finish", text=text))


def emit_error(callback: Optional[StreamCallback], error: Any) -> None:
    """功能：发送错误类流式事件。
    参数：
    - callback：流式事件回调。
    - error：错误或异常信息。
    返回值：
    - 无。
    """
    emit_event(callback, ReActEvent(kind="error", text=clip_text(error, limit=800)))


def _format_arguments(args: Any) -> str:
    """功能：将工具参数字典格式化为 tool_start 事件展示字符串。
    参数：
    - args：工具参数字典或任意值。
    返回值：
    - str：key=value 形式、值截断后的逗号分隔文本。
    """
    if not isinstance(args, Mapping):
        return sanitize_display_text(args)
    parts = []
    for key, value in args.items():
        if isinstance(value, (dict, list, tuple)):
            try:
                value_text = json.dumps(value, ensure_ascii=False)
            except Exception:
                value_text = str(value)
        else:
            value_text = str(value)
        parts.append(f"{key}={clip_text(value_text, limit=160)}")
    return ", ".join(parts)
