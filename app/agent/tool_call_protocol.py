"""工具调用参数协议校验。"""

from __future__ import annotations

from typing import Any, Mapping


class ToolCallProtocolError(ValueError):
    """功能：表示模型 tool_calls 参数不符合协议或 schema 的错误。
    参数：
    - 无。
    返回值：
    - 无。`reason` 字段标识具体错误类别。
    """

    def __init__(self, message: str, *, reason: str = "protocol_error"):
        """功能：构造带 reason 字段的协议错误。
        参数：
        - message：面向用户或日志的错误说明。
        - reason：机器可读的错误原因标识。
        返回值：
        - 无。
        """
        super().__init__(message)
        self.reason = reason


def validate_tool_arguments(
    tool_name: str,
    arguments: Mapping[str, Any],
    input_schema: Mapping[str, Any] | None,
) -> None:
    """功能：按 JSON Schema 校验 tool_calls 的 arguments 对象。
    参数：
    - tool_name：工具名称，用于错误信息。
    - arguments：模型传入的参数字典。
    - input_schema：工具的 input JSON Schema，可为 None。
    返回值：
    - 无。校验通过时不返回值。
    异常：
    - ToolCallProtocolError：缺少必填参数、未知参数或类型不匹配时抛出。
    """
    schema = dict(input_schema or {})
    properties = schema.get("properties")
    required = schema.get("required")
    additional_allowed = schema.get("additionalProperties", True)
    if not isinstance(properties, dict):
        properties = {}
    required_set = set(required) if isinstance(required, list) else set()

    missing = [name for name in required_set if name not in arguments]
    if missing:
        raise ToolCallProtocolError(
            f"工具 {tool_name} 缺少必填参数：{', '.join(missing)}",
            reason="missing_required_tool_argument",
        )

    if additional_allowed is False:
        extra = [name for name in arguments if name not in properties]
        if extra:
            raise ToolCallProtocolError(
                f"工具 {tool_name} 收到未知参数：{', '.join(extra)}",
                reason="unknown_tool_argument",
            )

    for name, prop in properties.items():
        if name not in arguments or not isinstance(prop, dict):
            continue
        expected = prop.get("type")
        value = arguments[name]
        if expected == "string" and not isinstance(value, str):
            raise ToolCallProtocolError(f"工具 {tool_name} 参数 {name} 必须是字符串。", reason="invalid_tool_argument_type")
        if expected == "object" and not isinstance(value, dict):
            raise ToolCallProtocolError(f"工具 {tool_name} 参数 {name} 必须是对象。", reason="invalid_tool_argument_type")
        if expected == "array" and not isinstance(value, list):
            raise ToolCallProtocolError(f"工具 {tool_name} 参数 {name} 必须是数组。", reason="invalid_tool_argument_type")
        if expected == "boolean" and not isinstance(value, bool):
            raise ToolCallProtocolError(f"工具 {tool_name} 参数 {name} 必须是布尔值。", reason="invalid_tool_argument_type")
        if expected == "integer" and not isinstance(value, int):
            raise ToolCallProtocolError(f"工具 {tool_name} 参数 {name} 必须是整数。", reason="invalid_tool_argument_type")
        if expected == "number" and not isinstance(value, (int, float)):
            raise ToolCallProtocolError(f"工具 {tool_name} 参数 {name} 必须是数字。", reason="invalid_tool_argument_type")
