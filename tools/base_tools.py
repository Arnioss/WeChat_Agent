#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""基础通用工具：当前日期查询等。"""
from datetime import datetime

from app.agent.tool_metadata import ToolRichMetadata


def get_current_date():
    """功能：返回当前本地日期字符串，供工具调用或模板渲染使用。
    参数：
    - 无。
    返回值：
    - str：当前日期，格式为 YYYY-MM-DD。
    """
    return datetime.now().strftime("%Y-%m-%d")


get_current_date.__tool_rich_metadata__ = ToolRichMetadata(
    summary="获取当前日期字符串（按运行环境本地时区的日历日）。",
    when_to_use=(
        "用户明确询问「今天几号」「当前日期」「现在是哪一天」等日历日问题。",
        "回答依赖「今天的日期」作为事实锚点，且不需要查项目文档。",
    ),
    when_not_to_use=(
        "需要查询知识库、手册、版本说明、流程规范等文档内容。",
        "需要当前时刻的精确时间（时分秒）或时区换算（本工具只返回日期）。",
        "与日期无关的一般对话或推理任务。",
    ),
    input_schema={
        "type": "object",
        "properties": {},
        "additionalProperties": False,
        "description": "无参数。",
    },
    output_description="str：格式为 YYYY-MM-DD 的日期。",
    output_schema={"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"},
    examples=(
        "get_current_date()",
    ),
    notes=(
        "日期来自服务器本地时间；若业务要求特定时区，应在最终回答中说明假设。",
        "不要与 rag_summarize 混用：文档里的「发布日期」类信息应走知识库。",
    ),
    priority=72,
    source="local",
)
