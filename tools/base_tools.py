#!/usr/bin/env python
# -*- coding: utf-8 -*-
from datetime import datetime


def get_current_date():
    """功能：返回当前本地日期字符串，供工具调用或模板渲染使用。
    参数：
    - 无。
    返回值：
    - str：当前日期，格式为 YYYY-MM-DD。
    """
    return datetime.now().strftime("%Y-%m-%d")
