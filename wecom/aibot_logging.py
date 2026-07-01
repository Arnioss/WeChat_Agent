"""企业微信 AiBot SDK 日志桥接：将 SDK 的 debug/info 输出接入项目日志级别控制。"""

from __future__ import annotations

import logging
import os
from typing import Any


_LEVEL_RANK = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "WARN": 30,
    "ERROR": 40,
    "CRITICAL": 50,
    "NONE": 100,
    "OFF": 100,
    "SILENT": 100,
}


class BridgedAiBotLogger:
    """功能：按配置级别过滤 AiBot SDK 日志，默认屏蔽 DEBUG 心跳/回执等噪音。
    参数：
    - min_level：允许输出的最低日志级别数值。
    - logger：项目业务 logger，用于实际写入日志。
    返回值：
    - 无。
    """

    def __init__(self, *, min_level: int, logger: logging.Logger):
        """功能：初始化桥接器，设置最低输出级别与目标 logger。
        参数：
        - min_level：允许输出的最低日志级别数值。
        - logger：项目业务 logger，用于实际写入日志。
        返回值：
        - 无。
        """
        self._min_level = min_level
        self._logger = logger

    def _emit(self, level: int, method: str, message: str, args: tuple[Any, ...]) -> None:
        """功能：按级别过滤后格式化消息并写入目标 logger。
        参数：
        - level：本条日志的级别数值。
        - method：logger 方法名（debug/info/warning/error）。
        - message：日志模板字符串。
        - args：模板占位符参数。
        返回值：
        - 无。
        """
        if level < self._min_level:
            return
        text = message % args if args else message
        getattr(self._logger, method)(text)

    def debug(self, message: str, *args: Any) -> None:
        """功能：输出 DEBUG 级别 AiBot SDK 日志（默认被过滤）。
        参数：
        - message：日志模板字符串。
        - args：模板占位符参数。
        返回值：
        - 无。
        """
        self._emit(10, "debug", message, args)

    def info(self, message: str, *args: Any) -> None:
        """功能：输出 INFO 级别 AiBot SDK 日志。
        参数：
        - message：日志模板字符串。
        - args：模板占位符参数。
        返回值：
        - 无。
        """
        self._emit(20, "info", message, args)

    def warn(self, message: str, *args: Any) -> None:
        """功能：输出 WARNING 级别 AiBot SDK 日志。
        参数：
        - message：日志模板字符串。
        - args：模板占位符参数。
        返回值：
        - 无。
        """
        self._emit(30, "warning", message, args)

    def error(self, message: str, *args: Any) -> None:
        """功能：输出 ERROR 级别 AiBot SDK 日志。
        参数：
        - message：日志模板字符串。
        - args：模板占位符参数。
        返回值：
        - 无。
        """
        self._emit(40, "error", message, args)


def build_aibot_logger(parent: logging.Logger) -> BridgedAiBotLogger:
    """功能：根据环境变量创建 AiBot SDK 日志桥接器。
    参数：
    - parent：项目业务 logger，用于继承 handler 与格式。
    返回值：
    - BridgedAiBotLogger：可传入 WSClientOptions.logger。
    环境变量：
    - WECHAT_AIBOT_SDK_LOG_LEVEL：DEBUG/INFO/WARNING/ERROR/NONE，默认 WARNING。
    """
    raw = (os.getenv("WECHAT_AIBOT_SDK_LOG_LEVEL") or "WARNING").strip().upper()
    min_level = _LEVEL_RANK.get(raw, 30)
    child = parent.getChild("aibot_sdk")
    return BridgedAiBotLogger(min_level=min_level, logger=child)
