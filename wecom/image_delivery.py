"""企业微信知识库图片异步投递。"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from app.agent.kb_body_images import normalize_kb_image_path
from wecom.media_uploader import upload_temp_image

logger = logging.getLogger(__name__)


@dataclass
class WecomImageDelivery:
    """功能：在独立线程中异步上传并发送知识库图片，不阻塞文字流式回复。
    参数：
    - loop：主 asyncio 事件循环，用于 run_coroutine_threadsafe。
    - ws_client：企业微信 WS 客户端，用于 reply 发送图片消息。
    - frame：当前消息帧，作为 reply 上下文。
    - sent_paths：已发送图片路径集合，用于去重。
    返回值：
    - 无。
    """
    loop: asyncio.AbstractEventLoop
    ws_client: Any
    frame: dict
    sent_paths: Set[str] = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def schedule_image(self, file_path: str) -> None:
        """功能：调度单张知识库图片的异步上传与发送。
        参数：
        - file_path：图片路径或 `[KB_IMAGE:...]` 中的相对/绝对路径。
        返回值：
        - 无。
        """
        resolved = normalize_kb_image_path(file_path)
        if resolved is None:
            logger.warning("企微知识库图片不存在 path=%s", file_path)
            return
        path = str(resolved)
        with self._lock:
            if path in self.sent_paths:
                return
            self.sent_paths.add(path)
        logger.info("企微知识库图片已排队上传 path=%s", path)

        def _runner() -> None:
            """功能：在后台线程中将图片上传协程提交到主事件循环执行。
            参数：
            - 无（闭包捕获 path 与 self.loop）。
            返回值：
            - 无。
            """
            asyncio.run_coroutine_threadsafe(self._send_image(path), self.loop)

        threading.Thread(target=_runner, name="wecom-kb-image", daemon=True).start()

    async def _send_image(self, file_path: str) -> None:
        """功能：上传图片并以 image 消息回复用户。
        参数：
        - file_path：已规范化且存在的本地图片绝对路径。
        返回值：
        - 无。
        """
        try:
            media_id = await upload_temp_image(self.ws_client._ws_manager, file_path)
            await self.ws_client.reply(
                self.frame,
                {
                    "msgtype": "image",
                    "image": {"media_id": media_id},
                },
            )
            logger.info("企微知识库图片已发送 path=%s media_id=%s", file_path, media_id[:16])
        except Exception:
            logger.exception("企微知识库图片发送失败 path=%s", file_path)


_delivery_var: threading.local = threading.local()


def set_wecom_image_delivery(delivery: Optional[WecomImageDelivery]) -> None:
    """功能：在当前线程上下文中绑定图片投递器实例。
    参数：
    - delivery：投递器实例；传 None 表示清除绑定。
    返回值：
    - 无。
    """
    _delivery_var.current = delivery


def get_wecom_image_delivery() -> Optional[WecomImageDelivery]:
    """功能：获取当前线程绑定的图片投递器。
    参数：
    - 无。
    返回值：
    - Optional[WecomImageDelivery]：已绑定时返回实例，否则 None。
    """
    return getattr(_delivery_var, "current", None)


def schedule_wecom_kb_image(file_path: str) -> None:
    """功能：若当前上下文已绑定投递器，则调度知识库图片发送。
    参数：
    - file_path：知识库图片路径。
    返回值：
    - 无。
    """
    delivery = get_wecom_image_delivery()
    if delivery is not None:
        delivery.schedule_image(file_path)
