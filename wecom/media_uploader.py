"""企业微信智能机器人临时素材分片上传。"""

from __future__ import annotations

import base64
import hashlib
import math
from pathlib import Path
from typing import Any

from aibot.utils import generate_req_id

_CHUNK_BYTES = 512 * 1024


def _ws_errcode(frame: dict) -> int:
    """功能：从 WebSocket 响应帧中解析 errcode。
    参数：
    - frame：WebSocket 响应帧字典。
    返回值：
    - int：errcode 整数值；缺失或非法时返回 -1。
    """
    try:
        return int(frame.get("errcode", 0))
    except Exception:
        return -1


async def upload_temp_image(ws_manager: Any, file_path: str | Path) -> str:
    """功能：通过长连接分片上传本地图片为企业微信临时素材。
    参数：
    - ws_manager：aibot WebSocket 管理器，负责 send_reply 交互。
    - file_path：本地图片路径。
    返回值：
    - str：临时素材 media_id（有效期 3 天）。
    异常：
    - FileNotFoundError：图片不存在。
    - ValueError：文件过小、超过 10MB 或分片数超过 100。
    - RuntimeError：初始化、分片上传或完成阶段失败。
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"图片不存在: {path}")

    data = path.read_bytes()
    total_size = len(data)
    if total_size < 5:
        raise ValueError("图片文件过小")
    if total_size > 10 * 1024 * 1024:
        raise ValueError("图片超过 10MB 上限")

    total_chunks = max(1, math.ceil(total_size / _CHUNK_BYTES))
    if total_chunks > 100:
        raise ValueError("图片分片数超过 100")

    md5 = hashlib.md5(data).hexdigest()
    init_body = {
        "type": "image",
        "filename": path.name[:256],
        "total_size": total_size,
        "total_chunks": total_chunks,
        "md5": md5,
    }
    init_frame = await ws_manager.send_reply(
        generate_req_id("aibot_upload_media_init"),
        init_body,
        "aibot_upload_media_init",
    )
    if _ws_errcode(init_frame):
        raise RuntimeError(f"上传初始化失败: {init_frame}")

    upload_id = (init_frame.get("body") or {}).get("upload_id")
    if not upload_id:
        raise RuntimeError(f"上传初始化未返回 upload_id: {init_frame}")

    for index in range(total_chunks):
        start = index * _CHUNK_BYTES
        chunk = data[start : start + _CHUNK_BYTES]
        chunk_frame = await ws_manager.send_reply(
            generate_req_id("aibot_upload_media_chunk"),
            {
                "upload_id": upload_id,
                "chunk_index": index,
                "base64_data": base64.b64encode(chunk).decode("ascii"),
            },
            "aibot_upload_media_chunk",
        )
        if _ws_errcode(chunk_frame):
            raise RuntimeError(f"上传分片 {index} 失败: {chunk_frame}")

    finish_frame = await ws_manager.send_reply(
        generate_req_id("aibot_upload_media_finish"),
        {"upload_id": upload_id},
        "aibot_upload_media_finish",
    )
    if _ws_errcode(finish_frame):
        raise RuntimeError(f"上传完成失败: {finish_frame}")

    media_id = (finish_frame.get("body") or {}).get("media_id")
    if not media_id:
        raise RuntimeError(f"上传完成未返回 media_id: {finish_frame}")
    return str(media_id)
