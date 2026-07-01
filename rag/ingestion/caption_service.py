"""基于视觉模型的图片说明生成与缓存服务。"""

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Optional

import requests

from rag.env_config import RagEnvConfig
from rag.http_key_pool import RagHttpKeyPool, is_rate_limit_error
from rag.logging_utils import rag_log

CAPTION_PROMPT = (
    "你是知识库索引助手。请用中文简洁描述这张图片中与文档、流程、界面、配置相关的内容，"
    "便于后续语义检索。输出 2-6 句话，不要编造图中没有的信息。若无法识别，回复「无法识别图示内容」。"
)


class CaptionService:
    """调用视觉 LLM 为图片生成中文说明，支持本地 JSON 缓存。

    功能：
        封装视觉 API 调用、caption 缓存读写及失败回退逻辑。

    参数：
        无（实例属性由 ``__init__`` 设置）。

    返回值：
        无（类定义）。
    """

    def __init__(self, config: RagEnvConfig, *, cache_dir: Path | None = None):
        """初始化 caption 服务。

        功能：
            绑定 RAG 环境配置，可选创建 caption 缓存目录与 HTTP 会话。

        参数：
            config: RAG 环境配置（API、vision 模型、超时等）。
            cache_dir: caption JSON 缓存目录；为 None 时不读写缓存。

        返回值：
            无。

        异常：
            无。
        """
        self.config = config
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._chat_url = config.base_url.rstrip("/") + "/chat/completions"
        self._key_pool = RagHttpKeyPool.from_env(fallback_key=config.api_key, service="图片识别")

    def _cache_path(self, asset_rel: str) -> Optional[Path]:
        """根据资源键生成缓存文件路径。

        功能：
            对 asset_rel 与 vision 模型名做 MD5，返回 ``{cache_dir}/{hash}.json``。

        参数：
            asset_rel: 图片路径或相对资源键字符串。

        返回值：
            缓存文件 ``Path``；未配置 cache_dir 时返回 None。

        异常：
            无。
        """
        if not self.cache_dir:
            return None
        key = hashlib_name(asset_rel + self.config.vision_model)
        return self.cache_dir / f"{key}.json"

    def has_cached(self, image_path: Path) -> bool:
        """检查图片是否已有 caption 缓存。

        功能：
            以图片绝对路径字符串为键查询缓存是否存在且可读。

        参数：
            image_path: 本地图片文件路径。

        返回值：
            缓存命中且含非空 caption 时返回 True。

        异常：
            无。
        """
        return bool(self._read_cache(str(image_path)))

    def caption_image_file(self, image_path: Path, *, alt_text: str | None = None) -> str:
        """为本地图片生成或读取说明文本。

        功能：
            优先返回 alt_text（vision 未启用时）或缓存；否则调用视觉 API 识别，
            成功后写入缓存；失败或超大文件时回退 alt_text 或占位符。

        参数：
            image_path: 本地图片文件路径。
            alt_text: 可选备用说明（如 Markdown alt 文本）。

        返回值：
            中文图片描述或占位符（如 ``[图-无描述]``、``[图-文件过大已跳过识别]``）。

        异常：
            无；API 或 IO 失败时记录日志并返回回退文本。
        """
        if alt_text and alt_text.strip() and not self.config.vision_enabled:
            return alt_text.strip()

        rel_key = str(image_path)
        cached = self._read_cache(rel_key)
        if cached:
            return cached

        if not self.config.vision_enabled:
            return (alt_text or "").strip() or "[图-无描述]"

        try:
            data = image_path.read_bytes()
            if len(data) > self.config.max_image_bytes:
                return (alt_text or "").strip() or "[图-文件过大已跳过识别]"
            mime = _mime_for_path(image_path)
            b64 = base64.standard_b64encode(data).decode("ascii")
            text = self._request_vision(b64, mime)
            if text:
                self._write_cache(rel_key, text)
                return text
        except Exception as exc:
            rag_log(f"[RAG] 图片识别失败 {image_path.name}: {exc}", flush=True)

        return (alt_text or "").strip() or "[图-无描述]"

    def _request_vision(self, b64: str, mime: str) -> str:
        """向视觉 chat/completions API 发送 base64 图片并获取说明。

        功能：
            构造多模态消息，最多重试 3 次（指数退避），解析 choices[0].message.content。

        参数：
            b64: 图片的 base64 编码字符串。
            mime: 图片 MIME 类型（如 ``image/png``）。

        返回值：
            模型返回的说明文本（已 strip）。

        异常：
            RuntimeError: 3 次请求均失败。
        """
        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": self.config.caption_max_tokens,
            "temperature": 0.2,
        }
        if not self._key_pool.api_keys:
            raise RuntimeError("未配置图片识别 API Key（OPENROUTER_API_KEY / OPENROUTER_API_KEYS）。")

        last_error: Exception | None = None
        key_indices = self._key_pool.iter_indices()
        max_attempts = 3
        for key_pos, key_idx in enumerate(key_indices):
            headers = {
                "Authorization": f"Bearer {self._key_pool.bearer(key_idx)}",
                "Content-Type": "application/json",
            }
            for attempt in range(max_attempts):
                try:
                    resp = self._session.post(
                        self._chat_url,
                        headers=headers,
                        json=payload,
                        timeout=self.config.http_timeout,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    content = data["choices"][0]["message"]["content"]
                    return (content or "").strip()
                except Exception as exc:
                    last_error = exc
                    if is_rate_limit_error(exc):
                        self._key_pool.mark_rate_limited(key_idx)
                        if key_pos + 1 < len(key_indices):
                            self._key_pool.log_switch(key_idx)
                        break
                    if attempt >= max_attempts - 1:
                        break
                    time.sleep(0.8 * (attempt + 1))
        raise RuntimeError(f"图片识别请求失败：{last_error}") from last_error

    def _read_cache(self, key: str) -> str:
        """从缓存文件读取 caption。

        功能：
            根据 key 定位缓存 JSON，返回 caption 字段。

        参数：
            key: 缓存键（通常为图片路径字符串）。

        返回值：
            caption 文本；无缓存或读取失败时返回空字符串。

        异常：
            无。
        """
        path = self._cache_path(key)
        if not path or not path.exists():
            return ""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return str(data.get("caption") or "")
        except Exception:
            return ""

    def _write_cache(self, key: str, caption: str) -> None:
        """将 caption 写入缓存文件。

        功能：
            以 JSON 格式保存 caption 与 vision 模型名。

        参数：
            key: 缓存键。
            caption: 说明文本。

        返回值：
            无。

        异常：
            无；未配置 cache_dir 时直接返回。
        """
        path = self._cache_path(key)
        if not path:
            return
        path.write_text(
            json.dumps({"caption": caption, "model": self.config.vision_model}, ensure_ascii=False),
            encoding="utf-8",
        )


def hashlib_name(text: str) -> str:
    """对文本做 MD5 哈希，返回 32 位十六进制字符串。

    功能：
        将 text 按 UTF-8 编码后计算 MD5 digest。

    参数：
        text: 待哈希的字符串。

    返回值：
        小写十六进制 MD5 摘要。

    异常：
        无。
    """
    import hashlib

    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _mime_for_path(path: Path) -> str:
    """根据文件后缀推断图片 MIME 类型。

    功能：
        映射常见图片后缀到 MIME；未知后缀默认 ``image/png``。

    参数：
        path: 图片文件路径。

    返回值：
        MIME 类型字符串。

    异常：
        无。
    """
    suffix = path.suffix.lower()
    mapping = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
    }
    return mapping.get(suffix, "image/png")
