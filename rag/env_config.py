"""用于知识检索与总结的 RAG 环境配置。"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from rag.logging_utils import rag_log

DEFAULT_EMBEDDING_MODEL = "text-embedding-v4"
DEFAULT_CHAT_MODEL = "deepseek-v4-flash"
DEFAULT_VISION_MODEL = "qwen3.6-plus"
DEFAULT_RERANK_MODEL = "qwen3-rerank"
PARSER_VERSION = "1.3.0"  # 1.3.0: docx/md/pdf preserve ordered text-image blocks
CAPTION_PROMPT_VERSION = "v1"
CHUNKING_VERSION = "4"


def env_bool(name: str, default: bool = False) -> bool:
    """功能：读取环境变量并解析为布尔值。
    参数：
    - name：环境变量名称。
    - default：变量未设置或为空时的默认值。
    返回值：
    - bool：值为 1/true/yes/on 时返回 True，否则 False。
    """
    value = (os.getenv(name) or "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


def env_int(name: str, default: int, *, minimum: int | None = None) -> int:
    """功能：读取环境变量并解析为整数，可选下限约束。
    参数：
    - name：环境变量名称。
    - default：未设置、为空或解析失败时的默认值。
    - minimum：解析成功后的最小值；为 None 时不做下限裁剪。
    返回值：
    - int：解析后的整数值。
    """
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        rag_log(f"[RAG] {name}={value!r} invalid, using {default}")
        return default
    if minimum is not None:
        return max(minimum, parsed)
    return parsed


def env_float(name: str, default: float, *, minimum: float | None = None) -> float:
    """功能：读取环境变量并解析为浮点数，可选下限约束。
    参数：
    - name：环境变量名称。
    - default：未设置、为空或解析失败时的默认值。
    - minimum：解析成功后的最小值；为 None 时不做下限裁剪。
    返回值：
    - float：解析后的浮点数值。
    """
    value = (os.getenv(name) or "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        rag_log(f"[RAG] {name}={value!r} invalid, using {default}")
        return default
    if minimum is not None:
        return max(minimum, parsed)
    return parsed


def resolve_api_key() -> str:
    """功能：从环境变量解析 OpenRouter/API 网关 Key。
    参数：
    - 无。
    返回值：
    - str：优先 `OPENROUTER_API_KEY`，否则取 `OPENROUTER_API_KEYS` 中第一个。
    异常：
    - ValueError：未配置任何 Key 时抛出。
    """
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if api_key:
        return api_key
    raw_keys = (os.getenv("OPENROUTER_API_KEYS") or "").strip()
    if raw_keys:
        return raw_keys.split(",")[0].strip()
    raise ValueError("缺少 OPENROUTER_API_KEY / OPENROUTER_API_KEYS，请在 .env 文件中设置。")


def resolve_base_url() -> str:
    """功能：读取 OpenAI 兼容网关地址。
    参数：
    - 无。
    返回值：
    - str：去除末尾斜杠的 `OPENAI_BASE_URL`。
    异常：
    - ValueError：未设置 `OPENAI_BASE_URL` 时抛出。
    """
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if not base_url:
        raise ValueError("缺少环境变量 OPENAI_BASE_URL，请在 .env 文件中设置。")
    return base_url.rstrip("/")


def resolve_embedding_model() -> str:
    """功能：解析嵌入模型名称，未配置时使用默认值。
    参数：
    - 无。
    返回值：
    - str：`RAG_EMBEDDING_MODEL` 或 `DEFAULT_EMBEDDING_MODEL`。
    """
    model = (os.getenv("RAG_EMBEDDING_MODEL") or "").strip()
    if not model:
        rag_log(f"[RAG] RAG_EMBEDDING_MODEL 未设置，使用默认 {DEFAULT_EMBEDDING_MODEL!r}")
        return DEFAULT_EMBEDDING_MODEL
    return model


def resolve_chat_model() -> str:
    """功能：解析 RAG 对话模型名称，未配置时使用默认值。
    参数：
    - 无。
    返回值：
    - str：`RAG_CHAT_MODEL`、`OPENROUTER_MODEL` 或 `DEFAULT_CHAT_MODEL`。
    """
    model = (os.getenv("RAG_CHAT_MODEL") or os.getenv("OPENROUTER_MODEL") or "").strip()
    if not model:
        rag_log(f"[RAG] RAG_CHAT_MODEL / OPENROUTER_MODEL 未设置，使用默认 {DEFAULT_CHAT_MODEL!r}")
        return DEFAULT_CHAT_MODEL
    return model


def resolve_vision_model() -> str:
    """功能：解析多模态识图模型名称，未配置时使用默认值。
    参数：
    - 无。
    返回值：
    - str：`RAG_VISION_MODEL` 或 `DEFAULT_VISION_MODEL`。
    """
    model = (os.getenv("RAG_VISION_MODEL") or "").strip()
    if not model:
        rag_log(f"[RAG] RAG_VISION_MODEL 未设置，使用默认 {DEFAULT_VISION_MODEL!r}")
        return DEFAULT_VISION_MODEL
    return model


def resolve_rerank_model() -> str:
    """功能：解析 Rerank 模型名称，未配置时使用默认值。
    参数：
    - 无。
    返回值：
    - str：`RAG_RERANK_MODEL` 或 `DEFAULT_RERANK_MODEL`。
    """
    model = (os.getenv("RAG_RERANK_MODEL") or "").strip()
    if not model:
        return DEFAULT_RERANK_MODEL
    return model


@dataclass(frozen=True)
class RagEnvConfig:
    """功能：聚合 RAG 运行所需的网关、模型与解析/分片相关配置。
    参数：
    - 无（字段由 `load` 从 `.env` 填充）。
    返回值：
    - 无。作为不可变配置对象在向量库、摄入管线间共享。
    """
    project_root: Path
    base_url: str
    api_key: str
    embedding_model: str
    chat_model: str
    vision_model: str
    multimodal_enabled: bool
    vision_enabled: bool
    parser_version: str
    http_timeout: float
    embedding_retries: int
    caption_max_tokens: int
    min_figure_area_ratio: float
    max_pages: int
    max_image_bytes: int
    caption_prompt_version: str
    chunking_version: str
    rerank_enabled: bool
    rerank_model: str
    rerank_endpoint: str
    retrieve_candidates: int
    rerank_top_n: int
    rerank_min_score: float
    keyword_enabled: bool
    keyword_top_n: int
    rrf_k: int

    @classmethod
    def load(cls, project_root: Path | None = None) -> "RagEnvConfig":
        """功能：从项目根目录加载 `.env` 并构造 RAG 配置快照。
        参数：
        - project_root：项目根路径；默认取本包上级目录。
        返回值：
        - RagEnvConfig：包含网关、模型与各解析阈值的配置实例。
        """
        root = project_root or Path(__file__).resolve().parents[1]
        load_dotenv(root / ".env", override=False)
        return cls(
            project_root=root,
            base_url=resolve_base_url(),
            api_key=resolve_api_key(),
            embedding_model=resolve_embedding_model(),
            chat_model=resolve_chat_model(),
            vision_model=resolve_vision_model(),
            multimodal_enabled=env_bool("RAG_MULTIMODAL_ENABLED", True),
            vision_enabled=env_bool("RAG_VISION_ENABLED", True),
            parser_version=(os.getenv("RAG_PARSER_VERSION") or PARSER_VERSION).strip(),
            http_timeout=float(os.getenv("RAG_HTTP_TIMEOUT_SECONDS", "120")),
            embedding_retries=max(0, int(os.getenv("RAG_EMBEDDING_RETRIES", "2"))),
            caption_max_tokens=int(os.getenv("RAG_CAPTION_MAX_TOKENS", "300")),
            min_figure_area_ratio=float(os.getenv("RAG_MIN_FIGURE_AREA_RATIO", "0.04")),
            max_pages=int(os.getenv("RAG_MAX_PAGES", "500")),
            max_image_bytes=int(os.getenv("RAG_MAX_IMAGE_BYTES", str(15 * 1024 * 1024))),
            caption_prompt_version=(os.getenv("RAG_CAPTION_PROMPT_VERSION") or CAPTION_PROMPT_VERSION).strip(),
            chunking_version=(os.getenv("RAG_CHUNKING_VERSION") or CHUNKING_VERSION).strip(),
            rerank_enabled=env_bool("RAG_RERANK_ENABLED", False),
            rerank_model=resolve_rerank_model(),
            rerank_endpoint=(os.getenv("RAG_RERANK_ENDPOINT") or "/rerank").strip() or "/rerank",
            retrieve_candidates=env_int("RAG_RETRIEVE_CANDIDATES", 20, minimum=1),
            rerank_top_n=env_int("RAG_RERANK_TOP_N", 5, minimum=1),
            rerank_min_score=env_float("RAG_RERANK_MIN_SCORE", 0.35, minimum=0.0),
            keyword_enabled=env_bool("RAG_KEYWORD_ENABLED", True),
            keyword_top_n=env_int("RAG_KEYWORD_TOP_N", 20, minimum=1),
            rrf_k=env_int("RAG_RRF_K", 60, minimum=1),
        )
