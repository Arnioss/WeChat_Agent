"""RAG 向量存储：Chroma 向量检索、HTTP 嵌入与多模态入库索引。"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import requests
from dotenv import load_dotenv

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

from rag.env_config import RagEnvConfig, env_bool
from rag.http_key_pool import RagHttpKeyPool, is_rate_limit_error
from rag.logging_utils import rag_log, rag_trace_log


def _env_bool(name: str, default: bool) -> bool:
    """功能：读取布尔型环境变量（委托 rag.env_config.env_bool）。

    参数：
        name: 环境变量名。
        default: 未设置或无法解析时的默认值。

    返回值：
        解析后的布尔值。

    异常：
        无。
    """
    return env_bool(name, default)


class VectorBackend(Protocol):
    """功能：Chroma 向量存储后端协议，定义分片增删查与按来源扩展接口。
    参数：
    - 无（Protocol 类型，由 ChromaVectorBackend 等实现类构造）。
    返回值：
    - 无。
    """

    def delete_source(self, source: str) -> None:
        """功能：删除指定来源（文件绝对路径 key）下的全部分片。

        参数：
            source: 文件绝对路径 key。

        返回值：
            无。

        异常：
            后端存储操作异常可能向上抛出。
        """
        ...

    def upsert(
        self,
        *,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, object]],
    ) -> None:
        """功能：批量写入或覆盖向量分片及其元数据。

        参数：
            ids: 分片 id 列表。
            documents: 分片文本列表。
            embeddings: 向量列表。
            metadatas: 元数据列表。

        返回值：
            无。

        异常：
            后端存储操作异常可能向上抛出。
        """
        ...

    def query(self, *, query_embedding: List[float], n_results: int) -> List[Dict[str, str]]:
        """功能：按查询向量检索最相关的 n_results 条分片。

        参数：
            query_embedding: 查询向量。
            n_results: 返回条数。

        返回值：
            标准化分片行列表。

        异常：
            后端存储操作异常可能向上抛出。
        """
        ...

    def get_by_source(self, source: str) -> List[Dict[str, str]]:
        """功能：按来源 key 取回该文件的全部分片（用于上下文扩展）。

        参数：
            source: 文件绝对路径 key。

        返回值：
            标准化分片行列表。

        异常：
            后端存储操作异常可能向上抛出。
        """
        ...

    def all_rows(self) -> List[Dict[str, str]]:
        """功能：取回集合内全部分片行（用于关键词召回语料构建）。

        参数：
            无。

        返回值：
            按 source、chunk_index 排序的标准化分片行列表。

        异常：
            后端存储操作异常可能向上抛出。
        """
        ...


class ChromaVectorBackend:
    """功能：Chroma 持久化向量后端，HNSW 余弦空间检索；初始化前注入 SQLite 兼容补丁。

    参数：
        persist_dir: Chroma 持久化目录。
        collection_name: 集合名称。

    返回值：
        无（构造器）。

    异常：
        RuntimeError: chromadb 导入失败或 PersistentClient/集合初始化失败（含环境诊断信息）。
    """

    def __init__(self, *, persist_dir: Path, collection_name: str):
        """功能：初始化 Chroma 持久化客户端与集合。
        参数：
        - persist_dir：Chroma 数据目录。
        - collection_name：集合名称。
        返回值：
        - 无。
        异常：
        - RuntimeError：chromadb 导入或初始化失败时抛出。
        """
        from rag.sqlite_chroma_compat import ensure_chroma_sqlite

        project_root = Path(__file__).resolve().parents[1]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        ensure_chroma_sqlite()
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
        os.environ.setdefault("CHROMADB_ANONYMIZED_TELEMETRY", "false")

        try:
            import chromadb
            from chromadb.config import Settings
        except Exception as exc:
            raise RuntimeError(
                "ChromaDB import failed. Use Python 3.10+ (3.12 recommended on Windows) "
                "and reinstall dependencies: python -m pip install -U -r requirements.txt"
            ) from exc

        settings_kwargs = self._chroma_settings_kwargs(persist_dir)
        try:
            self._client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(**settings_kwargs),
            )
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            version = getattr(chromadb, "__version__", "unknown")
            raise RuntimeError(
                "ChromaDB initialization failed. "
                f"python={sys.version.split()[0]}, executable={sys.executable}, "
                f"platform={platform.platform()}, chromadb={version}, persist_dir={persist_dir}. "
                "On Windows, prefer Python 3.12 x64, run `python check_env.py --chroma-only`, "
                "close other processes using the same .rag_store directory, and rebuild the Chroma "
                "store if it was created by a different Chroma version."
            ) from exc

    @staticmethod
    def _chroma_settings_kwargs(persist_dir: Path) -> Dict[str, Any]:
        """功能：组装 Chroma Settings 参数字典（禁用遥测、可配置 API 实现）。

        参数：
            persist_dir: 持久化目录路径。

        返回值：
            传给 chromadb.config.Settings 的 kwargs；chroma_api_impl 可由 RAG_CHROMA_API_IMPL 覆盖。

        异常：
            无。
        """
        no_op_telemetry = "rag.chroma_telemetry.NoOpProductTelemetry"
        api_impl = os.getenv("RAG_CHROMA_API_IMPL") or "chromadb.api.segment.SegmentAPI"
        return {
            "anonymized_telemetry": False,
            "chroma_api_impl": api_impl,
            "chroma_product_telemetry_impl": no_op_telemetry,
            "chroma_telemetry_impl": no_op_telemetry,
            "persist_directory": str(persist_dir),
            "is_persistent": True,
        }

    def delete_source(self, source: str) -> None:
        """功能：按 metadata.source 条件删除 Chroma 集合内该文件的全部分片。

        参数：
            source: 文件绝对路径 key。

        返回值：
            无。

        异常：
            Chroma 客户端/集合操作异常可能向上抛出。
        """
        self._collection.delete(where={"source": source})

    def upsert(
        self,
        *,
        ids: List[str],
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, object]],
    ) -> None:
        """功能：委托 Chroma collection.upsert 写入或覆盖分片。

        参数：
            ids: 分片 id 列表。
            documents: 分片文本列表。
            embeddings: 向量列表。
            metadatas: 元数据列表。

        返回值：
            无。

        异常：
            Chroma 客户端/集合操作异常可能向上抛出。
        """
        self._collection.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query(self, *, query_embedding: List[float], n_results: int) -> List[Dict[str, str]]:
        """功能：调用 Chroma 向量检索（HNSW 余弦），结果转为统一行格式。

        参数：
            query_embedding: 查询向量。
            n_results: 返回条数。

        返回值：
            标准化分片行列表（content、source 及 metadata 字段）。

        异常：
            Chroma 客户端/集合操作异常可能向上抛出。
        """
        result = self._collection.query(query_embeddings=[query_embedding], n_results=n_results)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        rows = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            rows.append(
                _row_from_hit(
                    str(doc),
                    str((meta or {}).get("source", "")),
                    dict(meta or {}),
                )
            )
        return rows

    def get_by_source(self, source: str) -> List[Dict[str, str]]:
        """功能：按 source 过滤取回全部分片，再按 chunk_index 排序。

        参数：
            source: 文件绝对路径 key。

        返回值：
            标准化分片行列表。

        异常：
            Chroma 客户端/集合操作异常可能向上抛出。
        """
        result = self._collection.get(
            where={"source": source},
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        rows = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            rows.append(_row_from_hit(str(doc), str((meta or {}).get("source", "")), dict(meta or {})))
        rows.sort(key=lambda item: int(item.get("chunk_index") or 0))
        return rows

    def all_rows(self) -> List[Dict[str, str]]:
        """功能：从 Chroma 集合读取全部分片并规范为统一行格式。

        参数：
            无。

        返回值：
            按 source、chunk_index 排序的标准化分片行列表。

        异常：
            Chroma 客户端/集合操作异常可能向上抛出。
        """
        result = self._collection.get(include=["documents", "metadatas"])
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        rows = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            rows.append(_row_from_hit(str(doc), str((meta or {}).get("source", "")), dict(meta or {})))
        rows.sort(key=lambda item: (item.get("source") or "", int(item.get("chunk_index") or 0)))
        return rows


def _parse_metadata_json(raw: object) -> Dict[str, object]:
    """功能：将 metadata JSON 字符串或已有 dict 解析为字典。

    参数：
        raw: JSON 字符串、dict 或空值。

    返回值：
        解析后的元数据 dict；空输入或解析失败时返回 {}。

    异常：
        无。
    """
    if not raw:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw))
    except Exception:
        return {}


def _row_from_hit(content: str, source: str, meta: Dict[str, object]) -> Dict[str, str]:
    """功能：将检索/查询原始结果规范为统一的 str 字段行（含多模态元数据）。

    参数：
        content: 分片正文。
        source: 来源 key。
        meta: 原始元数据 dict。

    返回值：
        含 content、source 及可选 chunk_index、doc_id、block_*、image_* 等字段的字典；list/dict 值 JSON 序列化。

    异常：
        无。
    """
    row: Dict[str, str] = {"content": content, "source": source}
    for key in ("chunk_index", "doc_id", "block_start", "block_end", "block_types", "image_assets", "image_captions"):
        value = meta.get(key)
        if value is None or value == "":
            continue
        if isinstance(value, (list, dict)):
            row[key] = json.dumps(value, ensure_ascii=False)
        else:
            row[key] = str(value)
    return row


def _chunk_metadata(file_key: str, record, *, doc_id: str = "") -> Dict[str, object]:
    """功能：由多模态入库分片 record 构建 Chroma metadata dict。

    参数：
        file_key: 文件绝对路径 key（写入 source）。
        record: 分片记录对象（含 chunk_index、block_*、image_* 等属性）。
        doc_id: 可选文档 id；缺省时从 record.doc_id 读取。

    返回值：
        含 source、chunk_index、doc_id、block 范围/类型及图片资产 JSON 的元数据 dict。

    异常：
        无。
    """
    return {
        "source": file_key,
        "chunk_index": record.chunk_index,
        "doc_id": doc_id or getattr(record, "doc_id", ""),
        "block_start": getattr(record, "block_start", 0),
        "block_end": getattr(record, "block_end", 0),
        "block_types": record.block_types,
        "image_assets": json.dumps(record.image_assets, ensure_ascii=False),
        "image_captions": json.dumps(record.image_captions, ensure_ascii=False),
    }


def _normalize_hash_entry(value: object) -> Dict[str, str]:
    """功能：将 indexed_files 中某文件的 hash 条目规范为 dict 结构。

    参数：
        value: 旧版纯 md5 字符串或已含版本字段的 dict。

    返回值：
        dict；非 dict 输入时包装为 {"md5": str(value)}。

    异常：
        无。
    """
    if isinstance(value, dict):
        return dict(value)
    return {"md5": str(value or "")}


def _is_index_current(entry: Dict[str, str], current_md5: str, config: RagEnvConfig) -> bool:
    """功能：判断文件是否无需重建索引（内容 md5 与解析/分片/嵌入/视觉模型版本均一致）。

    参数：
        entry: 已索引文件的 hash 条目。
        current_md5: 磁盘文件当前 md5。
        config: 当前 RagEnvConfig（parser、embedding、chunking、vision 版本）。

    返回值：
        True 表示可跳过该文件；任一版本或 md5 不匹配则 False。

    异常：
        无。
    """
    if entry.get("md5") != current_md5:
        return False
    if not entry.get("parser_version") or entry.get("parser_version") != config.parser_version:
        return False
    if not entry.get("embedding_model") or entry.get("embedding_model") != config.embedding_model:
        return False
    if entry.get("chunking_version") != config.chunking_version:
        return False
    if entry.get("vision_model") != config.vision_model:
        return False
    return True


def _make_index_entry(current_md5: str, config: RagEnvConfig) -> Dict[str, str]:
    """功能：生成写入 indexed_files 的完整索引版本条目。

    参数：
        current_md5: 文件内容 md5。
        config: 当前 RagEnvConfig。

    返回值：
        含 md5、parser_version、embedding_model、chunking_version、vision_model 的 dict。

    异常：
        无。
    """
    return {
        "md5": current_md5,
        "parser_version": config.parser_version,
        "embedding_model": config.embedding_model,
        "chunking_version": config.chunking_version,
        "vision_model": config.vision_model,
    }


class VectorStoreService:
    """功能：RAG 向量库门面：Chroma 持久化、扫知识库、HTTP 嵌入、检索与按来源扩展上下文。

    参数：
        无（从 .env 与 RagEnvConfig 加载配置）。

    返回值：
        无（构造器）。

    异常：
        RuntimeError: Chroma 后端初始化失败时由 ChromaVectorBackend 抛出。
    """

    def __init__(self):
        """功能：从 .env 加载 RAG 配置并初始化 Chroma 向量后端与嵌入客户端。
        参数：
        - 无。
        返回值：
        - 无。
        异常：
        - RuntimeError：Chroma 后端初始化失败时抛出。
        """
        self.project_root = Path(__file__).resolve().parents[1]
        load_dotenv(self.project_root / ".env", override=False)
        self.top_k = int(os.getenv("RAG_TOP_K", "4"))
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE", "700"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))
        self.data_dir = self._resolve_project_path(os.getenv("RAG_DATA_DIR") or "knowledge")
        self.persist_dir = self._resolve_project_path(os.getenv("RAG_PERSIST_DIR") or ".rag_store")
        self.collection_name = os.getenv("RAG_COLLECTION_NAME") or "knowledge"
        self.rag_config = RagEnvConfig.load(self.project_root)
        self.model = self.rag_config.embedding_model
        self.multimodal_enabled = self.rag_config.multimodal_enabled
        self.allowed_types = tuple(
            x.strip().lower() for x in (os.getenv("RAG_ALLOWED_TYPES") or "").split(",") if x.strip()
        )
        if not self.allowed_types:
            if self.multimodal_enabled:
                self.allowed_types = (
                    ".txt",
                    ".md",
                    ".pdf",
                    ".docx",
                    ".png",
                    ".jpg",
                    ".jpeg",
                    ".webp",
                )
            else:
                self.allowed_types = (".txt", ".md", ".pdf")
        self.enabled = _env_bool("RAG_ENABLED", False)

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_url = self.rag_config.base_url.rstrip("/") + "/embeddings"
        self._key_pool = RagHttpKeyPool.from_env(
            fallback_key=self.rag_config.api_key,
            service="嵌入",
        )
        self._active_key_idx = 0
        self._http_timeout = self.rag_config.http_timeout
        self._embedding_retries = self.rag_config.embedding_retries
        self._http = requests.Session()
        self._keyword_retriever = None
        self._keyword_corpus_size = -1
        self._rerank_client = None
        if self.rag_config.rerank_enabled:
            from rag.rerank import RerankClient

            self._rerank_client = RerankClient(self.rag_config)
        self._ingestion_pipeline = None
        if self.multimodal_enabled:
            from rag.ingestion.pipeline import DocumentIngestionPipeline

            self._ingestion_pipeline = DocumentIngestionPipeline(
                config=self.rag_config,
                persist_dir=self.persist_dir,
                data_dir=self.data_dir,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            rag_log(
                f"[RAG] 多模态入库已开启；嵌入={self.model!r} 图片识别={self.rag_config.vision_model!r}",
                flush=True,
            )

        self._hash_file = self.persist_dir / "indexed_files.json"
        rag_log(f"[RAG] 打开向量持久化目录: {self.persist_dir}", flush=True)
        self._backend: VectorBackend = ChromaVectorBackend(
            persist_dir=self.persist_dir,
            collection_name=self.collection_name,
        )
        rag_log("[RAG] Chroma 集合就绪。", flush=True)

    def _resolve_project_path(self, value: str) -> Path:
        """功能：将相对路径解析为基于项目根的绝对 Path。

        参数：
            value: 路径字符串（相对或绝对）。

        返回值：
            resolve() 后的绝对 Path。

        异常：
            无。
        """
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _file_hashes(self) -> Dict[str, str]:
        """功能：读取 indexed_files.json 增量索引状态。

        参数：
            无。

        返回值：
            文件 key → 版本条目 dict 的映射；文件不存在或 JSON 损坏时返回 {}。

        异常：
            无。
        """
        for path in (
            self._hash_file,
            self.persist_dir / "indexed_files_chroma.json",
            self.persist_dir / "indexed_files_sqlite.json",
        ):
            if not path.exists():
                continue
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
        return {}

    def _save_hashes(self, data: Dict[str, str]) -> None:
        """功能：将增量索引 hash 映射原子写回 indexed_files.json。

        参数：
            data: 完整 hash 映射。

        返回值：
            无。

        异常：
            OSError 等文件写入异常可能向上抛出。
        """
        self._hash_file.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(data, ensure_ascii=False, indent=2)
        tmp = self._hash_file.with_suffix(self._hash_file.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self._hash_file)

    def _commit_file_index(self, hashes: Dict[str, object], file_key: str, current_md5: str) -> None:
        """功能：标记单个文件索引完成并立即持久化进度。

        参数：
            hashes: 当前增量索引 hash 映射（就地更新）。
            file_key: 文件绝对路径 key。
            current_md5: 文件内容 md5。

        返回值：
            无。

        异常：
            OSError 等文件写入异常可能向上抛出。
        """
        hashes[file_key] = _make_index_entry(current_md5, self.rag_config)
        self._save_hashes(hashes)

    @staticmethod
    def _md5(path: Path) -> str:
        """功能：流式计算文件内容 MD5（1MB 块），用于判断文件是否变更。

        参数：
            path: 文件路径。

        返回值：
            十六进制 md5 字符串。

        异常：
            OSError 等文件读取异常可能向上抛出。
        """
        h = hashlib.md5()
        with path.open("rb") as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def _read_text(self, file_path: Path) -> str:
        """功能：非多模态路径下读取 .txt/.md/.pdf 纯文本（无 DocumentIngestionPipeline 时使用）。

        参数：
            file_path: 知识库文件路径。

        返回值：
            提取的文本；不支持的扩展名返回空字符串。

        异常：
            RuntimeError: PDF 且未安装 pypdf。
        """
        suffix = file_path.suffix.lower()
        if suffix in (".txt", ".md"):
            return file_path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            if PdfReader is None:
                raise RuntimeError("未安装 pypdf，无法解析 PDF 文件。")
            reader = PdfReader(str(file_path))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        return ""

    def _split_text(self, text: str) -> List[str]:
        """功能：固定窗口滑动分片（chunk_size / chunk_overlap），用于纯文本入库路径。

        参数：
            text: 原始文本。

        返回值：
            非空分片字符串列表；空白输入返回 []。

        异常：
            无。
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        chunks: List[str] = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(cleaned):
            chunks.append(cleaned[start : start + self.chunk_size])
            start += step
        return [x for x in chunks if x.strip()]

    def _display_path(self, file_path: Path) -> str:
        """功能：生成相对知识库目录的展示路径，便于日志阅读。

        参数：
            file_path: 绝对或相对文件路径。

        返回值：
            相对 data_dir 的路径字符串；无法 relativize 时返回 str(file_path)。

        异常：
            无。
        """
        try:
            return str(file_path.relative_to(self.data_dir))
        except ValueError:
            return str(file_path)

    def _get_keyword_retriever(self):
        """功能：懒加载 KeywordRetriever，语料规模变化时自动重建。

        参数：
            无。

        返回值：
            与当前向量库全量分片同步的 KeywordRetriever 实例。

        异常：
            无（KeywordRetriever 构造异常可能向上抛出）。
        """
        rows = self._backend.all_rows()
        if self._keyword_retriever is None or self._keyword_corpus_size != len(rows):
            from rag.keyword_retrieval import KeywordRetriever

            self._keyword_retriever = KeywordRetriever(rows)
            self._keyword_corpus_size = len(rows)
        return self._keyword_retriever

    def _embed(self, texts: List[str], *, progress_label: str = "") -> List[List[float]]:
        """功能：分批（每批 10 条）调用嵌入 HTTP 接口，可选输出进度日志。

        参数：
            texts: 待嵌入文本列表。
            progress_label: 非空时在 rag_log 中标注批次进度（建索引时用文件短路径）。

        返回值：
            与 texts 等长的 embedding 向量列表；空输入返回 []。

        异常：
            RuntimeError: 某批 _request_embeddings 重试耗尽后抛出。
        """
        if not texts:
            return []
        all_embeddings: List[List[float]] = []
        batch_size = 10
        total_batches = (len(texts) + batch_size - 1) // batch_size
        label = f" ({progress_label})" if progress_label else ""
        log_batches = bool(progress_label)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            bn = i // batch_size + 1
            if log_batches:
                rag_log(f"[RAG] 请求向量嵌入{label}：第 {bn}/{total_batches} 批，本批 {len(batch)} 条...", flush=True)
            all_embeddings.extend(self._request_embeddings(batch))
            if log_batches:
                rag_log(f"[RAG] 嵌入返回{label}：第 {bn}/{total_batches} 批完成。", flush=True)
        return all_embeddings

    def _request_embeddings(self, batch: List[str]) -> List[List[float]]:
        """功能：向 {base_url}/embeddings 发送 OpenAI 兼容 POST，带 Bearer 鉴权、429 换 key 与指数退避重试。

        参数：
            batch: 单批待嵌入文本（通常 ≤10 条）。

        返回值：
            与 batch 等长、按 index 排序的 embedding 列表。

        异常：
            RuntimeError: 返回数量/格式异常、SOCKS 代理缺 PySocks、或重试耗尽后的最后一次错误。
        """
        if not self._key_pool.api_keys:
            raise RuntimeError("未配置嵌入 API Key（OPENROUTER_API_KEY / OPENROUTER_API_KEYS）。")

        payload = {"model": self.model, "input": batch}
        pool_size = len(self._key_pool)
        key_indices = self._key_pool.iter_indices()
        last_error: Optional[Exception] = None

        for key_pos, key_idx in enumerate(key_indices):
            headers = {
                "Authorization": f"Bearer {self._key_pool.bearer(key_idx)}",
                "Content-Type": "application/json",
            }
            for attempt in range(self._embedding_retries + 1):
                try:
                    response = self._http.post(
                        self._embedding_url,
                        headers=headers,
                        json=payload,
                        timeout=self._http_timeout,
                    )
                    response.raise_for_status()
                    data = response.json()
                    items = data.get("data") or []
                    items = sorted(items, key=lambda item: int(item.get("index", 0)))
                    embeddings = [item.get("embedding") for item in items]
                    if len(embeddings) != len(batch) or any(not isinstance(item, list) for item in embeddings):
                        raise RuntimeError("嵌入接口返回数量或格式异常。")
                    self._active_key_idx = key_idx
                    return embeddings
                except Exception as exc:
                    last_error = exc
                    if is_rate_limit_error(exc):
                        self._key_pool.mark_rate_limited(key_idx)
                        if key_pos + 1 < len(key_indices):
                            self._key_pool.log_switch(key_idx)
                        break
                    if attempt >= self._embedding_retries:
                        break
                    hint = ""
                    if "Missing dependencies for SOCKS support" in str(exc):
                        hint = "；检测到 SOCKS 代理依赖缺失，请在当前 Python 环境执行：python -m pip install PySocks"
                    wait_seconds = 0.8 * (attempt + 1)
                    rag_log(
                        f"[RAG] 嵌入请求失败（key {key_idx + 1}/{pool_size}），{wait_seconds:.1f}s 后重试 "
                        f"({attempt + 1}/{self._embedding_retries})：{exc}{hint}",
                        flush=True,
                    )
                    time.sleep(wait_seconds)
        if last_error and "Missing dependencies for SOCKS support" in str(last_error):
            raise RuntimeError(
                "嵌入请求失败：当前 .env 配置了 SOCKS 代理，但当前 Python 环境缺少 PySocks。"
                "请执行 python -m pip install PySocks，或重新安装 requirements.txt。"
            ) from last_error
        raise RuntimeError(f"嵌入请求失败：{last_error}") from last_error

    def build_or_update_index(self) -> Dict[str, int]:
        """功能：增量扫描知识库：md5/版本未变则跳过；变更则删旧分片、多模态或纯文本入库并写 hash。

        参数：
            无。

        返回值：
            {"indexed_files": 成功文件数, "indexed_chunks": 写入分片总数}；RAG 未开启或目录不存在时均为 0。

        异常：
            无（单文件索引失败仅 rag_log 记录并 continue，不中断全盘扫描）。
        """
        if not self.enabled:
            rag_log("[RAG] RAG_ENABLED 未开启，跳过建索引。", flush=True)
            return {"indexed_files": 0, "indexed_chunks": 0}
        if not self.data_dir.exists():
            rag_log(f"[RAG] 知识目录不存在，跳过：{self.data_dir}", flush=True)
            return {"indexed_files": 0, "indexed_chunks": 0}

        rag_log(
            f"[RAG] 开始扫描知识库：{self.data_dir}（类型 {self.allowed_types}），向量目录 {self.persist_dir}",
            flush=True,
        )
        rag_log(
            "[RAG] 若长时间停在某一步：① 关闭占用同一 persist 目录的 Agent/Streamlit/企微进程；"
            "② 检查嵌入网关网络与 RAG_HTTP_TIMEOUT_SECONDS；"
            "③ Windows 上先运行 python check_env.py --chroma-only，确认 Python 3.12 x64、Chroma 与代理依赖正常。",
            flush=True,
        )

        prev_hashes = self._file_hashes()
        new_hashes = dict(prev_hashes)
        indexed_files = 0
        indexed_chunks = 0

        try:
            for file_path in sorted(self.data_dir.rglob("*")):
                if not file_path.is_file() or file_path.suffix.lower() not in self.allowed_types:
                    continue
                file_key = str(file_path.resolve())
                current_md5 = self._md5(file_path)
                entry = _normalize_hash_entry(prev_hashes.get(file_key))
                if _is_index_current(entry, current_md5, self.rag_config):
                    continue

                short = self._display_path(file_path)
                rag_log(f"[RAG] 索引文件: {short} ...", flush=True)
                if self._ingestion_pipeline:
                    self._ingestion_pipeline.delete_file_assets(file_path)
                try:
                    if self._ingestion_pipeline:
                        chunk_records = self._ingestion_pipeline.ingest_file(
                            file_path,
                            display_path=short,
                        )
                        if not chunk_records:
                            self._commit_file_index(new_hashes, file_key, current_md5)
                            rag_log(f"[RAG]   跳过（无有效分片）: {short}", flush=True)
                            continue
                        chunk_texts = [r.text for r in chunk_records]
                        rag_log(
                            f"[RAG]   分片数 {len(chunk_texts)}，正在请求嵌入模型 {self.model!r} ...",
                            flush=True,
                        )
                        embeddings = self._embed(chunk_texts, progress_label=short)
                        self._backend.delete_source(file_key)
                        ids = [
                            f"{file_key}:{r.chunk_index}:{current_md5[:8]}"
                            for r in chunk_records
                        ]
                        metadatas = [
                            _chunk_metadata(file_key, r) for r in chunk_records
                        ]
                        self._backend.upsert(
                            ids=ids,
                            documents=chunk_texts,
                            embeddings=embeddings,
                            metadatas=metadatas,
                        )
                        indexed_chunks += len(chunk_records)
                    else:
                        text = self._read_text(file_path)
                        chunks = self._split_text(text)
                        if not chunks:
                            self._commit_file_index(new_hashes, file_key, current_md5)
                            rag_log(f"[RAG]   跳过（无有效分片）: {short}", flush=True)
                            continue
                        rag_log(
                            f"[RAG]   分片数 {len(chunks)}，正在请求嵌入模型 {self.model!r} ...",
                            flush=True,
                        )
                        embeddings = self._embed(chunks, progress_label=short)
                        self._backend.delete_source(file_key)
                        ids = [f"{file_key}:{i}:{current_md5[:8]}" for i in range(len(chunks))]
                        metadatas = [{"source": file_key, "chunk_index": i} for i in range(len(chunks))]
                        self._backend.upsert(
                            ids=ids,
                            documents=chunks,
                            embeddings=embeddings,
                            metadatas=metadatas,
                        )
                        indexed_chunks += len(chunks)
                except Exception as exc:
                    rag_log(f"[RAG]   索引失败（已跳过）: {short} — {exc}", flush=True)
                    continue

                self._commit_file_index(new_hashes, file_key, current_md5)
                indexed_files += 1
                rag_log(f"[RAG]   完成: {short}", flush=True)
        except KeyboardInterrupt:
            rag_log(
                f"[RAG] 用户中断，进度已保存：已完成 {indexed_files} 个文件、{indexed_chunks} 个分片；"
                "下次运行将跳过已索引文件。",
                flush=True,
            )
            raise

        rag_log(f"[RAG] 全部完成：文件数={indexed_files}，分片数={indexed_chunks}", flush=True)
        return {"indexed_files": indexed_files, "indexed_chunks": indexed_chunks}

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, str]]:
        """功能：对 query 做嵌入后向后端检索 top-k；开启关键词召回时用 RRF 融合，再可选 rerank。

        参数：
            query: 用户查询文本。
            top_k: 返回条数；默认 self.top_k。

        返回值：
            标准化分片行列表；空 query 返回 []。

        异常：
            RuntimeError: 嵌入请求失败时由 _embed/_request_embeddings 抛出。
        """
        if not query.strip():
            return []
        final_k = top_k or self.top_k
        rerank_top_n = self.rag_config.rerank_top_n if self.rag_config.rerank_enabled else final_k
        target_k = max(1, top_k or rerank_top_n or final_k)
        candidate_k = max(target_k, self.rag_config.retrieve_candidates if self.rag_config.rerank_enabled else final_k)
        if self.rag_config.keyword_enabled:
            candidate_k = max(candidate_k, self.rag_config.keyword_top_n)

        query_emb = self._embed([query])[0]
        vector_rows = self._backend.query(query_embedding=query_emb, n_results=candidate_k)
        rows = list(vector_rows)
        if self.rag_config.keyword_enabled:
            try:
                keyword_rows = self._get_keyword_retriever().retrieve(
                    query,
                    top_k=max(self.rag_config.keyword_top_n, candidate_k),
                )
                rows = _rrf_fuse(vector_rows, keyword_rows, k=self.rag_config.rrf_k)
            except Exception as exc:
                rag_log(f"[RAG] keyword recall failed, vector recall only: {exc}", flush=True)
                rows = vector_rows

        if self._rerank_client is not None:
            n_rerank_in = len(rows)
            rerank_started = time.perf_counter()
            ranked = self._rerank_client.rerank(query, rows)
            rerank_ms = int((time.perf_counter() - rerank_started) * 1000)
            if any(row.get("rerank_score") is not None for row in ranked):
                rows = _filter_reranked_rows(
                    ranked,
                    min_score=self.rag_config.rerank_min_score,
                    top_n=target_k,
                )
                from rag.rerank import format_rerank_trace

                rag_trace_log(
                    format_rerank_trace(
                        ms=rerank_ms,
                        n_in=n_rerank_in,
                        n_out=len(rows),
                        rows=rows,
                        model=self.rag_config.rerank_model,
                    )
                )
            else:
                rows = ranked[:target_k]
                rag_trace_log(
                    f"Rerank {rerank_ms}ms | {n_rerank_in}→{len(rows)} 条（无分数，按召回截断）"
                )
        else:
            rows = rows[:target_k]
        if not self.multimodal_enabled:
            return rows
        from rag.doc_assets import enrich_retrieved_rows

        return enrich_retrieved_rows(
            rows,
            persist_dir=self.persist_dir,
            vision_model=self.rag_config.vision_model,
        )

    def expand_sources(
        self,
        rows: List[Dict[str, str]],
        *,
        max_chars: int,
        max_sources: int = 1,
    ) -> List[Dict[str, str]]:
        """功能：按命中行扩展同一 source 的邻近分片，在 max_chars 内拼接上下文（flow expand）。

        参数：
            rows: retrieve 返回的命中行（按顺序处理，去重 source）。
            max_chars: 每个 source 扩展内容总字符上限。
            max_sources: 最多扩展的文件来源数，默认 1。

        返回值：
            扩展后的分片行列表；多模态开启时同样经 enrich_retrieved_rows 补全图片资产。

        异常：
            无（后端 get_by_source 异常可能向上抛出）。
        """
        expanded: List[Dict[str, str]] = []
        seen_sources: set[str] = set()
        for row in rows:
            source = (row.get("source") or "").strip()
            if not source or source in seen_sources:
                continue
            seen_sources.add(source)
            source_rows = self._backend.get_by_source(source)
            ordered_rows = _order_rows_around_hit(source_rows, row)
            selected: List[Dict[str, str]] = []
            total_chars = 0
            for source_row in ordered_rows:
                content = (source_row.get("content") or "").strip()
                if not content:
                    continue
                if total_chars and total_chars + len(content) > max_chars:
                    break
                selected.append(source_row)
                total_chars += len(content)
            selected.sort(key=lambda item: int(item.get("chunk_index") or 0))
            expanded.extend(selected)
            if len(seen_sources) >= max_sources:
                break
        if not self.multimodal_enabled:
            return expanded
        from rag.doc_assets import enrich_retrieved_rows

        return enrich_retrieved_rows(
            expanded,
            persist_dir=self.persist_dir,
            vision_model=self.rag_config.vision_model,
        )


def _order_rows_around_hit(source_rows: List[Dict[str, str]], hit_row: Dict[str, str]) -> List[Dict[str, str]]:
    """功能：将同一文件的全部分片按与命中 chunk_index 的距离排序，优先扩展邻近段落。

    参数：
        source_rows: get_by_source 返回的分片列表。
        hit_row: 检索命中的那一行（含 chunk_index）。

    返回值：
        按 |chunk_index - hit| 升序、同距离再按 chunk_index 排序的列表；无 hit_index 时原序返回。

    异常：
        无。
    """
    if not source_rows:
        return []
    hit_index = _safe_int(hit_row.get("chunk_index"))
    if hit_index is None:
        return source_rows
    return sorted(
        source_rows,
        key=lambda item: (
            abs((_safe_int(item.get("chunk_index")) or 0) - hit_index),
            _safe_int(item.get("chunk_index")) or 0,
        ),
    )


def _row_key(row: Dict[str, str]) -> tuple[str, str]:
    """功能：生成分片行的去重键（source + chunk_index）。

    参数：
        row: 标准化分片行 dict。

    返回值：
        (source, chunk_index) 元组。

    异常：
        无。
    """
    return ((row.get("source") or "").strip(), str(row.get("chunk_index") or ""))


def _rrf_fuse(*ranked_groups: List[Dict[str, str]], k: int = 60) -> List[Dict[str, str]]:
    """功能：Reciprocal Rank Fusion，合并多路召回排序（score = sum(1 / (k + rank))）。

    参数：
        ranked_groups: 多路已排序的分片行列表。
        k: RRF 平滑常数，默认 60。

    返回值：
        按 rrf_score 降序排列的融合分片行列表。

    异常：
        无。
    """
    scores: Dict[tuple[str, str], float] = {}
    rows_by_key: Dict[tuple[str, str], Dict[str, str]] = {}
    for group in ranked_groups:
        for rank, row in enumerate(group, start=1):
            key = _row_key(row)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            existing = rows_by_key.get(key)
            if existing is None:
                rows_by_key[key] = dict(row)
            else:
                existing.update({field: value for field, value in row.items() if value not in (None, "")})
    fused: List[Dict[str, str]] = []
    for key, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        item = dict(rows_by_key[key])
        item["rrf_score"] = f"{score:.6f}"
        fused.append(item)
    return fused


def _filter_reranked_rows(rows: List[Dict[str, str]], *, min_score: float, top_n: int) -> List[Dict[str, str]]:
    """功能：按 rerank 分数阈值与 top_n 截断重排结果。

    参数：
        rows: 含 rerank_score 的分片行列表（已按分数排序）。
        min_score: 最低 rerank 分数阈值。
        top_n: 最多保留条数。

    返回值：
        满足 min_score 的前 top_n 条分片行。

    异常：
        无（无效 rerank_score 行被跳过）。
    """
    selected: List[Dict[str, str]] = []
    for row in rows:
        raw_score = row.get("rerank_score")
        if raw_score is None:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if score >= min_score:
            selected.append(row)
        if len(selected) >= top_n:
            break
    return selected


def _safe_int(value: object) -> int | None:
    """功能：将 chunk_index 等字段安全转为 int。

    参数：
        value: 任意可转 int 的值；空值按 0 尝试。

    返回值：
        整数；TypeError/ValueError 时返回 None。

    异常：
        无。
    """
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return None
