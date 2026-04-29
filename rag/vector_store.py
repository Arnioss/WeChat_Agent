import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from rag.sqlite_chroma_compat import ensure_chroma_sqlite

ensure_chroma_sqlite()

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None


def _env_bool(name: str, default: bool) -> bool:
    """功能：读取环境变量并解析为布尔值。
    参数：
    - name：环境变量名称或对象名称。
    - default：默认值。
    返回值：
    - bool：解析后的布尔结果；变量为空时返回 `default`。
    """
    value = (os.getenv(name) or "").strip().lower()
    if not value:
        return default
    return value in ("1", "true", "yes", "on")


class VectorStoreService:
    """功能：管理知识文件分片、向量化与相似度检索。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self):
        """功能：加载 RAG 配置并初始化向量库、Embedding 客户端与索引目录。
        参数：
        - 无。
        返回值：
        - 无。缺少关键环境变量（如 `OPENAI_BASE_URL`）会立即抛出异常阻止继续运行。
        """
        load_dotenv()
        self.top_k = int(os.getenv("RAG_TOP_K"))
        self.chunk_size = int(os.getenv("RAG_CHUNK_SIZE"))
        self.chunk_overlap = int(os.getenv("RAG_CHUNK_OVERLAP"))
        self.data_dir = Path(os.getenv("RAG_DATA_DIR")).resolve()
        self.persist_dir = Path(os.getenv("RAG_PERSIST_DIR")).resolve()
        self.collection_name = os.getenv("RAG_COLLECTION_NAME")
        self.model = os.getenv("RAG_EMBEDDING_MODEL")
        self.allowed_types = tuple(
            x.strip().lower() for x in (os.getenv("RAG_ALLOWED_TYPES") or "").split(",") if x.strip()
        )
        self.enabled = _env_bool("RAG_ENABLED", False)

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._hash_file = self.persist_dir / "indexed_files.json"

        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("缺少环境变量 OPENAI_BASE_URL，请在 .env 文件中设置。")
        api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            raw_keys = (os.getenv("OPENROUTER_API_KEYS") or "").strip()
            api_key = raw_keys.split(",")[0].strip() if raw_keys else ""
        self._openai_client = OpenAI(base_url=base_url, api_key=api_key)

        self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._chroma_client.get_or_create_collection(name=self.collection_name)

    def _file_hashes(self) -> Dict[str, str]:
        """功能：读取已索引文件哈希映射。
        参数：
        - 无。
        返回值：
        - Dict[str, str]：文件路径到 MD5 的映射字典。
        """
        if not self._hash_file.exists():
            return {}
        try:
            return json.loads(self._hash_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_hashes(self, data: Dict[str, str]) -> None:
        """功能：保存已索引文件哈希映射到磁盘。
        参数：
        - data：文件路径到 MD5 的映射字典。
        返回值：
        - 无。
        """
        self._hash_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _md5(path: Path) -> str:
        """功能：计算文件 MD5 指纹。
        参数：
        - path：文件路径。
        返回值：
        - str：文件 MD5 十六进制字符串。
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
        """功能：按文件类型读取文本内容（txt/md/pdf）。
        参数：
        - file_path：文件路径。
        返回值：
        - str：读取到的文本内容，不支持类型时返回空串。
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
        """功能：按配置切分文本为重叠分片。
        参数：
        - text：待处理文本内容。
        返回值：
        - List[str]：可用于向量化的文本分片列表。
        """
        cleaned = (text or "").strip()
        if not cleaned:
            return []
        chunks: List[str] = []
        start = 0
        step = max(1, self.chunk_size - self.chunk_overlap)
        while start < len(cleaned):
            chunks.append(cleaned[start:start + self.chunk_size])
            start += step
        return [x for x in chunks if x.strip()]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """功能：调用嵌入模型把文本批量转换为向量。
        参数：
        - texts：待向量化文本列表。
        返回值：
        - List[List[float]]：与输入文本一一对应的向量列表。
        """
        if not texts:
            return []
        # Some embedding gateways cap one request to <=10 inputs.
        all_embeddings: List[List[float]] = []
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            resp = self._openai_client.embeddings.create(model=self.model, input=batch)
            all_embeddings.extend(item.embedding for item in resp.data)
        return all_embeddings

    def build_or_update_index(self) -> Dict[str, int]:
        """功能：扫描知识文件并增量更新向量索引。
        参数：
        - 无。
        返回值：
        - Dict[str, int]：包含更新文件数与写入分片数的统计信息。
        """
        if not self.enabled:
            return {"indexed_files": 0, "indexed_chunks": 0}
        if not self.data_dir.exists():
            return {"indexed_files": 0, "indexed_chunks": 0}

        prev_hashes = self._file_hashes()
        new_hashes = dict(prev_hashes)
        indexed_files = 0
        indexed_chunks = 0

        for file_path in sorted(self.data_dir.rglob("*")):
            if not file_path.is_file() or file_path.suffix.lower() not in self.allowed_types:
                continue
            file_key = str(file_path)
            current_md5 = self._md5(file_path)
            if prev_hashes.get(file_key) == current_md5:
                continue

            self._collection.delete(where={"source": file_key})
            text = self._read_text(file_path)
            chunks = self._split_text(text)
            if not chunks:
                new_hashes[file_key] = current_md5
                continue

            embeddings = self._embed(chunks)
            ids = [f"{file_key}:{i}:{current_md5[:8]}" for i in range(len(chunks))]
            metadatas = [{"source": file_key, "chunk_index": i} for i in range(len(chunks))]
            self._collection.upsert(ids=ids, documents=chunks, embeddings=embeddings, metadatas=metadatas)

            new_hashes[file_key] = current_md5
            indexed_files += 1
            indexed_chunks += len(chunks)

        self._save_hashes(new_hashes)
        return {"indexed_files": indexed_files, "indexed_chunks": indexed_chunks}

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, str]]:
        """功能：按查询内容检索最相关的技能或文档。
        参数：
        - query：用户输入的问题文本。
        - top_k：可选召回条数，未传入时使用默认配置值。
        返回值：
        - List[Dict[str, str]]：检索命中文本及其来源路径列表。
        """
        if not query.strip():
            return []
        k = top_k or self.top_k
        query_emb = self._embed([query])[0]
        result = self._collection.query(query_embeddings=[query_emb], n_results=k)
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        rows = []
        for i, doc in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            rows.append({"content": doc, "source": str((meta or {}).get("source", ""))})
        return rows

