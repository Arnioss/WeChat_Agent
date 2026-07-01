"""Rerank 客户端：调用 OpenAI 兼容网关对召回结果重排序。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

import requests

from rag.env_config import RagEnvConfig
from rag.http_key_pool import RagHttpKeyPool, is_rate_limit_error
from rag.logging_utils import rag_log, rag_trace_log


@dataclass(frozen=True)
class RerankResult:
    """功能：单条 rerank 结果，含原文档索引与相关性分数。
    参数：
    - 无（dataclass 字段见 index、score）。
    返回值：
    - 无。
    """

    index: int
    score: float


class RerankClient:
    """功能：向 RAG 网关 /rerank 端点发送 query+documents 并重排召回行。
    参数：
    - 无（通过 __init__ 注入 config 与 session）。
    返回值：
    - 无。
    """

    def __init__(self, config: RagEnvConfig, *, session: requests.Session | None = None):
        """功能：初始化 Rerank 客户端与 Key 池。
        参数：
        - config：RAG 环境配置，含 base_url、rerank_model、http_timeout。
        - session：可选 requests Session；默认新建。
        返回值：
        - 无。
        """
        self.config = config
        self.session = session or requests.Session()
        endpoint = config.rerank_endpoint or "/rerank"
        if not endpoint.startswith("/"):
            endpoint = "/" + endpoint
        self.url = config.base_url.rstrip("/") + endpoint
        self._key_pool = RagHttpKeyPool.from_env(fallback_key=config.api_key, service="Rerank")

    def rerank(self, query: str, rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """功能：对召回行按 query 相关性重排序；失败时回退原顺序。
        参数：
        - query：用户查询文本。
        - rows：向量/关键词召回的行列表。
        返回值：
        - List[Dict[str, str]]：重排后的行；失败或未配置 Key 时返回原 rows。
        """
        if not rows or not query.strip():
            return rows
        if not self._key_pool.api_keys:
            rag_log("[RAG] Rerank 未配置 API Key，回退召回顺序。", flush=True)
            return rows
        started = time.perf_counter()
        payload = {
            "model": self.config.rerank_model,
            "query": query,
            "documents": [self._document_text(row) for row in rows],
        }
        key_indices = self._key_pool.iter_indices()
        last_error: Exception | None = None
        for key_pos, key_idx in enumerate(key_indices):
            try:
                response = self.session.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self._key_pool.bearer(key_idx)}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.config.http_timeout,
                )
                response.raise_for_status()
                results = _parse_rerank_response(response.json())
                if not results:
                    raise RuntimeError("empty rerank result")
                return self._apply_results(rows, results)
            except Exception as exc:
                last_error = exc
                if is_rate_limit_error(exc) and key_pos + 1 < len(key_indices):
                    self._key_pool.mark_rate_limited(key_idx)
                    self._key_pool.log_switch(key_idx)
                    continue
                break
        ms = int((time.perf_counter() - started) * 1000)
        rag_trace_log(f"Rerank 失败({ms}ms)，回退召回顺序：{last_error}")
        rag_log(f"[RAG] rerank failed, fallback to recall order: {last_error}", flush=True)
        return rows

    @staticmethod
    def _document_text(row: Dict[str, str]) -> str:
        """功能：将召回行格式化为 rerank API 的 document 文本。
        参数：
        - row：含 content、source、chunk_index 的字典。
        返回值：
        - str：带 source/chunk 前缀的正文。
        """
        content = (row.get("content") or "").strip()
        source = (row.get("source") or "").strip()
        chunk_index = (row.get("chunk_index") or "").strip()
        if source or chunk_index:
            return f"source: {source}\nchunk: {chunk_index}\n{content}"
        return content

    def _apply_results(self, rows: List[Dict[str, str]], results: List[RerankResult]) -> List[Dict[str, str]]:
        """功能：按 rerank 分数重排行并写入 rerank_score 字段。
        参数：
        - rows：原始召回行。
        - results：网关返回的 index/score 列表。
        返回值：
        - List[Dict[str, str]]：重排后的行；未命中 index 的行追加在末尾。
        """
        ranked: List[Dict[str, str]] = []
        used: set[int] = set()
        for result in sorted(results, key=lambda item: item.score, reverse=True):
            if result.index < 0 or result.index >= len(rows) or result.index in used:
                continue
            used.add(result.index)
            row = dict(rows[result.index])
            row["rerank_score"] = f"{result.score:.6f}"
            ranked.append(row)
        for index, row in enumerate(rows):
            if index not in used:
                ranked.append(dict(row))
        return ranked


def _parse_rerank_response(data: Any) -> List[RerankResult]:
    """功能：解析 rerank API JSON 为 RerankResult 列表。
    参数：
    - data：网关 JSON 响应体。
    返回值：
    - List[RerankResult]：有效 index/score 条目；无法解析的项跳过。
    """
    items = _extract_items(data)
    results: List[RerankResult] = []
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        raw_index = (
            item.get("index")
            if item.get("index") is not None
            else item.get("document_index")
            if item.get("document_index") is not None
            else item.get("rank")
        )
        try:
            index = int(raw_index if raw_index is not None else position)
        except (TypeError, ValueError):
            index = position
        score = _extract_score(item)
        if score is None:
            continue
        results.append(RerankResult(index=index, score=score))
    return results


def _extract_items(data: Any) -> Iterable[Any]:
    """功能：从多种 rerank 响应结构中提取结果列表。
    参数：
    - data：JSON 根对象或列表。
    返回值：
    - Iterable[Any]：结果项迭代器；无法识别时为空。
    """
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        return []
    for key in ("results", "data", "documents"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    output = data.get("output")
    if isinstance(output, dict):
        for key in ("results", "data", "documents"):
            value = output.get(key)
            if isinstance(value, list):
                return value
    if isinstance(output, list):
        return output
    return []


def format_rerank_trace(
    *,
    ms: int,
    n_in: int,
    n_out: int,
    rows: List[Dict[str, str]],
    model: str,
) -> str:
    """功能：格式化 rerank 链路追踪日志一行文本。
    参数：
    - ms：耗时毫秒。
    - n_in：输入条数。
    - n_out：输出条数。
    - rows：重排后的行（取 top rerank_score）。
    - model：rerank 模型名。
    返回值：
    - str：人类可读的 trace 摘要。
    """
    scores: List[str] = []
    for row in rows:
        raw = row.get("rerank_score")
        if raw is None:
            continue
        try:
            scores.append(f"{float(raw):.2f}")
        except (TypeError, ValueError):
            continue
        if len(scores) >= 4:
            break
    top = ",".join(scores) if scores else "-"
    return f"Rerank {ms}ms | {n_in}→{n_out} 条 | top {top} | {model}"


def _extract_score(item: Dict[str, Any]) -> float | None:
    """功能：从 rerank 结果项中提取浮点分数。
    参数：
    - item：单条结果 dict。
    返回值：
    - float | None：relevance_score/score 等字段；无有效值时 None。
    """
    for key in ("relevance_score", "score", "rerank_score", "similarity"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None
