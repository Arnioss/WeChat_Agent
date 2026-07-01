"""本地关键词/BM25 召回：精确标识符与短中文查询的补充检索。"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List


TOKEN_RE = re.compile(
    r"https?://[^\s)）]+|"
    r"[A-Za-z]+[A-Za-z0-9_.:/#*\-]*|"
    r"0x[0-9A-Fa-f]+|"
    r"[0-9A-Fa-f]{4,}|"
    r"\d+(?:\.\d+)*|"
    r"[\u4e00-\u9fff]"
)


@dataclass
class _IndexedRow:
    """功能：索引后的文档行，含分词与 BM25 统计字段。
    参数：
    - 无（dataclass 字段见 row、tokens 等）。
    返回值：
    - 无。
    """

    row: Dict[str, str]
    tokens: List[str]
    term_counts: Counter[str]
    length: int
    raw_lower: str


class KeywordRetriever:
    """功能：对向量库全量行建立 BM25 索引并执行关键词检索。
    参数：
    - 无。
    返回值：
    - 无。
    """

    def __init__(self, rows: Iterable[Dict[str, str]]):
        """功能：从召回行构建倒排索引与 IDF。
        参数：
        - rows：含 content 等字段的文档行可迭代对象。
        返回值：
        - 无。
        """
        self.docs = [_make_indexed_row(row) for row in rows if (row.get("content") or "").strip()]
        self.avg_len = sum(doc.length for doc in self.docs) / max(1, len(self.docs))
        doc_freq: Counter[str] = Counter()
        for doc in self.docs:
            doc_freq.update(set(doc.tokens))
        total = max(1, len(self.docs))
        self.idf = {
            token: math.log(1.0 + (total - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def retrieve(self, query: str, *, top_k: int) -> List[Dict[str, str]]:
        """功能：按 BM25 + 子串 boost 对 query 打分并返回 top_k 行。
        参数：
        - query：用户查询文本。
        - top_k：返回条数上限。
        返回值：
        - List[Dict[str, str]]：含 keyword_score 字段的行列表。
        """
        query_tokens = tokenize(query)
        if not query_tokens or not self.docs:
            return []
        query_counts = Counter(query_tokens)
        exact_terms = _exact_terms(query)
        scored = []
        for doc in self.docs:
            score = self._score_doc(doc, query_counts)
            score += _substring_boost(doc.raw_lower, exact_terms)
            if score > 0:
                scored.append((score, doc.row))
        scored.sort(key=lambda item: item[0], reverse=True)
        rows: List[Dict[str, str]] = []
        for score, row in scored[: max(1, top_k)]:
            item = dict(row)
            item["keyword_score"] = f"{score:.6f}"
            rows.append(item)
        return rows

    def _score_doc(self, doc: _IndexedRow, query_counts: Counter[str]) -> float:
        """功能：对单文档计算 BM25 分数。
        参数：
        - doc：索引文档。
        - query_counts：query 词频 Counter。
        返回值：
        - float：BM25 分数；无匹配词时为 0。
        """
        k1 = 1.5
        b = 0.75
        score = 0.0
        for token, qf in query_counts.items():
            tf = doc.term_counts.get(token, 0)
            if not tf:
                continue
            idf = self.idf.get(token, 0.0)
            denom = tf + k1 * (1.0 - b + b * doc.length / max(1.0, self.avg_len))
            score += idf * ((tf * (k1 + 1.0)) / max(denom, 1e-9)) * min(2, qf)
        return score


def tokenize(text: str) -> List[str]:
    """功能：将文本切分为检索用词元（URL、英文、十六进制、中文单字等）。
    参数：
    - text：待分词文本。
    返回值：
    - List[str]：小写词元列表。
    """
    tokens: List[str] = []
    for match in TOKEN_RE.finditer(text or ""):
        token = match.group(0).strip().lower()
        if not token:
            continue
        tokens.append(token)
    return tokens


def _make_indexed_row(row: Dict[str, str]) -> _IndexedRow:
    """功能：将单行 metadata 转为带分词统计的 _IndexedRow。
    参数：
    - row：向量库行 dict。
    返回值：
    - _IndexedRow：含 tokens、term_counts、length。
    """
    text = "\n".join(
        str(row.get(key) or "")
        for key in ("source", "chunk_index", "content", "image_captions")
    )
    tokens = tokenize(text)
    return _IndexedRow(
        row=dict(row),
        tokens=tokens,
        term_counts=Counter(tokens),
        length=max(1, len(tokens)),
        raw_lower=text.lower(),
    )


def _exact_terms(query: str) -> List[str]:
    """功能：从 query 提取英数字精确匹配子串（用于子串 boost）。
    参数：
    - query：用户查询。
    返回值：
    - List[str]：小写精确词列表。
    """
    terms = []
    for token in re.findall(r"[A-Za-z0-9_.:/#*\-]{2,}", query or ""):
        lowered = token.lower().strip()
        if lowered:
            terms.append(lowered)
    return terms


def _substring_boost(raw_lower: str, exact_terms: List[str]) -> float:
    """功能：对含 query 精确子串的文档追加加分。
    参数：
    - raw_lower：文档全文小写。
    - exact_terms：精确匹配词列表。
    返回值：
    - float：boost 分数累加值。
    """
    boost = 0.0
    for term in exact_terms:
        if term in raw_lower:
            if any(ch in term for ch in ("*", "#", "/", ".", "_", "-", ":")):
                boost += 6.0
            elif len(term) >= 6:
                boost += 4.0
            else:
                boost += 1.5
    return boost
