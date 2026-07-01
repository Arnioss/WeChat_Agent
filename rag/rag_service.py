import json
import os
import re
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from rag.block_store import DocumentBlockStore, render_blocks_inline
from rag.env_config import RagEnvConfig
from rag.rag_images import resolve_asset_path
from rag.vector_store import VectorStoreService


class RagSummarizeService:
    """功能：检索本地知识库参考资料，供 Agent 整理摘要回答。

    参数：
        无（通过构造器传入 project_directory）。

    返回值：
        无（构造器）。

    异常：
        无。
    """

    def __init__(self, project_directory: str):
        """功能：初始化 RAG 摘要服务，加载向量库、块存储与环境配置。

        参数：
            project_directory: 项目根目录路径。

        返回值：
            无。

        异常：
            无。
        """
        load_dotenv()
        self.project_directory = str(Path(project_directory).resolve())
        self.vector_store = VectorStoreService()
        self.persist_dir = self.vector_store.persist_dir
        self.rag_config = RagEnvConfig.load(Path(project_directory))
        self.block_store = DocumentBlockStore(self.persist_dir)

    def rag_summarize(self, query: str) -> str:
        """功能：检索知识库并组装参考资料上下文，供 Agent 生成最终回答。

        参数：
            query: 用户知识库查询问题。

        返回值：
            格式化的参考资料文本；空 query 或检索无结果时返回提示语。

        异常：
            无（内部检索/渲染失败时降级为空结果提示）。
        """
        if not query.strip():
            return "请输入你要查询的知识问题。"

        if _query_auto_index_enabled():
            self.vector_store.build_or_update_index()
        docs = self.vector_store.retrieve(query)
        if not docs:
            return "知识库暂无足够信息。请补充资料后重试。"

        flow_mode = _flow_expand_enabled() and _is_flow_query(query)
        if flow_mode:
            docs = self.vector_store.expand_sources(
                docs,
                max_chars=_flow_max_context_chars(),
                max_sources=1,
            ) or docs

        context_blocks = []
        for i, doc in enumerate(docs, start=1):
            source = doc.get("source", "")
            content = self._render_doc_content(doc, query=query)
            if not content:
                continue
            chunk_index = doc.get("chunk_index", "")
            block = [
                f"【参考资料{i}】",
                f"来源: {source}",
                f"分片: {chunk_index}",
                f"内容: {content}",
            ]
            if not _has_inline_image_marker(content):
                figures = _format_figures_section(doc, persist_dir=self.persist_dir)
                if figures:
                    block.append(figures)
            context_blocks.append("\n".join(block))

        if not context_blocks:
            return "知识库暂无足够信息。请补充资料后重试。"

        mode_hint = (
            "当前问题像流程/步骤类问题，已按来源扩展相邻资料；最终回答请尽量保留关键步骤、命令、参数和注意事项。"
            if flow_mode
            else "最终回答请基于这些资料整理结论；如需补充通用知识，必须明确标注为非知识库内容。"
        )
        return "\n\n".join(
            [
                "以下是知识库检索到的参考资料，不是最终答案。",
                f"用户问题：{query.strip()}",
                mode_hint,
                "\n\n".join(context_blocks),
            ]
        )

    def _render_doc_content(self, doc: dict, *, query: str = "") -> str:
        """功能：将检索命中的分片渲染为可读正文（块级渲染、查询压缩、补全图示）。

        参数：
            doc: 向量检索返回的分片行 dict。
            query: 用户查询，用于非流程类问题的块压缩。

        返回值：
            渲染后的正文文本；块存储不可用或失败时回退 doc.content。

        异常：
            无（块读取异常被捕获并回退）。
        """
        doc_id = (doc.get("doc_id") or "").strip()
        source = (doc.get("source") or "").strip()
        try:
            block_start = int(doc.get("block_start") or 0)
            block_end = int(doc.get("block_end") or block_start)
        except (TypeError, ValueError):
            return (doc.get("content") or "").strip()

        store = getattr(self, "block_store", None)
        if store is not None:
            try:
                blocks = store.get_range(
                    doc_id=doc_id,
                    source=source,
                    block_start=block_start,
                    block_end=block_end,
                )
                if not _is_flow_query(query):
                    blocks = _compress_blocks_for_query(blocks, query)
                rendered = render_blocks_inline(blocks, persist_dir=self.persist_dir)
                rendered = _append_missing_figures(rendered, doc, persist_dir=self.persist_dir)
                if rendered:
                    return rendered
            except Exception:
                pass
        return (doc.get("content") or "").strip()


def _format_figures_section(doc: dict, *, persist_dir: Path) -> str:
    """功能：从分片元数据生成「关联图示」Markdown 段落（含 KB_IMAGE 标记）。

    参数：
        doc: 含 image_assets、image_captions 的分片行 dict。
        persist_dir: RAG 持久化目录，用于解析图片资产路径。

    返回值：
        关联图示文本块；无图片资产时返回空字符串。

    异常：
        无（JSON 解析失败时返回空字符串）。
    """
    assets_raw = doc.get("image_assets") or "[]"
    captions_raw = doc.get("image_captions") or "[]"
    try:
        assets = json.loads(assets_raw) if isinstance(assets_raw, str) else assets_raw
        captions = json.loads(captions_raw) if isinstance(captions_raw, str) else captions_raw
    except Exception:
        return ""
    if not assets:
        return ""
    lines = ["关联图示:"]
    for idx, asset in enumerate(assets, start=1):
        caption = captions[idx - 1] if idx - 1 < len(captions) else ""
        resolved = resolve_asset_path(persist_dir, str(asset))
        if resolved is not None:
            lines.append(f"- 图{idx}: {caption}\n  [KB_IMAGE:{resolved.as_posix()}]")
        else:
            lines.append(f"- 图{idx}: {caption}（asset: {asset}）")
    return "\n".join(lines)


def _append_missing_figures(rendered: str, doc: dict, *, persist_dir: Path) -> str:
    """功能：在已渲染正文中补全尚未内联的图片引用。

    参数：
        rendered: 块渲染后的正文文本。
        doc: 含 image_assets、image_captions 的分片行 dict。
        persist_dir: RAG 持久化目录，用于解析图片资产路径。

    返回值：
        补全缺失图示后的正文；无缺失时原样返回 rendered。

    异常：
        无（JSON 解析失败时原样返回 rendered）。
    """
    assets_raw = doc.get("image_assets") or "[]"
    captions_raw = doc.get("image_captions") or "[]"
    try:
        assets = json.loads(assets_raw) if isinstance(assets_raw, str) else assets_raw
        captions = json.loads(captions_raw) if isinstance(captions_raw, str) else captions_raw
    except Exception:
        return rendered
    if not isinstance(assets, list):
        return rendered
    if not isinstance(captions, list):
        captions = []
    if not assets:
        return rendered

    existing_paths = _kb_image_paths_in_text(rendered)
    missing_lines = []
    for idx, asset in enumerate(assets, start=1):
        resolved = resolve_asset_path(persist_dir, str(asset))
        if resolved is None:
            continue
        resolved_path = resolved.as_posix()
        if resolved_path in existing_paths:
            continue
        caption = captions[idx - 1] if idx - 1 < len(captions) else ""
        missing_lines.append(f"- 图{idx}: {caption}\n  [KB_IMAGE:{resolved_path}]")
    if not missing_lines:
        return rendered

    rendered_text = (rendered or "").rstrip()
    figures_block = "关联图示:\n" + "\n".join(missing_lines)
    if rendered_text:
        return f"{rendered_text}\n\n{figures_block}"
    return figures_block


def _compress_blocks_for_query(blocks: List[dict], query: str) -> List[dict]:
    """功能：按查询相关性裁剪块列表，仅保留最相关章节边界内的块。

    参数：
        blocks: DocumentBlockStore 返回的块 dict 列表。
        query: 用户查询文本。

    返回值：
        裁剪后的块列表；流程类 query 或无法定位章节时返回原列表。

    异常：
        无。
    """
    if not blocks or not query.strip():
        return blocks
    if _is_flow_query(query):
        return blocks
    scored = [
        (_block_relevance_score(block, query), index)
        for index, block in enumerate(blocks)
        if (block.get("text") or block.get("image_caption") or "").strip()
    ]
    if not scored:
        return blocks
    best_score, best_index = max(scored, key=lambda item: (item[0], -item[1]))
    if best_score <= 0:
        return blocks

    heading_indexes = [
        index for index, block in enumerate(blocks)
        if _is_section_boundary(block)
    ]
    if not heading_indexes:
        return blocks

    start_index = 0
    for heading_index in heading_indexes:
        if heading_index <= best_index:
            start_index = heading_index
        else:
            break
    end_index = len(blocks)
    for heading_index in heading_indexes:
        if heading_index > start_index:
            end_index = heading_index
            break
    return blocks[start_index:end_index]


def _block_relevance_score(block: dict, query: str) -> float:
    """功能：计算单个块与用户 query 的文本相关性得分。

    参数：
        block: 含 section_title、text、image_caption 等字段的块 dict。
        query: 用户查询文本。

    返回值：
        非负浮点得分；无文本或 token 时返回 0.0。

    异常：
        无（分词模块不可用时降级为空白分词）。
    """
    try:
        from rag.keyword_retrieval import tokenize
    except Exception:
        tokenize = None
    text = " ".join(
        str(block.get(key) or "")
        for key in ("section_title", "text", "image_caption", "image_asset")
    ).lower()
    if not text:
        return 0.0
    if tokenize is None:
        query_tokens = [part.lower() for part in query.split() if part.strip()]
    else:
        query_tokens = tokenize(query)
    score = 0.0
    for token in query_tokens:
        if not token:
            continue
        if token in text:
            score += 2.0 if len(token) >= 4 else 1.0
    for exact in _query_exact_terms(query):
        if exact in text:
            score += 6.0
    return score


def _is_section_boundary(block: dict) -> bool:
    """功能：判断块是否为章节/小节边界（标题、编号、问句等）。

    参数：
        block: 块 dict（含 block_type、text 等）。

    返回值：
        True 表示可作为压缩时的章节切分点。

    异常：
        无。
    """
    if (block.get("block_type") or "") == "heading":
        return True
    text = (block.get("text") or "").strip()
    if not text or "\n" in text or len(text) > 90:
        return False
    import re

    if re.match(r"^[（(]?\d+[）).、]", text):
        return False
    if text.endswith(("?", "？")):
        return True
    if re.match(r"^\d+(\.\d+){0,4}\s*\S+", text):
        return True
    if text in {"切测试", "切正式"}:
        return True
    return text.startswith(("如何", "怎么", "怎样", "测试环境信息"))


def _query_exact_terms(query: str) -> List[str]:
    """功能：从 query 中提取需精确匹配的英文/数字/符号术语。

    参数：
        query: 用户查询文本。

    返回值：
        小写术语字符串列表。

    异常：
        无。
    """
    import re

    return [
        term.lower()
        for term in re.findall(r"[A-Za-z0-9_.:/#*\-]{2,}", query or "")
        if term.strip()
    ]


def _has_inline_image_marker(text: str) -> bool:
    """功能：检测正文是否已含内联 KB_IMAGE 图片标记。

    参数：
        text: 待检测正文。

    返回值：
        含 [KB_IMAGE: 时为 True。

    异常：
        无。
    """
    return "[KB_IMAGE:" in (text or "")


def _kb_image_paths_in_text(text: str) -> set[str]:
    """功能：从正文中解析所有 KB_IMAGE 标记内的图片路径。

    参数：
        text: 含 KB_IMAGE 标记的正文。

    返回值：
        已引用图片路径的集合。

    异常：
        无。
    """
    return set(re.findall(r"\[KB_IMAGE:([^\]]+)\]", text or ""))


def _query_auto_index_enabled() -> bool:
    """功能：读取 RAG_QUERY_AUTO_INDEX 环境变量，判断查询前是否自动建索引。

    参数：
        无。

    返回值：
        开启时为 True。

    异常：
        无。
    """
    return (os.getenv("RAG_QUERY_AUTO_INDEX") or "false").strip().lower() in ("1", "true", "yes", "on")


def _flow_expand_enabled() -> bool:
    """功能：读取 RAG_FLOW_EXPAND_ENABLED，判断流程类 query 是否扩展上下文。

    参数：
        无。

    返回值：
        开启时为 True（默认开启）。

    异常：
        无。
    """
    return (os.getenv("RAG_FLOW_EXPAND_ENABLED") or "true").strip().lower() in ("1", "true", "yes", "on")


def _flow_max_context_chars() -> int:
    """功能：读取 RAG_FLOW_MAX_CONTEXT_CHARS，返回流程扩展上下文字符上限。

    参数：
        无。

    返回值：
        不小于 2000 的整数；解析失败时默认 16000。

    异常：
        无。
    """
    raw = (os.getenv("RAG_FLOW_MAX_CONTEXT_CHARS") or "16000").strip()
    try:
        return max(2000, int(raw))
    except ValueError:
        return 16000


def _is_flow_query(query: str) -> bool:
    """功能：根据 RAG_FLOW_KEYWORDS 判断 query 是否为流程/步骤类问题。

    参数：
        query: 用户查询文本。

    返回值：
        命中任一关键词时为 True。

    异常：
        无。
    """
    keywords = os.getenv(
        "RAG_FLOW_KEYWORDS",
        "流程,步骤,教程,怎么操作,如何配置,切环境,安装,开通,设置,完整,详细,怎么开,怎么做",
    )
    text = (query or "").lower()
    return any(item.strip().lower() in text for item in keywords.split(",") if item.strip())
