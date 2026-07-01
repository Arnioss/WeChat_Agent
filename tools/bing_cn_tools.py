from __future__ import annotations

import html
import json
import os
import re
import uuid
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import requests

from app.agent.tool_metadata import ToolRichMetadata


BING_SEARCH_URL = "https://cn.bing.com/search"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)
CRAWLER_BLACKLIST = {
    "zhihu.com",
    "www.zhihu.com",
    "zhuanlan.zhihu.com",
    "xiaohongshu.com",
    "www.xiaohongshu.com",
    "xhs.com",
    "weibo.com",
    "www.weibo.com",
    "m.weibo.com",
    "weixin.qq.com",
    "mp.weixin.qq.com",
    "douyin.com",
    "www.douyin.com",
    "tiktok.com",
    "www.tiktok.com",
    "bilibili.com",
    "www.bilibili.com",
    "m.bilibili.com",
    "csdn.net",
    "www.csdn.net",
    "blog.csdn.net",
}


def _request_headers() -> Dict[str, str]:
    """功能：构造必应搜索与网页抓取共用的 HTTP 请求头。
    参数：
    - 无。
    返回值：
    - Dict[str, str]：包含 User-Agent、Accept 等字段的请求头字典。
    """
    return {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
    }


def _coerce_int(value: Any, *, default: int, minimum: int, maximum: Optional[int] = None) -> int:
    """功能：将任意值安全转换为整数并限制在指定范围内。
    参数：
    - value：待转换的原始值。
    - default：转换失败时使用的默认值。
    - minimum：允许的最小值。
    - maximum：允许的最大值；为 None 时不设上限。
    返回值：
    - int：规范化后的整数。
    """
    try:
        number = int(value)
    except Exception:
        number = default
    number = max(minimum, number)
    if maximum is not None:
        number = min(maximum, number)
    return number


def _clean_text(value: str) -> str:
    """功能：去除 HTML 标签、反转义并压缩空白，得到可读纯文本。
    参数：
    - value：含 HTML 的原始字符串。
    返回值：
    - str：清洗后的纯文本。
    """
    text = re.sub(r"<[^>]+>", " ", value or "")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_attr(tag: str, attr: str) -> str:
    """功能：从 HTML 标签字符串中提取指定属性值。
    参数：
    - tag：HTML 起始标签片段。
    - attr：属性名。
    返回值：
    - str：属性值；未找到时返回空字符串。
    """
    match = re.search(rf'\s{re.escape(attr)}=["\']([^"\']+)["\']', tag or "", re.IGNORECASE)
    return html.unescape(match.group(1)) if match else ""


def _split_bing_result_blocks(raw_html: str) -> List[str]:
    """功能：按必应搜索结果块 `b_algo` 切分 HTML 页面。
    参数：
    - raw_html：必应搜索结果页 HTML 文本。
    返回值：
    - List[str]：每个搜索结果 `<li>` 块的 HTML 片段列表。
    """
    blocks: List[str] = []
    pattern = re.compile(r'<li\b[^>]*class=["\'][^"\']*\bb_algo\b[^"\']*["\'][^>]*>', re.IGNORECASE)
    matches = list(pattern.finditer(raw_html or ""))
    for idx, match in enumerate(matches):
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(raw_html)
        block = raw_html[start:end]
        close_idx = block.lower().rfind("</li>")
        if close_idx != -1:
            block = block[: close_idx + len("</li>")]
        blocks.append(block)
    return blocks


def _parse_bing_results(raw_html: str, query: str, count: int) -> Dict[str, Any]:
    """功能：解析必应搜索结果 HTML，提取标题、链接、摘要与 UUID 映射。
    参数：
    - raw_html：必应搜索结果页 HTML 文本。
    - query：原始搜索关键词。
    - count：最多返回的结果条数。
    返回值：
    - Dict[str, Any]：包含 query、results、urlMap，以及可选 totalResults 的字典。
    """
    results: List[Dict[str, str]] = []
    for block in _split_bing_result_blocks(raw_html):
        title_match = re.search(r"<h2\b[^>]*>.*?(<a\b[^>]*>.*?</a>)", block, re.IGNORECASE | re.DOTALL)
        if not title_match:
            continue
        anchor = title_match.group(1)
        open_tag_match = re.match(r"(<a\b[^>]*>)", anchor, re.IGNORECASE | re.DOTALL)
        url = _extract_attr(open_tag_match.group(1), "href") if open_tag_match else ""
        title = _clean_text(anchor)
        if not title or not url:
            continue

        snippet = ""
        caption_match = re.search(
            r'<div\b[^>]*class=["\'][^"\']*\bb_caption\b[^"\']*["\'][^>]*>.*?<p\b[^>]*>(.*?)</p>',
            block,
            re.IGNORECASE | re.DOTALL,
        )
        if caption_match:
            snippet = _clean_text(caption_match.group(1))

        display_url = ""
        cite_match = re.search(r"<cite\b[^>]*>(.*?)</cite>", block, re.IGNORECASE | re.DOTALL)
        if cite_match:
            display_url = _clean_text(cite_match.group(1))

        results.append(
            {
                "uuid": str(uuid.uuid4()),
                "title": title,
                "url": url,
                "snippet": snippet,
                "displayUrl": display_url or url,
            }
        )
        if len(results) >= count:
            break

    total_results = None
    count_match = re.search(
        r'<span\b[^>]*class=["\'][^"\']*\bsb_count\b[^"\']*["\'][^>]*>(.*?)</span>',
        raw_html or "",
        re.IGNORECASE | re.DOTALL,
    )
    if count_match:
        number_match = re.search(r"[\d,，]+", _clean_text(count_match.group(1)))
        if number_match:
            try:
                total_results = int(number_match.group(0).replace(",", "").replace("，", ""))
            except Exception:
                total_results = None

    payload: Dict[str, Any] = {
        "query": query,
        "results": results,
        "urlMap": {item["uuid"]: item["url"] for item in results},
    }
    if total_results is not None:
        payload["totalResults"] = total_results
    return payload


def _coerce_search_payload(query: Any, count: Any, offset: Any) -> Tuple[str, int, int]:
    """功能：规范化 bing_search 入参，兼容 dict 与 positional 调用。
    参数：
    - query：搜索词或含 query/count/offset 的 dict。
    - count：期望结果数量。
    - offset：结果偏移量。
    返回值：
    - Tuple[str, int, int]：`(query_text, result_count, result_offset)`。
    异常：
    - ValueError：query 为空时抛出。
    """
    if isinstance(query, dict):
        payload = query
        query = payload.get("query")
        count = payload.get("count", count)
        offset = payload.get("offset", offset)
    query_text = str(query or "").strip()
    if not query_text:
        raise ValueError("bing_search 需要非空 query。")
    return (
        query_text,
        _coerce_int(count, default=10, minimum=1, maximum=50),
        _coerce_int(offset, default=0, minimum=0),
    )


def bing_search(query: str, count: int = 10, offset: int = 0) -> str:
    """功能：使用必应中文搜索引擎检索网页，返回结构化 JSON 字符串。
    参数：
    - query：搜索关键词或完整查询语句。
    - count：返回结果数量，默认 10，最多 50。
    - offset：分页偏移量，默认 0。
    返回值：
    - str：JSON 字符串，含 query、results、urlMap 及可选 totalResults。
    异常：
    - ValueError：query 为空时抛出。
    - requests.HTTPError：HTTP 请求失败时抛出。
    """
    query_text, result_count, result_offset = _coerce_search_payload(query, count, offset)
    timeout = float(os.getenv("BING_CN_SEARCH_TIMEOUT_SECONDS", "15"))
    response = requests.get(
        BING_SEARCH_URL,
        params={"q": query_text, "first": result_offset + 1, "count": result_count},
        headers=_request_headers(),
        timeout=timeout,
    )
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    payload = _parse_bing_results(response.text, query_text, result_count)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _hostname_is_blacklisted(url: str) -> bool:
    """功能：判断 URL 主机名是否命中爬虫黑名单。
    参数：
    - url：待检查的完整 URL。
    返回值：
    - bool：命中黑名单返回 True。
    """
    try:
        hostname = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    if not hostname:
        return False
    return any(hostname == domain or hostname.endswith("." + domain) for domain in CRAWLER_BLACKLIST)


class _VisibleTextParser(HTMLParser):
    """功能：从 HTML 中提取可见正文，跳过 script/style 等标签内容。
    参数：
    - 无。
    返回值：
    - 无。通过 text 属性获取合并后的可见纯文本。
    """
    def __init__(self):
        """功能：初始化 HTML 解析器，准备收集可见文本片段。
        参数：
        - 无。
        返回值：
        - 无。
        """
        super().__init__(convert_charrefs=True)
        self._skip_depth = 0
        self._parts: List[str] = []
        self._skip_tags = {"script", "style", "nav", "footer", "header", "iframe", "noscript"}

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        """功能：遇到需跳过的标签时递增跳过深度，忽略其内部文本。
        参数：
        - tag：HTML 起始标签名。
        - attrs：标签属性列表。
        返回值：
        - 无。
        """
        if tag.lower() in self._skip_tags:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        """功能：离开需跳过的标签时递减跳过深度。
        参数：
        - tag：HTML 结束标签名。
        返回值：
        - 无。
        """
        if tag.lower() in self._skip_tags and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        """功能：收集当前不在跳过区域内的可见文本数据。
        参数：
        - data：标签内的文本数据片段。
        返回值：
        - 无。
        """
        if self._skip_depth == 0 and data and data.strip():
            self._parts.append(data.strip())

    @property
    def text(self) -> str:
        """功能：返回已收集片段合并并压缩空白后的可见纯文本。
        参数：
        - 无。
        返回值：
        - str：合并后的可见正文文本。
        """
        return re.sub(r"\s+", " ", " ".join(self._parts)).strip()


def _extract_visible_text(raw_html: str) -> str:
    """功能：将 HTML 页面转换为压缩空白后的可见纯文本。
    参数：
    - raw_html：网页 HTML 源码。
    返回值：
    - str：提取到的可见文本。
    """
    parser = _VisibleTextParser()
    parser.feed(raw_html or "")
    return parser.text


def _clip_content(text: str) -> Tuple[str, bool]:
    """功能：按环境变量限制抓取正文最大长度，必要时截断。
    参数：
    - text：原始正文文本。
    返回值：
    - Tuple[str, bool]：`(正文, 是否被截断)`。
    """
    limit = _coerce_int(os.getenv("BING_CN_CRAWL_MAX_CHARS"), default=12000, minimum=1000)
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...(truncated)", True


def _fetch_webpage(url: str) -> Dict[str, Any]:
    """功能：抓取单个网页并提取可读正文。
    参数：
    - url：目标网页 URL。
    返回值：
    - Dict[str, Any]：成功时含 url/content/truncated；黑名单站点含 error/isBlacklisted。
    异常：
    - RuntimeError：正文过短或为空时抛出。
    - requests.HTTPError：HTTP 请求失败时抛出。
    """
    if _hostname_is_blacklisted(url):
        return {
            "url": url,
            "error": "该网站在爬虫黑名单中，禁止抓取",
            "isBlacklisted": True,
        }
    timeout = float(os.getenv("BING_CN_CRAWL_TIMEOUT_SECONDS", "30"))
    response = requests.get(
        url,
        headers=_request_headers(),
        timeout=timeout,
        allow_redirects=True,
    )
    response.raise_for_status()
    response.encoding = response.encoding or "utf-8"
    content, truncated = _clip_content(_extract_visible_text(response.text))
    if not content or len(content) < 50:
        raise RuntimeError("提取的内容太少或为空")
    return {
        "url": response.url or url,
        "content": content,
        "truncated": truncated,
    }


def _iter_crawl_targets(target: Any, url_map: Optional[Dict[str, str]]) -> Iterable[Tuple[str, str]]:
    """功能：将 crawl_webpage 的多种入参形式规范化为 `(uuid, url)` 迭代器。
    参数：
    - target：URL 字符串、dict、list 或 bing_search 返回结构。
    - url_map：可选 UUID 到 URL 的映射表。
    返回值：
    - Iterable[Tuple[str, str]]：待抓取目标的 UUID 与 URL 对。
    """
    if isinstance(target, dict):
        payload = target
        if "url" in payload:
            item_url = str(payload.get("url") or "").strip()
            if item_url:
                yield str(payload.get("uuid") or uuid.uuid4()), item_url
            return
        local_url_map = payload.get("urlMap") or payload.get("url_map") or url_map or {}
        uuids = payload.get("uuids") or payload.get("uuid")
        if isinstance(uuids, str):
            uuids = [uuids]
        if isinstance(uuids, list):
            for item in uuids:
                key = str(item)
                if isinstance(local_url_map, dict) and local_url_map.get(key):
                    yield key, str(local_url_map[key])
                else:
                    yield key, ""
            return
        if all(isinstance(key, str) and isinstance(value, str) for key, value in payload.items()):
            for key, item_url in payload.items():
                yield key, item_url
            return

    if isinstance(target, list):
        for item in target:
            key = str(item)
            if url_map and key in url_map:
                yield key, str(url_map[key])
            else:
                yield str(uuid.uuid4()), key
        return

    key = str(target or "").strip()
    if not key:
        return
    if url_map and key in url_map:
        yield key, str(url_map[key])
    else:
        yield str(uuid.uuid4()), key


def crawl_webpage(target, url_map: Optional[Dict[str, str]] = None) -> str:
    """功能：抓取一个或多个网页正文，支持直接 URL 或 bing_search 的 uuids + urlMap。
    参数：
    - target：URL、URL 列表，或含 uuids/urlMap 的结构。
    - url_map：可选 UUID 到 URL 映射；通常来自 bing_search 返回。
    返回值：
    - str：JSON 字符串数组，每项含 uuid、url、content 或 error 等字段。
    异常：
    - ValueError：无法解析出任何抓取目标时抛出。
    """
    results: List[Dict[str, Any]] = []
    targets = list(_iter_crawl_targets(target, url_map))
    if not targets:
        raise ValueError("crawl_webpage 需要 URL，或 uuids + urlMap。")

    for item_uuid, item_url in targets:
        if not item_url:
            results.append({"uuid": item_uuid, "url": "", "error": "UUID 在 urlMap 中不存在"})
            continue
        try:
            row = _fetch_webpage(item_url)
            row["uuid"] = item_uuid
            results.append(row)
        except Exception as exc:
            results.append({"uuid": item_uuid, "url": item_url, "error": str(exc)})
    return json.dumps(results, ensure_ascii=False, indent=2)


bing_search.__tool_rich_metadata__ = ToolRichMetadata(
    summary="使用必应中文搜索引擎搜索实时网页信息，返回搜索结果、摘要和 UUID 到 URL 的映射。",
    when_to_use=(
        "用户明确要求联网搜索、查最新资料、查公开网页资料时。",
        "本地知识库没有覆盖，且需要来自公开互联网的信息补充时。",
        "需要先找到候选网页，再用 crawl_webpage 抓取正文时。",
    ),
    when_not_to_use=(
        "内部知识库、项目文档或业务规范类问题，应优先 rag_summarize。",
        "问题可直接基于当前对话回答，且不需要公开网页信息。",
        "只需要当前日期时，应使用 get_current_date。",
    ),
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词或完整查询语句。"},
            "count": {"type": "integer", "description": "返回结果数量，默认 10，最多 50。", "default": 10},
            "offset": {"type": "integer", "description": "结果偏移量，用于分页，默认 0。", "default": 0},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
    output_description="JSON 字符串：query、results、urlMap、totalResults（若页面可解析）。results 内含 uuid/title/url/snippet/displayUrl。",
    output_schema={"type": "string"},
    examples=(
        'bing_search({"query": "Python 3.13 新特性"})',
        'bing_search({"query": "企业微信机器人 MCP 文档", "count": 5})',
    ),
    notes=(
        "无需 API key；请求直接发送到 cn.bing.com。",
        "若需要正文内容，使用返回的 uuid 和 urlMap 调用 crawl_webpage。",
    ),
    priority=70,
    source="local",
)


crawl_webpage.__tool_rich_metadata__ = ToolRichMetadata(
    summary="抓取网页并提取可读正文；支持直接 URL，也支持 bing_search 返回的 UUID + urlMap。",
    when_to_use=(
        "已经通过 bing_search 找到候选网页，需要读取网页正文时。",
        "用户直接给出公开网页 URL 并要求总结、提取或分析内容时。",
    ),
    when_not_to_use=(
        "搜索结果摘要已经足够回答，无需打开网页时。",
        "目标站点在黑名单中，例如知乎、小红书、微博、微信公众号、抖音、B站、CSDN。",
        "登录后页面、强反爬页面或非公开内容。",
    ),
    input_schema={
        "type": "object",
        "properties": {
            "target": {
                "anyOf": [
                    {"type": "string"},
                    {"type": "object"},
                    {"type": "array"},
                ],
                "description": "URL 字符串，或 {\"url\": \"...\"}，或 {\"uuids\": [...], \"urlMap\": {...}}。",
            },
            "url_map": {
                "type": "object",
                "description": "可选，UUID 到 URL 的映射；通常来自 bing_search 返回的 urlMap。",
            },
        },
        "required": ["target"],
        "additionalProperties": True,
    },
    output_description="JSON 字符串数组：每项包含 uuid、url、content 或 error、isBlacklisted、truncated。",
    output_schema={"type": "string"},
    examples=(
        'crawl_webpage({"target": "https://example.com/article"})',
        'crawl_webpage({"target": {"uuids": ["uuid1"], "urlMap": {"uuid1": "https://example.com"}}})',
    ),
    notes=(
        "正文默认最多返回 12000 字符，可用 BING_CN_CRAWL_MAX_CHARS 调整。",
        "网络失败、黑名单或正文过短会在单项 result 中返回 error。",
    ),
    priority=68,
    source="local",
)
