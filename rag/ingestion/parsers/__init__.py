"""文档格式解析器包，按文件类型路由至具体解析实现。"""

from rag.ingestion.parsers.router import parse_file

__all__ = ["parse_file"]
