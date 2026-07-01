"""HTTP 请求客户端 IP 解析工具。"""

from __future__ import annotations

from typing import Any, Optional


def _first_xff_ip(xff: str) -> Optional[str]:
    """功能：从 X-Forwarded-For 头中提取首个客户端 IP。
    参数：
    - xff：X-Forwarded-For 原始字符串，格式为「客户端 IP, 代理1, 代理2…」。
    返回值：
    - Optional[str]：首段 IP；输入无效或为空时返回 None。
    """
    if not xff or not isinstance(xff, str):
        return None
    # X-Forwarded-For 格式：客户端 IP, 代理1, 代理2…（取首段）
    part = xff.split(",")[0].strip()
    return part or None


def client_ip_from_wsgi_environ(environ: dict) -> str:
    """功能：从 WSGI environ 解析客户端 IP（适用于任意 WSGI 与测试）。
    参数：
    - environ：WSGI 环境字典。
    返回值：
    - str：优先 X-Forwarded-For 首段，其次 X-Real-IP，再次 REMOTE_ADDR；均缺失时返回 `unknown`。
    """
    xff = environ.get("HTTP_X_FORWARDED_FOR")
    if xff:
        first = _first_xff_ip(str(xff))
        if first:
            return first
    real = environ.get("HTTP_X_REAL_IP")
    if real and str(real).strip():
        return str(real).strip()
    addr = environ.get("REMOTE_ADDR")
    if addr:
        return str(addr)
    return "unknown"


def client_ip_from_flask_request(request: Any) -> str:
    """功能：从 Flask request 解析客户端 IP（显式读取代理头，与 ProxyFix 无关）。
    参数：
    - request：Flask 请求对象。
    返回值：
    - str：优先 X-Forwarded-For 首段，其次 X-Real-IP，再次 remote_addr；解析失败时返回 `unknown`。
    """
    try:
        xff = request.headers.get("X-Forwarded-For")
        if xff:
            first = _first_xff_ip(str(xff))
            if first:
                return first
        real = request.headers.get("X-Real-IP")
        if real and str(real).strip():
            return str(real).strip()
        ra = getattr(request, "remote_addr", None)
        if ra:
            return str(ra)
    except Exception:
        pass
    return "unknown"
