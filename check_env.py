"""环境一键检查：MCP / Chroma / RAG 多模态依赖。"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

TELEMETRY_IMPL = "rag.chroma_telemetry.NoOpProductTelemetry"
STABLE_API_IMPL = "chromadb.api.segment.SegmentAPI"
RUST_API_IMPL = "chromadb.api.rust.RustBindingsAPI"


def _env_value(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _module_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "未安装"
    except Exception:
        return "未知"


def _proxy_settings() -> dict[str, str]:
    settings: dict[str, str] = {}
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY"):
        value = _env_value(name)
        if value:
            settings[name] = value
    return settings


def _uses_socks_proxy() -> bool:
    for value in _proxy_settings().values():
        lower = value.lower()
        if lower.startswith(("socks://", "socks4://", "socks4a://", "socks5://", "socks5h://")):
            return True
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        value = _env_value(name).lower()
        if value.startswith(("socks://", "socks4://", "socks4a://", "socks5://", "socks5h://")):
            return True
    return False


def _redact_url(url: str) -> str:
    try:
        parts = urlsplit(url)
    except Exception:
        return url
    if not parts.query:
        return url

    redacted_pairs = []
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        if key.lower() in {"apikey", "api_key", "key", "token", "secret", "access_token"}:
            redacted_pairs.append((key, "***"))
        else:
            redacted_pairs.append((key, value))

    return urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            parts.path,
            urlencode(redacted_pairs),
            parts.fragment,
        )
    )


def _probe_tcp_endpoint(host: str, port: int, timeout_seconds: float = 3.0) -> tuple[bool, str]:
    try:
        with socket.create_connection((host, port), timeout=timeout_seconds):
            return True, "正常"
    except Exception as exc:  # pragma: no cover - depends on local network
        return False, str(exc)


def _prepare_chroma_sqlite() -> None:
    from rag.sqlite_chroma_compat import ensure_chroma_sqlite

    ensure_chroma_sqlite()


def _exit_message_for_code(code: int) -> str:
    unsigned = code & 0xFFFFFFFF
    if unsigned == 0xC0000005:
        return "原生访问冲突 0xC0000005，通常是本机 Chroma/Rust/HNSW 原生库崩溃。"
    return f"退出码 {code}"


def _run_chroma_smoke_child(api_impl: str) -> int:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child-smoke", api_impl]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("ANONYMIZED_TELEMETRY", "false")
    env.setdefault("CHROMADB_ANONYMIZED_TELEMETRY", "false")
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return int(proc.returncode)


def _chroma_smoke(api_impl: str) -> None:
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")
    os.environ.setdefault("CHROMADB_ANONYMIZED_TELEMETRY", "false")

    _prepare_chroma_sqlite()
    import chromadb
    from chromadb.config import Settings

    persist_dir = Path(tempfile.gettempdir()) / (
        "wechat_agent_chroma_check_" + api_impl.rsplit(".", 1)[-1] + f"_{os.getpid()}"
    )
    if persist_dir.exists():
        shutil.rmtree(persist_dir)

    settings = Settings(
        anonymized_telemetry=False,
        chroma_api_impl=api_impl,
        chroma_product_telemetry_impl=TELEMETRY_IMPL,
        chroma_telemetry_impl=TELEMETRY_IMPL,
        is_persistent=True,
        persist_directory=str(persist_dir),
    )
    client = chromadb.PersistentClient(path=str(persist_dir), settings=settings)
    collection = client.get_or_create_collection(
        name="wechat_agent_chroma_check",
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(
        ids=["a", "b"],
        documents=["alpha", "beta"],
        embeddings=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        metadatas=[{"source": "smoke-a"}, {"source": "smoke-b"}],
    )
    result = collection.query(query_embeddings=[[1.0, 0.0, 0.0]], n_results=1)
    print("[Chroma检查] 冒烟测试结果=" + json.dumps(result.get("ids"), ensure_ascii=False), flush=True)


def check_mcp() -> None:
    from app.mcp.registry import McpToolRegistry

    print(f"[MCP检查] executable={sys.executable}", flush=True)
    print(f"[MCP检查] python={sys.version.split()[0]} platform={platform.platform()}", flush=True)
    print(f"[MCP检查] mcp={_module_version('mcp')}", flush=True)
    print(f"[MCP检查] httpx={_module_version('httpx')}", flush=True)
    print(f"[MCP检查] project_root={PROJECT_ROOT}", flush=True)
    print(f"[MCP检查] cwd={Path.cwd()}", flush=True)

    proxy_settings = _proxy_settings()
    if proxy_settings:
        print("[MCP检查] 代理环境变量:", flush=True)
        for name, value in proxy_settings.items():
            print(f"[MCP检查]   {name}={value}", flush=True)
    else:
        print("[MCP检查] 代理环境变量: 无", flush=True)

    if _module_version("mcp") == "未安装":
        raise SystemExit(
            "[MCP检查] 缺少 mcp 包，请先在当前 Python 环境执行: "
            'python -m pip install -U "mcp>=1.12"'
        )
    if _module_version("httpx") == "未安装":
        raise SystemExit(
            "[MCP检查] 缺少 httpx 包，请先在当前 Python 环境执行: "
            'python -m pip install -U "httpx[socks]"'
        )
    if _uses_socks_proxy() and importlib.util.find_spec("socksio") is None:
        raise SystemExit(
            "[MCP检查] 检测到 SOCKS 代理，但 socksio 不存在，请安装: "
            'python -m pip install -U "httpx[socks]"'
        )

    registry = McpToolRegistry(project_directory=str(PROJECT_ROOT))
    cache_path = registry._cache_path()
    raw_configs = registry._load_raw_server_configs()
    configs = list(getattr(registry, "_configs", []) or [])
    cache_payload = registry._read_cache()

    env_payload = _env_value("MCP_SERVERS_JSON")
    if env_payload:
        config_source = "MCP_SERVERS_JSON"
    else:
        config_path = _env_value("MCP_CONFIG_PATH") or str(PROJECT_ROOT / "config" / "mcp_servers.json")
        path = Path(config_path)
        if not path.is_absolute():
            path = (PROJECT_ROOT / path).resolve()
        config_source = str(path)

    print(f"[MCP检查] config_source={config_source}", flush=True)
    print(f"[MCP检查] cache_path={cache_path}", flush=True)
    print(f"[MCP检查] raw_servers={len(raw_configs)} enabled_servers={len(configs)}", flush=True)

    if _uses_socks_proxy():
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            value = _env_value(name)
            if not value:
                continue
            parts = urlsplit(value)
            if not parts.scheme.lower().startswith("socks"):
                continue
            if not parts.hostname or not parts.port:
                print(f"[MCP检查] {name} 格式无法解析", flush=True)
                continue
            ok, detail = _probe_tcp_endpoint(parts.hostname, int(parts.port))
            if ok:
                print(f"[MCP检查] 代理可达 {parts.hostname}:{parts.port} 正常", flush=True)
            else:
                print(
                    f"[MCP检查] 代理可达 {parts.hostname}:{parts.port} 失败: {detail}",
                    flush=True,
                )

    cache_dir = cache_path.parent
    created_dir = False
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        created_dir = True
    probe_path = cache_dir / ".write_probe"
    try:
        probe_path.write_text("ok", encoding="utf-8")
        print(f"[MCP检查] 缓存目录可写: {cache_dir}", flush=True)
    except Exception as exc:
        print(f"[MCP检查] 缓存目录写入失败: {exc}", flush=True)
    finally:
        try:
            if probe_path.exists():
                probe_path.unlink()
        except Exception:
            pass
        if created_dir:
            try:
                if cache_dir.exists() and not any(cache_dir.iterdir()):
                    cache_dir.rmdir()
            except Exception:
                pass

    issues = 0
    if not configs:
        print("[MCP检查] 没有找到可用的 MCP 服务配置", flush=True)
        issues += 1

    for config in configs:
        print(
            f"[MCP检查] server={config.name} url={_redact_url(config.url)} timeout={config.timeout_seconds}s",
            flush=True,
        )
        try:
            tools = registry._discover_server_tools(config)
        except Exception as exc:
            print(f"[MCP检查] 服务 {config.name} 发现失败: {exc}", flush=True)
            issues += 1
            continue

        print(f"[MCP检查] 服务 {config.name} 发现成功: {len(tools)} 个工具", flush=True)
        registry._write_server_cache(cache_payload, config, tools)
        if not tools:
            print("[MCP检查] 该服务返回 0 个工具，但缓存仍会被写入", flush=True)
        else:
            for tool in tools[:10]:
                print(f"[MCP检查]   - {tool.local_name} <- {tool.remote_name}", flush=True)
            if len(tools) > 10:
                print(f"[MCP检查]   … 另有 {len(tools) - 10} 个", flush=True)

    if cache_path.exists():
        stat = cache_path.stat()
        print(
            f"[MCP检查] 缓存文件已存在 size={stat.st_size} 字节 mtime={stat.st_mtime}",
            flush=True,
        )
    else:
        print("[MCP检查] 缓存文件尚不存在，通常表示发现或写入失败。", flush=True)
        issues += 1

    if issues:
        raise SystemExit(f"[MCP检查] 失败，共 {issues} 个问题")

    print("[MCP检查] 通过", flush=True)


def check_chroma(*, test_rust: bool = False) -> None:
    if _uses_socks_proxy():
        try:
            import socks  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                "[Chroma检查] 已配置 SOCKS 代理，但未安装 PySocks。"
                "请在本 Python 环境执行: python -m pip install PySocks"
            ) from exc

    if importlib.util.find_spec("hnswlib") is None:
        raise SystemExit(
            "[Chroma检查] 缺少 hnswlib，Chroma SegmentAPI 需要 chroma-hnswlib。"
            "Windows / Python 3.12 可安装预编译包: "
            "python -m pip install chroma-hnswlib==0.7.5"
        )

    _prepare_chroma_sqlite()
    import sqlite3

    print(f"[Chroma检查] executable={sys.executable}", flush=True)
    print(f"[Chroma检查] python={sys.version.split()[0]} platform={platform.platform()}", flush=True)
    print(f"[Chroma检查] sqlite={sqlite3.sqlite_version}", flush=True)
    if sys.version_info < (3, 10):
        print("[Chroma检查] 警告: 建议使用 Python 3.10+；Windows 请用 Python 3.12 x64。", flush=True)
    try:
        import chromadb
    except Exception as exc:
        raise SystemExit(f"[Chroma检查] 导入失败: {exc!r}") from exc
    print(f"[Chroma检查] chromadb={getattr(chromadb, '__version__', '未知')}", flush=True)

    print(f"[Chroma检查] 正在测试稳定 API={STABLE_API_IMPL}", flush=True)
    stable_code = _run_chroma_smoke_child(STABLE_API_IMPL)
    if stable_code != 0:
        raise SystemExit(f"[Chroma检查] 稳定 API 冒烟失败: {_exit_message_for_code(stable_code)}")
    print("[Chroma检查] 稳定 API 正常", flush=True)

    if test_rust:
        print(f"[Chroma检查] 正在测试 Rust API={RUST_API_IMPL}", flush=True)
        rust_code = _run_chroma_smoke_child(RUST_API_IMPL)
        if rust_code != 0:
            print(f"[Chroma检查] Rust API 失败: {_exit_message_for_code(rust_code)}", flush=True)
            print("[Chroma检查] 通过: 项目默认使用稳定 SegmentAPI，非 RustBindingsAPI。", flush=True)
            return
        print("[Chroma检查] Rust API 正常", flush=True)

    print("[Chroma检查] 通过", flush=True)


def check_rag() -> None:
    errors: list[str] = []

    try:
        from rag.env_config import RagEnvConfig

        cfg = RagEnvConfig.load(PROJECT_ROOT)
        print(f"[RAG检查] OPENAI_BASE_URL: {cfg.base_url}")
        print(f"[RAG检查] 嵌入模型: {cfg.embedding_model}")
        print(f"[RAG检查] 对话模型: {cfg.chat_model}")
        print(f"[RAG检查] 视觉模型: {cfg.vision_model}")
        print(f"[RAG检查] 多模态: {cfg.multimodal_enabled}")
    except Exception as exc:
        errors.append(f"环境: {exc}")

    for mod, pip_name in [
        ("fitz", "pymupdf"),
        ("docx", "python-docx"),
        ("PIL", "Pillow"),
    ]:
        try:
            __import__(mod)
            print(f"[RAG检查] 已导入 {pip_name}")
        except ImportError:
            errors.append(f"缺少依赖 {pip_name}，请 pip install {pip_name}")

    if errors:
        for item in errors:
            print(f"[RAG检查] 错误: {item}", file=sys.stderr)
        raise SystemExit(1)

    print("[RAG检查] 通过")


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 MCP / Chroma / RAG 环境")
    parser.add_argument("--mcp-only", action="store_true", help="仅检查 MCP")
    parser.add_argument("--chroma-only", action="store_true", help="仅检查 Chroma")
    parser.add_argument("--rag-only", action="store_true", help="仅检查 RAG")
    parser.add_argument("--test-rust", action="store_true", help="Chroma 额外测试 Rust API")
    parser.add_argument("--child-smoke", choices=[STABLE_API_IMPL, RUST_API_IMPL])
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env", override=False)

    if args.child_smoke:
        _chroma_smoke(args.child_smoke)
        return 0

    selected = sum(int(flag) for flag in (args.mcp_only, args.chroma_only, args.rag_only))
    run_mcp = args.mcp_only or selected == 0
    run_chroma = args.chroma_only or selected == 0
    run_rag = args.rag_only or selected == 0

    if selected == 0:
        print("[环境检查] 开始检查 MCP / Chroma / RAG ...", flush=True)

    if run_mcp:
        check_mcp()
    if run_chroma:
        check_chroma(test_rust=args.test_rust)
    if run_rag:
        check_rag()

    if selected == 0:
        print("[环境检查] 全部通过", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
