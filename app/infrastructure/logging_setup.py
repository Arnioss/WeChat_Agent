import logging
import os
import sys
from pathlib import Path


class _MaxLevelFilter(logging.Filter):
    """功能：限制日志仅通过不高于指定级别的记录。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self, level: int):
        """功能：设置日志过滤上限，用于分流 stdout/stderr 与文件输出级别。
        参数：
        - level：允许通过的最高日志级别。
        返回值：
        - 无。仅控制 `levelno <= level` 的记录通过，不改写日志内容。
        """
        super().__init__()
        self.level = level

    def filter(self, record: logging.LogRecord) -> bool:
        """功能：判断当前日志记录是否可通过过滤器。
        参数：
        - record：待判断的日志记录对象。
        返回值：
        - bool：日志级别不高于阈值时返回 True。
        """
        return record.levelno <= self.level


class _ConsoleStreamHandler(logging.StreamHandler):
    """功能：将日志写入真实终端流；emit 时解析 __stdout__/__stderr__，兼容 Streamlit 包装与 worker 线程。"""

    def __init__(self, *, stderr: bool = False):
        self._stderr = stderr
        super().__init__(stream=self._resolve_stream())

    def _resolve_stream(self):
        if self._stderr:
            return getattr(sys, "__stderr__", None) or sys.stderr
        return getattr(sys, "__stdout__", None) or sys.stdout

    def emit(self, record: logging.LogRecord) -> None:
        self.stream = self._resolve_stream()
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
            except UnicodeEncodeError:
                enc = getattr(stream, "encoding", None) or "utf-8"
                safe = msg.encode(enc, errors="replace").decode(enc, errors="replace")
                stream.write(safe + self.terminator)
            self.flush()
        except OSError:
            pass
        except Exception:
            self.handleError(record)


_CONFIGURED_SIGNATURE: str | None = None


def _logging_enabled(name: str, *, default: bool = True) -> bool:
    """功能：读取布尔型环境变量，判断是否启用指定日志输出通道。
    参数：
    - name：环境变量名。
    - default：变量未设置时的默认返回值。
    返回值：
    - bool：值为 1/true/yes/on 时返回 True，否则返回 False。
    """
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "on")


def resolve_log_paths(project_root: Path, log_basename: str | None = None) -> tuple[Path, Path]:
    """功能：按服务名解析 stdout/stderr 日志文件路径。
    参数：
    - project_root：项目根目录。
    - log_basename：日志文件名前缀；未传时读 `LOG_BASENAME`，默认 `wechat_server`。
    返回值：
    - tuple[Path, Path]：`{basename}.out.log` 与 `{basename}.err.log`。
    """
    name = (log_basename or os.getenv("LOG_BASENAME") or "wechat_server").strip()
    if not name:
        name = "wechat_server"
    return project_root / f"{name}.out.log", project_root / f"{name}.err.log"


def _config_signature(
    project_root: Path,
    log_basename: str | None,
    *,
    console_enabled: bool,
    file_enabled: bool,
    level_name: str,
) -> str:
    name = (log_basename or os.getenv("LOG_BASENAME") or "wechat_server").strip() or "wechat_server"
    return "|".join(
        (
            str(project_root.resolve()),
            name,
            level_name,
            "1" if console_enabled else "0",
            "1" if file_enabled else "0",
        )
    )


def configure_project_logging(
    project_root: Path,
    logger_name: str,
    *,
    log_basename: str | None = None,
) -> logging.Logger:
    """功能：配置项目日志输出到控制台与文件，并返回业务 logger。
    参数：
    - project_root：项目根目录，用于放置日志文件。
    - logger_name：需要返回的日志器名称。
    - log_basename：日志文件前缀；Web 用 `streamlit`，企微用 `wechat_server`。
    返回值：
    - logging.Logger：配置完成后的日志器实例。
    异常：
    - ValueError：未配置 `LOG_LEVEL` 环境变量时抛出。
    """
    global _CONFIGURED_SIGNATURE

    log_level = os.getenv("LOG_LEVEL")
    if not log_level:
        raise ValueError("缺少环境变量 LOG_LEVEL，请在 .env 文件中设置。")
    level = getattr(logging, log_level.upper(), logging.INFO)
    console_enabled = _logging_enabled("LOG_CONSOLE_ENABLED", default=True)
    file_enabled = _logging_enabled("LOG_FILE_ENABLED", default=True)

    signature = _config_signature(
        project_root,
        log_basename,
        console_enabled=console_enabled,
        file_enabled=file_enabled,
        level_name=log_level.upper(),
    )
    if _CONFIGURED_SIGNATURE == signature:
        return logging.getLogger(logger_name)

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(message)s")

    out_path, err_path = resolve_log_paths(project_root, log_basename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch(exist_ok=True)
    err_path.touch(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    if console_enabled:
        stdout_handler = _ConsoleStreamHandler(stderr=False)
        stdout_handler.setLevel(level)
        stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
        stdout_handler.setFormatter(fmt)
        root_logger.addHandler(stdout_handler)

        stderr_handler = _ConsoleStreamHandler(stderr=True)
        stderr_handler.setLevel(logging.WARNING)
        stderr_handler.setFormatter(fmt)
        root_logger.addHandler(stderr_handler)

    if file_enabled:
        out_file_handler = logging.FileHandler(out_path, encoding="utf-8")
        out_file_handler.setLevel(level)
        out_file_handler.addFilter(_MaxLevelFilter(logging.INFO))
        out_file_handler.setFormatter(fmt)
        root_logger.addHandler(out_file_handler)

        err_file_handler = logging.FileHandler(err_path, encoding="utf-8")
        err_file_handler.setLevel(logging.WARNING)
        err_file_handler.setFormatter(fmt)
        root_logger.addHandler(err_file_handler)

    if not root_logger.handlers:
        fallback = _ConsoleStreamHandler(stderr=True)
        fallback.setLevel(level)
        fallback.setFormatter(fmt)
        root_logger.addHandler(fallback)

    logging.captureWarnings(True)

    for noisy_logger in ("httpx", "httpcore"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    _CONFIGURED_SIGNATURE = signature
    return logging.getLogger(logger_name)
