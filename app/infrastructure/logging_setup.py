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


def configure_project_logging(project_root: Path, logger_name: str) -> logging.Logger:
    """功能：配置项目日志输出到控制台与文件，并返回业务 logger。
    参数：
    - project_root：项目根目录，用于放置日志文件。
    - logger_name：需要返回的日志器名称。
    返回值：
    - logging.Logger：配置完成后的日志器实例。
    异常：
    - ValueError：未配置 `LOG_LEVEL` 环境变量时抛出。
    """
    log_level = os.getenv("LOG_LEVEL")
    if not log_level:
        raise ValueError("缺少环境变量 LOG_LEVEL，请在 .env 文件中设置。")
    level = getattr(logging, log_level.upper(), logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(threadName)s - %(message)s")

    out_path = project_root / "wechat_server.out.log"
    err_path = project_root / "wechat_server.err.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch(exist_ok=True)
    err_path.touch(exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.addFilter(_MaxLevelFilter(logging.INFO))
    stdout_handler.setFormatter(fmt)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(fmt)

    out_file_handler = logging.FileHandler(out_path, encoding="utf-8")
    out_file_handler.setLevel(level)
    out_file_handler.addFilter(_MaxLevelFilter(logging.INFO))
    out_file_handler.setFormatter(fmt)

    err_file_handler = logging.FileHandler(err_path, encoding="utf-8")
    err_file_handler.setLevel(logging.WARNING)
    err_file_handler.setFormatter(fmt)

    root_logger.addHandler(stdout_handler)
    root_logger.addHandler(stderr_handler)
    root_logger.addHandler(out_file_handler)
    root_logger.addHandler(err_file_handler)

    logging.captureWarnings(True)
    return logging.getLogger(logger_name)
