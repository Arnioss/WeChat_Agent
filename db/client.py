import os
import queue
import threading
import logging
from contextlib import contextmanager
from typing import Any, Dict, Iterator

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

try:
    import pymysql
except Exception:  # pragma: no cover - optional dependency
    pymysql = None

load_dotenv()


def load_mysql_config() -> Dict[str, Any]:
    """功能：从环境变量读取并校验 MySQL 连接配置。
    参数：
    - 无。
    返回值：
    - Dict[str, Any]：标准化后的 MySQL 配置字典。
    异常：
    - ValueError：端口/超时/连接池大小格式错误或缺少必填配置时抛出。
    """
    enabled = (os.getenv("MYSQL_ENABLED") or "").lower() in ("1", "true", "yes", "on")
    raw_port = os.getenv("MYSQL_PORT")
    raw_timeout = os.getenv("MYSQL_CONNECT_TIMEOUT")
    raw_pool_size = os.getenv("MYSQL_POOL_SIZE")

    try:
        port = int(raw_port)
    except ValueError as e:
        raise ValueError("MYSQL_PORT 必须是整数") from e

    try:
        timeout = int(raw_timeout)
    except ValueError as e:
        raise ValueError("MYSQL_CONNECT_TIMEOUT 必须是整数") from e

    try:
        pool_size = int(raw_pool_size) if raw_pool_size else 5
    except ValueError as e:
        raise ValueError("MYSQL_POOL_SIZE 必须是整数") from e

    host = (os.getenv("MYSQL_HOST") or "").strip()
    user = (os.getenv("MYSQL_USER") or os.getenv("MYSQL_USERNAME") or "").strip()
    password = (os.getenv("MYSQL_PASSWORD") or "").strip()
    database = (os.getenv("MYSQL_DATABASE") or "").strip()

    if enabled:
        missing = []
        if not host:
            missing.append("MYSQL_HOST")
        if not user:
            missing.append("MYSQL_USER（或 MYSQL_USERNAME）")
        if not password:
            missing.append("MYSQL_PASSWORD")
        if not database:
            missing.append("MYSQL_DATABASE")
        if missing:
            raise ValueError("MySQL 已启用，但缺少必要环境变量: " + ", ".join(missing))

    return {
        "enabled": enabled,
        "host": host,
        "port": port,
        "user": user,
        "password": password,
        "database": database,
        "timeout": timeout,
        "pool_size": max(1, pool_size),
    }


class MySQLPool:
    """功能：管理MySQL连接池，提供连接复用、健康检查与安全回收能力。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self, config: Dict[str, Any]):
        """功能：根据配置初始化连接池（懒创建，避免进程启动时占满 MySQL 连接数）。
        参数：
        - config：数据库配置字典，需包含主机、端口、账号、库名、超时与池大小。
        返回值：
        - 无。
        """
        if not config.get("enabled"):
            raise RuntimeError("MYSQL_ENABLED=false，无法初始化数据库连接池")
        if pymysql is None:
            raise RuntimeError("未安装 pymysql，请先执行: pip install pymysql")
        self._cfg = config
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=config["pool_size"])
        self._lock = threading.Lock()
        self._created = 0

    def _new_connection(self):
        """功能：创建单个新的MySQL连接对象。
        参数：
        - 无。
        返回值：
        - Connection：可用的pymysql连接。
        """
        return pymysql.connect(
            host=self._cfg["host"],
            port=self._cfg["port"],
            user=self._cfg["user"],
            password=self._cfg["password"],
            database=self._cfg["database"],
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=self._cfg["timeout"],
            read_timeout=self._cfg["timeout"],
            write_timeout=self._cfg["timeout"],
            autocommit=True,
        )

    def _create_connection(self):
        """功能：在池上限内创建新连接并更新计数。
        参数：
        - 无。
        返回值：
        - Connection：新创建的 pymysql 连接对象。
        异常：
        - RuntimeError：连接池已达 pool_size 上限时抛出。
        """
        with self._lock:
            if self._created >= self._cfg["pool_size"]:
                raise RuntimeError("MySQL 连接池已满")
            conn = self._new_connection()
            self._created += 1
            return conn

    def _discard_connection(self, conn) -> None:
        """功能：关闭连接并从池计数中扣除。
        参数：
        - conn：待关闭的数据库连接对象。
        返回值：
        - 无。
        """
        try:
            conn.close()
        except Exception:
            pass
        with self._lock:
            self._created = max(0, self._created - 1)

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """功能：提供上下文连接接口，自动完成取连接、探活与归还。
        参数：
        - 无。
        返回值：
        - Iterator[Any]：上下文中的可用数据库连接。
        """
        conn = None
        try:
            conn = self._acquire()
            try:
                conn.ping(reconnect=True)
            except Exception:
                self._discard_connection(conn)
                conn = self._create_connection()
            yield conn
        finally:
            if conn is not None:
                self._release(conn)

    def _acquire(self):
        """功能：从连接池获取一个连接；池空且未达上限则懒创建。
        参数：
        - 无。
        返回值：
        - Connection：从池中取出的数据库连接。
        """
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            return self._create_connection()
        except RuntimeError:
            return self._queue.get(timeout=max(1, self._cfg["timeout"] + 1))

    def _release(self, conn):
        """功能：将连接归还到连接池；若池已满则关闭该连接。
        参数：
        - conn：待归还的数据库连接对象。
        返回值：
        - 无。
        """
        try:
            self._queue.put_nowait(conn)
        except queue.Full:
            self._discard_connection(conn)


_POOL = None
_POOL_LOCK = threading.Lock()
_MYSQL_WARMED = False


def get_mysql_pool() -> MySQLPool:
    """功能：获取全局单例连接池；首次调用时自动初始化。
    参数：
    - 无。
    返回值：
    - MySQLPool：全局共享的连接池实例。
    """
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = MySQLPool(load_mysql_config())
    return _POOL


def warm_mysql_pool() -> Dict[str, Any]:
    """功能：初始化连接池并探活，供 Web 启动预热使用（同进程内只记录一次日志）。
    参数：
    - 无。
    返回值：
    - Dict[str, Any]：含 enabled、available 及连接信息；未启用时 enabled/available 均为 False。
    """
    global _MYSQL_WARMED

    cfg = load_mysql_config()
    if not cfg.get("enabled"):
        return {"enabled": False, "available": False}

    pool = get_mysql_pool()
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT 1")

    if not _MYSQL_WARMED:
        logger.info(
            "MySQL 已连接 host=%s port=%s db=%s pool_size=%s",
            cfg["host"],
            cfg["port"],
            cfg["database"],
            cfg["pool_size"],
        )
        _MYSQL_WARMED = True

    return {
        "enabled": True,
        "available": True,
        "host": cfg["host"],
        "port": cfg["port"],
        "database": cfg["database"],
        "pool_size": cfg["pool_size"],
    }
