import os
import queue
import threading
from contextlib import contextmanager
from typing import Any, Dict, Iterator

from dotenv import load_dotenv

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
        pool_size = int(raw_pool_size)
    except ValueError as e:
        raise ValueError("MYSQL_POOL_SIZE 必须是整数") from e

    host = (os.getenv("MYSQL_HOST") or "").strip()
    user = (os.getenv("MYSQL_USER") or "").strip()
    password = (os.getenv("MYSQL_PASSWORD") or "").strip()
    database = (os.getenv("MYSQL_DATABASE") or "").strip()

    if enabled:
        missing = []
        if not host:
            missing.append("MYSQL_HOST")
        if not user:
            missing.append("MYSQL_USER")
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
    """功能：维护线程安全的 MySQL 连接池。
    参数：
    - 无。
    返回值：
    - 无。
    """
    def __init__(self, config: Dict[str, Any]):
        """功能：校验运行条件并按配置预热线程安全 MySQL 连接池。
        参数：
        - config：MySQL 连接与连接池配置字典。
        返回值：
        - 无。MySQL 未启用或缺少 `pymysql` 依赖时会直接抛错，避免运行期隐式失败。
        """
        if not config.get("enabled"):
            raise RuntimeError("MYSQL_ENABLED=false，无法初始化数据库连接池")
        if pymysql is None:
            raise RuntimeError("未安装 pymysql，请先执行: pip install pymysql")
        self._cfg = config
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=config["pool_size"])
        self._lock = threading.Lock()
        self._initialize_pool()

    def _new_connection(self):
        """功能：创建新的 MySQL 连接对象。
        参数：
        - 无。
        返回值：
        - Connection：pymysql 连接对象。
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

    def _initialize_pool(self):
        """功能：按配置预热连接池。
        参数：
        - 无。
        返回值：
        - 无。
        """
        with self._lock:
            if not self._queue.empty():
                return
            for _ in range(self._cfg["pool_size"]):
                self._queue.put(self._new_connection())

    @contextmanager
    def connection(self) -> Iterator[Any]:
        """功能：提供可自动归还连接的上下文管理器。
        参数：
        - 无。
        返回值：
        - 无。
        """
        conn = None
        try:
            conn = self._acquire()
            try:
                conn.ping(reconnect=True)
            except Exception:
                try:
                    conn.close()
                except Exception:
                    pass
                conn = self._new_connection()
            yield conn
        finally:
            if conn is not None:
                self._release(conn)

    def _acquire(self):
        """功能：从连接池获取一个可用连接。
        参数：
        - 无。
        返回值：
        - Connection：可用数据库连接对象。
        """
        return self._queue.get(timeout=max(1, self._cfg["timeout"] + 1))

    def _release(self, conn):
        """功能：归还连接到连接池，池满时关闭连接。
        参数：
        - conn：待归还的数据库连接对象。
        返回值：
        - 无。
        """
        try:
            self._queue.put_nowait(conn)
        except queue.Full:
            try:
                conn.close()
            except Exception:
                pass


_POOL = None
_POOL_LOCK = threading.Lock()


def get_mysql_pool() -> MySQLPool:
    """功能：获取全局单例 MySQL 连接池。
    参数：
    - 无。
    返回值：
    - MySQLPool：全局连接池实例。
    """
    global _POOL
    if _POOL is None:
        with _POOL_LOCK:
            if _POOL is None:
                _POOL = MySQLPool(load_mysql_config())
    return _POOL
