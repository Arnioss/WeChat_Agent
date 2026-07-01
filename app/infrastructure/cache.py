import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import redis
except Exception:  # pragma: no cover
    redis = None


class RedisCache:
    """功能：封装 Redis 读写与分布式锁操作，支持降级不可用状态。
    参数：
    - 无（单例类，通过 __new__/__init__ 按环境变量初始化）。
    返回值：
    - 无。同一进程内复用单例，避免 Streamlit 启动预热与首访重复建连。
    """
    _shared_instance: "RedisCache | None" = None

    def __new__(cls) -> "RedisCache":
        """功能：返回进程内共享的 RedisCache 单例实例。
        参数：
        - 无。
        返回值：
        - RedisCache：首次调用时创建实例，后续调用复用同一对象。
        """
        if cls._shared_instance is None:
            cls._shared_instance = super().__new__(cls)
        return cls._shared_instance

    def __init__(self):
        """功能：根据环境变量建立 Redis 连接，并确定缓存可用性降级状态。
        参数：
        - 无。
        返回值：
        - 无。未启用或连接失败时实例仍可用，但所有缓存操作会安全返回失败。
        """
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self.enabled = (os.getenv("REDIS_ENABLED") or "").lower() in ("1", "true", "yes", "on")
        self.prefix = os.getenv("REDIS_KEY_PREFIX")
        if not self.prefix:
            raise ValueError("缺少环境变量 REDIS_KEY_PREFIX，请在 .env 文件中设置。")
        self._client = None
        self._available = False
        if not self.enabled:
            return
        if redis is None:
            logger.warning("Redis 未启用：未安装 redis 包")
            return
        try:
            self._client = redis.Redis(
                host=os.getenv("REDIS_HOST"),
                port=int(os.getenv("REDIS_PORT")),
                db=int(os.getenv("REDIS_DB")),
                password=os.getenv("REDIS_PASSWORD") or None,
                decode_responses=True,
                socket_timeout=float(os.getenv("REDIS_SOCKET_TIMEOUT_SECONDS")),
                socket_connect_timeout=float(os.getenv("REDIS_CONNECT_TIMEOUT_SECONDS")),
            )
            self._client.ping()
            self._available = True
            logger.info("Redis 已连接 host=%s port=%s db=%s", os.getenv("REDIS_HOST"), os.getenv("REDIS_PORT"), os.getenv("REDIS_DB"))
        except Exception as e:
            logger.warning("Redis 不可用，回退本地缓存 error=%s", e)
            self._available = False
            self._client = None

    @property
    def available(self) -> bool:
        """功能：判断 Redis 缓存当前是否可用。
        参数：
        - 无。
        返回值：
        - bool：启用且连接可用时返回 True。
        """
        return self.enabled and self._available and self._client is not None

    def make_key(self, namespace: str, suffix: str) -> str:
        """功能：按统一前缀规则生成 Redis 键名。
        参数：
        - namespace：业务命名空间。
        - suffix：键后缀（通常为业务 ID）。
        返回值：
        - str：拼接后的完整 Redis 键名。
        """
        return f"{self.prefix}:{namespace}:{suffix}"

    def get_json(self, key: str) -> Optional[dict]:
        """功能：读取并反序列化 JSON 缓存值。
        参数：
        - key：缓存键名。
        返回值：
        - Optional[dict]：命中时返回字典，未命中或失败返回 None。
        """
        if not self.available:
            return None
        try:
            value = self._client.get(key)
            if not value:
                return None
            return json.loads(value)
        except Exception:
            logger.exception("Redis 读取 JSON 失败 key=%s", key)
            return None

    def set_json(self, key: str, value: dict, ttl_seconds: Optional[int] = None) -> bool:
        """功能：写入 JSON 缓存，可选设置过期时间。
        参数：
        - key：缓存键名。
        - value：待写入的字典对象。
        - ttl_seconds：可选过期秒数；为空或非正数时写入永久键。
        返回值：
        - bool：写入成功返回 True，否则返回 False。
        """
        if not self.available:
            return False
        try:
            payload = json.dumps(value, ensure_ascii=False)
            if ttl_seconds and ttl_seconds > 0:
                self._client.set(key, payload, ex=int(ttl_seconds))
            else:
                self._client.set(key, payload)
            return True
        except Exception:
            logger.exception("Redis 写入 JSON 失败 key=%s", key)
            return False

    def set_json_if_absent(self, key: str, value: dict, ttl_seconds: int) -> bool:
        """功能：仅在键不存在时写入 JSON 缓存。
        参数：
        - key：缓存键名。
        - value：待写入的字典对象。
        - ttl_seconds：过期秒数。
        返回值：
        - bool：写入成功返回 True；键已存在或失败返回 False。
        """
        if not self.available:
            return False
        try:
            payload = json.dumps(value, ensure_ascii=False)
            return bool(self._client.set(key, payload, ex=max(1, int(ttl_seconds)), nx=True))
        except Exception:
            logger.exception("Redis 条件写入失败 key=%s", key)
            return False

    def delete(self, key: str) -> bool:
        """功能：删除指定缓存键。
        参数：
        - key：缓存键名。
        返回值：
        - bool：删除请求成功返回 True，否则返回 False。
        """
        if not self.available:
            return False
        try:
            self._client.delete(key)
            return True
        except Exception:
            logger.exception("Redis 删除失败 key=%s", key)
            return False

    def acquire_lock(self, key: str, owner: str, ttl_seconds: int) -> bool:
        """功能：尝试获取分布式锁。
        参数：
        - key：锁键名。
        - owner：锁持有者标识。
        - ttl_seconds：锁自动过期秒数。
        返回值：
        - bool：成功加锁返回 True，失败返回 False。
        """
        if not self.available:
            return False
        try:
            return bool(self._client.set(key, owner, ex=max(1, int(ttl_seconds)), nx=True))
        except Exception:
            logger.exception("Redis 获取锁失败 key=%s", key)
            return False

    def release_lock(self, key: str, owner: str) -> bool:
        """功能：按持有者校验后释放分布式锁。
        参数：
        - key：锁键名。
        - owner：期望的锁持有者标识。
        返回值：
        - bool：仅当当前持有者匹配时删除锁并返回 True，否则返回 False。
        """
        if not self.available:
            return False
        try:
            current = self._client.get(key)
            if current == owner:
                self._client.delete(key)
                return True
            return False
        except Exception:
            logger.exception("Redis 释放锁失败 key=%s", key)
            return False
