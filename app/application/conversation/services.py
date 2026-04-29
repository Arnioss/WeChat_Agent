import logging
import os
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from db.client import get_mysql_pool, load_mysql_config
from app.infrastructure.cache import RedisCache

logger = logging.getLogger(__name__)


@dataclass
class SessionState:
    """功能：保存单个用户会话在内存中的运行态对象与并发控制锁。
    参数：
    - 无。
    返回值：
    - 无。会随 TTL 过期被 `SessionManager` 清理，不作为永久存储。
    """
    agent: Any
    user_id: str = ""
    session_id: str = ""
    last_active: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


class ChatStore:
    """功能：负责会话消息持久化、历史读取与消息去重状态落库。
    参数：
    - 无。
    返回值：
    - 无。依赖 MySQL，配置未启用时会在初始化阶段直接报错。
    """
    def __init__(self):
        """功能：读取会话存储配置并完成表结构与清理线程初始化。
        参数：
        - 无。
        返回值：
        - 无。会立即校验配置并尝试建表，失败时向上抛出异常阻止服务启动。
        """
        cfg = load_mysql_config()
        self.enabled = cfg["enabled"]
        self.history_limit = int(os.getenv("CHAT_HISTORY_LOAD_LIMIT"))
        self.retention_days = int(os.getenv("CHAT_MESSAGE_RETENTION_DAYS"))
        self.session_inactive_seconds = int(
            os.getenv("CHAT_SESSION_INACTIVE_RETENTION_SECONDS")
        )
        self.cleanup_interval_seconds = int(os.getenv("CHAT_CLEANUP_INTERVAL_SECONDS"))
        self.dedup_ttl_seconds = int(os.getenv("CHAT_MESSAGE_DEDUP_TTL_SECONDS"))
        self._lock = threading.Lock()
        self._ensure_ready()
        self._init_tables()
        self._start_cleanup_worker()

    def _ensure_ready(self):
        """功能：校验聊天持久化前置条件，确保 MySQL 持久化已启用。
        参数：
        - 无。
        返回值：
        - 无。
        """
        if not self.enabled:
            raise ValueError("MYSQL_ENABLED=false，无法启用会话持久化")

    def _init_tables(self):
        """功能：初始化会话、消息和去重三张核心数据表。
        参数：
        - 无。
        返回值：
        - 无。
        """
        with self._lock:
            with get_mysql_pool().connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_sessions (
                            id BIGINT PRIMARY KEY AUTO_INCREMENT,
                            user_id VARCHAR(128) NOT NULL,
                            session_id VARCHAR(128) NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                            last_active TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE KEY uk_user_session (user_id, session_id),
                            KEY idx_last_active (last_active)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_messages (
                            id BIGINT PRIMARY KEY AUTO_INCREMENT,
                            user_id VARCHAR(128) NOT NULL,
                            session_id VARCHAR(128) NOT NULL,
                            role VARCHAR(32) NOT NULL,
                            content TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            KEY idx_user_session_id (user_id, session_id, id),
                            KEY idx_created_at (created_at)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_message_dedup (
                            msgid VARCHAR(128) PRIMARY KEY,
                            status VARCHAR(32) NOT NULL,
                            answer TEXT NULL,
                            stream_id VARCHAR(64) NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                            KEY idx_updated_at (updated_at)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        """
                    )

    def _run_cleanup_once(self):
        """功能：执行一次清理任务，删除过期会话、消息和去重记录。
        参数：
        - 无。
        返回值：
        - 无。
        """
        with self._lock:
            with get_mysql_pool().connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        DELETE m
                        FROM chat_messages m
                        JOIN chat_sessions s
                          ON s.user_id = m.user_id
                         AND s.session_id = m.session_id
                        WHERE s.last_active < DATE_SUB(NOW(), INTERVAL %s SECOND)
                        """,
                        (self.session_inactive_seconds,),
                    )
                    cursor.execute(
                        """
                        DELETE FROM chat_sessions
                        WHERE last_active < DATE_SUB(NOW(), INTERVAL %s SECOND)
                        """,
                        (self.session_inactive_seconds,),
                    )
                    cursor.execute(
                        """
                        DELETE FROM chat_messages
                        WHERE created_at < DATE_SUB(NOW(), INTERVAL %s DAY)
                        """,
                        (self.retention_days,),
                    )
                    cursor.execute(
                        """
                        DELETE FROM chat_message_dedup
                        WHERE updated_at < DATE_SUB(NOW(), INTERVAL %s SECOND)
                        """,
                        (self.dedup_ttl_seconds,),
                    )

    def _cleanup_loop(self):
        """功能：后台循环执行数据清理任务，并在失败时记录异常日志。
        参数：
        - 无。
        返回值：
        - 无。
        """
        interval = max(30, self.cleanup_interval_seconds)
        while True:
            try:
                self._run_cleanup_once()
            except Exception:
                logger.exception("CHAT_STORE_CLEANUP_FAILED")
            time.sleep(interval)

    def _start_cleanup_worker(self):
        """功能：启动守护线程定时执行历史数据清理。
        参数：
        - 无。
        返回值：
        - 无。
        """
        worker = threading.Thread(
            target=self._cleanup_loop,
            name="chat-cleanup-worker",
            daemon=True,
        )
        worker.start()

    def touch_session(self, user_id: str, session_id: str):
        """功能：刷新会话活跃时间，必要时创建会话记录。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        返回值：
        - 无。
        """
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_sessions (user_id, session_id, last_active)
                    VALUES (%s, %s, NOW())
                    ON DUPLICATE KEY UPDATE last_active = NOW()
                    """,
                    (user_id, session_id),
                )

    def append_turn(self, user_id: str, session_id: str, user_text: str, answer: str):
        """功能：持久化一轮问答消息并刷新会话活跃时间。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        - user_text：用户输入文本。
        - answer：模型输出的最终回答文本。
        返回值：
        - 无。
        """
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_messages (user_id, session_id, role, content)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, session_id, "user", user_text or ""),
                )
                cursor.execute(
                    """
                    INSERT INTO chat_messages (user_id, session_id, role, content)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, session_id, "assistant", answer or ""),
                )
                cursor.execute(
                    """
                    INSERT INTO chat_sessions (user_id, session_id, last_active)
                    VALUES (%s, %s, NOW())
                    ON DUPLICATE KEY UPDATE last_active = NOW()
                    """,
                    (user_id, session_id),
                )

    def load_recent_turns(self, user_id: str, session_id: str) -> List[Tuple[str, str]]:
        """功能：读取会话最近问答轮次并配对为 `(问题,回答)` 列表。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        返回值：
        - List[Tuple[str, str]]：按时间顺序排列的历史问答轮次。
        """
        limit_rows = max(1, self.history_limit * 2)
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT role, content
                    FROM chat_messages
                    WHERE user_id = %s AND session_id = %s
                    ORDER BY id DESC
                    LIMIT %s
                    """,
                    (user_id, session_id, limit_rows),
                )
                rows = cursor.fetchall() or []

        rows = list(reversed(rows))
        turns: List[Tuple[str, str]] = []
        pending_user: Optional[str] = None
        for row in rows:
            role = str((row or {}).get("role") or "")
            content = str((row or {}).get("content") or "")
            if role == "user":
                pending_user = content
            elif role == "assistant" and pending_user is not None:
                turns.append((pending_user, content))
                pending_user = None
        if len(turns) > self.history_limit:
            turns = turns[-self.history_limit :]
        return turns

    def get_dedup(self, msgid: str) -> Optional[dict]:
        """功能：按消息 ID 查询去重状态记录。
        参数：
        - msgid：消息唯一 ID。
        返回值：
        - Optional[dict]：命中时返回状态字典，未命中返回 None。
        """
        if not msgid:
            return None
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT status, answer, stream_id, updated_at
                    FROM chat_message_dedup
                    WHERE msgid = %s
                    """,
                    (msgid,),
                )
                return cursor.fetchone()

    def mark_processing(self, msgid: str, stream_id: str) -> bool:
        """功能：将消息标记为处理中，避免重复消费。
        参数：
        - msgid：消息唯一 ID。
        - stream_id：本次处理对应的流式响应 ID。
        返回值：
        - bool：首次写入成功返回 True，已存在返回 False。
        """
        if not msgid:
            return True
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT IGNORE INTO chat_message_dedup (msgid, status, answer, stream_id)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (msgid, "processing", None, stream_id),
                )
                return cursor.rowcount > 0

    def mark_done(self, msgid: str, answer: str):
        """功能：将消息去重状态更新为已完成并写入答案。
        参数：
        - msgid：消息唯一 ID。
        - answer：模型输出的最终回答文本。
        返回值：
        - 无。
        """
        if not msgid:
            return
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE chat_message_dedup
                    SET status = %s, answer = %s
                    WHERE msgid = %s
                    """,
                    ("done", answer or "", msgid),
                )


class DedupService:
    """功能：统一管理消息去重状态，协调 Redis 缓存与数据库双写。
    参数：
    - 无。
    返回值：
    - 无。优先走缓存提升并发性能，缓存不可用时自动降级到数据库。
    """
    def __init__(self, chat_store: ChatStore, cache: Optional[RedisCache] = None):
        """功能：注入去重存储依赖并设置去重状态缓存 TTL。
        参数：
        - chat_store：数据库持久化存储服务。
        - cache：可选 Redis 缓存实例。
        返回值：
        - 无。未传入 `cache` 时会创建默认 `RedisCache`，但可在不可用时自动降级。
        """
        self.chat_store = chat_store
        self.cache = cache or RedisCache()
        self.ttl_seconds = int(os.getenv("CHAT_MESSAGE_DEDUP_TTL_SECONDS"))

    def get(self, msgid: str) -> Optional[dict]:
        """功能：读取消息去重状态，优先命中缓存。
        参数：
        - msgid：消息唯一 ID。
        返回值：
        - Optional[dict]：去重状态记录；不存在时返回 None。
        """
        if msgid and self.cache.available:
            cached = self.cache.get_json(self.cache.make_key("dedup", msgid))
            if cached:
                return cached
        return self.chat_store.get_dedup(msgid)

    def mark_processing(self, msgid: str, stream_id: str) -> bool:
        """功能：标记消息进入处理中状态，并同步缓存与数据库。
        参数：
        - msgid：消息唯一 ID。
        - stream_id：流式响应 ID。
        返回值：
        - bool：成功占位返回 True，重复消息返回 False。
        """
        if not msgid:
            return True
        if self.cache.available:
            ok = self.cache.set_json_if_absent(
                self.cache.make_key("dedup", msgid),
                {
                    "status": "processing",
                    "answer": None,
                    "stream_id": stream_id,
                    "updated_at": time.time(),
                },
                ttl_seconds=self.ttl_seconds,
            )
            if ok:
                self.chat_store.mark_processing(msgid, stream_id)
            return ok
        return self.chat_store.mark_processing(msgid, stream_id)

    def mark_done(self, msgid: str, answer: str):
        """功能：标记消息处理完成，并回写最终答案。
        参数：
        - msgid：消息唯一 ID。
        - answer：模型输出的最终回答文本。
        返回值：
        - 无。
        """
        if not msgid:
            return
        current = self.get(msgid) or {}
        self.chat_store.mark_done(msgid, answer)
        if self.cache.available:
            self.cache.set_json(
                self.cache.make_key("dedup", msgid),
                {
                    "status": "done",
                    "answer": answer or "",
                    "stream_id": current.get("stream_id"),
                    "updated_at": time.time(),
                },
                ttl_seconds=self.ttl_seconds,
            )


class SessionManager:
    """功能：管理会话级 Agent 实例生命周期，并提供会话分布式锁能力。
    参数：
    - 无。
    返回值：
    - 无。仅维护短期内存态，会话历史仍以 `ChatStore` 为准。
    """
    def __init__(
        self,
        agent_factory: Callable[[], Any],
        chat_store: ChatStore,
        ttl_seconds: int = 1800,
        cache: Optional[RedisCache] = None,
    ):
        """功能：配置会话工厂、持久化依赖和本地/分布式会话治理参数。
        参数：
        - agent_factory：创建智能体实例的工厂函数。
        - chat_store：聊天历史存储服务。
        - ttl_seconds：本地会话状态保活秒数。
        - cache：可选 Redis 缓存实例。
        返回值：
        - 无。`ttl_seconds` 控制内存回收，锁 TTL 由环境变量 `SESSION_LOCK_TTL_SECONDS` 决定。
        """
        self.agent_factory = agent_factory
        self.chat_store = chat_store
        self.ttl_seconds = ttl_seconds
        self.cache = cache or RedisCache()
        self.lock_ttl_seconds = int(os.getenv("SESSION_LOCK_TTL_SECONDS"))
        self._sessions: Dict[Tuple[str, str], SessionState] = {}
        self._lock = threading.Lock()

    def _cleanup_expired(self) -> None:
        """功能：清理本地内存中超过 TTL 的会话状态对象。
        参数：
        - 无。
        返回值：
        - 无。
        """
        now = time.time()
        expired = [
            key
            for key, state in self._sessions.items()
            if now - state.last_active > self.ttl_seconds
        ]
        for key in expired:
            self._sessions.pop(key, None)

    def get(self, user_id: str, session_id: str) -> SessionState:
        """功能：获取会话状态，必要时创建新会话并加载历史上下文。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        返回值：
        - SessionState：可用于当前请求处理的会话状态对象。
        """
        key = (user_id, session_id)
        with self._lock:
            self._cleanup_expired()
            state = self._sessions.get(key)
            if state is None:
                agent = self.agent_factory()
                turns = self.chat_store.load_recent_turns(user_id=user_id, session_id=session_id)
                if turns:
                    agent.conversation_turns = turns
                state = SessionState(
                    agent=agent,
                    user_id=user_id,
                    session_id=session_id,
                )
                self._sessions[key] = state
            self.chat_store.touch_session(user_id=user_id, session_id=session_id)
            if self.cache.available:
                self.cache.set_json(
                    self.cache.make_key("session", f"{user_id}:{session_id}"),
                    {
                        "user_id": user_id,
                        "session_id": session_id,
                        "last_active": time.time(),
                    },
                    ttl_seconds=self.ttl_seconds,
                )
            state.last_active = time.time()
            return state

    def acquire_lease(self, user_id: str, session_id: str, owner: str) -> bool:
        """功能：尝试获取会话级分布式处理锁。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        - owner：锁持有者标识。
        返回值：
        - bool：成功获取锁返回 True。
        """
        if not self.cache.available:
            return True
        return self.cache.acquire_lock(
            self.cache.make_key("session-lock", f"{user_id}:{session_id}"),
            owner=owner,
            ttl_seconds=self.lock_ttl_seconds,
        )

    def release_lease(self, user_id: str, session_id: str, owner: str) -> bool:
        """功能：释放会话级分布式处理锁。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        - owner：锁持有者标识。
        返回值：
        - bool：释放成功返回 True。
        """
        if not self.cache.available:
            return True
        return self.cache.release_lock(
            self.cache.make_key("session-lock", f"{user_id}:{session_id}"),
            owner=owner,
        )

    def local_size(self) -> int:
        """功能：返回本地内存中当前有效会话状态数量。
        参数：
        - 无。
        返回值：
        - int：清理过期项后剩余的会话状态条数。
        """
        with self._lock:
            self._cleanup_expired()
            return len(self._sessions)


@dataclass
class StreamState:
    """功能：描述单个流式响应任务的内容、状态与时效信息。
    参数：
    - 无。
    返回值：
    - 无。用于内存和缓存之间同步流式进度，不承载业务决策逻辑。
    """
    content: str = ""
    finish: bool = False
    template_card: Optional[dict] = None
    session_id: str = ""
    status: str = "processing"
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class StreamManager:
    """功能：维护流式响应状态的创建、更新、读取和过期清理。
    参数：
    - 无。
    返回值：
    - 无。支持本地内存与 Redis 双层存储，保证跨请求可恢复。
    """
    def __init__(self, ttl_seconds: int = 600, cache: Optional[RedisCache] = None):
        """功能：设置流式状态保活策略并初始化缓存与本地状态容器。
        参数：
        - ttl_seconds：流式状态保活秒数。
        - cache：可选 Redis 缓存实例。
        返回值：
        - 无。超过 TTL 的状态会在读写路径上被惰性清理。
        """
        self.ttl_seconds = ttl_seconds
        self.cache = cache or RedisCache()
        self._data: Dict[str, StreamState] = {}
        self._lock = threading.Lock()

    def _cleanup(self):
        """功能：清理本地内存中超过 TTL 的流式状态对象。
        参数：
        - 无。
        返回值：
        - 无。
        """
        now = time.time()
        expired = [
            key
            for key, value in self._data.items()
            if now - value.last_active > self.ttl_seconds
        ]
        for key in expired:
            self._data.pop(key, None)

    def create(
        self,
        stream_id: str,
        content: str = "",
        finish: bool = False,
        template_card: Optional[dict] = None,
        session_id: str = "",
        status: str = "processing",
    ):
        """功能：创建新的流式响应状态并写入缓存。
        参数：
        - stream_id：流式响应 ID。
        - content：当前展示内容。
        - finish：是否结束流式输出。
        - template_card：可选模板卡片内容。
        - session_id：所属会话 ID。
        - status：处理状态标识。
        返回值：
        - 无。
        """
        with self._lock:
            self._cleanup()
            state = StreamState(
                content=content,
                finish=finish,
                template_card=template_card,
                session_id=session_id,
                status=status,
                created_at=time.time(),
                last_active=time.time(),
            )
            self._data[stream_id] = state
            self._persist(stream_id, state)

    def update(
        self,
        stream_id: str,
        content: str,
        finish: bool,
        template_card: Optional[dict] = None,
        status: Optional[str] = None,
    ):
        """功能：更新已有流式状态并持久化。
        参数：
        - stream_id：流式响应 ID。
        - content：最新展示内容。
        - finish：是否结束流式输出。
        - template_card：可选模板卡片内容。
        - status：可选状态更新值。
        返回值：
        - 无。
        """
        with self._lock:
            self._cleanup()
            state = self._data.get(stream_id)
            if state is None:
                state = StreamState()
                self._data[stream_id] = state
            state.content = content
            state.finish = finish
            if template_card is not None:
                state.template_card = template_card
            if status:
                state.status = status
            state.last_active = time.time()
            self._persist(stream_id, state)

    def get(self, stream_id: str) -> Optional[StreamState]:
        """功能：读取流式状态，必要时从缓存回填本地内存。
        参数：
        - stream_id：流式响应 ID。
        返回值：
        - Optional[StreamState]：命中时返回流式状态对象，未命中返回 None。
        """
        if self.cache.available:
            cached = self.cache.get_json(self.cache.make_key("stream", stream_id))
            if cached:
                state = StreamState(
                    content=str(cached.get("content") or ""),
                    finish=bool(cached.get("finish")),
                    template_card=cached.get("template_card"),
                    session_id=str(cached.get("session_id") or ""),
                    status=str(cached.get("status") or "processing"),
                    created_at=float(cached.get("created_at") or time.time()),
                    last_active=float(cached.get("last_active") or time.time()),
                )
                with self._lock:
                    self._data[stream_id] = state
        with self._lock:
            self._cleanup()
            state = self._data.get(stream_id)
            if state:
                state.last_active = time.time()
                self._persist(stream_id, state)
            return state

    def mark_failed(self, stream_id: str, content: str):
        """功能：将流式任务标记为失败并结束输出。
        参数：
        - stream_id：流式响应 ID。
        - content：失败提示内容。
        返回值：
        - 无。
        """
        self.update(stream_id=stream_id, content=content, finish=True, status="failed")

    def mark_timeout(self, stream_id: str, content: str, template_card: Optional[dict] = None):
        """功能：将流式任务标记为超时并结束输出。
        参数：
        - stream_id：流式响应 ID。
        - content：超时提示内容。
        - template_card：可选超时卡片内容。
        返回值：
        - 无。
        """
        self.update(
            stream_id=stream_id,
            content=content,
            finish=True,
            template_card=template_card,
            status="timeout",
        )

    def _persist(self, stream_id: str, state: StreamState) -> None:
        """功能：把流式状态写入缓存存储。
        参数：
        - stream_id：流式响应 ID。
        - state：待持久化的流式状态对象。
        返回值：
        - 无。
        """
        if not self.cache.available:
            return
        self.cache.set_json(
            self.cache.make_key("stream", stream_id),
            {
                "content": state.content,
                "finish": state.finish,
                "template_card": state.template_card,
                "session_id": state.session_id,
                "status": state.status,
                "created_at": state.created_at,
                "last_active": state.last_active,
            },
            ttl_seconds=self.ttl_seconds,
        )

    def local_size(self) -> int:
        """功能：返回本地内存中当前有效流式状态数量。
        参数：
        - 无。
        返回值：
        - int：清理过期项后剩余的流式状态条数。
        """
        with self._lock:
            self._cleanup()
            return len(self._data)
