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

_CLEANUP_WORKER_LOCK = threading.Lock()
_CLEANUP_WORKER_STARTED = False


def _normalize_bot_id(bot_id: Optional[str]) -> str:
    """功能：规范化 bot_id 字符串，去除首尾空白。
    参数：
    - bot_id：可选机器人标识。
    返回值：
    - str：规范化后的 bot_id；None 或空值时返回空字符串。
    """
    return str(bot_id or "").strip()


@dataclass
class SessionState:
    """功能：保存单个用户会话在内存中的运行态对象与并发控制锁。
    参数：
    - 无。
    返回值：
    - 无。会随 TTL 过期被 `SessionManager` 清理，不作为永久存储。
    """
    agent: Any
    bot_id: str = ""
    user_id: str = ""
    session_id: str = ""
    last_active: float = field(default_factory=time.time)
    lock: threading.Lock = field(default_factory=threading.Lock)


@dataclass
class ChatSessionSummary:
    """功能：描述单个聊天会话的摘要信息，供列表展示使用。
    参数：
    - 无。
    返回值：
    - 无。作为只读数据载体，由 `ChatStore.list_sessions` 构造。
    """
    session_id: str
    title: str
    created_at: Any = None
    updated_at: Any = None
    last_active: Any = None
    message_count: int = 0
    last_message: str = ""


class ChatStore:
    """功能：负责会话消息持久化、历史读取与消息去重状态落库（按 bot_id 隔离）。
    参数：
    - 无。
    返回值：
    - 无。依赖 MySQL，配置未启用时会在初始化阶段直接报错。
    """
    def __init__(self, bot_id: Optional[str] = None):
        """功能：读取会话存储配置、绑定 bot_id 并完成表结构与清理线程初始化。
        参数：
        - bot_id：可选机器人标识；未传时从环境变量 `WECHAT_BOT_ID` 读取。
        返回值：
        - 无。会立即校验配置并尝试建表/迁移，失败时向上抛出异常阻止服务启动。
        """
        cfg = load_mysql_config()
        self.enabled = cfg["enabled"]
        self.bot_id = _normalize_bot_id(bot_id or os.getenv("WECHAT_BOT_ID"))
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
        """功能：校验 MySQL 会话存储是否已启用。
        参数：
        - 无。
        返回值：
        - 无。未启用时抛出 ValueError。
        """
        if not self.enabled:
            raise ValueError("MYSQL_ENABLED=false，无法启用会话持久化")

    def _init_tables(self):
        """功能：创建会话相关数据表并在必要时执行结构迁移。
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
                            bot_id VARCHAR(128) NOT NULL DEFAULT '',
                            user_id VARCHAR(128) NOT NULL,
                            session_id VARCHAR(128) NOT NULL,
                            title VARCHAR(255) NOT NULL DEFAULT '',
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                            last_active TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE KEY uk_bot_user_session (bot_id, user_id, session_id),
                            KEY idx_last_active (last_active)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_messages (
                            id BIGINT PRIMARY KEY AUTO_INCREMENT,
                            bot_id VARCHAR(128) NOT NULL DEFAULT '',
                            user_id VARCHAR(128) NOT NULL,
                            session_id VARCHAR(128) NOT NULL,
                            role VARCHAR(32) NOT NULL,
                            content TEXT NOT NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            KEY idx_bot_user_session_id (bot_id, user_id, session_id, id),
                            KEY idx_created_at (created_at)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        """
                    )
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS chat_message_dedup (
                            bot_id VARCHAR(128) NOT NULL DEFAULT '',
                            msgid VARCHAR(128) NOT NULL,
                            status VARCHAR(32) NOT NULL,
                            answer TEXT NULL,
                            stream_id VARCHAR(64) NULL,
                            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                            PRIMARY KEY (bot_id, msgid),
                            KEY idx_updated_at (updated_at)
                        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
                        """
                    )
                self._migrate_tables(conn)

    def _column_exists(self, cursor, table_name: str, column_name: str) -> bool:
        """功能：检查指定数据表是否包含某列。
        参数：
        - cursor：MySQL 游标。
        - table_name：表名。
        - column_name：列名。
        返回值：
        - bool：列存在返回 True，否则 False。
        """
        cursor.execute(
            """
            SELECT 1
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
              AND COLUMN_NAME = %s
            LIMIT 1
            """,
            (table_name, column_name),
        )
        return cursor.fetchone() is not None

    def _index_columns(self, cursor, table_name: str, index_name: str) -> List[str]:
        """功能：读取指定索引包含的列名列表。
        参数：
        - cursor：MySQL 游标。
        - table_name：表名。
        - index_name：索引名。
        返回值：
        - List[str]：索引列名列表，按 SHOW INDEX 返回顺序排列。
        """
        cursor.execute(f"SHOW INDEX FROM `{table_name}` WHERE Key_name = %s", (index_name,))
        rows = cursor.fetchall() or []
        return [str((row or {}).get("Column_name") or "") for row in rows]

    def _primary_key_columns(self, cursor, table_name: str) -> List[str]:
        """功能：读取数据表主键列名列表。
        参数：
        - cursor：MySQL 游标。
        - table_name：表名。
        返回值：
        - List[str]：主键列名列表，按序位排列。
        """
        cursor.execute(
            """
            SELECT COLUMN_NAME
            FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME = %s
              AND CONSTRAINT_NAME = 'PRIMARY'
            ORDER BY ORDINAL_POSITION
            """,
            (table_name,),
        )
        rows = cursor.fetchall() or []
        return [str((row or {}).get("COLUMN_NAME") or "") for row in rows]

    def _add_column_if_missing(self, cursor, table_name: str, column_definition: str) -> None:
        """功能：在列不存在时向数据表追加新列。
        参数：
        - cursor：MySQL 游标。
        - table_name：表名。
        - column_definition：ALTER TABLE ADD COLUMN 片段（含列名与类型）。
        返回值：
        - 无。
        """
        column_name = column_definition.split()[0].strip("`")
        if not self._column_exists(cursor, table_name, column_name):
            cursor.execute(f"ALTER TABLE `{table_name}` ADD COLUMN {column_definition}")

    def _migrate_tables(self, conn) -> None:
        """功能：依次迁移会话、消息与去重表结构。
        参数：
        - conn：MySQL 连接对象。
        返回值：
        - 无。
        """
        with conn.cursor() as cursor:
            self._migrate_chat_sessions(cursor)
            self._migrate_chat_messages(cursor)
            self._migrate_chat_message_dedup(cursor)

    def _migrate_chat_sessions(self, cursor) -> None:
        """功能：迁移 chat_sessions 表，补齐 bot_id 并调整唯一索引。
        参数：
        - cursor：MySQL 游标。
        返回值：
        - 无。
        """
        self._add_column_if_missing(cursor, "chat_sessions", "`bot_id` VARCHAR(128) NOT NULL DEFAULT ''")
        self._add_column_if_missing(cursor, "chat_sessions", "`title` VARCHAR(255) NOT NULL DEFAULT ''")
        if self._index_columns(cursor, "chat_sessions", "uk_user_session"):
            cursor.execute("ALTER TABLE `chat_sessions` DROP INDEX `uk_user_session`")
        if not self._index_columns(cursor, "chat_sessions", "uk_bot_user_session"):
            cursor.execute(
                "ALTER TABLE `chat_sessions` ADD UNIQUE KEY `uk_bot_user_session` (`bot_id`, `user_id`, `session_id`)"
            )

    @staticmethod
    def _make_session_title(text: str, *, limit: int = 48) -> str:
        """功能：从用户首条消息生成会话标题。
        参数：
        - text：用户输入文本。
        - limit：标题最大字符数。
        返回值：
        - str：规范化并截断后的标题；空输入返回空字符串。
        """
        value = " ".join(str(text or "").strip().split())
        if not value:
            return ""
        return value[:limit]

    @staticmethod
    def _summary_from_row(row: dict) -> ChatSessionSummary:
        """功能：将数据库查询行转换为 ChatSessionSummary 对象。
        参数：
        - row：包含会话字段的字典行。
        返回值：
        - ChatSessionSummary：会话摘要对象。
        """
        return ChatSessionSummary(
            session_id=str((row or {}).get("session_id") or ""),
            title=str((row or {}).get("title") or "") or "新会话",
            created_at=(row or {}).get("created_at"),
            updated_at=(row or {}).get("updated_at"),
            last_active=(row or {}).get("last_active"),
            message_count=int((row or {}).get("message_count") or 0),
            last_message=str((row or {}).get("last_message") or ""),
        )

    def _migrate_chat_messages(self, cursor) -> None:
        """功能：迁移 chat_messages 表，补齐 bot_id 并调整索引。
        参数：
        - cursor：MySQL 游标。
        返回值：
        - 无。
        """
        self._add_column_if_missing(cursor, "chat_messages", "`bot_id` VARCHAR(128) NOT NULL DEFAULT ''")
        if not self._index_columns(cursor, "chat_messages", "idx_bot_user_session_id"):
            cursor.execute(
                "ALTER TABLE `chat_messages` ADD KEY `idx_bot_user_session_id` (`bot_id`, `user_id`, `session_id`, `id`)"
            )
        if self._index_columns(cursor, "chat_messages", "idx_user_session_id"):
            cursor.execute("ALTER TABLE `chat_messages` DROP INDEX `idx_user_session_id`")

    def _migrate_chat_message_dedup(self, cursor) -> None:
        """功能：迁移 chat_message_dedup 表，确保主键为 (bot_id, msgid)。
        参数：
        - cursor：MySQL 游标。
        返回值：
        - 无。
        """
        self._add_column_if_missing(cursor, "chat_message_dedup", "`bot_id` VARCHAR(128) NOT NULL DEFAULT ''")
        primary_columns = self._primary_key_columns(cursor, "chat_message_dedup")
        if primary_columns != ["bot_id", "msgid"]:
            if primary_columns:
                cursor.execute("ALTER TABLE `chat_message_dedup` DROP PRIMARY KEY")
            cursor.execute(
                "ALTER TABLE `chat_message_dedup` ADD PRIMARY KEY (`bot_id`, `msgid`)"
            )

    def _run_cleanup_once(self):
        """功能：执行一次会话、消息与去重记录的过期清理。
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
                          ON s.bot_id = m.bot_id
                         AND s.user_id = m.user_id
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
        """功能：后台循环定期执行会话存储清理任务。
        参数：
        - 无。
        返回值：
        - 无。异常时记录日志并继续下一轮。
        """
        interval = max(30, self.cleanup_interval_seconds)
        while True:
            try:
                self._run_cleanup_once()
            except Exception:
                logger.exception("会话存储清理失败")
            time.sleep(interval)

    def _start_cleanup_worker(self):
        """功能：启动全局唯一的后台清理守护线程。
        参数：
        - 无。
        返回值：
        - 无。已启动时直接返回。
        """
        global _CLEANUP_WORKER_STARTED
        with _CLEANUP_WORKER_LOCK:
            if _CLEANUP_WORKER_STARTED:
                return
            _CLEANUP_WORKER_STARTED = True
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
                    INSERT INTO chat_sessions (bot_id, user_id, session_id, title, last_active)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON DUPLICATE KEY UPDATE last_active = NOW()
                    """,
                    (self.bot_id, user_id, session_id, ""),
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
        title = self._make_session_title(user_text)
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO chat_messages (bot_id, user_id, session_id, role, content)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (self.bot_id, user_id, session_id, "user", user_text or ""),
                )
                cursor.execute(
                    """
                    INSERT INTO chat_messages (bot_id, user_id, session_id, role, content)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (self.bot_id, user_id, session_id, "assistant", answer or ""),
                )
                cursor.execute(
                    """
                    INSERT INTO chat_sessions (bot_id, user_id, session_id, title, last_active)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON DUPLICATE KEY UPDATE
                        last_active = NOW(),
                        title = IF(title = '' AND VALUES(title) <> '', VALUES(title), title)
                    """,
                    (self.bot_id, user_id, session_id, title),
                )

    def list_sessions(self, user_id: str, *, query: str = "", limit: int = 50) -> List[ChatSessionSummary]:
        """功能：按活跃时间倒序列出用户会话摘要，支持标题与消息内容搜索。
        参数：
        - user_id：用户标识。
        - query：可选搜索关键词。
        - limit：返回条数上限（1–200）。
        返回值：
        - List[ChatSessionSummary]：会话摘要列表。
        """
        limit = max(1, min(int(limit or 50), 200))
        search = " ".join(str(query or "").strip().split())
        params: List[Any] = [self.bot_id, user_id]
        where = "s.bot_id = %s AND s.user_id = %s"
        if search:
            like = f"%{search}%"
            where += (
                " AND (s.title LIKE %s OR EXISTS ("
                "SELECT 1 FROM chat_messages mx "
                "WHERE mx.bot_id = s.bot_id AND mx.user_id = s.user_id "
                "AND mx.session_id = s.session_id AND mx.content LIKE %s"
                "))"
            )
            params.extend([like, like])
        params.append(limit)
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    SELECT
                        s.session_id,
                        s.title,
                        s.created_at,
                        s.updated_at,
                        s.last_active,
                        (
                            SELECT COUNT(*)
                            FROM chat_messages mc
                            WHERE mc.bot_id = s.bot_id
                              AND mc.user_id = s.user_id
                              AND mc.session_id = s.session_id
                        ) AS message_count,
                        (
                            SELECT ml.content
                            FROM chat_messages ml
                            WHERE ml.bot_id = s.bot_id
                              AND ml.user_id = s.user_id
                              AND ml.session_id = s.session_id
                            ORDER BY ml.id DESC
                            LIMIT 1
                        ) AS last_message
                    FROM chat_sessions s
                    WHERE {where}
                    ORDER BY s.last_active DESC, s.id DESC
                    LIMIT %s
                    """,
                    tuple(params),
                )
                rows = cursor.fetchall() or []
        return [self._summary_from_row(row) for row in rows]

    def load_messages(self, user_id: str, session_id: str, *, limit: int = 200) -> List[dict]:
        """功能：按时间顺序读取会话消息列表。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        - limit：返回条数上限（1–1000）。
        返回值：
        - List[dict]：仅含 user/assistant 角色的消息字典列表。
        """
        limit = max(1, min(int(limit or 200), 1000))
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT role, content, created_at
                    FROM chat_messages
                    WHERE bot_id = %s AND user_id = %s AND session_id = %s
                    ORDER BY id ASC
                    LIMIT %s
                    """,
                    (self.bot_id, user_id, session_id, limit),
                )
                rows = cursor.fetchall() or []
        return [
            {
                "role": str((row or {}).get("role") or ""),
                "content": str((row or {}).get("content") or ""),
                "created_at": (row or {}).get("created_at"),
            }
            for row in rows
            if str((row or {}).get("role") or "") in {"user", "assistant"}
        ]

    def clear_session_messages(self, user_id: str, session_id: str) -> None:
        """功能：清空指定会话的全部消息并重置会话标题。
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
                    DELETE FROM chat_messages
                    WHERE bot_id = %s AND user_id = %s AND session_id = %s
                    """,
                    (self.bot_id, user_id, session_id),
                )
                cursor.execute(
                    """
                    UPDATE chat_sessions
                    SET title = '', last_active = NOW()
                    WHERE bot_id = %s AND user_id = %s AND session_id = %s
                    """,
                    (self.bot_id, user_id, session_id),
                )

    def delete_session(self, user_id: str, session_id: str) -> None:
        """功能：删除指定会话及其全部消息记录。
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
                    DELETE FROM chat_messages
                    WHERE bot_id = %s AND user_id = %s AND session_id = %s
                    """,
                    (self.bot_id, user_id, session_id),
                )
                cursor.execute(
                    """
                    DELETE FROM chat_sessions
                    WHERE bot_id = %s AND user_id = %s AND session_id = %s
                    """,
                    (self.bot_id, user_id, session_id),
                )

    def delete_user_sessions(self, user_id: str) -> None:
        """功能：删除指定用户在当前 bot 下的全部会话与消息。
        参数：
        - user_id：用户标识。
        返回值：
        - 无。
        """
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM chat_messages
                    WHERE bot_id = %s AND user_id = %s
                    """,
                    (self.bot_id, user_id),
                )
                cursor.execute(
                    """
                    DELETE FROM chat_sessions
                    WHERE bot_id = %s AND user_id = %s
                    """,
                    (self.bot_id, user_id),
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
                rows = self._load_recent_turn_rows(
                    cursor=cursor,
                    bot_id=self.bot_id,
                    user_id=user_id,
                    session_id=session_id,
                    limit_rows=limit_rows,
                )

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

    def _load_recent_turn_rows(self, cursor, *, bot_id: str, user_id: str, session_id: str, limit_rows: int):
        """功能：从数据库读取会话最近的消息行（按 id 倒序）。
        参数：
        - cursor：MySQL 游标。
        - bot_id：机器人标识。
        - user_id：用户标识。
        - session_id：会话标识。
        - limit_rows：最大行数。
        返回值：
        - list：消息行列表，每项含 role 与 content 字段。
        """
        cursor.execute(
            """
            SELECT role, content
            FROM chat_messages
            WHERE bot_id = %s AND user_id = %s AND session_id = %s
            ORDER BY id DESC
            LIMIT %s
            """,
            (bot_id, user_id, session_id, limit_rows),
        )
        return cursor.fetchall() or []

    def get_dedup(self, msgid: str) -> Optional[dict]:
        """功能：按消息 ID 查询去重状态记录（限定当前 bot_id）。
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
                    WHERE bot_id = %s AND msgid = %s
                    """,
                    (self.bot_id, msgid),
                )
                return cursor.fetchone()

    def mark_processing(self, msgid: str, stream_id: str) -> bool:
        """功能：将消息标记为处理中，避免重复消费。
        参数：
        - msgid：消息唯一 ID。
        - stream_id：本次处理对应的流式响应 ID。
        返回值：
        - bool：首次写入成功返回 True，已存在返回 False；空 msgid 时返回 True。
        """
        if not msgid:
            return True
        with get_mysql_pool().connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT IGNORE INTO chat_message_dedup (bot_id, msgid, status, answer, stream_id)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (self.bot_id, msgid, "processing", None, stream_id),
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
                    WHERE bot_id = %s AND msgid = %s
                    """,
                    ("done", answer or "", self.bot_id, msgid),
                )


class DedupService:
    """功能：统一管理消息去重状态，协调 Redis 缓存与数据库双写（按 bot_id 隔离）。
    参数：
    - 无。
    返回值：
    - 无。优先走缓存提升并发性能，缓存不可用时自动降级到数据库。
    """
    def __init__(
        self,
        chat_store: ChatStore,
        cache: Optional[RedisCache] = None,
        bot_id: Optional[str] = None,
    ):
        """功能：注入去重存储依赖并设置去重状态缓存 TTL。
        参数：
        - chat_store：数据库持久化存储服务。
        - cache：可选 Redis 缓存实例。
        - bot_id：可选机器人标识；未传时继承 chat_store 或环境变量。
        返回值：
        - 无。未传入 `cache` 时会创建默认 `RedisCache`，但可在不可用时自动降级。
        """
        self.chat_store = chat_store
        self.cache = cache or RedisCache()
        self.bot_id = _normalize_bot_id(bot_id or getattr(chat_store, "bot_id", "") or os.getenv("WECHAT_BOT_ID"))
        self.ttl_seconds = int(os.getenv("CHAT_MESSAGE_DEDUP_TTL_SECONDS"))

    def _cache_key(self, msgid: str) -> str:
        """功能：生成消息去重记录在 Redis 中的缓存键。
        参数：
        - msgid：消息唯一 ID。
        返回值：
        - str：带 bot_id 前缀的缓存键。
        """
        return self.cache.make_key("dedup", f"{self.bot_id}:{msgid}")

    def get(self, msgid: str) -> Optional[dict]:
        """功能：读取消息去重状态，优先命中缓存。
        参数：
        - msgid：消息唯一 ID。
        返回值：
        - Optional[dict]：去重状态记录；不存在时返回 None。
        """
        if msgid and self.cache.available:
            cached = self.cache.get_json(self._cache_key(msgid))
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
                self._cache_key(msgid),
                {
                    "status": "processing",
                    "answer": None,
                    "stream_id": stream_id,
                    "bot_id": self.bot_id,
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
                self._cache_key(msgid),
                {
                    "status": "done",
                    "answer": answer or "",
                    "stream_id": current.get("stream_id"),
                    "bot_id": self.bot_id,
                    "updated_at": time.time(),
                },
                ttl_seconds=self.ttl_seconds,
            )


class SessionManager:
    """功能：管理会话级 Agent 实例生命周期，并提供会话分布式锁能力（按 bot_id 隔离）。
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
        bot_id: Optional[str] = None,
    ):
        """功能：配置会话工厂、持久化依赖和本地/分布式会话治理参数。
        参数：
        - agent_factory：创建智能体实例的工厂函数。
        - chat_store：聊天历史存储服务。
        - ttl_seconds：本地会话状态保活秒数。
        - cache：可选 Redis 缓存实例。
        - bot_id：可选机器人标识；未传时继承 chat_store 或环境变量。
        返回值：
        - 无。`ttl_seconds` 控制内存回收，锁 TTL 由环境变量 `SESSION_LOCK_TTL_SECONDS` 决定。
        """
        self.agent_factory = agent_factory
        self.chat_store = chat_store
        self.ttl_seconds = ttl_seconds
        self.cache = cache or RedisCache()
        self.bot_id = _normalize_bot_id(bot_id or getattr(chat_store, "bot_id", "") or os.getenv("WECHAT_BOT_ID"))
        self.lock_ttl_seconds = int(os.getenv("SESSION_LOCK_TTL_SECONDS"))
        self._sessions: Dict[Tuple[str, str, str], SessionState] = {}
        self._lock = threading.Lock()
        self._key_locks: Dict[Tuple[str, str, str], threading.Lock] = {}

    def _session_key_lock(self, key: Tuple[str, str, str]) -> threading.Lock:
        """功能：获取或创建指定会话键的进程内互斥锁。
        参数：
        - key：`(bot_id, user_id, session_id)` 三元组。
        返回值：
        - threading.Lock：该会话键对应的锁对象。
        """
        with self._lock:
            lock = self._key_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._key_locks[key] = lock
            return lock

    def _cleanup_expired(self) -> None:
        """功能：从本地内存中移除超过 TTL 的会话状态。
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

    def get(self, user_id: str, session_id: str, *, touch: bool = True) -> SessionState:
        """功能：获取会话状态，必要时创建新会话并加载历史上下文。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        返回值：
        - SessionState：可用于当前请求处理的会话状态对象。
        """
        state, _created = self.get_with_meta(user_id, session_id, touch=touch)
        return state

    def get_with_meta(self, user_id: str, session_id: str, *, touch: bool = True) -> Tuple[SessionState, bool]:
        """功能：获取会话状态，并返回本次是否新建了 Agent 实例。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        - touch：是否刷新会话活跃时间。
        返回值：
        - Tuple[SessionState, bool]：会话状态与是否新建 Agent。
        """
        key = (self.bot_id, user_id, session_id)
        with self._session_key_lock(key):
            with self._lock:
                self._cleanup_expired()
                state = self._sessions.get(key)
            if state is None:
                agent = self.agent_factory()
                turns = self.chat_store.load_recent_turns(user_id=user_id, session_id=session_id)
                if turns:
                    agent.conversation_turns = turns
                candidate = SessionState(
                    agent=agent,
                    bot_id=self.bot_id,
                    user_id=user_id,
                    session_id=session_id,
                )
                with self._lock:
                    state = self._sessions.get(key)
                    if state is None:
                        self._sessions[key] = candidate
                        state = candidate
                        created = True
                    else:
                        created = False
            else:
                created = False
            if touch:
                self.chat_store.touch_session(user_id=user_id, session_id=session_id)
            if touch and self.cache.available:
                self.cache.set_json(
                    self.cache.make_key("session", f"{self.bot_id}:{user_id}:{session_id}"),
                    {
                        "bot_id": self.bot_id,
                        "user_id": user_id,
                        "session_id": session_id,
                        "last_active": time.time(),
                    },
                    ttl_seconds=self.ttl_seconds,
                )
            state.last_active = time.time()
            return state, created

    def acquire_lease(self, user_id: str, session_id: str, owner: str) -> bool:
        """功能：尝试获取会话级分布式处理锁。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        - owner：锁持有者标识。
        返回值：
        - bool：成功获取锁返回 True；缓存不可用时直接返回 True。
        """
        if not self.cache.available:
            return True
        return self.cache.acquire_lock(
            self.cache.make_key("session-lock", f"{self.bot_id}:{user_id}:{session_id}"),
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
        - bool：释放成功返回 True；缓存不可用时直接返回 True。
        """
        if not self.cache.available:
            return True
        return self.cache.release_lock(
            self.cache.make_key("session-lock", f"{self.bot_id}:{user_id}:{session_id}"),
            owner=owner,
        )

    def discard(self, user_id: str, session_id: str) -> None:
        """功能：从本地内存中丢弃指定会话的 Agent 状态。
        参数：
        - user_id：用户标识。
        - session_id：会话标识。
        返回值：
        - 无。不影响数据库中的持久化记录。
        """
        key = (self.bot_id, user_id, session_id)
        with self._lock:
            self._sessions.pop(key, None)

    def discard_user(self, user_id: str) -> None:
        """功能：从本地内存中丢弃指定用户的全部会话 Agent 状态。
        参数：
        - user_id：用户标识。
        返回值：
        - 无。
        """
        with self._lock:
            stale_keys = [key for key in self._sessions if key[0] == self.bot_id and key[1] == user_id]
            for key in stale_keys:
                self._sessions.pop(key, None)

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
    bot_id: str = ""
    content: str = ""
    finish: bool = False
    template_card: Optional[dict] = None
    session_id: str = ""
    status: str = "processing"
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)


class StreamManager:
    """功能：维护流式响应状态的创建、更新、读取和过期清理（按 bot_id 隔离）。
    参数：
    - 无。
    返回值：
    - 无。支持本地内存与 Redis 双层存储，保证跨请求可恢复。
    """
    def __init__(
        self,
        ttl_seconds: int = 600,
        cache: Optional[RedisCache] = None,
        bot_id: Optional[str] = None,
    ):
        """功能：设置流式状态保活策略并初始化缓存与本地状态容器。
        参数：
        - ttl_seconds：流式状态保活秒数。
        - cache：可选 Redis 缓存实例。
        - bot_id：可选机器人标识；未传时从环境变量 `WECHAT_BOT_ID` 读取。
        返回值：
        - 无。超过 TTL 的状态会在读写路径上被惰性清理。
        """
        self.ttl_seconds = ttl_seconds
        self.cache = cache or RedisCache()
        self.bot_id = _normalize_bot_id(bot_id or os.getenv("WECHAT_BOT_ID"))
        self._data: Dict[str, StreamState] = {}
        self._lock = threading.Lock()
        self._persist_interval = float(os.getenv("STREAM_PERSIST_INTERVAL_SECONDS", "1.0"))
        self._last_persist_at: Dict[str, float] = {}

    def _scoped_stream_id(self, stream_id: str) -> str:
        """功能：为流式 ID 加上 bot_id 前缀，实现多租户隔离。
        参数：
        - stream_id：原始流式响应 ID。
        返回值：
        - str：带 bot_id 前缀的作用域流式 ID。
        """
        return f"{self.bot_id}:{stream_id}"

    def _cleanup(self):
        """功能：从本地内存中移除超过 TTL 的流式状态。
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
            scoped_stream_id = self._scoped_stream_id(stream_id)
            state = StreamState(
                bot_id=self.bot_id,
                content=content,
                finish=finish,
                template_card=template_card,
                session_id=session_id,
                status=status,
                created_at=time.time(),
                last_active=time.time(),
            )
            self._data[scoped_stream_id] = state
        self._maybe_persist(scoped_stream_id, state, force=True)

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
            scoped_stream_id = self._scoped_stream_id(stream_id)
            state = self._data.get(scoped_stream_id)
            if state is None:
                state = StreamState(bot_id=self.bot_id)
                self._data[scoped_stream_id] = state
            state.content = content
            state.finish = finish
            if template_card is not None:
                state.template_card = template_card
            if status:
                state.status = status
            state.last_active = time.time()
        self._maybe_persist(scoped_stream_id, state, force=finish)

    def get(self, stream_id: str) -> Optional[StreamState]:
        """功能：读取流式状态，必要时从缓存回填本地内存。
        参数：
        - stream_id：流式响应 ID。
        返回值：
        - Optional[StreamState]：命中时返回流式状态对象，未命中返回 None。
        """
        if self.cache.available:
            scoped_stream_id = self._scoped_stream_id(stream_id)
            cached = self.cache.get_json(self.cache.make_key("stream", scoped_stream_id))
            if cached:
                state = StreamState(
                    bot_id=str(cached.get("bot_id") or self.bot_id),
                    content=str(cached.get("content") or ""),
                    finish=bool(cached.get("finish")),
                    template_card=cached.get("template_card"),
                    session_id=str(cached.get("session_id") or ""),
                    status=str(cached.get("status") or "processing"),
                    created_at=float(cached.get("created_at") or time.time()),
                    last_active=float(cached.get("last_active") or time.time()),
                )
                with self._lock:
                    self._data[scoped_stream_id] = state
        with self._lock:
            self._cleanup()
            scoped_stream_id = self._scoped_stream_id(stream_id)
            state = self._data.get(scoped_stream_id)
            if state:
                state.last_active = time.time()
        if state:
            self._maybe_persist(scoped_stream_id, state)
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

    def _maybe_persist(self, stream_id: str, state: StreamState, *, force: bool = False) -> None:
        """功能：按节流策略将流式状态持久化到 Redis。
        参数：
        - stream_id：作用域流式 ID。
        - state：流式状态对象。
        - force：为 True 时跳过节流立即持久化。
        返回值：
        - 无。
        """
        if not self.cache.available:
            return
        now = time.time()
        if not force and not state.finish:
            last = self._last_persist_at.get(stream_id, 0.0)
            if now - last < self._persist_interval:
                return
        self._last_persist_at[stream_id] = now
        self._persist(stream_id, state)
        if state.finish:
            self._last_persist_at.pop(stream_id, None)

    def _persist(self, stream_id: str, state: StreamState) -> None:
        """功能：将流式状态写入 Redis 缓存。
        参数：
        - stream_id：作用域流式 ID。
        - state：流式状态对象。
        返回值：
        - 无。缓存不可用时直接返回。
        """
        if not self.cache.available:
            return
        self.cache.set_json(
            self.cache.make_key("stream", stream_id),
            {
                "bot_id": state.bot_id or self.bot_id,
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
