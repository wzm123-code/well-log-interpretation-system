"""
对话历史持久化 - 按 conversation_id 保存问答到 SQLite

【用途】多轮对话上下文，供 LLM 生成回答时参考。进程重启后仍可恢复。
"""
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DB_PATH = DATA_DIR / "conversations.db"
MAX_PAIRS = 50  # 每个会话保留最近 50 对

# 作用：在调用其他功能前确保数据库文件、表结构和索引都已存在。
# 执行流程：
# 创建数据目录（如果不存在）。
# 通过 _get_conn() 获取数据库连接。
# 执行 CREATE TABLE IF NOT EXISTS 创建表，字段包括自增主键 id、会话 ID、用户消息、助手消息、创建时间（Unix 时间戳，默认值为当前秒数）。
# 在 conversation_id 列上创建索引，加速按会话查询。
# 提交事务，确保表结构持久化。
def _ensure_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                user_msg TEXT NOT NULL,
                assistant_msg TEXT NOT NULL,
                created_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_id ON conversation_history(conversation_id)")
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT '',
                task_id TEXT,
                file_path TEXT,
                file_name TEXT,
                created_at REAL DEFAULT (strftime('%s', 'now')),
                updated_at REAL DEFAULT (strftime('%s', 'now'))
            )
        """)
        conn.commit()

#获取数据库连接，自动管理连接的打开和关闭，并返回连接对象
@contextmanager
def _get_conn() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

#获取指定会话的最近 N 对问答，按时间正序排列
def get_history(conversation_id: str, max_pairs: int = MAX_PAIRS) -> List[dict]:
    """获取指定会话的最近 N 对问答"""
    if not conversation_id:
        return []
    _ensure_db()
    with _get_conn() as conn:
        cur = conn.execute(
            """SELECT user_msg, assistant_msg FROM conversation_history
               WHERE conversation_id = ? ORDER BY created_at DESC LIMIT ?""",
            (conversation_id, max_pairs),
        )
        rows = list(cur.fetchall())
    rows.reverse()
    return [
        msg
        for r in rows
        for msg in (
            {"role": "user", "content": r["user_msg"] or ""},
            {"role": "assistant", "content": r["assistant_msg"] or ""},
        )
    ]

#
def append_pair(conversation_id: str, user_msg: str, assistant_msg: str) -> None:
    """追加一轮问答，持久化到 SQLite"""
    if not conversation_id or (not user_msg and not assistant_msg):
        return
    _ensure_db()
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO conversation_history (conversation_id, user_msg, assistant_msg) VALUES (?, ?, ?)",
            (conversation_id, user_msg or "", assistant_msg or ""),
        )
        cur = conn.execute(
            "SELECT COUNT(*) FROM conversation_history WHERE conversation_id = ?",
            (conversation_id,),
        )
        count = cur.fetchone()[0]
        if count > MAX_PAIRS:
            n_delete = count - MAX_PAIRS
            conn.execute(
                """DELETE FROM conversation_history WHERE id IN (
                    SELECT id FROM conversation_history WHERE conversation_id = ?
                    ORDER BY created_at ASC LIMIT ?)""",
                (conversation_id, n_delete),
            )
        conn.commit()


def list_saved_sessions() -> List[Dict[str, Any]]:
    """列出已保存的会话元数据，按更新时间倒序。"""
    _ensure_db()
    with _get_conn() as conn:
        cur = conn.execute(
            """SELECT conversation_id, title, task_id, file_path, file_name, created_at, updated_at
               FROM conversation_sessions ORDER BY updated_at DESC"""
        )
        rows = cur.fetchall()
    return [dict(r) for r in rows]


def save_session_meta(
    conversation_id: str,
    title: str,
    task_id: Optional[str] = None,
    file_path: Optional[str] = None,
    file_name: Optional[str] = None,
) -> None:
    """插入或更新会话书签（标题与可选的上传任务信息）。"""
    if not conversation_id:
        return
    _ensure_db()
    t = (title or "").strip() or "未命名会话"
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO conversation_sessions
                (conversation_id, title, task_id, file_path, file_name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?,
                    COALESCE((SELECT created_at FROM conversation_sessions WHERE conversation_id = ?), strftime('%s', 'now')),
                    strftime('%s', 'now'))
            ON CONFLICT(conversation_id) DO UPDATE SET
                title = excluded.title,
                task_id = excluded.task_id,
                file_path = excluded.file_path,
                file_name = excluded.file_name,
                updated_at = strftime('%s', 'now')
            """,
            (conversation_id, t, task_id or "", file_path or "", file_name or "", conversation_id),
        )
        conn.commit()


def update_session_title(conversation_id: str, title: str) -> bool:
    """仅更新已存在会话书签的标题。"""
    if not conversation_id:
        return False
    t = (title or "").strip()
    if not t:
        return False
    _ensure_db()
    with _get_conn() as conn:
        cur = conn.execute(
            "UPDATE conversation_sessions SET title = ?, updated_at = strftime('%s', 'now') WHERE conversation_id = ?",
            (t, conversation_id),
        )
        conn.commit()
        return cur.rowcount > 0


def get_session_meta(conversation_id: str) -> Optional[Dict[str, Any]]:
    """读取单个已保存会话的元数据。"""
    if not conversation_id:
        return None
    _ensure_db()
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT conversation_id, title, task_id, file_path, file_name, created_at, updated_at "
            "FROM conversation_sessions WHERE conversation_id = ?",
            (conversation_id,),
        )
        row = cur.fetchone()
    return dict(row) if row else None


def delete_session(conversation_id: str) -> None:
    """删除会话书签及该会话在 conversation_history 中的全部问答。"""
    if not conversation_id:
        return
    _ensure_db()
    with _get_conn() as conn:
        conn.execute("DELETE FROM conversation_history WHERE conversation_id = ?", (conversation_id,))
        conn.execute("DELETE FROM conversation_sessions WHERE conversation_id = ?", (conversation_id,))
        conn.commit()


def get_messages_json(conversation_id: str) -> List[Dict[str, str]]:
    """供前端恢复的完整消息列表（user/assistant 交替，不含欢迎语）。"""
    return get_history(conversation_id, max_pairs=MAX_PAIRS)
