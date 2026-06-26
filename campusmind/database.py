import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from campusmind.auth import hash_password, verify_password
from campusmind.config import get_settings
from campusmind.schemas import SignupRequest


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session() -> Iterator[sqlite3.Connection]:
    settings = get_settings()
    settings.ensure_storage()
    conn = _connect(settings.database_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with db_session() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                age INTEGER,
                gender TEXT,
                education_level TEXT,
                interests TEXT,
                mental_health_sensitive INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS user_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                memory_value TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                message TEXT NOT NULL,
                domain TEXT DEFAULT 'Unknown',
                confidence REAL DEFAULT 0,
                latency_ms INTEGER DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
            );
            """
        )


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def create_user(payload: SignupRequest) -> dict[str, Any]:
    user_id = str(uuid.uuid4())
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO users (
                user_id, name, email, password_hash, age, gender,
                education_level, interests, mental_health_sensitive
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                payload.name,
                payload.email.lower(),
                hash_password(payload.password),
                payload.age,
                payload.gender,
                payload.education_level,
                payload.interests,
                int(payload.mental_health_sensitive),
            ),
        )
        user = get_user_by_id(user_id, conn)
    assert user is not None
    user.pop("password_hash", None)
    return user


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    with db_session() as conn:
        user = get_user_by_email(email.lower(), conn)
    if not user or not verify_password(password, user["password_hash"]):
        return None
    user.pop("password_hash", None)
    return user


def get_user_by_email(email: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    owns_conn = conn is None
    conn = conn or _connect(get_settings().database_path)
    try:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email.lower(),)).fetchone()
        return row_to_dict(row)
    finally:
        if owns_conn:
            conn.close()


def get_user_by_id(user_id: str, conn: sqlite3.Connection | None = None) -> dict[str, Any] | None:
    owns_conn = conn is None
    conn = conn or _connect(get_settings().database_path)
    try:
        row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        return row_to_dict(row)
    finally:
        if owns_conn:
            conn.close()


def get_user_memory(user_id: str) -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            "SELECT memory_type, memory_value, created_at FROM user_memory WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def get_recent_history(user_id: str, session_id: str, limit_messages: int = 10) -> list[dict[str, Any]]:
    with db_session() as conn:
        rows = conn.execute(
            """
            SELECT role, message, domain, confidence, created_at
            FROM chat_history
            WHERE user_id = ? AND session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, session_id, limit_messages),
        ).fetchall()
    return [dict(row) for row in reversed(rows)]


def append_chat(
    user_id: str,
    session_id: str,
    role: str,
    message: str,
    domain: str = "Unknown",
    confidence: float = 0.0,
    latency_ms: int = 0,
) -> None:
    with db_session() as conn:
        conn.execute(
            """
            INSERT INTO chat_history (user_id, session_id, role, message, domain, confidence, latency_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (user_id, session_id, role, message, domain, confidence, latency_ms),
        )
        rows = conn.execute(
            """
            SELECT id FROM chat_history
            WHERE user_id = ? AND session_id = ?
            ORDER BY id DESC
            LIMIT -1 OFFSET 10
            """,
            (user_id, session_id),
        ).fetchall()
        stale_ids = [row["id"] for row in rows]
        if stale_ids:
            conn.executemany("DELETE FROM chat_history WHERE id = ?", [(row_id,) for row_id in stale_ids])


def analytics_for_user(user_id: str) -> dict[str, Any]:
    with db_session() as conn:
        domain_rows = conn.execute(
            """
            SELECT domain, COUNT(*) AS count
            FROM chat_history
            WHERE user_id = ? AND role = 'assistant'
            GROUP BY domain
            """,
            (user_id,),
        ).fetchall()
        stats = conn.execute(
            """
            SELECT COUNT(*) AS queries, AVG(confidence) AS avg_confidence, AVG(latency_ms) AS avg_latency_ms
            FROM chat_history
            WHERE user_id = ? AND role = 'assistant'
            """,
            (user_id,),
        ).fetchone()
    return {
        "domain_distribution": [dict(row) for row in domain_rows],
        "queries": int(stats["queries"] or 0),
        "avg_confidence": float(stats["avg_confidence"] or 0),
        "avg_latency_ms": float(stats["avg_latency_ms"] or 0),
    }

