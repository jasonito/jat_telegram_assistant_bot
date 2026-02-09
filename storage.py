import sqlite3
from datetime import datetime
from pathlib import Path

from config import (
    DB_PATH, DATA_DIR, NOTES_DIR, TELEGRAM_MD_DIR, SLACK_MD_DIR, NEWS_MD_DIR,
)


def init_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    TELEGRAM_MD_DIR.mkdir(parents=True, exist_ok=True)
    SLACK_MD_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_MD_DIR.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                chat_id TEXT NOT NULL,
                chat_title TEXT,
                user_id TEXT,
                user_name TEXT,
                text TEXT,
                message_ts TEXT,
                received_ts TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_date TEXT NOT NULL,
                cluster_seq INTEGER NOT NULL,
                canonical_title TEXT,
                canonical_url TEXT,
                canonical_source TEXT,
                canonical_published_at TEXT,
                canonical_summary TEXT,
                canonical_summary_source TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cluster_id INTEGER NOT NULL,
                source TEXT,
                title TEXT,
                title_norm TEXT,
                url TEXT,
                summary TEXT,
                published_at TEXT,
                hash_url TEXT,
                hash_title TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (cluster_id) REFERENCES news_clusters(id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS news_subscriptions (
                chat_id TEXT PRIMARY KEY,
                enabled INTEGER NOT NULL,
                interval_minutes INTEGER,
                last_sent_at TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                user_id TEXT,
                user_name TEXT,
                text TEXT,
                message_ts TEXT,
                received_ts TEXT NOT NULL,
                meeting_id TEXT,
                bucket TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_news_items_hash_url ON news_items(hash_url)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_items_published_at ON news_items(published_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_clusters_date_seq ON news_clusters(cluster_date, cluster_seq)"
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Message operations
# ---------------------------------------------------------------------------
def store_message(
    platform: str,
    chat_id: str,
    chat_title: str,
    user_id: str,
    user_name: str,
    text: str,
    message_ts: datetime,
) -> None:
    received_ts = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO messages
            (platform, chat_id, chat_title, user_id, user_name, text, message_ts, received_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                platform,
                chat_id,
                chat_title,
                user_id,
                user_name,
                text,
                message_ts.isoformat(timespec="seconds") if message_ts else None,
                received_ts,
            ),
        )
        conn.commit()


def search_messages(keyword: str, limit: int = 10, day: str | None = None) -> list[tuple]:
    kw = f"%{keyword}%"
    params = [kw]
    where = "text LIKE ?"
    if day:
        where += " AND received_ts LIKE ?"
        params.append(f"{day}%")
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT platform, chat_title, user_name, text, received_ts
            FROM messages
            WHERE {where}
            ORDER BY received_ts DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    return rows


def summarize_day(day: str) -> tuple[int, dict, list[tuple]]:
    with sqlite3.connect(DB_PATH) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE received_ts LIKE ?",
            (f"{day}%",),
        ).fetchone()[0]
        by_platform = dict(
            conn.execute(
                """
                SELECT platform, COUNT(*)
                FROM messages
                WHERE received_ts LIKE ?
                GROUP BY platform
                ORDER BY COUNT(*) DESC
                """,
                (f"{day}%",),
            ).fetchall()
        )
        recent = conn.execute(
            """
            SELECT platform, chat_title, user_name, text, received_ts
            FROM messages
            WHERE received_ts LIKE ?
            ORDER BY received_ts DESC
            LIMIT 5
            """,
            (f"{day}%",),
        ).fetchall()
    return total, by_platform, recent


# ---------------------------------------------------------------------------
# Markdown file operations
# ---------------------------------------------------------------------------
def append_markdown(
    platform: str,
    chat_title: str,
    user_name: str,
    text: str,
    message_ts: datetime,
) -> None:
    day = (message_ts or datetime.now()).strftime("%Y-%m-%d")
    if platform == "telegram":
        md_dir = TELEGRAM_MD_DIR
    elif platform == "slack":
        md_dir = SLACK_MD_DIR
    else:
        md_dir = NOTES_DIR / platform
        md_dir.mkdir(parents=True, exist_ok=True)
    md_path = md_dir / f"{day}_{platform}.md"
    if not md_path.exists():
        md_path.write_text(f"# {day} {platform}\n\n", encoding="utf-8")

    time_str = (message_ts or datetime.now()).strftime("%H:%M:%S")
    line = f"- [{time_str}] ({platform}) {chat_title} | {user_name}: {text}\n"
    with md_path.open("a", encoding="utf-8") as f:
        f.write(line)


def append_slack_note_markdown(
    channel_id: str,
    user_name: str,
    text: str,
    message_ts: datetime,
) -> None:
    notes_dir = NOTES_DIR / "slack"
    notes_dir.mkdir(parents=True, exist_ok=True)
    day = (message_ts or datetime.now()).strftime("%Y-%m-%d")
    md_path = notes_dir / f"{day}_slack.md"
    if not md_path.exists():
        md_path.write_text(f"# {day} slack\n\n", encoding="utf-8")
    time_str = (message_ts or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    line = f"- [{time_str}] {user_name}: {text}\n"
    with md_path.open("a", encoding="utf-8") as f:
        f.write(line)


# ---------------------------------------------------------------------------
# Note operations
# ---------------------------------------------------------------------------
def store_note(
    platform: str,
    channel_id: str,
    user_id: str,
    user_name: str,
    text: str,
    message_ts: datetime,
    meeting_id: str | None,
    bucket: str,
) -> None:
    received_ts = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO notes
            (platform, channel_id, user_id, user_name, text, message_ts, received_ts, meeting_id, bucket)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                platform,
                channel_id,
                user_id,
                user_name,
                text,
                message_ts.isoformat(timespec="seconds") if message_ts else None,
                received_ts,
                meeting_id,
                bucket,
            ),
        )
        conn.commit()


def search_notes(keyword: str, channel_id: str | None, limit: int = 8) -> list[tuple]:
    kw = f"%{keyword}%"
    params = [kw]
    where = "text LIKE ?"
    if channel_id:
        where += " AND channel_id = ?"
        params.append(channel_id)
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            f"""
            SELECT platform, channel_id, user_name, text, received_ts, bucket, meeting_id
            FROM notes
            WHERE {where}
            ORDER BY received_ts DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
    return rows


def process_slack_note(
    channel_id: str,
    user_id: str,
    user_name: str,
    text: str,
    message_ts: datetime,
) -> tuple[bool, str]:
    if channel_id.startswith("D"):
        bucket = "inbox"
    elif channel_id.startswith(("C", "G")):
        bucket = "channel"
    else:
        bucket = "unknown"
    meeting_id = None

    store_message("slack", channel_id, channel_id, user_id, user_name, text, message_ts)
    store_note("slack", channel_id, user_id, user_name, text, message_ts, meeting_id, bucket)
    append_slack_note_markdown(channel_id, user_name, text, message_ts)
    return True, "已記錄。"
