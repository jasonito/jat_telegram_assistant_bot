import os
import re
import webbrowser
from pathlib import Path
import subprocess
import shutil
import sqlite3
import hashlib
import json
import time
from urllib.parse import quote
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
import threading
from threading import Event
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
from dotenv import load_dotenv
import feedparser
from rapidfuzz import fuzz

from slack_bolt import App as SlackApp
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"
TELEGRAM_LONG_POLLING = os.getenv("TELEGRAM_LONG_POLLING", "0").lower() in {
    "1",
    "true",
    "yes",
}
TELEGRAM_LOCAL_WEBHOOK_URL = os.getenv(
    "TELEGRAM_LOCAL_WEBHOOK_URL", "http://127.0.0.1:8000/telegram"
)

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data")).resolve()
DB_PATH = DATA_DIR / "messages.sqlite"
MD_DIR = DATA_DIR / "markdown"
NEWS_MD_DIR = DATA_DIR / "news"
ALLOWED_GROUPS = {
    g.strip()
    for g in os.getenv("TELEGRAM_ALLOWED_GROUPS", "").split(",")
    if g.strip()
}

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")
SLACK_USER_ID = os.getenv("SLACK_USER_ID", "")

NEWS_RSS_URLS_ENV = os.getenv("NEWS_RSS_URLS", "")
NEWS_RSS_URLS_FILE = os.getenv("NEWS_RSS_URLS_FILE", "")
NEWS_FETCH_INTERVAL_MINUTES = int(os.getenv("NEWS_FETCH_INTERVAL_MINUTES", "180"))
NEWS_PUSH_MAX_ITEMS = int(os.getenv("NEWS_PUSH_MAX_ITEMS", "10"))
NEWS_GNEWS_QUERY = os.getenv("NEWS_GNEWS_QUERY", "site:reuters.com semiconductors technology")
NEWS_GNEWS_HL = os.getenv("NEWS_GNEWS_HL", "en-US")
NEWS_GNEWS_GL = os.getenv("NEWS_GNEWS_GL", "US")
NEWS_GNEWS_CEID = os.getenv("NEWS_GNEWS_CEID", "US:en")
try:
    NEWS_TZ = ZoneInfo("Asia/Taipei")
except ZoneInfoNotFoundError:
    # Fallback for Windows without tzdata installed.
    NEWS_TZ = timezone(timedelta(hours=8))
    print("[WARN] tzdata not found; falling back to fixed UTC+08:00")

app = FastAPI()


def init_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)
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
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_news_items_hash_url ON news_items(hash_url)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_items_published_at ON news_items(published_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_news_clusters_date_seq ON news_clusters(cluster_date, cluster_seq)"
        )
        conn.commit()


init_storage()


def open_notepad() -> None:
    subprocess.Popen(["notepad.exe"], close_fds=True)


def sort_downloads() -> str:
    downloads = Path.home() / "Downloads"
    if not downloads.exists():
        return "Downloads folder not found."

    sorted_root = downloads / "_sorted"
    sorted_root.mkdir(exist_ok=True)

    moved = 0
    for item in downloads.iterdir():
        if item.is_dir():
            continue
        ext = item.suffix.lower().lstrip(".") or "no_ext"
        dest_dir = sorted_root / ext
        dest_dir.mkdir(exist_ok=True)
        dest_path = dest_dir / item.name
        try:
            shutil.move(str(item), str(dest_path))
            moved += 1
        except Exception:
            continue

    return f"Sorted {moved} file(s) into {sorted_root}."


async def send_message(chat_id: int, text: str) -> None:
    payload = {"chat_id": chat_id, "text": text}
    resp = requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)
    print(f"sendMessage status={resp.status_code} body={resp.text}")


def delete_telegram_webhook(drop_pending: bool = False) -> None:
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/deleteWebhook",
            json={"drop_pending_updates": drop_pending},
            timeout=10,
        )
        print(f"deleteWebhook status={resp.status_code} body={resp.text}")
    except Exception as e:
        print(f"deleteWebhook error: {e}")


def handle_command(text: str) -> str:
    text = text.strip()

    if text.lower().startswith("open "):
        url = text[5:].strip()
        if not re.match(r"^https?://", url, re.I):
            url = "https://" + url
        webbrowser.open(url)
        return f"Opened: {url}"

    if text.lower() in {"notepad", "open notepad"}:
        open_notepad()
        return "Opened Notepad."

    if text.lower() == "sort downloads":
        return sort_downloads()

    if text.lower() in {"help", "/help"}:
        return "Commands: open <url>, notepad, sort downloads"

    return "已成功紀錄"


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


def append_markdown(
    platform: str,
    chat_title: str,
    user_name: str,
    text: str,
    message_ts: datetime,
) -> None:
    day = (message_ts or datetime.now()).strftime("%Y-%m-%d")
    md_path = MD_DIR / f"{day}.md"
    if not md_path.exists():
        md_path.write_text(f"# {day}\n\n", encoding="utf-8")

    time_str = (message_ts or datetime.now()).strftime("%H:%M:%S")
    line = f"- [{time_str}] ({platform}) {chat_title} | {user_name}: {text}\n"
    with md_path.open("a", encoding="utf-8") as f:
        f.write(line)


def get_local_tz():
    return NEWS_TZ


def build_gnews_url(query: str, hl: str, gl: str, ceid: str) -> str:
    q = quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def get_news_rss_urls() -> list[str]:
    if NEWS_RSS_URLS_FILE:
        path = Path(NEWS_RSS_URLS_FILE)
        if path.exists():
            raw = path.read_text(encoding="utf-8").splitlines()
            return [u.strip() for u in raw if u.strip()]

    if NEWS_RSS_URLS_ENV:
        raw = NEWS_RSS_URLS_ENV.splitlines()
        return [u.strip() for u in raw if u.strip()]

    gnews_url = build_gnews_url(NEWS_GNEWS_QUERY, NEWS_GNEWS_HL, NEWS_GNEWS_GL, NEWS_GNEWS_CEID)
    return [
        "https://www.reuters.com/technology/rss",
        gnews_url,
        "https://feeds.bloomberg.com/technology/news.rss",
        "https://feeds.bloomberg.com/technology/ai/news.rss",
        "https://semianalysis.com/feed/",
        "https://asia.nikkei.com/rss/feed/nar",
        "https://asia.nikkei.com/rss/feed/technology",
    ]


def normalize_title(title: str) -> str:
    t = (title or "").strip().lower()
    if not t:
        return ""
    t = re.sub(
        r"^\\s*(reuters|bloomberg|nikkei|nikkei asia|asia nikkei|semianalysis)\\s*[-:|]\\s*",
        "",
        t,
    )
    t = re.sub(
        r"\\s*[-:|]\\s*(reuters|bloomberg|nikkei|nikkei asia|asia nikkei|semianalysis)\\s*$",
        "",
        t,
    )
    t = re.sub(r"[^a-z0-9\\u4e00-\\u9fff\\s]", " ", t)
    t = re.sub(r"\\s+", " ", t).strip()
    return t


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_entry_datetime(entry) -> datetime | None:
    dt_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if not dt_struct:
        return None
    ts = time.mktime(dt_struct)
    return datetime.fromtimestamp(ts, tz=get_local_tz())


def get_news_source_name(feed, url: str) -> str:
    if "news.google.com/rss/search" in (url or "") and "reuters" in NEWS_GNEWS_QUERY.lower():
        return "Reuters(GNews)"
    title = getattr(feed, "feed", {}).get("title") if feed else None
    if title:
        return title
    return url


def get_source_weight(source: str | None) -> int:
    s = (source or "").lower()
    if "reuters" in s:
        return 3
    if "bloomberg" in s:
        return 3
    if "nikkei" in s:
        return 2
    if "semianalysis" in s:
        return 2
    return 1


def fetch_news_entries() -> list[dict]:
    items = []
    for url in get_news_rss_urls():
        feed = feedparser.parse(url)
        source_name = get_news_source_name(feed, url)
        for entry in getattr(feed, "entries", []):
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            if not title or not link:
                continue
            summary = entry.get("summary") or entry.get("description") or ""
            published_at = parse_entry_datetime(entry)
            items.append(
                {
                    "source": source_name,
                    "title": title,
                    "url": link,
                    "summary": summary.strip(),
                    "published_at": published_at,
                }
            )
    items.sort(key=lambda x: x["published_at"] or datetime.now(tz=get_local_tz()))
    return items


def find_similar_cluster(recent_rows: list[tuple], title_norm: str) -> int | None:
    if not title_norm:
        return None
    best_cluster = None
    best_score = 0
    threshold = 95 if len(title_norm) < 40 else 92
    for cluster_id, existing_norm in recent_rows:
        if not existing_norm:
            continue
        score = fuzz.token_set_ratio(title_norm, existing_norm)
        if score >= threshold and score > best_score:
            best_score = score
            best_cluster = cluster_id
    return best_cluster


def ensure_cluster(
    conn: sqlite3.Connection,
    item: dict,
    recent_rows: list[tuple],
) -> int:
    title_norm = item["title_norm"]
    cluster_id = find_similar_cluster(recent_rows, title_norm)
    published_at = item["published_at"] or datetime.now(tz=get_local_tz())
    published_iso = published_at.isoformat()
    summary_len = len(item["summary"] or "")
    source_weight = get_source_weight(item["source"])
    if cluster_id:
        row = conn.execute(
            """
            SELECT canonical_published_at, canonical_summary, canonical_title, canonical_url, canonical_source
            FROM news_clusters
            WHERE id = ?
            """,
            (cluster_id,),
        ).fetchone()
        if row:
            canonical_published_at, canonical_summary, canonical_title, canonical_url, canonical_source = row
        else:
            canonical_published_at = None
            canonical_summary = None
            canonical_title = None
            canonical_url = None
            canonical_source = None

        update_fields = {}
        canonical_summary_len = len(canonical_summary or "")
        canonical_weight = get_source_weight(canonical_source)
        canonical_time = canonical_published_at or published_iso

        replace_canonical = False
        if summary_len > canonical_summary_len:
            replace_canonical = True
        elif summary_len == canonical_summary_len:
            if source_weight > canonical_weight:
                replace_canonical = True
            elif source_weight == canonical_weight and published_iso < canonical_time:
                replace_canonical = True

        if replace_canonical:
            update_fields["canonical_title"] = item["title"]
            update_fields["canonical_url"] = item["url"]
            update_fields["canonical_source"] = item["source"]
            update_fields["canonical_published_at"] = published_iso
            update_fields["canonical_summary"] = item["summary"] or None
            update_fields["canonical_summary_source"] = item["source"] if item["summary"] else None
        elif not canonical_summary and item["summary"]:
            update_fields["canonical_summary"] = item["summary"]
            update_fields["canonical_summary_source"] = item["source"]

        if update_fields:
            sets = ", ".join(f"{k} = ?" for k in update_fields.keys())
            params = list(update_fields.values()) + [cluster_id]
            conn.execute(f"UPDATE news_clusters SET {sets} WHERE id = ?", params)
        return cluster_id

    cluster_date = published_at.strftime("%Y%m%d")
    row = conn.execute(
        "SELECT COALESCE(MAX(cluster_seq), 0) + 1 FROM news_clusters WHERE cluster_date = ?",
        (cluster_date,),
    ).fetchone()
    next_seq = row[0] if row else 1
    conn.execute(
        """
        INSERT INTO news_clusters
        (cluster_date, cluster_seq, canonical_title, canonical_url, canonical_source, canonical_published_at, canonical_summary, canonical_summary_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cluster_date,
            next_seq,
            item["title"],
            item["url"],
            item["source"],
            published_iso,
            item["summary"] or None,
            item["source"] if item["summary"] else None,
        ),
    )
    cluster_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    recent_rows.append((cluster_id, title_norm))
    return cluster_id


def fetch_and_store_news() -> set[str]:
    changed_dates: set[str] = set()
    entries = fetch_news_entries()
    if not entries:
        return changed_dates

    now = datetime.now(tz=get_local_tz())
    cutoff_dt = now - timedelta(hours=12)
    cutoff_iso = cutoff_dt.isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        recent_rows = conn.execute(
            """
            SELECT cluster_id, title_norm
            FROM news_items
            WHERE published_at IS NOT NULL AND published_at >= ?
            ORDER BY published_at DESC
            LIMIT 500
            """,
            (cutoff_iso,),
        ).fetchall()

        for item in entries:
            published_dt = item.get("published_at")
            if not published_dt:
                continue
            published_dt = published_dt.astimezone(get_local_tz())
            if published_dt < cutoff_dt:
                continue
            item["title_norm"] = normalize_title(item["title"])
            item["hash_url"] = hash_text(item["url"])
            item["hash_title"] = hash_text(item["title_norm"] or item["title"])
            exists = conn.execute(
                "SELECT 1 FROM news_items WHERE hash_url = ?",
                (item["hash_url"],),
            ).fetchone()
            if exists:
                continue

            cluster_id = ensure_cluster(conn, item, recent_rows)
            published_iso = published_dt.isoformat()
            created_iso = now.isoformat()
            conn.execute(
                """
                INSERT INTO news_items
                (cluster_id, source, title, title_norm, url, summary, published_at, hash_url, hash_title, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cluster_id,
                    item["source"],
                    item["title"],
                    item["title_norm"],
                    item["url"],
                    item["summary"],
                    published_iso,
                    item["hash_url"],
                    item["hash_title"],
                    created_iso,
                ),
            )
            row = conn.execute(
                "SELECT cluster_date FROM news_clusters WHERE id = ?",
                (cluster_id,),
            ).fetchone()
            if row and row[0]:
                changed_dates.add(row[0])
        conn.commit()

    for cluster_date in changed_dates:
        write_news_markdown_for_date(cluster_date)
    return changed_dates


def format_cluster_id(cluster_date: str, cluster_seq: int) -> str:
    return f"{cluster_date}-{int(cluster_seq):04d}"


def yaml_escape(text: str | None) -> str:
    if not text:
        return ""
    return text.replace('"', '\\"')


def write_news_markdown_for_date(cluster_date: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        clusters = conn.execute(
            """
            SELECT id, cluster_seq, canonical_title, canonical_url, canonical_source,
                   canonical_published_at, canonical_summary
            FROM news_clusters
            WHERE cluster_date = ?
            ORDER BY canonical_published_at ASC
            """,
            (cluster_date,),
        ).fetchall()

        blocks = []
        for (
            cluster_id,
            cluster_seq,
            canonical_title,
            canonical_url,
            canonical_source,
            canonical_published_at,
            canonical_summary,
        ) in clusters:
            sources = conn.execute(
                """
                SELECT DISTINCT source, url
                FROM news_items
                WHERE cluster_id = ?
                ORDER BY source ASC
                """,
                (cluster_id,),
            ).fetchall()

            cluster_id_str = format_cluster_id(cluster_date, cluster_seq)
            lines = [
                "---",
                f'cluster_id: "{cluster_id_str}"',
                f'published_at: "{yaml_escape(canonical_published_at)}"',
                "canonical:",
                f'  source: "{yaml_escape(canonical_source)}"',
                f'  url: "{yaml_escape(canonical_url)}"',
                "sources:",
            ]
            for source, url in sources:
                lines.append(f'  - source: "{yaml_escape(source)}"')
                lines.append(f'    url: "{yaml_escape(url)}"')
            lines.append(f'title: "{yaml_escape(canonical_title)}"')
            lines.append("---")
            summary = canonical_summary or ""
            blocks.append("\n".join(lines) + "\n" + summary.strip() + "\n")

    md_path = NEWS_MD_DIR / f"{cluster_date}.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(blocks))


def upsert_news_subscription(chat_id: str, enabled: bool) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT chat_id FROM news_subscriptions WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE news_subscriptions SET enabled = ?, interval_minutes = ? WHERE chat_id = ?",
                (1 if enabled else 0, NEWS_FETCH_INTERVAL_MINUTES, chat_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO news_subscriptions (chat_id, enabled, interval_minutes, last_sent_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, 1 if enabled else 0, NEWS_FETCH_INTERVAL_MINUTES, None),
            )
        conn.commit()


def get_latest_clusters(limit: int) -> list[tuple]:
    now = datetime.now(tz=get_local_tz())
    cutoff_iso = (now - timedelta(hours=12)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, canonical_title, canonical_url, canonical_source, canonical_published_at,
                   cluster_date, cluster_seq
            FROM news_clusters
            WHERE canonical_published_at >= ?
            ORDER BY canonical_published_at DESC
            LIMIT ?
            """,
            (cutoff_iso, limit),
        ).fetchall()
    return rows


def search_clusters(keyword: str, limit: int) -> list[tuple]:
    kw = f"%{keyword}%"
    now = datetime.now(tz=get_local_tz())
    cutoff_iso = (now - timedelta(hours=12)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT c.id, c.canonical_title, c.canonical_url, c.canonical_source,
                   c.canonical_published_at, c.cluster_date, c.cluster_seq
            FROM news_clusters c
            JOIN news_items i ON i.cluster_id = c.id
            WHERE (i.title LIKE ? OR i.summary LIKE ?)
              AND c.canonical_published_at >= ?
            ORDER BY c.canonical_published_at DESC
            LIMIT ?
            """,
            (kw, kw, cutoff_iso, limit),
        ).fetchall()
    return rows


def format_cluster_list(rows: list[tuple]) -> str:
    lines = []
    for (
        _cid,
        title,
        url,
        source,
        published_at,
        cluster_date,
        cluster_seq,
    ) in rows:
        cluster_id_str = format_cluster_id(cluster_date, cluster_seq)
        lines.append(f"- [{source}] {title}")
        lines.append(f"  {url}")
        lines.append(f"  id={cluster_id_str} time={published_at}")
    return "\n".join(lines) if lines else "No news items found."


def handle_news_command(text: str, chat_id: str) -> list[str]:
    tokens = text.strip().split()
    sub = tokens[1].lower() if len(tokens) > 1 else "latest"
    if sub in {"subscribe", "sub"}:
        upsert_news_subscription(chat_id, True)
        return ["已訂閱新聞推播。"]
    if sub in {"unsubscribe", "unsub"}:
        upsert_news_subscription(chat_id, False)
        return ["已取消新聞推播。"]
    if sub == "sources":
        srcs = get_news_rss_urls()
        return ["News sources:\n" + "\n".join(srcs)]
    if sub == "search":
        if len(tokens) < 3:
            return ["用法：/news search <keywords>"]
        fetch_and_store_news()
        rows = search_clusters(" ".join(tokens[2:]), 10)
        return [format_cluster_list(rows)]
    if sub == "debug":
        fetch_and_store_news()
        now = datetime.now(tz=get_local_tz())
        cutoff_iso = (now - timedelta(hours=12)).isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """
                SELECT source, COUNT(*) AS cnt
                FROM news_items
                WHERE published_at >= ?
                GROUP BY source
                ORDER BY cnt DESC
                """
            , (cutoff_iso,)).fetchall()
        if not rows:
            return ["No news items ingested in the last 12 hours."]
        lines = ["News items by source (last 12 hours):"]
        for source, cnt in rows:
            lines.append(f"- {source}: {cnt}")
        return ["\n".join(lines)]
    if sub == "latest":
        limit = 5
        if len(tokens) >= 3 and tokens[2].isdigit():
            limit = max(1, min(20, int(tokens[2])))
        fetch_and_store_news()
        rows = get_latest_clusters(limit)
        return [format_cluster_list(rows)]
    if sub == "help":
        return [
            "News commands: /news latest [N], /news search <keywords>, /news sources, /news debug, /news subscribe, /news unsubscribe"
        ]
    return ["未知指令。用 /news help 查看指令。"]


NEWS_SHORTCUTS = {
    "/news_latest": "/news latest",
    "/news_search": "/news search",
    "/news_sources": "/news sources",
    "/news_debug": "/news debug",
    "/news_subscribe": "/news subscribe",
    "/news_unsubscribe": "/news unsubscribe",
    "/news_help": "/news help",
}


def set_telegram_commands() -> None:
    commands = [
        {"command": "news_latest", "description": "最新新聞"},
        {"command": "news_search", "description": "搜尋新聞"},
        {"command": "news_sources", "description": "新聞來源"},
        {"command": "news_debug", "description": "來源統計"},
        {"command": "news_subscribe", "description": "訂閱推播"},
        {"command": "news_unsubscribe", "description": "取消推播"},
        {"command": "news_help", "description": "指令說明"},
    ]
    try:
        resp = requests.post(
            f"{TELEGRAM_API}/setMyCommands",
            json={"commands": commands},
            timeout=10,
        )
        print(f"setMyCommands status={resp.status_code} body={resp.text}")
    except Exception as e:
        print(f"setMyCommands error: {e}")


def send_message_sync(chat_id: str, text: str) -> None:
    payload = {"chat_id": chat_id, "text": text}
    resp = requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)
    print(f"sendMessage status={resp.status_code} body={resp.text}")


def push_news_to_subscribers() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        subs = conn.execute(
            "SELECT chat_id, enabled, last_sent_at FROM news_subscriptions WHERE enabled = 1"
        ).fetchall()

    if not subs:
        return

    now = datetime.now(tz=get_local_tz())
    cutoff_iso = (now - timedelta(hours=12)).isoformat()
    now_iso = now.isoformat()
    for chat_id, _enabled, last_sent_at in subs:
        effective_last = last_sent_at
        if last_sent_at:
            try:
                last_dt = datetime.fromisoformat(last_sent_at)
                if last_dt < datetime.fromisoformat(cutoff_iso):
                    effective_last = cutoff_iso
            except Exception:
                effective_last = cutoff_iso
        with sqlite3.connect(DB_PATH) as conn:
            if effective_last:
                rows = conn.execute(
                    """
                    SELECT id, canonical_title, canonical_url, canonical_source, canonical_published_at,
                           cluster_date, cluster_seq
                    FROM news_clusters
                    WHERE canonical_published_at > ?
                      AND canonical_published_at >= ?
                    ORDER BY canonical_published_at ASC
                    LIMIT ?
                    """,
                    (effective_last, cutoff_iso, NEWS_PUSH_MAX_ITEMS),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, canonical_title, canonical_url, canonical_source, canonical_published_at,
                           cluster_date, cluster_seq
                    FROM news_clusters
                    WHERE canonical_published_at >= ?
                    ORDER BY canonical_published_at DESC
                    LIMIT ?
                    """,
                    (cutoff_iso, NEWS_PUSH_MAX_ITEMS),
                ).fetchall()
            if rows:
                msg = "News update:\n" + format_cluster_list(rows)
                send_message_sync(chat_id, msg)
            conn.execute(
                "UPDATE news_subscriptions SET last_sent_at = ? WHERE chat_id = ?",
                (now_iso, chat_id),
            )
            conn.commit()


def start_news_thread() -> None:
    def loop():
        while True:
            try:
                fetch_and_store_news()
                push_news_to_subscribers()
            except Exception as e:
                print(f"news worker error: {e}")
            time.sleep(max(1, NEWS_FETCH_INTERVAL_MINUTES) * 60)

    t = threading.Thread(target=loop, daemon=True)
    t.start()


def telegram_get_file_info(file_id: str) -> tuple[str, str] | None:
    try:
        resp = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=10)
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data.get("ok"):
            return None
        file_path = data.get("result", {}).get("file_path")
        if not file_path:
            return None
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
        return file_url, file_path
    except Exception:
        return None


def telegram_polling_loop() -> None:
    print("[INFO] Telegram long polling enabled.")
    delete_telegram_webhook(drop_pending=False)
    offset = None
    while True:
        try:
            params = {"timeout": 20}
            if offset is not None:
                params["offset"] = offset
            resp = requests.get(
                f"{TELEGRAM_API}/getUpdates",
                params=params,
                timeout=30,
            )
            if resp.status_code != 200:
                time.sleep(2)
                continue
            data = resp.json()
            if not data.get("ok"):
                time.sleep(2)
                continue
            for update in data.get("result", []):
                update_id = update.get("update_id")
                if update_id is not None:
                    offset = update_id + 1
                try:
                    local_resp = requests.post(
                        TELEGRAM_LOCAL_WEBHOOK_URL,
                        json=update,
                        timeout=10,
                    )
                    if local_resp.status_code != 200:
                        print(
                            f"[WARN] local webhook status={local_resp.status_code} body={local_resp.text}"
                        )
                except Exception as e:
                    print(f"[WARN] local webhook error: {e}")
        except Exception as e:
            print(f"[ERROR] Telegram polling error: {e}")
            time.sleep(2)


def start_telegram_polling_thread() -> None:
    if not TELEGRAM_LONG_POLLING:
        return
    t = threading.Thread(target=telegram_polling_loop, daemon=True)
    t.start()


@app.post("/telegram")
async def telegram_webhook(request: Request):
    update = await request.json()
    message = update.get("message", {})
    chat = message.get("chat", {})
    chat_type = chat.get("type")
    chat_id = chat.get("id")
    chat_title = chat.get("title") or chat.get("username") or ""
    text = message.get("text")

    sender = message.get("from", {})
    if sender.get("is_bot"):
        return JSONResponse({"ok": True})

    user_id = str(sender.get("id")) if sender.get("id") else ""
    user_name = (
        sender.get("username")
        or " ".join(filter(None, [sender.get("first_name"), sender.get("last_name")])).strip()
        or "unknown"
    )

    msg_ts = None
    if message.get("date"):
        msg_ts = datetime.fromtimestamp(message.get("date"))

    if chat_type in {"group", "supergroup"}:
        if text:
            store_message("telegram", str(chat_id), chat_title, user_id, user_name, text, msg_ts)
            append_markdown("telegram", chat_title, user_name, text, msg_ts)
        return JSONResponse({"ok": True})

    if chat_type == "private" and chat_id and text:
        text_stripped = text.strip()
        text_lower = text_stripped.lower()
        for shortcut, full_cmd in NEWS_SHORTCUTS.items():
            if text_lower.startswith(shortcut):
                text = full_cmd + text_stripped[len(shortcut):]
                break
        store_message("telegram", str(chat_id), chat_title or "private", user_id, user_name, text, msg_ts)
        append_markdown("telegram", chat_title or "private", user_name, text, msg_ts)
        if text.strip().lower().startswith("/news"):
            replies = handle_news_command(text, str(chat_id))
            for reply in replies:
                await send_message(chat_id, reply)
        else:
            reply = handle_command(text)
            await send_message(chat_id, reply)
        return JSONResponse({"ok": True})

    doc = message.get("document") or {}
    is_image_doc = doc.get("mime_type", "").startswith("image/")
    if chat_type == "private" and chat_id and (message.get("photo") or is_image_doc):
        file_id = None
        file_unique_id = None
        msg_id = message.get("message_id")
        if message.get("photo"):
            photos = message.get("photo") or []
            photo = photos[-1] if photos else None
            if photo:
                file_id = photo.get("file_id")
                file_unique_id = photo.get("file_unique_id")
        else:
            file_id = doc.get("file_id")
            file_unique_id = doc.get("file_unique_id")
        if file_id and file_unique_id and msg_id:
            file_info = telegram_get_file_info(file_id)
            if file_info:
                file_url, file_path = file_info
                try:
                    img_resp = requests.get(file_url, timeout=20)
                    if img_resp.status_code != 200:
                        raise RuntimeError(f"image download failed: {img_resp.status_code}")
                    day = (msg_ts or datetime.now()).strftime("%Y-%m-%d")
                    img_dir = DATA_DIR / "inbox" / "images" / day
                    img_dir.mkdir(parents=True, exist_ok=True)
                    suffix = Path(file_path).suffix or ".jpg"
                    filename = f"{chat_id}_{msg_id}_{file_unique_id}{suffix}"
                    out_path = img_dir / filename
                    out_path.write_bytes(img_resp.content)
                    await send_message(chat_id, f"已存檔：{filename}")
                    return JSONResponse({"ok": True})
                except Exception:
                    pass
        await send_message(chat_id, "存檔失敗，請稍後再試。")
        return JSONResponse({"ok": True})

    return JSONResponse({"ok": True})


def start_slack_socket_mode() -> None:
    if not (SLACK_BOT_TOKEN and SLACK_APP_TOKEN and SLACK_USER_ID):
        print("Slack tokens/user not set; skipping Slack Socket Mode.")
        return

    slack_app = SlackApp(token=SLACK_BOT_TOKEN)

    @slack_app.event("message")
    def handle_slack_message(event, say):
        if event.get("subtype"):
            return
        if event.get("bot_id"):
            return
        if event.get("channel_type") != "im":
            return
        if event.get("user") != SLACK_USER_ID:
            return

        text = event.get("text")
        if not text:
            return

        msg_ts = None
        if event.get("ts"):
            try:
                msg_ts = datetime.fromtimestamp(float(event.get("ts")))
            except Exception:
                msg_ts = None

        store_message("slack", event.get("channel", ""), "DM", event.get("user", ""), "Jason", text, msg_ts)
        append_markdown("slack", "DM", "Jason", text, msg_ts)

    handler = SocketModeHandler(slack_app, SLACK_APP_TOKEN)
    handler.connect()
    Event().wait()


def start_slack_thread() -> None:
    t = threading.Thread(target=start_slack_socket_mode, daemon=True)
    t.start()


set_telegram_commands()
start_news_thread()
start_slack_thread()
start_telegram_polling_thread()
