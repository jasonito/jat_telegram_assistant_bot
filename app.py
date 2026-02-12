import os
import asyncio
import re
import webbrowser
from html import unescape
from pathlib import Path, PurePosixPath
import subprocess
import shutil
import sqlite3
import hashlib
import json
import time
from typing import Iterator
from urllib.parse import quote
from urllib.parse import urlparse
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
try:
    import dropbox
except Exception:
    dropbox = None

try:
    from google.cloud import vision
except Exception:
    vision = None

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
NEWS_MD_DIR = DATA_DIR / "news"
NOTES_DIR = DATA_DIR / "notes"
TELEGRAM_MD_DIR = NOTES_DIR / "telegram"
SLACK_MD_DIR = NOTES_DIR / "slack"
OCR_MD_DIR = NOTES_DIR / "ocr"
INBOX_IMAGES_DIR = DATA_DIR / "inbox" / "images"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"
ALLOWED_GROUPS = {
    g.strip()
    for g in os.getenv("TELEGRAM_ALLOWED_GROUPS", "").split(",")
    if g.strip()
}

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")
SLACK_USER_ID = os.getenv("SLACK_USER_ID", "")
SLACK_DEBUG = os.getenv("SLACK_DEBUG", "0").lower() in {"1", "true", "yes"}

AI_SUMMARY_ENABLED = os.getenv("AI_SUMMARY_ENABLED", "0").lower() in {"1", "true", "yes"}
AI_SUMMARY_PROVIDER = os.getenv("AI_SUMMARY_PROVIDER", "openai").strip().lower()
AI_SUMMARY_TIMEOUT_SECONDS = int(
    os.getenv("AI_SUMMARY_TIMEOUT_SECONDS", os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
)
AI_SUMMARY_MAX_CHARS = int(os.getenv("AI_SUMMARY_MAX_CHARS", "6000"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "openai/gpt-oss-120b")
HUGGINGFACE_BASE_URL = os.getenv("HUGGINGFACE_BASE_URL", "https://router.huggingface.co/v1")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
NEWS_URL_FETCH_MAX_ARTICLES = int(os.getenv("NEWS_URL_FETCH_MAX_ARTICLES", "3"))
NEWS_URL_FETCH_MAX_CHARS = int(os.getenv("NEWS_URL_FETCH_MAX_CHARS", "3000"))
NEWS_URL_FETCH_TIMEOUT_SECONDS = int(os.getenv("NEWS_URL_FETCH_TIMEOUT_SECONDS", "6"))
NEWS_DIGEST_MAX_ITEMS = int(os.getenv("NEWS_DIGEST_MAX_ITEMS", "6"))
NEWS_DIGEST_AI_ITEMS = int(os.getenv("NEWS_DIGEST_AI_ITEMS", "2"))
NEWS_DIGEST_FETCH_ARTICLE_ITEMS = int(os.getenv("NEWS_DIGEST_FETCH_ARTICLE_ITEMS", "1"))
NOTE_DIGEST_MAX_ITEMS = int(os.getenv("NOTE_DIGEST_MAX_ITEMS", "5"))

OCR_PROVIDER = os.getenv("OCR_PROVIDER", "google_vision").strip().lower()
OCR_LANG_HINTS = [x.strip() for x in os.getenv("OCR_LANG_HINTS", "zh-TW,en").split(",") if x.strip()]

DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN", "").strip()
DROPBOX_REFRESH_TOKEN = os.getenv("DROPBOX_REFRESH_TOKEN", "").strip()
DROPBOX_APP_KEY = os.getenv("DROPBOX_APP_KEY", "").strip()
DROPBOX_APP_SECRET = os.getenv("DROPBOX_APP_SECRET", "").strip()
DROPBOX_TOKEN_REFRESH_LEEWAY_SECONDS = int(os.getenv("DROPBOX_TOKEN_REFRESH_LEEWAY_SECONDS", "300"))
DROPBOX_TRANSCRIPTS_PATH = os.getenv("DROPBOX_TRANSCRIPTS_PATH", "/Transcripts").strip() or "/Transcripts"
DROPBOX_TRANSCRIPTS_SYNC_ENABLED = os.getenv("DROPBOX_TRANSCRIPTS_SYNC_ENABLED", "1").lower() in {
    "1",
    "true",
    "yes",
}
DROPBOX_ROOT_PATH = os.getenv("DROPBOX_ROOT_PATH", "/read").strip() or "/read"
DROPBOX_SYNC_ENABLED = os.getenv("DROPBOX_SYNC_ENABLED", "1").lower() in {"1", "true", "yes"}
DROPBOX_SYNC_TIME = os.getenv("DROPBOX_SYNC_TIME", "00:10").strip() or "00:10"
DROPBOX_SYNC_TZ_NAME = os.getenv("DROPBOX_SYNC_TZ", "Asia/Taipei").strip() or "Asia/Taipei"
DROPBOX_SYNC_ON_STARTUP = os.getenv("DROPBOX_SYNC_ON_STARTUP", "1").lower() in {"1", "true", "yes"}

NEWS_RSS_URLS_ENV = os.getenv("NEWS_RSS_URLS", "")
NEWS_RSS_URLS_FILE = os.getenv("NEWS_RSS_URLS_FILE", "")
NEWS_FETCH_INTERVAL_MINUTES = int(os.getenv("NEWS_FETCH_INTERVAL_MINUTES", "180"))
NEWS_PUSH_MAX_ITEMS = int(os.getenv("NEWS_PUSH_MAX_ITEMS", "10"))
NEWS_PUSH_ENABLED = os.getenv("NEWS_PUSH_ENABLED", "0").lower() in {"1", "true", "yes"}
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
_vision_client = None
_dropbox_client = None
_dropbox_token_lock = threading.Lock()
_dropbox_access_token = DROPBOX_ACCESS_TOKEN
_dropbox_access_token_expires_at = 0.0


def init_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    TELEGRAM_MD_DIR.mkdir(parents=True, exist_ok=True)
    SLACK_MD_DIR.mkdir(parents=True, exist_ok=True)
    OCR_MD_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_MD_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sync_state (
                provider TEXT NOT NULL,
                local_path TEXT NOT NULL,
                fingerprint TEXT NOT NULL,
                last_synced_at TEXT NOT NULL,
                PRIMARY KEY(provider, local_path)
            )
            """
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


async def send_message(
    chat_id: int,
    text: str,
    parse_mode: str | None = None,
    disable_web_page_preview: bool | None = None,
) -> None:
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    if disable_web_page_preview is not None:
        payload["disable_web_page_preview"] = disable_web_page_preview
    resp = requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)
    if resp.status_code >= 300 and parse_mode:
        fallback_payload = {"chat_id": chat_id, "text": text}
        resp = requests.post(f"{TELEGRAM_API}/sendMessage", json=fallback_payload, timeout=10)
    print(f"sendMessage status={resp.status_code} body={resp.text}")


def _chunk_text_for_telegram(text: str, limit: int = 3500) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for line in text.splitlines():
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(line) <= limit:
            current = line
            continue
        start = 0
        while start < len(line):
            chunks.append(line[start : start + limit])
            start += limit

    if current:
        chunks.append(current)
    return chunks if chunks else [text]


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
    text_lower = text.lower()

    if text_lower.startswith("/summary_news"):
        day = _parse_day_arg(text)
        parts = build_scoped_summary(day, "news")
        return "\n".join(parts)

    if text_lower.startswith("/summary_note"):
        day = _parse_day_arg(text)
        parts = build_scoped_summary(day, "note", recent_days=3)
        return "\n".join(parts)

    if text_lower.startswith("/summary_weekly"):
        day = _parse_day_arg(text)
        parts = summary_weekly(day)
        return "\n".join(parts)

    if text_lower.startswith("/summary"):
        scope, day = _parse_summary_args(text)
        parts = summary_ai(day, scope=scope)
        return "\n".join(parts)

    if text_lower.startswith("/status"):
        return build_status_report()

    if text_lower.startswith("open "):
        url = text[5:].strip()
        if not re.match(r"^https?://", url, re.I):
            url = "https://" + url
        webbrowser.open(url)
        return f"Opened: {url}"

    if text_lower in {"notepad", "open notepad"}:
        open_notepad()
        return "Opened Notepad."

    if text_lower == "sort downloads":
        return sort_downloads()

    if text_lower in {"help", "/help"}:
        return "Commands: open <url>, notepad, sort downloads, /status"

    if text.startswith("/"):
        return "Unsupported command."
    return "已成功紀錄"


def route_user_text_command(text: str, chat_id: str) -> tuple[list[str], str | None, bool | None]:
    cmd_text = (text or "").strip()
    lower = cmd_text.lower()

    if lower.startswith("/news"):
        replies = handle_news_command(cmd_text, chat_id)
        tokens = cmd_text.split()
        sub = tokens[1].lower() if len(tokens) > 1 else "latest"
        parse_mode = "Markdown" if sub == "latest" else None
        disable_preview = True if sub == "latest" else None
        return replies, parse_mode, disable_preview

    reply = handle_command(cmd_text)
    parse_mode = None
    disable_preview = None
    if lower.startswith("/summary_news"):
        parse_mode = "Markdown"
        disable_preview = True
    elif lower.startswith("/summary"):
        scope, _ = _parse_summary_args(cmd_text)
        if scope == "news":
            parse_mode = "Markdown"
            disable_preview = True
    return [reply], parse_mode, disable_preview


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


def append_ocr_markdown(
    chat_id: str,
    msg_id: str | int,
    image_path: Path,
    text: str,
    message_ts: datetime | None,
) -> Path:
    day = (message_ts or datetime.now()).strftime("%Y-%m-%d")
    md_path = OCR_MD_DIR / f"{day}_ocr.md"
    if not md_path.exists():
        md_path.write_text(f"# {day} ocr\n\n", encoding="utf-8")
    try:
        rel_path = image_path.relative_to(DATA_DIR).as_posix()
    except Exception:
        rel_path = str(image_path)
    time_str = (message_ts or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    normalized_text = (text or "").strip()
    with md_path.open("a", encoding="utf-8") as f:
        f.write(f"## [{time_str}] telegram chat={chat_id} message={msg_id}\n")
        f.write(f"- image: `{rel_path}`\n")
        if normalized_text:
            f.write("- text:\n\n")
            f.write(normalized_text)
            f.write("\n\n")
        else:
            f.write("- text: [no text detected]\n\n")
    return md_path


def _get_google_vision_client():
    global _vision_client
    if _vision_client is not None:
        return _vision_client
    if vision is None:
        raise RuntimeError("google-cloud-vision is not installed")
    _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


def extract_text_from_image(image_bytes: bytes) -> str:
    if not image_bytes:
        return ""
    if OCR_PROVIDER != "google_vision":
        raise RuntimeError(f"unsupported OCR_PROVIDER: {OCR_PROVIDER}")
    client = _get_google_vision_client()
    image = vision.Image(content=image_bytes)
    image_context = vision.ImageContext(language_hints=OCR_LANG_HINTS) if OCR_LANG_HINTS else None
    result = client.text_detection(image=image, image_context=image_context, timeout=20)
    if result.error and result.error.message:
        raise RuntimeError(result.error.message)
    text = ""
    if result.full_text_annotation and result.full_text_annotation.text:
        text = result.full_text_annotation.text
    elif result.text_annotations:
        text = result.text_annotations[0].description or ""
    return text.strip()


def normalize_dropbox_path(path: str) -> str:
    p = (path or "").strip().replace("\\", "/")
    if not p.startswith("/"):
        p = "/" + p
    return p.rstrip("/") or "/read"


def _dropbox_create_folder_if_missing(dbx, path: str) -> None:
    try:
        dbx.files_create_folder_v2(path)
    except Exception as e:
        msg = str(e).lower()
        if "conflict" in msg or "already" in msg:
            return
        raise


def ensure_dropbox_folders(dbx, root_path: str) -> None:
    root = normalize_dropbox_path(root_path)
    _dropbox_create_folder_if_missing(dbx, root)
    for folder in ("news", "notes", "images"):
        _dropbox_create_folder_if_missing(dbx, f"{root}/{folder}")


def iter_sync_files() -> Iterator[tuple[str, Path, str]]:
    roots = [
        ("news", NEWS_MD_DIR),
        ("notes", NOTES_DIR),
        ("images", INBOX_IMAGES_DIR),
    ]
    for category, root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            yield category, p, rel


TRANSCRIPT_TEXT_EXTENSIONS = {
    ".md",
    ".markdown",
    ".txt",
    ".srt",
    ".vtt",
    ".ass",
    ".ssa",
    ".json",
}


def _is_transcript_text_file(path: str) -> bool:
    return Path(path).suffix.lower() in TRANSCRIPT_TEXT_EXTENSIONS


def _dropbox_list_folder_entries_recursive(remote_path: str) -> list[object]:
    root = normalize_dropbox_path(remote_path)
    result = _dropbox_call_with_retry(lambda dbx: dbx.files_list_folder(root, recursive=True))
    entries = list(result.entries or [])
    while getattr(result, "has_more", False):
        cursor = result.cursor
        result = _dropbox_call_with_retry(lambda dbx, c=cursor: dbx.files_list_folder_continue(c))
        entries.extend(result.entries or [])
    return entries


def _safe_transcript_relpath(root_path: str, remote_path: str) -> str:
    root = normalize_dropbox_path(root_path).rstrip("/")
    full = normalize_dropbox_path(remote_path)
    rel = full[len(root) :].lstrip("/") if full.startswith(f"{root}/") else full.lstrip("/")
    parts = [p for p in PurePosixPath(rel).parts if p not in {"", ".", ".."}]
    return "/".join(parts)


def _transcript_fingerprint(entry) -> str:
    rev = str(getattr(entry, "rev", "") or "")
    content_hash = str(getattr(entry, "content_hash", "") or "")
    modified = getattr(entry, "server_modified", None)
    modified_text = modified.isoformat() if modified else ""
    return f"{rev}:{content_hash}:{modified_text}"


def sync_dropbox_transcripts_to_local(full_scan: bool = False) -> dict[str, int]:
    stats = {
        "transcripts_scanned": 0,
        "transcripts_downloaded": 0,
        "transcripts_skipped": 0,
        "transcripts_failed": 0,
    }
    if not DROPBOX_TRANSCRIPTS_SYNC_ENABLED:
        return stats

    remote_root = normalize_dropbox_path(DROPBOX_TRANSCRIPTS_PATH)
    try:
        entries = _dropbox_list_folder_entries_recursive(remote_root)
    except Exception as e:
        print(f"[WARN] Dropbox transcript listing failed for {remote_root}: {e}")
        return stats

    for entry in entries:
        if not getattr(entry, "is_file", False):
            continue

        remote_path = str(getattr(entry, "path_display", "") or getattr(entry, "path_lower", "") or "")
        if not remote_path or not _is_transcript_text_file(remote_path):
            continue

        stats["transcripts_scanned"] += 1
        rel = _safe_transcript_relpath(remote_root, remote_path)
        if not rel:
            stats["transcripts_skipped"] += 1
            continue

        local_path = TRANSCRIPTS_DIR / Path(rel)
        state_key = normalize_dropbox_path(str(getattr(entry, "path_lower", "") or remote_path))
        fingerprint = _transcript_fingerprint(entry)
        last_fp = get_sync_state("dropbox_transcripts", state_key)
        if not full_scan and last_fp == fingerprint and local_path.exists():
            stats["transcripts_skipped"] += 1
            continue

        try:
            data = _dropbox_call_with_retry(
                lambda dbx, p=remote_path: dbx.files_download(p)[1].content
            )
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            server_modified = getattr(entry, "server_modified", None)
            if server_modified:
                ts = float(server_modified.timestamp())
                os.utime(local_path, (ts, ts))
            upsert_sync_state("dropbox_transcripts", state_key, fingerprint)
            stats["transcripts_downloaded"] += 1
        except Exception as e:
            stats["transcripts_failed"] += 1
            print(f"[WARN] Dropbox transcript download failed for {remote_path}: {e}")

    return stats


def compute_file_fingerprint(path: Path) -> str:
    st = path.stat()
    raw = f"{st.st_size}:{st.st_mtime_ns}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def get_sync_state(provider: str, local_path: str) -> str | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            """
            SELECT fingerprint
            FROM sync_state
            WHERE provider = ? AND local_path = ?
            """,
            (provider, local_path),
        ).fetchone()
    return row[0] if row else None


def upsert_sync_state(provider: str, local_path: str, fingerprint: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO sync_state(provider, local_path, fingerprint, last_synced_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(provider, local_path)
            DO UPDATE SET
                fingerprint = excluded.fingerprint,
                last_synced_at = excluded.last_synced_at
            """,
            (provider, local_path, fingerprint, datetime.now().isoformat(timespec="seconds")),
        )
        conn.commit()


def _dropbox_can_refresh_token() -> bool:
    return bool(DROPBOX_REFRESH_TOKEN and DROPBOX_APP_KEY and DROPBOX_APP_SECRET)


def _dropbox_is_auth_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "invalid_access_token" in msg or "expired_access_token" in msg or "401" in msg


def _dropbox_fetch_access_token() -> tuple[str, float]:
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": DROPBOX_REFRESH_TOKEN,
        "client_id": DROPBOX_APP_KEY,
        "client_secret": DROPBOX_APP_SECRET,
    }
    resp = requests.post("https://api.dropboxapi.com/oauth2/token", data=payload, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"Dropbox token refresh failed ({resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    access_token = (data.get("access_token") or "").strip()
    if not access_token:
        raise RuntimeError("Dropbox token refresh response missing access_token")
    expires_in = int(data.get("expires_in", 14400))
    # Refresh a bit earlier to avoid racing against token expiry during long operations.
    expires_at = time.time() + max(60, expires_in - max(0, DROPBOX_TOKEN_REFRESH_LEEWAY_SECONDS))
    return access_token, expires_at


def _dropbox_refresh_access_token(force: bool = False) -> str:
    global _dropbox_access_token, _dropbox_access_token_expires_at, _dropbox_client
    if not _dropbox_can_refresh_token():
        if _dropbox_access_token:
            return _dropbox_access_token
        raise RuntimeError(
            "Dropbox credential missing: set DROPBOX_ACCESS_TOKEN or "
            "(DROPBOX_REFRESH_TOKEN + DROPBOX_APP_KEY + DROPBOX_APP_SECRET)."
        )

    with _dropbox_token_lock:
        if not force and _dropbox_access_token and time.time() < _dropbox_access_token_expires_at:
            return _dropbox_access_token
        token, expires_at = _dropbox_fetch_access_token()
        _dropbox_access_token = token
        _dropbox_access_token_expires_at = expires_at
        _dropbox_client = None
        return _dropbox_access_token


def _dropbox_rebuild_client() -> None:
    global _dropbox_client
    token = _dropbox_refresh_access_token(force=False)
    _dropbox_client = dropbox.Dropbox(token, timeout=30)


def _get_dropbox_client():
    global _dropbox_client
    if _dropbox_client is not None:
        return _dropbox_client
    if dropbox is None:
        raise RuntimeError("dropbox SDK is not installed")
    if not (_dropbox_access_token or _dropbox_can_refresh_token()):
        raise RuntimeError(
            "Dropbox credential missing: set DROPBOX_ACCESS_TOKEN or "
            "(DROPBOX_REFRESH_TOKEN + DROPBOX_APP_KEY + DROPBOX_APP_SECRET)."
        )
    _dropbox_rebuild_client()
    return _dropbox_client


def _dropbox_call_with_retry(func):
    try:
        return func(_get_dropbox_client())
    except Exception as e:
        if not _dropbox_can_refresh_token() or not _dropbox_is_auth_error(e):
            raise
        _dropbox_refresh_access_token(force=True)
        _dropbox_rebuild_client()
        return func(_get_dropbox_client())


def sync_file_to_dropbox(local_path: Path, remote_path: str) -> None:
    content = local_path.read_bytes()
    def _upload(dbx):
        dbx.files_upload(
            content,
            remote_path,
            mode=dropbox.files.WriteMode.overwrite,
            mute=True,
        )
    _dropbox_call_with_retry(_upload)


def run_dropbox_sync(full_scan: bool = False) -> dict[str, int]:
    stats = {
        "scanned": 0,
        "uploaded": 0,
        "skipped": 0,
        "failed": 0,
        "transcripts_scanned": 0,
        "transcripts_downloaded": 0,
        "transcripts_skipped": 0,
        "transcripts_failed": 0,
    }
    if not DROPBOX_SYNC_ENABLED:
        return stats
    try:
        _get_dropbox_client()
    except Exception as e:
        print(f"[WARN] Dropbox sync unavailable: {e}")
        return stats
    root = normalize_dropbox_path(DROPBOX_ROOT_PATH)
    try:
        _dropbox_call_with_retry(lambda dbx: ensure_dropbox_folders(dbx, root))
    except Exception as e:
        print(f"[WARN] Dropbox folder bootstrap failed: {e}")
        return stats
    transcript_stats = sync_dropbox_transcripts_to_local(full_scan=full_scan)
    stats.update({k: stats.get(k, 0) + transcript_stats.get(k, 0) for k in transcript_stats})
    for category, local_path, rel in iter_sync_files():
        stats["scanned"] += 1
        local_key = local_path.relative_to(DATA_DIR).as_posix()
        try:
            fingerprint = compute_file_fingerprint(local_path)
            last_fp = get_sync_state("dropbox", local_key)
            if not full_scan and last_fp == fingerprint:
                stats["skipped"] += 1
                continue
            remote_path = f"{root}/{category}/{rel}".replace("//", "/")
            sync_file_to_dropbox(local_path, remote_path)
            upsert_sync_state("dropbox", local_key, fingerprint)
            stats["uploaded"] += 1
        except Exception as e:
            stats["failed"] += 1
            print(f"[WARN] Dropbox upload failed for {local_path}: {e}")
    print(
        "[INFO] Dropbox sync finished "
        f"scanned={stats['scanned']} uploaded={stats['uploaded']} "
        f"skipped={stats['skipped']} failed={stats['failed']} "
        f"transcripts_scanned={stats['transcripts_scanned']} "
        f"transcripts_downloaded={stats['transcripts_downloaded']} "
        f"transcripts_skipped={stats['transcripts_skipped']} "
        f"transcripts_failed={stats['transcripts_failed']}"
    )
    return stats


def get_dropbox_sync_tz():
    try:
        return ZoneInfo(DROPBOX_SYNC_TZ_NAME)
    except ZoneInfoNotFoundError:
        return get_local_tz()


def parse_hhmm(value: str, default_hour: int = 0, default_minute: int = 10) -> tuple[int, int]:
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", (value or "").strip())
    if not m:
        return default_hour, default_minute
    hour = int(m.group(1))
    minute = int(m.group(2))
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return default_hour, default_minute
    return hour, minute


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
    return True, "Saved."


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
            """
            SELECT COUNT(*)
            FROM messages
            WHERE received_ts LIKE ?
            """,
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


def detect_mode(context_text: str) -> str:
    text = (context_text or "").strip()
    total_len = len(text)

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    max_para_len = max((len(p) for p in paragraphs), default=0)
    avg_para_len = total_len / max(len(paragraphs), 1)

    date_marks = re.findall(r"\b\d{1,2}/\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b", text)
    if len(date_marks) >= 2:
        return "article_summary"

    if max_para_len >= 300 and total_len >= 800:
        return "article_summary"

    urls = re.findall(r"https?://", text)
    if len(urls) >= 2 and avg_para_len <= 120:
        return "slack_daily"

    article_score = 0
    non_ascii_hits = sum(1 for ch in text if ord(ch) > 127)
    if non_ascii_hits >= (total_len / 100) * 3:
        article_score += 1
    if len(re.findall(r"(analysis|summary|market|earnings|revenue|guidance)", text, flags=re.I)) >= 3:
        article_score += 1
    if article_score >= 2:
        return "article_summary"

    return "slack_daily"


def generate_ai_summary(
    day: str,
    msg_total: int,
    note_total: int,
    msg_by_platform: dict,
    note_by_platform: dict,
    recent_msgs: list[tuple],
    recent_notes: list[tuple],
    news_titles: list[tuple],
    news_rows: list[tuple] | None = None,
) -> str | None:
    if not AI_SUMMARY_ENABLED:
        return None

    provider = AI_SUMMARY_PROVIDER
    if provider == "antropic":
        provider = "anthropic"
    if provider == "hf":
        provider = "huggingface"
    if provider == "local":
        provider = "ollama"

    def clip(text: str, max_len: int = 180) -> str:
        t = re.sub(r"\s+", " ", str(text or "").strip())
        return t if len(t) <= max_len else t[: max_len - 3] + "..."

    msg_platform = ", ".join(f"{k}={v}" for k, v in msg_by_platform.items()) or "none"
    note_platform = ", ".join(f"{k}={v}" for k, v in note_by_platform.items()) or "none"

    activity_lines = []
    for platform, chat_title, user_name, msg_text, received_ts in recent_msgs:
        ts = (received_ts or "")[:19]
        activity_lines.append(f"- [{ts}] ({platform}) {chat_title} | {user_name}: {clip(msg_text)}")

    note_lines = []
    for platform, ch_id, user_name, msg_text, received_ts in recent_notes:
        ts = (received_ts or "")[:19]
        note_lines.append(f"- [{ts}] ({platform}) {ch_id} | {user_name}: {clip(msg_text)}")

    normalized_news_rows = []
    source_news = news_rows if news_rows is not None else news_titles
    for row in source_news or []:
        if not row:
            continue
        if isinstance(row, (list, tuple)):
            title = row[0] if len(row) > 0 else ""
            url = row[1] if len(row) > 1 else ""
        else:
            title = str(row)
            url = ""
        title = str(title or "").strip()
        url = str(url or "").strip()
        if title:
            normalized_news_rows.append((title, url))

    news_lines = []
    for title, url in normalized_news_rows:
        url = url.strip()
        has_url = bool(re.match(r"^https?://", url, re.I))
        source = url if has_url else "靘?蝻箏仃"
        news_lines.append(f"- {clip(title, 180)} | {source}")

    context_lines = [
        f"Date: {day}",
        f"Message total: {msg_total}",
        f"Note total: {note_total}",
        f"Messages by platform: {msg_platform}",
        f"Notes by platform: {note_platform}",
        "Recent activity:",
        *activity_lines,
        "Recent notes:",
        *note_lines,
        "News:",
        *news_lines,
    ]

    context_text = "\n".join(context_lines)
    if len(context_text) > AI_SUMMARY_MAX_CHARS:
        context_text = context_text[:AI_SUMMARY_MAX_CHARS]

    system_prompt = (
        "You are an operations and project decision assistant. "
        "Use only the provided input. Do not invent facts. "
        "Output must be in Traditional Chinese."
    )
    SLACK_DAILY_PROMPT = (
        f"隢 {day} ?Ｗ銝隞?Slack 瘥??嚗?雿輻頛詨鞈?嚗?擃葉???n\n"
        "頛詨?澆?嚗?潮摰?嚗n"
        "A. 隞??\n"
        "- 3-5 暺????data/news 銝剖祕??曄?鈭?嚗蒂?? URL\n"
        "- 瘥?銝銵??踹?瘜??膩\n\n"
        "B. 敺齒???n"
        "- 3-5 暺?隞亙?瑁????\n"
        "- ?亥?閮?頞喉?隢神????頞喋?銝??冽葫\n\n"
        "?釭閬?嚗n"
        "- 銝?雿輻?望?璅?\n"
        "- 銝?頛詨??頠賊?蝝?n"
        "- 蝮賡摨衣? 300-500 摮n\n"
        "頛詨鞈?嚗n"
        f"{context_text}\n"
    )

    ARTICLE_SUMMARY_PROMPT = (
        "隢?隞乩??批捆?渡??箔?蝭?瑽???嚗?擃葉???n\n"
        "頛詨?澆?嚗n"
        "1. 銝?亥店銝餅嚗? ?伐?\n"
        "2. ?詨??批捆??嚗?-6 暺?瘥? 2-3 ?伐?\n"
        "   - ?亙?????畾菜?蝡?嚗? 2/6??/6嚗?隢??府蝯?\n"
        "   - 敹?靽???銝剔??琿??詨??極?瑕?蝔梯?蝟餌絞閮剛?蝝啁?\n"
        "3. 雿??詨?閫暺?蝯?嚗?-2 ?伐?嚗蒂蝯虫?撖阡??臬銵??寞?\n\n"
        "閬?嚗n"
        "- ?蝙?典???閮?銝??啣?鈭辣??暺n"
        "- ?交?畾菔?閮?頞喉?隢??銝?鋆神\n"
        "???批捆嚗n"
        f"{context_text}\n"
    )

    mode = detect_mode(context_text)
    if mode == "article_summary":
        user_prompt = ARTICLE_SUMMARY_PROMPT
    else:
        user_prompt = SLACK_DAILY_PROMPT

    try:
        if provider == "openai":
            if not OPENAI_API_KEY:
                print("[WARN] AI summary enabled but OPENAI_API_KEY is empty; using fallback summary")
                return None
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                print(f"[WARN] OpenAI summary API failed: status={resp.status_code} body={resp.text[:300]}")
                return None
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return content or None

        if provider == "gemini":
            if not GEMINI_API_KEY:
                print("[WARN] AI summary enabled but GEMINI_API_KEY is empty; using fallback summary")
                return None
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
            resp = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                    "generationConfig": {"temperature": 0.2},
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                print(f"[WARN] Gemini summary API failed: status={resp.status_code} body={resp.text[:300]}")
                return None
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts") or []
            content = "\n".join((part.get("text") or "").strip() for part in parts if part.get("text"))
            return content.strip() or None

        if provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                print("[WARN] AI summary enabled but ANTHROPIC_API_KEY is empty; using fallback summary")
                return None
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 800,
                    "temperature": 0.2,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                print(f"[WARN] Anthropic summary API failed: status={resp.status_code} body={resp.text[:300]}")
                return None
            data = resp.json()
            blocks = data.get("content") or []
            content = "\n".join((b.get("text") or "").strip() for b in blocks if b.get("type") == "text")
            return content.strip() or None

        if provider == "huggingface":
            if not HUGGINGFACE_API_KEY:
                print("[WARN] AI summary enabled but HUGGINGFACE_API_KEY is empty; using fallback summary")
                return None
            url = f"{HUGGINGFACE_BASE_URL.rstrip('/')}/chat/completions"
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": HUGGINGFACE_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                print(f"[WARN] HuggingFace summary API failed: status={resp.status_code} body={resp.text[:300]}")
                return None
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            return content or None

        if provider == "ollama":
            url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat"
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                print(f"[WARN] Ollama summary API failed: status={resp.status_code} body={resp.text[:300]}")
                return None
            data = resp.json()
            content = data.get("message", {}).get("content", "").strip()
            return content or None

        print(f"[WARN] unsupported AI_SUMMARY_PROVIDER={AI_SUMMARY_PROVIDER}; using fallback summary")
        return None

    except Exception as e:
        print(f"[WARN] AI summary exception provider={provider}: {e}")
        return None


def merge_all(day: str) -> list[str]:
    notes_files = sorted(NOTES_DIR.rglob(f"{day}*.md"))

    day_compact = day.replace("-", "")
    news_candidates = [
        NEWS_MD_DIR / f"{day}_news.md",
        NEWS_MD_DIR / f"{day_compact}.md",
        NEWS_MD_DIR / f"{day_compact}_news.md",
    ]
    news_files = [fp for fp in news_candidates if fp.exists()]
    if not news_files:
        news_files = sorted(NEWS_MD_DIR.glob(f"{day_compact}*.md"))

    parts: list[str] = [f"{day} merged raw data"]

    if notes_files:
        parts.append("== Notes ==")
        for fp in notes_files:
            parts.append(f"# file: {fp}")
            try:
                content = fp.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                content = f"[read error] {e}"
            parts.append(content if content else "[empty]")
    else:
        parts.append("== Notes ==")
        parts.append("[no notes files]")

    if news_files:
        parts.append("== News ==")
        for fp in news_files:
            parts.append(f"# file: {fp}")
            try:
                content = fp.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                content = f"[read error] {e}"
            parts.append(content if content else "[empty]")
    else:
        parts.append("== News ==")
        parts.append("[no news files]")

    return parts


def _resolve_ai_provider() -> str:
    provider = (AI_SUMMARY_PROVIDER or "").strip().lower()
    if provider == "antropic":
        return "anthropic"
    if provider == "hf":
        return "huggingface"
    if provider == "local":
        return "ollama"
    return provider


def _run_ai_chat(system_prompt: str, user_prompt: str) -> str | None:
    provider = _resolve_ai_provider()
    try:
        if provider == "openai":
            if not OPENAI_API_KEY:
                return None
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                return None
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None

        if provider == "gemini":
            if not GEMINI_API_KEY:
                return None
            resp = requests.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                    "generationConfig": {"temperature": 0.2},
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                return None
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts") or []
            return "\n".join((p.get("text") or "").strip() for p in parts if p.get("text")).strip() or None

        if provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                return None
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 800,
                    "temperature": 0.2,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                return None
            data = resp.json()
            blocks = data.get("content") or []
            return "\n".join((b.get("text") or "").strip() for b in blocks if b.get("type") == "text").strip() or None

        if provider == "huggingface":
            if not HUGGINGFACE_API_KEY:
                return None
            resp = requests.post(
                f"{HUGGINGFACE_BASE_URL.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": HUGGINGFACE_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                return None
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None

        if provider == "ollama":
            resp = requests.post(
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
                headers={"Content-Type": "application/json"},
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                return None
            data = resp.json()
            return data.get("message", {}).get("content", "").strip() or None
    except Exception:
        return None
    return None


def _strip_markdown_noise(text: str) -> str:
    if not text:
        return text
    return text.replace("**", "").replace("__", "").replace("`", "")


def _normalize_news_output(text: str) -> str:
    insufficient = "鞈?銝雲"
    if not text:
        return insufficient

    def clean_prefix(line: str) -> str:
        return re.sub(r"^\s*(?:[-*?]|\d+[.)])\s*", "", line).strip()

    lines_out: list[str] = []
    for raw in text.splitlines():
        line = _strip_markdown_noise(raw).strip()
        if not line:
            continue
        line = clean_prefix(line)
        if not line:
            continue
        if line.lower().startswith(("a.", "news", "today")):
            continue

        md = re.search(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", line)
        if md:
            title = md.group(1).strip()
            url = md.group(2).strip()
            lines_out.append(f"- <{url}|{title}> | URL: {url}")
            continue

        slack_link = re.search(r"<(https?://[^|>]+)\|([^>]+)>", line)
        if slack_link:
            url = slack_link.group(1).strip()
            title = slack_link.group(2).strip()
            lines_out.append(f"- <{url}|{title}> | URL: {url}")
            continue

        m = re.search(r"(https?://\S+)", line)
        url = m.group(1).rstrip(").,;") if m else ""
        title = line
        title = re.sub(r"\(https?://[^)\s]+\)", "", title).strip()
        title = re.sub(r"https?://\S+", "", title).strip(" -|:")
        if not title:
            title = "untitled"

        if url:
            lines_out.append(f"- <{url}|{title}> | URL: {url}")
        else:
            lines_out.append(f"- {title} | URL: ?")

    if not lines_out:
        return insufficient

    return "\n".join(lines_out[:12])


def _normalize_notes_output(text: str) -> str:
    insufficient = "鞈?銝雲"
    if not text:
        return insufficient

    def clean_prefix(line: str) -> str:
        return re.sub(r"^\s*(?:[-*?]|\d+[.)])\s*", "", line).strip()

    out: list[str] = []
    for raw in text.splitlines():
        line = _strip_markdown_noise(raw).strip()
        if not line:
            continue
        line = clean_prefix(line)
        if not line:
            continue
        if line.lower().startswith(("b.", "notes", "highlights")):
            continue
        out.append(f"- {line}")

    if not out:
        return insufficient
    return "\n".join(out[:15])


def _normalize_three_points_output(text: str) -> str:
    insufficient = "鞈?銝雲"
    if not text:
        return insufficient

    def clean_prefix(line: str) -> str:
        return re.sub(r"^\s*(?:[-*?]|\d+[.)])\s*", "", line).strip()

    out: list[str] = []
    for raw in text.splitlines():
        line = _strip_markdown_noise(raw).strip()
        if not line:
            continue
        line = clean_prefix(line)
        if not line:
            continue
        out.append(f"- {line}")

    if not out:
        return insufficient
    return "\n".join(out[:3])


def _parse_summary_args(text: str) -> tuple[str, str]:
    tokens = text.strip().split()
    scope = "all"
    day = datetime.now().strftime("%Y-%m-%d")
    day_token_idx = 1

    if len(tokens) > 1:
        t1 = tokens[1].strip().lower()
        if t1 in {"note", "notes"}:
            scope = "note"
            day_token_idx = 2
        elif t1 == "news":
            scope = "news"
            day_token_idx = 2
        elif re.fullmatch(r"\d{4}-\d{2}-\d{2}", t1):
            day = t1
            day_token_idx = 2

    if len(tokens) > day_token_idx and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[day_token_idx]):
        day = tokens[day_token_idx]

    return scope, day


def _parse_day_arg(text: str) -> str:
    tokens = text.strip().split()
    if len(tokens) > 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[1]):
        return tokens[1]
    return datetime.now().strftime("%Y-%m-%d")


def build_scoped_summary(day: str, scope: str, *, recent_days: int | None = None) -> list[str]:
    scope_norm = (scope or "all").strip().lower()
    if scope_norm == "notes":
        scope_norm = "note"
    days = max(1, int(recent_days or 1))
    if scope_norm == "news":
        if days > 1:
            return build_news_digest_recent(day, days=days)
        return build_news_digest(day)
    if scope_norm == "note":
        if days > 1:
            return build_note_digest_recent(day, days=days)
        return build_note_digest(day)
    return summary_ai(day, scope="all")


def _collect_transcript_files_for_day(day: str) -> list[Path]:
    if not TRANSCRIPTS_DIR.exists():
        return []

    day_compact = day.replace("-", "")
    target_date = datetime.strptime(day, "%Y-%m-%d").date()
    out: list[Path] = []
    seen: set[str] = set()
    for fp in TRANSCRIPTS_DIR.rglob("*"):
        if not fp.is_file():
            continue
        if not _is_transcript_text_file(fp.name):
            continue
        name = fp.name.lower()
        include = day in name or day_compact in name
        if not include:
            try:
                local_dt = datetime.fromtimestamp(fp.stat().st_mtime, tz=get_local_tz())
                include = local_dt.date() == target_date
            except Exception:
                include = False
        if not include:
            continue
        key = str(fp)
        if key in seen:
            continue
        seen.add(key)
        out.append(fp)
    out.sort()
    return out


def _summary_files_for_day(day: str) -> tuple[list[Path], list[Path]]:
    notes_files = sorted(NOTES_DIR.rglob(f"{day}*.md"))
    ocr_files = sorted(OCR_MD_DIR.glob(f"{day}*_ocr.md"))
    if ocr_files:
        seen = {str(fp) for fp in notes_files}
        for fp in ocr_files:
            if str(fp) not in seen:
                notes_files.append(fp)
        notes_files.sort()
    transcript_files = _collect_transcript_files_for_day(day)
    if transcript_files:
        seen = {str(fp) for fp in notes_files}
        for fp in transcript_files:
            if str(fp) not in seen:
                notes_files.append(fp)
        notes_files.sort()

    day_compact = day.replace("-", "")
    news_candidates = [
        NEWS_MD_DIR / f"{day}_news.md",
        NEWS_MD_DIR / f"{day_compact}.md",
        NEWS_MD_DIR / f"{day_compact}_news.md",
    ]
    news_files = [fp for fp in news_candidates if fp.exists()]
    if not news_files:
        news_files = sorted(NEWS_MD_DIR.glob(f"{day_compact}*.md"))
    return notes_files, news_files


def _load_raw_summary_files(files: list[Path]) -> str:
    chunks: list[str] = []
    for fp in files:
        try:
            content = fp.read_text(encoding="utf-8", errors="replace").strip()
        except Exception as e:
            content = f"[read error] {e}"
        chunks.append(f"# file: {fp}\n{content or '[empty]'}")
    raw = "\n\n".join(chunks)
    return raw[:AI_SUMMARY_MAX_CHARS] if len(raw) > AI_SUMMARY_MAX_CHARS else raw


def _extract_news_urls(news_raw: str, limit: int = 20) -> list[str]:
    if not news_raw:
        return []
    candidates = re.findall(r"https?://[^\s\"'<>)]+", news_raw, flags=re.I)
    urls: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        url = item.strip().rstrip(".,;")
        if not re.match(r"^https?://", url, re.I):
            continue
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _fetch_article_text(url: str) -> str:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (summary-bot)"},
            timeout=NEWS_URL_FETCH_TIMEOUT_SECONDS,
            allow_redirects=True,
        )
        if resp.status_code >= 400:
            return ""
        raw = resp.text or ""
        if not raw:
            return ""
        raw = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
        raw = re.sub(r"(?is)<style.*?>.*?</style>", " ", raw)
        raw = re.sub(r"(?is)<[^>]+>", " ", raw)
        raw = unescape(raw)
        raw = re.sub(r"\s+", " ", raw).strip()
        if len(raw) > NEWS_URL_FETCH_MAX_CHARS:
            raw = raw[:NEWS_URL_FETCH_MAX_CHARS]
        return raw
    except Exception:
        return ""


def _summarize_news_from_urls(day: str, news_raw: str) -> str:
    urls = _extract_news_urls(news_raw, limit=max(NEWS_URL_FETCH_MAX_ARTICLES * 4, 8))
    snippets: list[str] = []
    for url in urls[:NEWS_URL_FETCH_MAX_ARTICLES]:
        text = _fetch_article_text(url)
        if not text:
            continue
        snippets.append(f"URL: {url}\n{text}")

    if not snippets:
        return "目前沒有可用的新聞 URL 內容可摘要。"

    system_prompt = (
        "Use only provided content. Do not invent facts. "
        "Output must be in Traditional Chinese."
    )
    user_prompt = (
        f"隢??{day} ??摰對??渡?????暺n"
        "?澆?閬?嚗n"
        "- ?芾撓?箔?暺n"
        "- 瘥? 1-2 ?功n"
        "- 瘥??敺?銝???URL嚗????皞?券???嚗n"
        "隢蝺券n\n"
        f"{chr(10).join(snippets)}"
    )
    out = _run_ai_chat(system_prompt, user_prompt)
    return _normalize_three_points_output(out or "")


def _clean_plain_text(text: str) -> str:
    if not text:
        return ""
    t = unescape(text)
    t = re.sub(r"(?is)<script.*?>.*?</script>", " ", t)
    t = re.sub(r"(?is)<style.*?>.*?</style>", " ", t)
    t = re.sub(r"(?is)<[^>]+>", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _domain_of(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _source_name_from_url(url: str) -> str:
    host = _domain_of(url).replace("www.", "")
    if not host:
        return "Unknown"
    mapping = {
        "axios.com": "Axios",
        "reuters.com": "Reuters",
        "bloomberg.com": "Bloomberg",
        "technews.tw": "TechNews",
        "nikkei.com": "Nikkei",
        "semianalysis.com": "SemiAnalysis",
    }
    for suffix, name in mapping.items():
        if host.endswith(suffix):
            return name
    return host.split(".")[0].capitalize()


def _pick_best_news_url(canonical_url: str, source_urls: list[str], summary_text: str) -> str:
    urls: list[str] = []
    seen: set[str] = set()
    for u in [canonical_url, *source_urls, *_extract_news_urls(summary_text, limit=12)]:
        if not u or not re.match(r"^https?://", u, re.I):
            continue
        if u in seen:
            continue
        seen.add(u)
        urls.append(u)

    if not urls:
        return ""

    def is_aggregator(url: str) -> bool:
        host = _domain_of(url)
        return (
            "news.google.com" in host
            or "google.com" in host and "/rss/articles/" in url
            or "techmeme.com" in host
        )

    for u in urls:
        if not is_aggregator(u):
            return u
    return urls[0]


def _safe_parse_iso(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def _escape_md_link_text(text: str) -> str:
    t = (text or "").replace("\\", "\\\\")
    t = t.replace("[", "\\[").replace("]", "\\]")
    return t


def _escape_md_url(url: str) -> str:
    return (url or "").replace(" ", "%20").replace(")", "%29")


def _extract_point_lines(text: str) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    for raw in text.splitlines():
        line = _clean_plain_text(_strip_markdown_noise(raw))
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+[.)]|(?:point|\u91cd\u9ede)\s*\d+\s*[:\uff1a])\s*", "", line).strip()
        if line:
            lines.append(line)
    return lines


def _fallback_three_points(text: str) -> list[str]:
    clean = _clean_plain_text(text)
    if not clean:
        return ["鞈?銝雲", "鞈?銝雲", "鞈?銝雲"]
    parts = [x.strip() for x in re.split(r"[?.!????;]\s*", clean) if x.strip()]
    points = parts[:3]
    while len(points) < 3:
        points.append("鞈?銝雲")
    return points


def _extract_note_lines(notes_raw: str, limit: int = 80) -> list[str]:
    if not notes_raw:
        return []
    lines: list[str] = []
    for raw in notes_raw.splitlines():
        line = _clean_plain_text(raw)
        if not line:
            continue
        if line.startswith("# file:"):
            continue
        if re.match(r"^\d{4}-\d{2}-\d{2}", line):
            continue
        if line in {"[empty]", "[no notes files]"}:
            continue
        line = re.sub(r"^\s*-\s*\[[^\]]+\]\s*\([^)]*\)\s*[^|]*\|\s*[^:]+:\s*", "", line).strip()
        line = re.sub(r"^\s*-\s*", "", line).strip()
        if not line:
            continue
        if line.startswith("/"):
            continue
        lines.append(line)
        if len(lines) >= limit:
            break
    return lines


def _extract_digest_tags(text: str, min_tags: int = 5, max_tags: int = 10) -> list[str]:
    def norm_token(token: str) -> str:
        t = _clean_plain_text(token).strip().lstrip("#")
        t = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_+-]", "", t)
        return t

    found: list[str] = []
    seen: set[str] = set()

    for m in re.findall(r"#([0-9A-Za-z\u4e00-\u9fff_+-]{2,24})", text or ""):
        t = norm_token(m)
        if not t:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(f"#{t}")
        if len(found) >= max_tags:
            return found

    clean = _clean_plain_text(text or "").lower()
    en_stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "over", "after", "before",
        "about", "today", "latest", "notes", "summary", "agent", "agents", "model", "models",
    }
    for m in re.findall(r"\b[a-z][a-z0-9+\-]{2,20}\b", clean):
        if m in en_stop:
            continue
        key = m.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(f"#{m}")
        if len(found) >= max_tags:
            return found

    for m in re.findall(r"[\u4e00-\u9fff]{2,6}", _clean_plain_text(text or "")):
        key = m.lower()
        if key in seen:
            continue
        seen.add(key)
        found.append(f"#{m}")
        if len(found) >= max_tags:
            break

    while len(found) < min_tags:
        filler = f"#Note{len(found) + 1}"
        if filler.lower() not in seen:
            seen.add(filler.lower())
            found.append(filler)
    return found[:max_tags]


def _parse_note_digest_ai(text: str) -> tuple[list[tuple[str, list[str]]], list[str]]:
    items: list[tuple[str, list[str]]] = []
    tags: list[str] = []
    cur_title = ""
    cur_points: list[str] = []

    def flush_item() -> None:
        nonlocal cur_title, cur_points
        if not cur_title and cur_points:
            cur_title = cur_points[0][:40]
        if not cur_title:
            return
        dedup_points: list[str] = []
        seen_points: set[str] = set()
        for p in cur_points:
            pv = p.strip()
            if not pv:
                continue
            k = pv.lower()
            if k in seen_points:
                continue
            seen_points.add(k)
            dedup_points.append(pv)
        if len(dedup_points) < 3:
            return
        items.append((cur_title.strip(), dedup_points[:5]))
        cur_title = ""
        cur_points = []

    title_re = re.compile(r"^(?:title|\u6a19\u984c)\s*[:\uff1a]\s*(.+)$", re.I)
    point_re = re.compile(r"^(?:point\s*[1-5]|\u91cd\u9ede\s*[1-5])\s*[:\uff1a]\s*(.+)$", re.I)
    tags_re = re.compile(r"^(?:tag|tags|\u6a19\u7c64)\s*[:\uff1a]\s*(.+)$", re.I)
    heading_re = re.compile(r"^(?:#{1,3}\s*|\d+[.)]\s+)(.+)$")
    bullet_re = re.compile(r"^\s*[-*•]\s*(.+)$")

    for raw in (text or "").splitlines():
        line = _clean_plain_text(raw).strip()
        if not line:
            continue
        if line.startswith("---"):
            flush_item()
            continue

        m_title = title_re.match(line)
        if m_title:
            flush_item()
            cur_title = m_title.group(1).strip()
            continue

        m_point = point_re.match(line)
        if m_point:
            cur_points.append(m_point.group(1).strip())
            continue

        m_tags = tags_re.match(line)
        if m_tags:
            tags = _extract_digest_tags(m_tags.group(1), min_tags=5, max_tags=10)
            continue

        m_heading = heading_re.match(line)
        if m_heading:
            flush_item()
            cur_title = m_heading.group(1).strip()
            continue

        m_bullet = bullet_re.match(line)
        if m_bullet and cur_title:
            cur_points.append(m_bullet.group(1).strip())
            continue

    flush_item()
    if not tags:
        tags = _extract_digest_tags(text or "", min_tags=5, max_tags=10)
    return items[:NOTE_DIGEST_MAX_ITEMS], tags


def build_note_digest(day: str) -> list[str]:
    notes_files, _ = _summary_files_for_day(day)
    notes_raw = _load_raw_summary_files(notes_files)
    if not notes_raw:
        return [f"{day} 筆記/逐字稿摘要", "當日沒有可用筆記或逐字稿資料。"]

    note_items: list[tuple[str, list[str]]] = []
    tags: list[str] = []

    if AI_SUMMARY_ENABLED:
        system_prompt = (
            "Use only provided content. Do not invent facts. "
            "Output must be in Traditional Chinese."
        )
        user_prompt = (
            f"Summarize notes and transcripts for {day} with scan-friendly structure.\n"
            f"- Create 3 to {NOTE_DIGEST_MAX_ITEMS} topics.\n"
            "- Format each topic exactly:\n"
            "Title: <max 22 chars>\n"
            "Point1: ...\n"
            "Point2: ...\n"
            "Point3: ...\n"
            "Point4: ... (optional)\n"
            "Point5: ... (optional)\n"
            "---\n"
            "- Last line must be: tags: #tag1 #tag2 ... (5 to 10 tags)\n"
            "- No links, no YAML/HTML, no duplicated fields.\n\n"
            f"{notes_raw}"
        )
        out = _run_ai_chat(system_prompt, user_prompt)
        ai_items, ai_tags = _parse_note_digest_ai(out or "")
        note_items = ai_items
        tags = ai_tags

    if not note_items:
        lines = _extract_note_lines(notes_raw, limit=120)
        if not lines:
            return [f"{day} 筆記/逐字稿摘要", "沒有可用的筆記或逐字稿文字。"]
        for line in lines[:NOTE_DIGEST_MAX_ITEMS]:
            title = re.sub(r"https?://\S+", "", line).strip()
            if len(title) > 28:
                title = f"{title[:28].rstrip()}..."
            sentence_parts = [x.strip() for x in re.split(r"[。！？!?]\s*", line) if x.strip()]
            points = sentence_parts[:5]
            while len(points) < 3:
                points.append("待補充細節")
            note_items.append((title or "當日筆記", points[:5]))
        tags = _extract_digest_tags("\n".join(lines), min_tags=5, max_tags=10)

    point_label = "\u91cd\u9ede"
    tag_label = "標籤"
    out_lines: list[str] = [f"{day} 筆記/逐字稿摘要"]
    for title, points in note_items[:NOTE_DIGEST_MAX_ITEMS]:
        out_lines.append(title)
        for idx, p in enumerate(points[:5], start=1):
            out_lines.append(f"{point_label}{idx}\uff1a{p}")
        out_lines.append("")
    out_lines.append(f"{tag_label}\uff1a{' '.join(tags[:10])}")
    return out_lines


def build_note_digest_recent(end_day: str, days: int = 3) -> list[str]:
    days = max(1, min(7, int(days)))
    end_dt = datetime.strptime(end_day, "%Y-%m-%d").replace(tzinfo=get_local_tz())
    start_dt = (end_dt - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    day_list = [(start_dt + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]

    raw_blocks: list[str] = []
    for d in day_list:
        notes_files, _ = _summary_files_for_day(d)
        day_raw = _load_raw_summary_files(notes_files)
        if day_raw:
            raw_blocks.append(f"[{d}]\n{day_raw}")

    header = f"{start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} 筆記/逐字稿摘要"
    if not raw_blocks:
        return [header, "指定期間沒有可用筆記或逐字稿資料。"]

    merged_raw = "\n\n".join(raw_blocks)
    note_items: list[tuple[str, list[str]]] = []
    tags: list[str] = []

    if AI_SUMMARY_ENABLED:
        system_prompt = (
            "Use only provided content. Do not invent facts. "
            "Output must be in Traditional Chinese."
        )
        user_prompt = (
            f"Summarize notes and transcripts from {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')} with scan-friendly structure.\n"
            f"- Create 4 to {max(NOTE_DIGEST_MAX_ITEMS, 6)} topics.\n"
            "- Format each topic exactly:\n"
            "Title: <max 22 chars>\n"
            "Point1: ...\n"
            "Point2: ...\n"
            "Point3: ...\n"
            "Point4: ... (optional)\n"
            "Point5: ... (optional)\n"
            "---\n"
            "- Last line must be: tags: #tag1 #tag2 ... (5 to 10 tags)\n"
            "- No links, no YAML/HTML, no duplicated fields.\n\n"
            f"{merged_raw}"
        )
        out = _run_ai_chat(system_prompt, user_prompt)
        ai_items, ai_tags = _parse_note_digest_ai(out or "")
        note_items = ai_items[: max(NOTE_DIGEST_MAX_ITEMS, 6)]
        tags = ai_tags

    if not note_items:
        lines = _extract_note_lines(merged_raw, limit=180)
        if not lines:
            return [header, "沒有可用的筆記或逐字稿文字。"]
        limit_items = max(NOTE_DIGEST_MAX_ITEMS, 6)
        for line in lines[:limit_items]:
            title = re.sub(r"https?://\S+", "", line).strip()
            if len(title) > 28:
                title = f"{title[:28].rstrip()}..."
            sentence_parts = [x.strip() for x in re.split(r"[。！？!?]\s*", line) if x.strip()]
            points = sentence_parts[:5]
            while len(points) < 3:
                points.append("待補充細節")
            note_items.append((title or "近期筆記", points[:5]))
        tags = _extract_digest_tags("\n".join(lines), min_tags=5, max_tags=10)

    point_label = "\u91cd\u9ede"
    tag_label = "標籤"
    out_lines: list[str] = [header]
    for title, points in note_items[: max(NOTE_DIGEST_MAX_ITEMS, 6)]:
        out_lines.append(title)
        for idx, p in enumerate(points[:5], start=1):
            out_lines.append(f"{point_label}{idx}\uff1a{p}")
        out_lines.append("")
    out_lines.append(f"{tag_label}\uff1a{' '.join(tags[:10])}")
    return out_lines


def _summarize_single_news_item(
    day: str,
    title: str,
    source: str,
    url: str,
    summary: str,
    *,
    use_ai: bool,
    fetch_article: bool,
) -> list[str]:
    # Fast path: skip network+LLM for most items.
    if not use_ai:
        return _fallback_three_points(summary or title)

    article_text = _fetch_article_text(url) if (fetch_article and url) else ""
    context = "\n\n".join(
        [
            f"Title: {title}",
            f"Source: {source}",
            f"URL: {url or 'N/A'}",
            f"Feed Summary: {_clean_plain_text(summary) or 'N/A'}",
            f"Article Content: {article_text or 'N/A'}",
        ]
    )

    if AI_SUMMARY_ENABLED:
        system_prompt = (
            "Use only provided content. Do not invent facts. "
            "Output must be in Traditional Chinese."
        )
        user_prompt = (
            f"For news date {day}, summarize this single item into exactly three key points in Traditional Chinese.\n"
            "Output rules:\n"
            "- Exactly 3 lines\n"
            "- One concise sentence per line\n"
            "- No markdown markers\n\n"
            f"{context}"
        )
        out = _run_ai_chat(system_prompt, user_prompt)
        lines = _extract_point_lines(out or "")
        if len(lines) >= 3:
            return lines[:3]

    return _fallback_three_points(summary or context)


def build_news_digest(day: str) -> list[str]:
    day_compact = day.replace("-", "")
    with sqlite3.connect(DB_PATH) as conn:
        clusters = conn.execute(
            """
            SELECT id, cluster_seq, canonical_title, canonical_url, canonical_source,
                   canonical_published_at, canonical_summary, cluster_date
            FROM news_clusters
            WHERE cluster_date IN (?, ?)
            ORDER BY canonical_published_at ASC, cluster_seq ASC
            LIMIT ?
            """,
            (day, day_compact, NEWS_DIGEST_MAX_ITEMS),
        ).fetchall()

        if not clusters:
            return [f"# {day} News Digest", "---", "\u4eca\u65e5\u7121\u53ef\u7528\u65b0\u805e\u8cc7\u6599\u3002"]

        lines: list[str] = [f"# {day} News Digest", "---"]
        for idx, (
            cluster_id,
            _cluster_seq,
            canonical_title,
            canonical_url,
            canonical_source,
            canonical_published_at,
            canonical_summary,
            _cluster_date,
        ) in enumerate(clusters, start=1):
            src_rows = conn.execute(
                """
                SELECT DISTINCT source, url
                FROM news_items
                WHERE cluster_id = ?
                ORDER BY source ASC
                """,
                (cluster_id,),
            ).fetchall()
            source_urls = [str(u or "").strip() for _, u in src_rows if u]

            title = _clean_plain_text(str(canonical_title or "")) or "Untitled"
            summary_text = _clean_plain_text(str(canonical_summary or ""))
            best_url = _pick_best_news_url(str(canonical_url or ""), source_urls, summary_text)

            source_name = _clean_plain_text(str(canonical_source or "")) or "Unknown"
            if "techmeme" in source_name.lower() and best_url and "techmeme.com" not in _domain_of(best_url):
                source_name = f"{_source_name_from_url(best_url)} (via Techmeme)"

            published_dt = _safe_parse_iso(str(canonical_published_at or ""))
            hhmm = published_dt.astimezone(get_local_tz()).strftime("%H:%M") if published_dt else "--:--"
            use_ai = AI_SUMMARY_ENABLED and idx <= max(0, NEWS_DIGEST_AI_ITEMS)
            fetch_article = idx <= max(0, NEWS_DIGEST_FETCH_ARTICLE_ITEMS)
            points = _summarize_single_news_item(
                day,
                title,
                source_name,
                best_url,
                summary_text,
                use_ai=use_ai,
                fetch_article=fetch_article,
            )

            badge = f"{idx}."
            if best_url:
                md_title = _escape_md_link_text(title)
                md_url = _escape_md_url(best_url)
                lines.append(f"## {badge} [{md_title}]({md_url})")
            else:
                lines.append(f"## {badge} {title}")
            lines.append(f"\u767c\u5e03\u6642\u9593\uff1a{hhmm}\uff5c\u4f86\u6e90\uff1a{source_name}")
            lines.append("")
            lines.append(f"\u91cd\u9ede1\uff1a{points[0]}")
            lines.append(f"\u91cd\u9ede2\uff1a{points[1]}")
            lines.append(f"\u91cd\u9ede3\uff1a{points[2]}")
            lines.append("")
            lines.append("---")
    return lines


def build_news_digest_recent(end_day: str, days: int = 3) -> list[str]:
    days = max(1, min(7, int(days)))
    end_dt = datetime.strptime(end_day, "%Y-%m-%d").replace(tzinfo=get_local_tz())
    start_dt = (end_dt - timedelta(days=days - 1)).replace(hour=0, minute=0, second=0, microsecond=0)
    end_bound = end_dt.replace(hour=23, minute=59, second=59, microsecond=999999)

    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, cluster_seq, canonical_title, canonical_url, canonical_source,
                   canonical_published_at, canonical_summary, cluster_date
            FROM news_clusters
            WHERE canonical_published_at IS NOT NULL
            ORDER BY canonical_published_at DESC
            LIMIT 400
            """
        ).fetchall()

        clusters: list[tuple] = []
        for row in rows:
            ts = _safe_parse_iso(str(row[5] or ""))
            if not ts:
                continue
            ts_local = ts.astimezone(get_local_tz())
            if start_dt <= ts_local <= end_bound:
                clusters.append(row)
            if len(clusters) >= max(NEWS_DIGEST_MAX_ITEMS * days, NEWS_DIGEST_MAX_ITEMS):
                break

        header = f"# {start_dt.strftime('%Y-%m-%d')} ~ {end_dt.strftime('%Y-%m-%d')} News Digest"
        if not clusters:
            return [header, "---", "\u6307\u5b9a\u671f\u9593\u7121\u53ef\u7528\u65b0\u805e\u8cc7\u6599\u3002"]

        lines: list[str] = [header, "---"]
        for idx, (
            cluster_id,
            _cluster_seq,
            canonical_title,
            canonical_url,
            canonical_source,
            canonical_published_at,
            canonical_summary,
            _cluster_date,
        ) in enumerate(clusters, start=1):
            src_rows = conn.execute(
                """
                SELECT DISTINCT source, url
                FROM news_items
                WHERE cluster_id = ?
                ORDER BY source ASC
                """,
                (cluster_id,),
            ).fetchall()
            source_urls = [str(u or "").strip() for _, u in src_rows if u]

            title_text = _clean_plain_text(str(canonical_title or "")) or "Untitled"
            summary_text = _clean_plain_text(str(canonical_summary or ""))
            best_url = _pick_best_news_url(str(canonical_url or ""), source_urls, summary_text)

            source_name = _clean_plain_text(str(canonical_source or "")) or "Unknown"
            if "techmeme" in source_name.lower() and best_url and "techmeme.com" not in _domain_of(best_url):
                source_name = f"{_source_name_from_url(best_url)} (via Techmeme)"

            published_dt = _safe_parse_iso(str(canonical_published_at or ""))
            ts_text = published_dt.astimezone(get_local_tz()).strftime("%m-%d %H:%M") if published_dt else "--:--"
            use_ai = AI_SUMMARY_ENABLED and idx <= max(0, NEWS_DIGEST_AI_ITEMS)
            fetch_article = idx <= max(0, NEWS_DIGEST_FETCH_ARTICLE_ITEMS)
            points = _summarize_single_news_item(
                end_day,
                title_text,
                source_name,
                best_url,
                summary_text,
                use_ai=use_ai,
                fetch_article=fetch_article,
            )

            badge = f"{idx}."
            if best_url:
                md_title = _escape_md_link_text(title_text)
                md_url = _escape_md_url(best_url)
                lines.append(f"## {badge} [{md_title}]({md_url})")
            else:
                lines.append(f"## {badge} {title_text}")
            lines.append(f"\u767c\u5e03\u6642\u9593\uff1a{ts_text}\uff5c\u4f86\u6e90\uff1a{source_name}")
            lines.append("")
            lines.append(f"\u91cd\u9ede1\uff1a{points[0]}")
            lines.append(f"\u91cd\u9ede2\uff1a{points[1]}")
            lines.append(f"\u91cd\u9ede3\uff1a{points[2]}")
            lines.append("")
            lines.append("---")
    return lines


def summary_weekly(day: str) -> list[str]:
    end_day = datetime.strptime(day, "%Y-%m-%d")
    days = [(end_day - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(6, -1, -1)]

    daily_sections: list[str] = []
    for d in days:
        parts = summary_ai(d, scope="all")
        if len(parts) >= 6 and parts[0].endswith("(AI)"):
            daily_sections.append(
                "\n".join(
                    [
                        f"## {d}",
                        parts[1],
                        parts[2],
                        parts[4],
                        parts[5],
                    ]
                )
            )
            continue

        fallback = merge_all(d)
        fallback_text = "\n".join(fallback)
        daily_sections.append(f"## {d}\n{fallback_text[:600]}")

    context_text = "\n\n".join(daily_sections)
    if len(context_text) > AI_SUMMARY_MAX_CHARS:
        context_text = context_text[:AI_SUMMARY_MAX_CHARS]

    system_prompt = (
        "Use only provided content. Do not invent facts. "
        "Output must be in Traditional Chinese."
    )
    weekly_prompt = (
        f"隢?撠??{day} ?? 7 憭拙摰寧?粹望?閬n"
        "頛詨?澆?嚗n"
        "A. ?啗?\n"
        "- 5-10 暺???頞典???萄?貉?鈭辣嚗?雯?隢??n\n"
        "B. 蝑?\n"
        "- 5-10 暺?瘙箇???颲艾◢?芾?銝梯??n\n"
        "隢蝺券?閮n\n"
        f"{context_text}"
    )
    weekly = _run_ai_chat(system_prompt, weekly_prompt)

    if not weekly:
        return [f"{day} summary_weekly", context_text]
    return [f"{day} summary_weekly (AI)", weekly]


def summary_ai(day: str, scope: str = "all") -> list[str]:
    scope = (scope or "all").strip().lower()
    if scope == "notes":
        scope = "note"

    notes_files, news_files = _summary_files_for_day(day)
    news_raw = _load_raw_summary_files(news_files)
    notes_raw = _load_raw_summary_files(notes_files)

    if not AI_SUMMARY_ENABLED:
        if scope == "news":
            return [f"{day} summary news", news_raw or "[no news files]"]
        if scope == "note":
            return build_note_digest(day)
        return merge_all(day)

    system_prompt = (
        "Use only provided content. Do not invent facts. "
        "Output must be in Traditional Chinese."
    )

    a_prompt = (
        f"Summarize news for {day}. Output 8-12 items. If source is limited, output all available items and prioritize at least 6.\n"
        "Each item must contain title and URL. If URL is missing, write URL: ?.\n"
        "Do NOT use title(url) inline parenthesis style.\n"
        "Do not invent facts.\n\n"
        f"{news_raw or '[no news files]'}"
    )
    b_prompt = (
        f"Summarize notes and transcripts for {day}. Output 8-15 key points.\n"
        "Focus on decisions, TODOs, risks, and next actions.\n"
        "Do NOT use markdown emphasis symbols like **.\n"
        "Do not invent facts.\n\n"
        f"{notes_raw or '[no notes files]'}"
    )

    if scope == "news":
        return build_scoped_summary(day, "news")
    if scope == "note":
        return build_scoped_summary(day, "note")

    a = _run_ai_chat(system_prompt, a_prompt) if scope == "all" else None
    b = _run_ai_chat(system_prompt, b_prompt) if scope == "all" else None


    if not a and not b:
        return merge_all(day)

    a_fmt = _normalize_news_output(a or "")
    b_fmt = _normalize_notes_output(b or "")
    return [
        f"{day} summary (AI)",
        "A. ?啗?",
        a_fmt,
        "",
        "B. 蝑?",
        b_fmt,
    ]


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

    md_path = NEWS_MD_DIR / f"{cluster_date}_news.md"
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
    if sub == "sources":
        srcs = get_news_rss_urls()
        return ["News sources:\n" + "\n".join(srcs)]
    if sub == "search":
        if len(tokens) < 3:
            return ["Usage: /news search <keywords>"]
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
        fetch_and_store_news()
        end_day = datetime.now(tz=get_local_tz()).strftime("%Y-%m-%d")
        if len(tokens) >= 3 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[2]):
            end_day = tokens[2]
        return ["\n".join(build_scoped_summary(end_day, "news", recent_days=3))]
    if sub == "help":
        return [
            "News commands: /news latest [N], /news search <keywords>, /news sources, /news debug"
        ]
    return ["Unknown /news subcommand. Use /news help."]


NEWS_SHORTCUTS = {
    "/news_latest": "/news latest",
    "/news_sources": "/news sources",
    "/news_debug": "/news debug",
    "/news_help": "/news help",
    "/status": "/status",
}


def set_telegram_commands() -> None:
    commands = [
        {"command": "summary_note", "description": "3-day note digest"},
        {"command": "news_latest", "description": "Latest digest (3 days)"},
        {"command": "news_sources", "description": "List news sources"},
        {"command": "news_debug", "description": "Debug ingestion"},
        {"command": "news_help", "description": "News command help"},
        {"command": "status", "description": "Bot health status"},
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


def send_message_sync(chat_id: str, text: str, parse_mode: str | None = None) -> None:
    payload = {"chat_id": chat_id, "text": text}
    if parse_mode:
        payload["parse_mode"] = parse_mode
    resp = requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)
    if resp.status_code >= 300 and parse_mode:
        resp = requests.post(f"{TELEGRAM_API}/sendMessage", json={"chat_id": chat_id, "text": text}, timeout=10)
    print(f"sendMessage status={resp.status_code} body={resp.text}")


def _resolve_ai_model_name() -> str:
    provider = AI_SUMMARY_PROVIDER
    if provider == "openai":
        return OPENAI_MODEL
    if provider == "gemini":
        return GEMINI_MODEL
    if provider == "anthropic":
        return ANTHROPIC_MODEL
    if provider == "huggingface":
        return HUGGINGFACE_MODEL
    if provider == "ollama":
        return OLLAMA_MODEL
    return "unknown"


def _ollama_health() -> str:
    if AI_SUMMARY_PROVIDER != "ollama":
        return "n/a"
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code != 200:
            return f"http_{resp.status_code}"
        data = resp.json() if resp.content else {}
        models = data.get("models") or []
        target = (OLLAMA_MODEL or "").strip().lower()
        found = False
        for model in models:
            name = str(model.get("name") or "").strip().lower()
            if name == target:
                found = True
                break
        return "ready" if found else "model_missing"
    except Exception as e:
        return f"error:{type(e).__name__}"


def build_status_report() -> str:
    now = datetime.now(tz=get_local_tz())
    day = now.strftime("%Y-%m-%d")
    start_3d = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    news_24h = 0
    notes_3d = 0
    try:
        with sqlite3.connect(DB_PATH) as conn:
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM news_clusters
                WHERE canonical_published_at IS NOT NULL
                  AND canonical_published_at >= ?
                """,
                ((now - timedelta(hours=24)).isoformat(),),
            ).fetchone()
            news_24h = int(row[0] if row else 0)
    except Exception:
        news_24h = -1

    try:
        for d in range(3):
            target_day = (start_3d + timedelta(days=d)).strftime("%Y-%m-%d")
            day_files, _ = _summary_files_for_day(target_day)
            notes_3d += len(day_files)
    except Exception:
        notes_3d = -1

    lines = [
        f"status time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"ai summary: {'on' if AI_SUMMARY_ENABLED else 'off'}",
        f"ai provider/model: {AI_SUMMARY_PROVIDER}/{_resolve_ai_model_name()}",
        f"ollama health: {_ollama_health()}",
        f"db path exists: {'yes' if DB_PATH.exists() else 'no'}",
        f"news clusters (24h): {news_24h}",
        f"note markdown files (3d): {notes_3d}",
        f"dropbox sync: {'on' if DROPBOX_SYNC_ENABLED else 'off'} ({DROPBOX_SYNC_TIME} {DROPBOX_SYNC_TZ_NAME})",
        f"news push: {'on' if NEWS_PUSH_ENABLED else 'off'} (interval={NEWS_FETCH_INTERVAL_MINUTES}m)",
        f"today: {day}",
    ]
    return "\n".join(lines)


def push_news_to_subscribers() -> None:
    if not NEWS_PUSH_ENABLED:
        return
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


def start_dropbox_sync_thread() -> None:
    if not DROPBOX_SYNC_ENABLED:
        return
    if dropbox is None:
        print("[WARN] Dropbox sync disabled: dropbox SDK not installed.")
        return
    if not (_dropbox_access_token or _dropbox_can_refresh_token()):
        print(
            "[WARN] Dropbox sync disabled: set DROPBOX_ACCESS_TOKEN "
            "or (DROPBOX_REFRESH_TOKEN + DROPBOX_APP_KEY + DROPBOX_APP_SECRET)."
        )
        return

    run_hour, run_minute = parse_hhmm(DROPBOX_SYNC_TIME, default_hour=0, default_minute=10)

    def loop():
        startup_done = False
        last_daily_run = ""
        while True:
            try:
                now = datetime.now(tz=get_dropbox_sync_tz())
                day_key = now.strftime("%Y-%m-%d")
                if DROPBOX_SYNC_ON_STARTUP and not startup_done:
                    run_dropbox_sync(full_scan=True)
                    startup_done = True
                if now.hour == run_hour and now.minute == run_minute and last_daily_run != day_key:
                    run_dropbox_sync(full_scan=False)
                    last_daily_run = day_key
            except Exception as e:
                print(f"[WARN] Dropbox sync worker error: {e}")
            time.sleep(60)

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
                    asyncio.run(process_telegram_update(update))
                except Exception as e:
                    print(f"[WARN] telegram update process error: {e}")
        except Exception as e:
            print(f"[ERROR] Telegram polling error: {e}")
            time.sleep(2)


def start_telegram_polling_thread() -> None:
    if not TELEGRAM_LONG_POLLING:
        return
    t = threading.Thread(target=telegram_polling_loop, daemon=True)
    t.start()


async def process_telegram_update(update: dict) -> None:
    message = update.get("message") or update.get("edited_message") or {}
    if not message:
        return

    chat = message.get("chat", {})
    chat_type = chat.get("type")
    chat_id = chat.get("id")
    chat_title = chat.get("title") or chat.get("username") or ""
    text = (message.get("text") or message.get("caption") or "").strip()

    sender = message.get("from", {})
    if sender.get("is_bot"):
        return

    user_id = str(sender.get("id")) if sender.get("id") else ""
    user_name = (
        sender.get("username")
        or " ".join(filter(None, [sender.get("first_name"), sender.get("last_name")])).strip()
        or "unknown"
    )

    msg_ts = None
    if message.get("date"):
        msg_ts = datetime.fromtimestamp(message.get("date"))

    is_group = chat_type in {"group", "supergroup"}
    is_private = chat_type == "private"
    if is_group and ALLOWED_GROUPS:
        allowed = {x.strip() for x in ALLOWED_GROUPS if x.strip()}
        if str(chat_id) not in allowed and chat_title not in allowed:
            return

    doc = message.get("document") or {}
    is_image_doc = doc.get("mime_type", "").startswith("image/")
    has_image = bool(message.get("photo") or is_image_doc)

    if (is_group or is_private) and chat_id and text:
        store_message("telegram", str(chat_id), chat_title or chat_type or "", user_id, user_name, text, msg_ts)
        append_markdown("telegram", chat_title or ("private" if is_private else str(chat_id)), user_name, text, msg_ts)

    if is_private and chat_id and text and not has_image:
        text_stripped = text.strip()
        cmd_text = text_stripped
        for shortcut, full_cmd in NEWS_SHORTCUTS.items():
            if text_stripped.lower().startswith(shortcut):
                cmd_text = full_cmd + text_stripped[len(shortcut) :]
                break

        replies, parse_mode, disable_preview = route_user_text_command(cmd_text, str(chat_id))
        for reply in replies:
            for chunk in _chunk_text_for_telegram(reply):
                await send_message(
                    chat_id,
                    chunk,
                    parse_mode=parse_mode,
                    disable_web_page_preview=disable_preview,
                )
        return

    if (is_private or is_group) and chat_id and has_image:
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
                    img_dir = INBOX_IMAGES_DIR / day
                    img_dir.mkdir(parents=True, exist_ok=True)
                    suffix = Path(file_path).suffix or ".jpg"
                    filename = f"{chat_id}_{msg_id}_{file_unique_id}{suffix}"
                    out_path = img_dir / filename
                    out_path.write_bytes(img_resp.content)

                    ocr_ok = False
                    md_saved = False
                    ocr_text = ""
                    try:
                        ocr_text = extract_text_from_image(img_resp.content)
                        if ocr_text.strip():
                            ocr_ok = True
                        else:
                            ocr_text = "[OCR failed] no text detected"
                    except Exception as e:
                        ocr_text = f"[OCR failed] {e}"
                        print(f"[WARN] OCR failed for {out_path}: {e}")
                    try:
                        append_ocr_markdown(str(chat_id), msg_id, out_path, ocr_text, msg_ts)
                        md_saved = True
                    except Exception as e:
                        print(f"[WARN] OCR markdown save failed for {out_path}: {e}")

                    is_ocr_success = ocr_ok and md_saved
                    if is_private:
                        if is_ocr_success:
                            await send_message(chat_id, "img saved, ocr succeed")
                        else:
                            await send_message(chat_id, "img saved, but ocr failed")
                    else:
                        group_status = "ocr succeed" if is_ocr_success else "ocr failed"
                        print(f"[INFO] Telegram group image saved: {out_path} ({group_status})")
                    return
                except Exception as e:
                    print(f"[WARN] Telegram image save failed: {e}")

        if is_private:
            await send_message(chat_id, "Image save failed. Please retry.")
        return


@app.post("/telegram")
async def telegram_webhook(request: Request):
    update = await request.json()
    await process_telegram_update(update)
    return JSONResponse({"ok": True})


def start_slack_socket_mode() -> None:
    if not (SLACK_BOT_TOKEN and SLACK_APP_TOKEN and SLACK_USER_ID):
        print("Slack tokens/user not set; skipping Slack Socket Mode.")
        return

    slack_app = SlackApp(token=SLACK_BOT_TOKEN)
    if SLACK_DEBUG:
        print("[SLACK] Socket Mode starting...")

    @slack_app.command("/note")
    def handle_slack_note(ack, respond, command):
        ack()
        text = (command.get("text") or "").strip()
        if not text:
            respond("Usage: /note <text>")
            return
        channel_id = command.get("channel_id", "")
        user_id = command.get("user_id", "")
        user_name = command.get("user_name") or user_id
        msg_ts = None
        if command.get("command_ts"):
            try:
                msg_ts = datetime.fromtimestamp(float(command.get("command_ts")))
            except Exception:
                msg_ts = None
        ok, msg = process_slack_note(channel_id, user_id, user_name, text, msg_ts)
        respond(msg)

    @slack_app.command("/summary_news")
    def handle_slack_summary_news(ack, respond, command):
        ack()
        text = (command.get("text") or "").strip()
        day = text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text or "") else datetime.now().strftime("%Y-%m-%d")
        parts = build_scoped_summary(day, "news")
        respond("\n".join(parts))

    @slack_app.command("/summary_note")
    def handle_slack_summary_note(ack, respond, command):
        ack()
        text = (command.get("text") or "").strip()
        day = text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text or "") else datetime.now().strftime("%Y-%m-%d")
        parts = build_scoped_summary(day, "note")
        respond("\n".join(parts))

    @slack_app.command("/status")
    def handle_slack_status(ack, respond, command):
        ack()
        respond(build_status_report())

    @slack_app.command("/summary_weekly")
    def handle_slack_summary_weekly(ack, respond, command):
        ack()
        text = (command.get("text") or "").strip()
        day = text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text or "") else datetime.now().strftime("%Y-%m-%d")
        parts = summary_weekly(day)
        respond("\n".join(parts))

    @slack_app.event("message")
    def handle_slack_message(event, say):
        if SLACK_DEBUG:
            print(f"[SLACK] event received: keys={list(event.keys())}")
        if event.get("subtype"):
            if SLACK_DEBUG:
                print("[SLACK] ignored: subtype present")
            return
        if event.get("bot_id"):
            if SLACK_DEBUG:
                print("[SLACK] ignored: bot message")
            return
        channel_type = event.get("channel_type")
        if channel_type and channel_type != "im":
            if SLACK_DEBUG:
                print(f"[SLACK] ignored: channel_type={channel_type}")
            return
        if event.get("user") != SLACK_USER_ID:
            if SLACK_DEBUG:
                print(f"[SLACK] ignored: user mismatch {event.get('user')}")
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

        # DM text commands fallback (when slash commands are not configured)
        if text.startswith("/note "):
            payload = text[len("/note ") :].strip()
            if payload:
                channel_id = event.get("channel", "")
                user_id = event.get("user", "")
                user_name = user_id
                ok, msg = process_slack_note(channel_id, user_id, user_name, payload, msg_ts)
                say(msg)
                return
        if text.startswith("/summary_weekly"):
            tokens = text.split()
            day = tokens[1] if len(tokens) > 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[1]) else datetime.now().strftime("%Y-%m-%d")
            parts = summary_weekly(day)
            say("\n".join(parts))
            return
        if text.startswith("/summary_news"):
            tokens = text.split()
            day = tokens[1] if len(tokens) > 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[1]) else datetime.now().strftime("%Y-%m-%d")
            parts = build_scoped_summary(day, "news")
            say("\n".join(parts))
            return
        if text.startswith("/summary_note"):
            tokens = text.split()
            day = tokens[1] if len(tokens) > 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[1]) else datetime.now().strftime("%Y-%m-%d")
            parts = build_scoped_summary(day, "note")
            say("\n".join(parts))
            return
        if text.startswith("/status"):
            say(build_status_report())
            return

        channel_id = event.get("channel", "")
        user_id = event.get("user", "")
        user_name = user_id
        store_message("slack", channel_id, channel_id, user_id, user_name, text, msg_ts)
        append_markdown("slack", channel_id, user_name, text, msg_ts)

    handler = SocketModeHandler(slack_app, SLACK_APP_TOKEN)
    handler.connect()
    Event().wait()


def start_slack_thread() -> None:
    t = threading.Thread(target=start_slack_socket_mode, daemon=True)
    t.start()


set_telegram_commands()
start_news_thread()
start_slack_thread()
start_dropbox_sync_thread()
start_telegram_polling_thread()
