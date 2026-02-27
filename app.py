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
import uuid
import traceback
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


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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
TELEGRAM_FILE_FETCH_RETRIES = max(1, int(os.getenv("TELEGRAM_FILE_FETCH_RETRIES", "4")))
TELEGRAM_FILE_FETCH_CONNECT_TIMEOUT = max(1, int(os.getenv("TELEGRAM_FILE_FETCH_CONNECT_TIMEOUT", "5")))
TELEGRAM_FILE_FETCH_READ_TIMEOUT = max(3, int(os.getenv("TELEGRAM_FILE_FETCH_READ_TIMEOUT", "12")))
TELEGRAM_FILE_FETCH_RETRY_DELAY_SECONDS = max(
    0.1, float(os.getenv("TELEGRAM_FILE_FETCH_RETRY_DELAY_SECONDS", "0.25"))
)

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "read")).resolve()
DB_PATH = DATA_DIR / "messages.sqlite"
NEWS_MD_DIR = DATA_DIR / "news"
NOTES_DIR = DATA_DIR / "notes"
TELEGRAM_MD_DIR = NOTES_DIR / "telegram"
INBOX_IMAGES_DIR = DATA_DIR / "images"
TRANSCRIPTS_DIR = DATA_DIR / "_runtime" / "transcribe"
TRANSCRIPTS_TMP_DIR = DATA_DIR / "_runtime" / "transcribe_tmp"
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
NEWS_ENABLED = os.getenv("NEWS_ENABLED", "1").lower() in {"1", "true", "yes"}
NEWS_FETCH_INTERVAL_MINUTES = int(os.getenv("NEWS_FETCH_INTERVAL_MINUTES", "180"))
NEWS_PUSH_MAX_ITEMS = int(os.getenv("NEWS_PUSH_MAX_ITEMS", "10"))
NEWS_PUSH_ENABLED = os.getenv("NEWS_PUSH_ENABLED", "0").lower() in {"1", "true", "yes"}
NEWS_GNEWS_QUERY = os.getenv("NEWS_GNEWS_QUERY", "site:reuters.com semiconductors technology")
NEWS_GNEWS_HL = os.getenv("NEWS_GNEWS_HL", "en-US")
NEWS_GNEWS_GL = os.getenv("NEWS_GNEWS_GL", "US")
NEWS_GNEWS_CEID = os.getenv("NEWS_GNEWS_CEID", "US:en")
FEATURE_NEWS_ENABLED = _env_flag("FEATURE_NEWS_ENABLED", NEWS_ENABLED)
FEATURE_TRANSCRIBE_ENABLED = _env_flag("FEATURE_TRANSCRIBE_ENABLED", True)
FEATURE_TRANSCRIBE_AUTO_URL = _env_flag("FEATURE_TRANSCRIBE_AUTO_URL", False)
TRANSCRIBE_PROGRESS_HEARTBEAT_SECONDS = max(10, int(os.getenv("TRANSCRIBE_PROGRESS_HEARTBEAT_SECONDS", "30")))
FEATURE_OCR_ENABLED = _env_flag("FEATURE_OCR_ENABLED", True)
FEATURE_OCR_CHOICE_ENABLED = _env_flag("FEATURE_OCR_CHOICE_ENABLED", False)
OCR_CHOICE_SCOPE = (os.getenv("OCR_CHOICE_SCOPE", "private") or "private").strip().lower()
OCR_CHOICE_TIMEOUT_SECONDS = int(os.getenv("OCR_CHOICE_TIMEOUT_SECONDS", "60"))
OCR_CHOICE_TIMEOUT_DEFAULT = (os.getenv("OCR_CHOICE_TIMEOUT_DEFAULT", "skip") or "skip").strip().lower()
FEATURE_SLACK_ENABLED = _env_flag("FEATURE_SLACK_ENABLED", True)
APP_PROFILE = (os.getenv("APP_PROFILE", "main") or "main").strip().lower()

NOTION_ENABLED = _env_flag("NOTION_ENABLED", False)
NOTION_TOKEN = os.getenv("NOTION_TOKEN", "").strip()
NOTION_VERSION = os.getenv("NOTION_VERSION", "2022-06-28").strip() or "2022-06-28"
NOTION_CHATLOG_YEAR_PAGES_JSON = os.getenv("NOTION_CHATLOG_YEAR_PAGES_JSON", "").strip()
NOTION_CHATLOG_FALLBACK_PAGE_ID = os.getenv("NOTION_CHATLOG_FALLBACK_PAGE_ID", "").strip()
NOTION_CHATLOG_IMAGE_MODE = (os.getenv("NOTION_CHATLOG_IMAGE_MODE", "link") or "link").strip().lower()
NOTION_CHATLOG_OCR_MODE = (os.getenv("NOTION_CHATLOG_OCR_MODE", "optional") or "optional").strip().lower()
NOTION_CHATLOG_INCLUDE_TIME = _env_flag("NOTION_CHATLOG_INCLUDE_TIME", False)

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
_notion_top_anchor_cache: dict[str, str] = {}
_notion_date_heading_cache: dict[str, str] = {}
_startup_lock = threading.Lock()
_startup_done = False
_telegram_poll_last_ok_at = 0.0
_telegram_poll_last_update_at = 0.0
_telegram_poll_last_update_id: int | None = None
_telegram_poll_last_error = ""
_telegram_send_last_ok_at = 0.0
_telegram_send_last_status = ""
_telegram_send_last_error = ""

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".aac", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".opus"}
URL_RE = re.compile(r"https?://[^\s<>()]+", flags=re.I)
OCR_CHOICE_CALLBACK_RE = re.compile(r"^ocr_choice:([a-f0-9-]{8,64}):(run|save)$", flags=re.I)


def _load_transcription_module():
    try:
        import transcription as tx
    except Exception as e:
        raise RuntimeError(
            "transcription module unavailable. Install dependencies for transcription first."
        ) from e
    return tx


def init_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    TELEGRAM_MD_DIR.mkdir(parents=True, exist_ok=True)
    INBOX_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    TRANSCRIPTS_TMP_DIR.mkdir(parents=True, exist_ok=True)
    if FEATURE_NEWS_ENABLED:
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
        if FEATURE_NEWS_ENABLED:
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
        if FEATURE_NEWS_ENABLED:
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
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ocr_pending_choices (
                job_id TEXT PRIMARY KEY,
                chat_id TEXT NOT NULL,
                prompt_message_id INTEGER,
                image_path TEXT NOT NULL,
                file_unique_id TEXT,
                source_msg_id INTEGER,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                status TEXT NOT NULL,
                ocr_done INTEGER NOT NULL DEFAULT 0,
                ocr_md_saved INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ocr_pending_status_expires ON ocr_pending_choices(status, expires_at)"
        )
        try:
            conn.execute("ALTER TABLE ocr_pending_choices ADD COLUMN caption TEXT")
        except sqlite3.OperationalError:
            pass
        try:
            conn.execute("ALTER TABLE ocr_pending_choices ADD COLUMN message_ts TEXT")
        except sqlite3.OperationalError:
            pass
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
    reply_markup: dict | None = None,
) -> int | None:
    def _send() -> int | None:
        global _telegram_send_last_ok_at, _telegram_send_last_status, _telegram_send_last_error
        try:
            payload = {"chat_id": chat_id, "text": text}
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if disable_web_page_preview is not None:
                payload["disable_web_page_preview"] = disable_web_page_preview
            if reply_markup is not None:
                payload["reply_markup"] = reply_markup
            resp = requests.post(f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)
            if resp.status_code >= 300 and parse_mode:
                fallback_payload = {"chat_id": chat_id, "text": text}
                resp = requests.post(f"{TELEGRAM_API}/sendMessage", json=fallback_payload, timeout=10)
        except Exception as e:
            _telegram_send_last_status = "request_error"
            _telegram_send_last_error = f"{type(e).__name__}: {e}"
            print(f"sendMessage error: {type(e).__name__}: {e}")
            return None
        try:
            data = resp.json() if resp.content else {}
        except Exception:
            data = {}
        _telegram_send_last_status = str(resp.status_code)
        if data.get("ok"):
            _telegram_send_last_ok_at = time.time()
            _telegram_send_last_error = ""
        else:
            _telegram_send_last_error = str(data.get("description") or "send_not_ok")
        print(f"sendMessage status={resp.status_code} ok={bool(data.get('ok'))}")
        if data.get("ok"):
            return data.get("result", {}).get("message_id")
        return None

    return await asyncio.to_thread(_send)


async def edit_message(chat_id: int, message_id: int, text: str) -> bool:
    def _edit() -> bool:
        payload = {"chat_id": chat_id, "message_id": message_id, "text": text}
        try:
            resp = requests.post(f"{TELEGRAM_API}/editMessageText", json=payload, timeout=10)
            data = resp.json() if resp.content else {}
            print(f"editMessageText status={resp.status_code} ok={bool(data.get('ok'))}")
            return bool(resp.status_code < 300 and data.get("ok"))
        except Exception:
            return False

    return await asyncio.to_thread(_edit)


async def clear_message_inline_keyboard(chat_id: int, message_id: int) -> bool:
    def _clear() -> bool:
        payload = {"chat_id": chat_id, "message_id": message_id, "reply_markup": {"inline_keyboard": []}}
        try:
            resp = requests.post(f"{TELEGRAM_API}/editMessageReplyMarkup", json=payload, timeout=10)
            data = resp.json() if resp.content else {}
            print(f"editMessageReplyMarkup status={resp.status_code} ok={bool(data.get('ok'))}")
            return bool(resp.status_code < 300 and data.get("ok"))
        except Exception:
            return False

    return await asyncio.to_thread(_clear)


async def send_document(chat_id: int, file_path: Path, caption: str | None = None) -> bool:
    def _send() -> bool:
        data = {"chat_id": str(chat_id)}
        if caption:
            data["caption"] = caption
        try:
            with file_path.open("rb") as fh:
                resp = requests.post(
                    f"{TELEGRAM_API}/sendDocument",
                    data=data,
                    files={"document": (file_path.name, fh, "text/markdown")},
                    timeout=60,
                )
            body = resp.json() if resp.content else {}
            print(f"sendDocument status={resp.status_code} ok={bool(body.get('ok'))}")
            return bool(resp.status_code < 300 and body.get("ok"))
        except Exception:
            return False

    return await asyncio.to_thread(_send)


async def answer_callback_query(callback_query_id: str, text: str = "", show_alert: bool = False) -> bool:
    def _answer() -> bool:
        payload = {"callback_query_id": callback_query_id, "text": text, "show_alert": show_alert}
        try:
            resp = requests.post(f"{TELEGRAM_API}/answerCallbackQuery", json=payload, timeout=10)
            body = resp.json() if resp.content else {}
            print(f"answerCallbackQuery status={resp.status_code} ok={bool(body.get('ok'))}")
            return bool(resp.status_code < 300 and body.get("ok"))
        except Exception:
            return False

    return await asyncio.to_thread(_answer)


def _is_ocr_choice_enabled_for_chat(is_private: bool, is_group: bool) -> bool:
    if not (FEATURE_OCR_ENABLED and FEATURE_OCR_CHOICE_ENABLED):
        return False
    scope = OCR_CHOICE_SCOPE
    if scope == "private":
        return is_private
    if scope in {"all", "private_group", "both"}:
        return is_private or is_group
    return is_private


def _save_telegram_image(
    *,
    file_id: str,
    file_unique_id: str,
    chat_id: int,
    msg_id: int,
    msg_ts: datetime | None,
) -> tuple[Path, bytes]:
    file_url, file_path = telegram_get_file_info(file_id)
    last_err: Exception | None = None
    img_resp = None
    for attempt in range(TELEGRAM_FILE_FETCH_RETRIES):
        try:
            img_resp = requests.get(
                file_url,
                timeout=(TELEGRAM_FILE_FETCH_CONNECT_TIMEOUT, TELEGRAM_FILE_FETCH_READ_TIMEOUT),
            )
            if img_resp.status_code == 200 and img_resp.content:
                break
            raise RuntimeError(f"image download failed: HTTP {img_resp.status_code}")
        except Exception as e:
            last_err = e
            if attempt < TELEGRAM_FILE_FETCH_RETRIES - 1:
                time.sleep(TELEGRAM_FILE_FETCH_RETRY_DELAY_SECONDS * (attempt + 1))
                continue
            break
    if not img_resp or img_resp.status_code != 200 or not img_resp.content:
        if last_err:
            raise RuntimeError(f"圖片下載失敗：{last_err}") from last_err
        raise RuntimeError("圖片下載失敗：empty response")
    day = (msg_ts or datetime.now()).strftime("%Y-%m-%d")
    img_dir = INBOX_IMAGES_DIR / day
    img_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file_path).suffix or ".jpg"
    filename = f"{chat_id}_{msg_id}_{file_unique_id}{suffix}"
    out_path = img_dir / filename
    out_path.write_bytes(img_resp.content)
    return out_path, img_resp.content


def _run_ocr_on_image(
    *,
    chat_id: int | str,
    msg_id: int | str,
    image_path: Path,
    msg_ts: datetime | None,
) -> tuple[bool, bool, str]:
    ocr_ok = False
    md_saved = False
    ocr_text = ""
    try:
        image_bytes = image_path.read_bytes()
    except Exception as e:
        ocr_text = f"[OCR failed] cannot read image: {e}"
    else:
        try:
            ocr_text = extract_text_from_image(image_bytes)
            if ocr_text.strip():
                ocr_ok = True
            else:
                ocr_text = "[OCR failed] no text detected"
        except Exception as e:
            ocr_text = f"[OCR failed] {e}"
            print(f"[WARN] OCR failed for {image_path}: {e}")
    try:
        append_ocr_markdown(str(chat_id), msg_id, image_path, ocr_text, msg_ts)
        md_saved = True
    except Exception as e:
        print(f"[WARN] OCR markdown save failed for {image_path}: {e}")
    return ocr_ok, md_saved, (ocr_text or "").strip()


def _create_ocr_choice_job(
    *,
    job_id: str,
    chat_id: int,
    prompt_message_id: int | None,
    image_path: Path,
    file_unique_id: str,
    source_msg_id: int | None,
    caption: str | None = None,
    message_ts: datetime | None = None,
) -> str:
    now = datetime.now()
    expires = now + timedelta(seconds=max(5, OCR_CHOICE_TIMEOUT_SECONDS))
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO ocr_pending_choices
            (job_id, chat_id, prompt_message_id, image_path, file_unique_id, source_msg_id,
             created_at, expires_at, status, ocr_done, ocr_md_saved, caption, message_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0, 0, ?, ?)
            """,
            (
                job_id,
                str(chat_id),
                prompt_message_id,
                str(image_path),
                file_unique_id,
                source_msg_id,
                now.isoformat(timespec="seconds"),
                expires.isoformat(timespec="seconds"),
                (caption or "").strip() or None,
                message_ts.isoformat(timespec="seconds") if message_ts else None,
            ),
        )
        conn.commit()
    return job_id


def _get_ocr_choice_job(job_id: str) -> tuple | None:
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            """
            SELECT job_id, chat_id, prompt_message_id, image_path, source_msg_id,
                   expires_at, status, ocr_done, ocr_md_saved, caption, message_ts
            FROM ocr_pending_choices
            WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()


def _update_ocr_choice_job_status(job_id: str, status: str, ocr_done: int = 0, ocr_md_saved: int = 0) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            UPDATE ocr_pending_choices
            SET status = ?, ocr_done = ?, ocr_md_saved = ?
            WHERE job_id = ?
            """,
            (status, int(ocr_done), int(ocr_md_saved), job_id),
        )
        conn.commit()


def _list_expired_ocr_choice_jobs(now_dt: datetime | None = None) -> list[tuple]:
    now = (now_dt or datetime.now()).isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute(
            """
            SELECT job_id, chat_id, prompt_message_id, image_path, caption, message_ts
            FROM ocr_pending_choices
            WHERE status = 'pending' AND expires_at <= ?
            """,
            (now,),
        ).fetchall()


def _build_ocr_choice_keyboard(job_id: str) -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "進行 OCR", "callback_data": f"ocr_choice:{job_id}:run"},
                {"text": "只存圖", "callback_data": f"ocr_choice:{job_id}:save"},
            ]
        ]
    }


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


def _extract_transcribe_target(text: str) -> str:
    m = re.match(r"^/transcribe(?:@\w+)?\s*(.*)$", (text or "").strip(), flags=re.I | re.S)
    if not m:
        return ""
    return m.group(1).strip()


def _extract_supported_transcribe_url(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    try:
        tx = _load_transcription_module()
    except Exception:
        return ""
    for match in URL_RE.findall(raw):
        candidate = match.rstrip(").,;!?")
        _url_type, error = tx.classify_url(candidate)
        if not error:
            return candidate
    return ""


def _build_transcript_ai_summary(transcript_path: Path) -> str | None:
    try:
        content = transcript_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    if not content.strip():
        return None

    transcript_text = content
    if "\n---\n" in transcript_text:
        transcript_text = transcript_text.split("\n---\n", 1)[1]
    transcript_text = transcript_text.strip()
    if not transcript_text:
        return None

    clipped = transcript_text[: min(len(transcript_text), AI_SUMMARY_MAX_CHARS)]
    fallback = _fallback_three_points(clipped)
    if not AI_SUMMARY_ENABLED:
        return fallback

    system_prompt = (
        "Use only provided transcript content. Do not invent facts. "
        "Output must be in Traditional Chinese."
    )
    user_prompt = (
        "請根據以下逐字稿輸出重點摘要：\n"
        "1. 先用 3-5 點條列重點\n"
        "2. 補一段 2-3 句的整體結論\n\n"
        f"{clipped}"
    )
    ai = _run_ai_chat(system_prompt, user_prompt)
    return ai or fallback


def _prepend_summary_to_transcript(transcript_path: Path, summary_text: str) -> None:
    summary = (summary_text or "").strip()
    if not summary:
        return
    raw = transcript_path.read_text(encoding="utf-8", errors="replace")
    if raw.lstrip().startswith("## AI 摘要"):
        return
    block = f"## AI 摘要\n\n{summary}\n\n---\n\n"
    transcript_path.write_text(block + raw, encoding="utf-8")


def _append_transcript_to_telegram_markdown(
    chat_id: int,
    title: str,
    source: str,
    transcript_path: Path,
    message_ts: datetime | None,
) -> Path:
    day = (message_ts or datetime.now()).strftime("%Y-%m-%d")
    md_path = TELEGRAM_MD_DIR / f"{day}_telegram.md"
    if not md_path.exists():
        md_path.write_text(f"# {day} telegram\n\n", encoding="utf-8")
    content = transcript_path.read_text(encoding="utf-8", errors="replace").strip()
    time_str = (message_ts or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    normalized_title = (title or "").strip() or "untitled"
    normalized_source = (source or "").strip() or "unknown"
    with md_path.open("a", encoding="utf-8") as f:
        f.write(f"## [{time_str}] transcript chat={chat_id}\n")
        f.write(f"- title: {normalized_title}\n")
        f.write(f"- source: {normalized_source}\n\n")
        if content:
            f.write(content)
            f.write("\n\n")
        else:
            f.write("[empty transcript]\n\n")
    return md_path


def _localize_transcribe_status(status: str) -> str:
    s = (status or "").strip()
    if not s:
        return ""
    lower = s.lower()
    if (
        lower.startswith("checking video info")
        or lower.startswith("downloading")
        or lower.startswith("preparing")
        or lower.startswith("transcribing")
    ):
        return "進度：處理中..."
    return "進度：處理中..."


def _build_transcribe_status_message(intro_text: str, status_text: str, title: str | None = None) -> str:
    lines = [intro_text.strip(), status_text.strip()]
    if title:
        lines.append(f"轉錄完成：{title}")
    return "\n".join(x for x in lines if x)


async def _run_transcribe_job_with_progress(chat_id: int, worker, intro_text: str):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str] = asyncio.Queue()

    def on_status(msg: str) -> None:
        text = (msg or "").strip()
        if not text:
            return
        loop.call_soon_threadsafe(queue.put_nowait, text)

    progress_text = _build_transcribe_status_message(intro_text, "進度：處理中...")
    progress_message_id = await send_message(chat_id, progress_text)
    job = asyncio.create_task(asyncio.to_thread(worker, on_status))
    last_status = ""
    last_sent_at = 0.0
    started_at = time.time()
    last_heartbeat_at = started_at
    while True:
        if job.done() and queue.empty():
            break
        try:
            raw_status = await asyncio.wait_for(queue.get(), timeout=0.8)
        except asyncio.TimeoutError:
            now = time.time()
            if job.done() or not last_status:
                continue
            if (now - last_heartbeat_at) < TRANSCRIBE_PROGRESS_HEARTBEAT_SECONDS:
                continue
            elapsed_sec = int(now - started_at)
            heartbeat = f"進度：處理中（已等待 {elapsed_sec // 60} 分 {elapsed_sec % 60} 秒）"
            heartbeat_text = _build_transcribe_status_message(intro_text, heartbeat)
            if progress_message_id:
                ok = await edit_message(chat_id, progress_message_id, heartbeat_text)
                if not ok:
                    progress_message_id = await send_message(chat_id, heartbeat_text)
            else:
                progress_message_id = await send_message(chat_id, heartbeat_text)
            last_heartbeat_at = now
            continue
        localized = _localize_transcribe_status(raw_status)
        now = time.time()
        if not localized:
            continue
        if localized == last_status and (now - last_sent_at) < 4:
            continue
        if (now - last_sent_at) < 1.2:
            continue
        if progress_message_id:
            text = _build_transcribe_status_message(intro_text, localized)
            ok = await edit_message(chat_id, progress_message_id, text)
            if not ok:
                progress_message_id = await send_message(chat_id, text)
        else:
            progress_message_id = await send_message(chat_id, _build_transcribe_status_message(intro_text, localized))
        last_status = localized
        last_sent_at = now
        last_heartbeat_at = now

    result = await job
    return result, progress_message_id


async def _run_transcribe_url_flow(chat_id: int, target: str, message_ts: datetime | None = None) -> bool:
    try:
        tx = _load_transcription_module()
    except Exception as e:
        await send_message(chat_id, f"轉錄模組不可用：{e}")
        return True
    _url_type, error = tx.classify_url(target)
    if error:
        await send_message(chat_id, f"URL 不支援：{error}")
        return True

    intro_text = ""
    try:
        (title, out_path), progress_msg_id = await _run_transcribe_job_with_progress(
            chat_id,
            lambda on_status: tx.transcribe_url_to_markdown(
                target,
                TRANSCRIPTS_DIR,
                TRANSCRIPTS_TMP_DIR,
                on_status=on_status,
            ),
            intro_text,
        )
        if progress_msg_id:
            done_text = _build_transcribe_status_message(intro_text, f"轉錄完成：{title}")
            await edit_message(chat_id, progress_msg_id, done_text)
        await asyncio.to_thread(
            _append_transcript_to_telegram_markdown,
            chat_id,
            title,
            target,
            out_path,
            message_ts,
        )
        await asyncio.to_thread(
            notion_append_chitchat_transcript,
            title=title,
            source=target,
            transcript_path=out_path,
            msg_ts=message_ts,
        )
        summary = await asyncio.to_thread(_build_transcript_ai_summary, out_path)
        if summary:
            await asyncio.to_thread(_prepend_summary_to_transcript, out_path, summary)
            for chunk in _chunk_text_for_telegram(f"摘要：\n{summary}"):
                await send_message(chat_id, chunk)
        else:
            await send_message(chat_id, "已成功紀錄")
    except Exception as e:
        err_text = _build_transcribe_status_message(intro_text, f"轉錄失敗：{e}")
        if 'progress_msg_id' in locals() and progress_msg_id:
            ok = await edit_message(chat_id, progress_msg_id, err_text)
            if not ok:
                await send_message(chat_id, err_text)
        else:
            await send_message(chat_id, err_text)
    finally:
        try:
            if 'out_path' in locals() and out_path and Path(out_path).exists():
                Path(out_path).unlink()
        except Exception:
            pass
    return True


def _extract_audio_attachment(message: dict) -> tuple[str, str] | None:
    audio = message.get("audio") or {}
    if audio.get("file_id"):
        name = audio.get("file_name") or f"audio_{audio.get('file_unique_id', 'upload')}.mp3"
        return audio["file_id"], name

    voice = message.get("voice") or {}
    if voice.get("file_id"):
        return voice["file_id"], f"voice_{voice.get('file_unique_id', 'upload')}.ogg"

    doc = message.get("document") or {}
    file_id = doc.get("file_id")
    if not file_id:
        return None
    mime_type = (doc.get("mime_type") or "").lower()
    filename = doc.get("file_name") or f"audio_{doc.get('file_unique_id', 'upload')}"
    ext = Path(filename).suffix.lower()
    if mime_type.startswith("audio/") or ext in ALLOWED_AUDIO_EXTENSIONS:
        if not ext:
            filename = f"{filename}.mp3"
        return file_id, filename
    return None


def _download_telegram_file(file_id: str, dest_path: Path) -> None:
    file_url, _file_path = telegram_get_file_info(file_id)
    resp = requests.get(file_url, timeout=300)
    if resp.status_code != 200:
        raise RuntimeError(f"Telegram file download failed: {resp.status_code}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    dest_path.write_bytes(resp.content)


async def handle_transcribe_text_command(chat_id: int, text: str, message_ts: datetime | None = None) -> bool:
    if not FEATURE_TRANSCRIBE_ENABLED:
        return False
    target = _extract_transcribe_target(text)
    if not target:
        if (text or "").strip().lower().startswith("/transcribe"):
            await send_message(chat_id, "用法：/transcribe <YouTube URL | Podcast URL | 音訊 URL>")
            return True
        return False
    return await _run_transcribe_url_flow(chat_id, target, message_ts=message_ts)


async def handle_transcribe_auto_url_message(chat_id: int, text: str, message_ts: datetime | None = None) -> bool:
    if not FEATURE_TRANSCRIBE_ENABLED or not FEATURE_TRANSCRIBE_AUTO_URL:
        return False
    raw = (text or "").strip()
    if not raw or raw.startswith("/"):
        return False
    target = _extract_supported_transcribe_url(raw)
    if not target:
        return False
    return await _run_transcribe_url_flow(chat_id, target, message_ts=message_ts)


async def handle_transcribe_audio_message(chat_id: int, message: dict, message_ts: datetime | None = None) -> bool:
    if not FEATURE_TRANSCRIBE_ENABLED:
        return False
    extracted = _extract_audio_attachment(message)
    if not extracted:
        return False

    file_id, original_filename = extracted
    ext = Path(original_filename).suffix.lower() or ".mp3"
    if ext not in ALLOWED_AUDIO_EXTENSIONS:
        await send_message(
            chat_id,
            f"不支援的檔案格式：{ext}\n支援：{', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
        )
        return True

    temp_input = TRANSCRIPTS_TMP_DIR / f"{uuid.uuid4()}{ext}"
    intro_text = ""
    try:
        tx = _load_transcription_module()
        await asyncio.to_thread(_download_telegram_file, file_id, temp_input)
        if not temp_input.exists() or temp_input.stat().st_size == 0:
            raise RuntimeError("下載到的音訊檔為空。")

        (title, out_path), progress_msg_id = await _run_transcribe_job_with_progress(
            chat_id,
            lambda on_status: tx.transcribe_upload_to_markdown(
                temp_input,
                original_filename,
                TRANSCRIPTS_DIR,
                TRANSCRIPTS_TMP_DIR,
                on_status=on_status,
            ),
            intro_text,
        )
        if progress_msg_id:
            done_text = _build_transcribe_status_message(intro_text, f"轉錄完成：{title}")
            await edit_message(chat_id, progress_msg_id, done_text)
        await asyncio.to_thread(
            _append_transcript_to_telegram_markdown,
            chat_id,
            title,
            original_filename,
            out_path,
            message_ts,
        )
        await asyncio.to_thread(
            notion_append_chitchat_transcript,
            title=title,
            source=original_filename,
            transcript_path=out_path,
            msg_ts=message_ts,
        )
        summary = await asyncio.to_thread(_build_transcript_ai_summary, out_path)
        if summary:
            await asyncio.to_thread(_prepend_summary_to_transcript, out_path, summary)
            for chunk in _chunk_text_for_telegram(f"摘要：\n{summary}"):
                await send_message(chat_id, chunk)
        else:
            await send_message(chat_id, "已成功紀錄")
    except Exception as e:
        err_text = _build_transcribe_status_message(intro_text, f"轉錄失敗：{e}")
        if 'progress_msg_id' in locals() and progress_msg_id:
            ok = await edit_message(chat_id, progress_msg_id, err_text)
            if not ok:
                await send_message(chat_id, err_text)
        else:
            await send_message(chat_id, err_text)
    finally:
        try:
            if temp_input.exists():
                temp_input.unlink()
        except Exception:
            pass
        try:
            if 'out_path' in locals() and out_path and Path(out_path).exists():
                Path(out_path).unlink()
        except Exception:
            pass
    return True


def handle_command(text: str) -> str:
    text = text.strip()
    text_lower = text.lower()

    if text_lower.startswith("/summary_news"):
        if not FEATURE_NEWS_ENABLED:
            return "新聞功能已關閉。"
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

    if FEATURE_NEWS_ENABLED and text_lower.startswith("/news_latest"):
        suffix = text.strip()[len("/news_latest") :].strip()
        return "\n".join(handle_news_command(f"/news latest {suffix}".strip(), ""))
    if FEATURE_NEWS_ENABLED and text_lower.startswith("/news_sources"):
        return "\n".join(handle_news_command("/news sources", ""))
    if FEATURE_NEWS_ENABLED and text_lower.startswith("/news_debug"):
        return "\n".join(handle_news_command("/news debug", ""))
    if FEATURE_NEWS_ENABLED and text_lower.startswith("/news_help"):
        return "\n".join(handle_news_command("/news help", ""))

    if text_lower.startswith("/news"):
        if not FEATURE_NEWS_ENABLED:
            return "新聞功能已關閉。"
        return "用法：/news latest | /news search <keywords> | /news sources | /news help"

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
        cmds = ["open <url>", "notepad", "sort downloads", "/status", "/summary", "/summary_note"]
        if FEATURE_NEWS_ENABLED:
            cmds.extend(["/news latest", "/summary news"])
        if FEATURE_TRANSCRIBE_ENABLED:
            cmds.append("/transcribe <url>")
        return "Commands: " + ", ".join(cmds)

    if text.startswith("/"):
        return "Unsupported command."
    return "已成功紀錄"


def route_user_text_command(text: str, chat_id: str) -> tuple[list[str], str | None, bool | None]:
    cmd_text = (text or "").strip()
    lower = cmd_text.lower()

    if FEATURE_NEWS_ENABLED and lower.startswith("/news"):
        replies = handle_news_command(cmd_text, chat_id)
        tokens = cmd_text.split()
        sub = tokens[1].lower() if len(tokens) > 1 else "latest"
        parse_mode = "Markdown" if sub == "latest" else None
        disable_preview = True if sub == "latest" else None
        return replies, parse_mode, disable_preview

    reply = handle_command(cmd_text)
    parse_mode = None
    disable_preview = None
    if FEATURE_NEWS_ENABLED and lower.startswith("/summary_news"):
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
    md_path = TELEGRAM_MD_DIR / f"{day}_telegram.md"
    if not md_path.exists():
        md_path.write_text(f"# {day} telegram\n\n", encoding="utf-8")
    try:
        rel_path = image_path.relative_to(DATA_DIR).as_posix()
    except Exception:
        rel_path = str(image_path)
    time_str = (message_ts or datetime.now()).strftime("%Y-%m-%d %H:%M:%S")
    normalized_text = (text or "").strip()
    with md_path.open("a", encoding="utf-8") as f:
        f.write(f"## [{time_str}] OCR chat={chat_id} message={msg_id}\n")
        f.write(f"- image: `{rel_path}`\n")
        if normalized_text:
            f.write("- ocr_text:\n\n")
            f.write(normalized_text)
            f.write("\n\n")
        else:
            f.write("- ocr_text: [no text detected]\n\n")
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
    folders = ["notes", "images"]
    if FEATURE_NEWS_ENABLED:
        folders.insert(0, "news")
    for folder in folders:
        _dropbox_create_folder_if_missing(dbx, f"{root}/{folder}")


def iter_sync_files() -> Iterator[tuple[str, Path, str]]:
    roots = [("notes", NOTES_DIR), ("images", INBOX_IMAGES_DIR)]
    if FEATURE_NEWS_ENABLED:
        roots.insert(0, ("news", NEWS_MD_DIR))
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


def _dropbox_download_file_bytes(remote_path: str) -> bytes | None:
    normalized = normalize_dropbox_path(remote_path)

    def _download(dbx):
        return dbx.files_download(normalized)[1].content

    try:
        return _dropbox_call_with_retry(_download)
    except Exception as e:
        msg = str(e).lower()
        if "not_found" in msg or "path/not_found" in msg:
            return None
        raise


def _split_markdown_blocks(text: str) -> list[str]:
    if not text:
        return []
    chunks = re.split(r"(?:\r?\n){2,}", text.replace("\r\n", "\n").strip())
    return [c.strip() for c in chunks if c and c.strip()]


def merge_markdown_content(remote_text: str | None, local_text: str) -> str:
    remote_text = (remote_text or "").replace("\r\n", "\n")
    local_text = (local_text or "").replace("\r\n", "\n")

    remote_blocks = _split_markdown_blocks(remote_text)
    local_blocks = _split_markdown_blocks(local_text)

    heading = ""
    if local_blocks and local_blocks[0].startswith("# "):
        heading = local_blocks.pop(0)
    elif remote_blocks and remote_blocks[0].startswith("# "):
        heading = remote_blocks.pop(0)

    merged_blocks: list[str] = []
    seen: set[str] = set()
    for block in remote_blocks + local_blocks:
        key = hashlib.sha1(block.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        merged_blocks.append(block)

    out_parts: list[str] = []
    if heading:
        out_parts.append(heading.strip())
    if merged_blocks:
        if out_parts:
            out_parts.append("")
        out_parts.append("\n\n".join(merged_blocks).strip())
    merged = "\n".join(out_parts).strip()
    return (merged + "\n") if merged else ""


def sync_markdown_with_merge(local_path: Path, remote_path: str) -> tuple[bool, bool]:
    local_text = local_path.read_text(encoding="utf-8")
    remote_bytes = _dropbox_download_file_bytes(remote_path)
    remote_text = remote_bytes.decode("utf-8", errors="replace") if remote_bytes is not None else None

    merged_text = merge_markdown_content(remote_text, local_text)
    local_changed = merged_text != local_text
    remote_changed = (remote_text or "") != merged_text

    if local_changed:
        local_path.write_text(merged_text, encoding="utf-8")
    if remote_changed:
        sync_file_to_dropbox(local_path, remote_path)

    return local_changed, remote_changed


def _dropbox_get_or_create_shared_link(remote_path: str) -> str | None:
    if dropbox is None:
        return None
    normalized = normalize_dropbox_path(remote_path)

    def _list_or_create(dbx):
        try:
            listed = dbx.sharing_list_shared_links(path=normalized, direct_only=True)
            links = list(getattr(listed, "links", []) or [])
            if links:
                return links[0].url
        except Exception:
            pass
        created = dbx.sharing_create_shared_link_with_settings(normalized)
        return getattr(created, "url", None)

    try:
        url = _dropbox_call_with_retry(_list_or_create)
    except Exception as e:
        print(f"[WARN] Dropbox shared link failed for {normalized}: {e}")
        return None
    if not url:
        return None
    return str(url).replace("?dl=0", "?raw=1")


def _dropbox_get_temporary_link(remote_path: str) -> str | None:
    if dropbox is None:
        return None
    normalized = normalize_dropbox_path(remote_path)

    def _fetch(dbx):
        result = dbx.files_get_temporary_link(normalized)
        return getattr(result, "link", None)

    try:
        url = _dropbox_call_with_retry(_fetch)
    except Exception as e:
        print(f"[WARN] Dropbox temporary link failed for {normalized}: {e}")
        return None
    return str(url).strip() if url else None


def _dropbox_remote_path_for_local_data_file(local_path: Path) -> str | None:
    try:
        rel = local_path.resolve().relative_to(DATA_DIR).as_posix()
    except Exception:
        return None
    root = normalize_dropbox_path(DROPBOX_ROOT_PATH)
    return f"{root}/{rel}".replace("//", "/")


def get_or_create_dropbox_shared_link_for_local_file(
    local_path: Path,
    *,
    prefer_temporary: bool = False,
) -> str | None:
    if not DROPBOX_SYNC_ENABLED:
        return None
    remote_path = _dropbox_remote_path_for_local_data_file(local_path)
    if not remote_path:
        return None
    try:
        _dropbox_call_with_retry(lambda dbx: ensure_dropbox_folders(dbx, normalize_dropbox_path(DROPBOX_ROOT_PATH)))
        sync_file_to_dropbox(local_path, remote_path)
    except Exception as e:
        print(f"[WARN] Dropbox upload for shared link failed ({local_path}): {e}")
        return None
    if prefer_temporary:
        tmp = _dropbox_get_temporary_link(remote_path)
        if tmp:
            return tmp
    return _dropbox_get_or_create_shared_link(remote_path)


def _parse_notion_year_page_map() -> dict[str, str]:
    if not NOTION_CHATLOG_YEAR_PAGES_JSON:
        return {}
    try:
        raw = json.loads(NOTION_CHATLOG_YEAR_PAGES_JSON)
    except Exception as e:
        print(f"[WARN] NOTION_CHATLOG_YEAR_PAGES_JSON parse failed: {e}")
        return {}
    if not isinstance(raw, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in raw.items():
        key = str(k).strip()
        val = str(v).replace("-", "").strip()
        if key and val:
            out[key] = val
    return out


def _notion_is_chitchat_enabled() -> bool:
    if APP_PROFILE != "chitchat":
        return False
    if not (NOTION_ENABLED and NOTION_TOKEN):
        return False
    return True


def _notion_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {NOTION_TOKEN}",
        "Notion-Version": NOTION_VERSION,
        "Content-Type": "application/json",
    }


def _notion_request(method: str, path: str, payload: dict | None = None) -> dict:
    url = f"https://api.notion.com/v1/{path.lstrip('/')}"
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            resp = requests.request(method.upper(), url, headers=_notion_headers(), json=payload, timeout=20)
            if resp.status_code in {429, 500, 502, 503, 504}:
                time.sleep(0.6 * (attempt + 1))
                continue
            if resp.status_code >= 400:
                raise RuntimeError(f"Notion API {resp.status_code}: {resp.text[:300]}")
            if not resp.content:
                return {}
            return resp.json()
        except Exception as e:
            last_exc = e
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            break
    raise RuntimeError(str(last_exc) if last_exc else "Notion request failed")


def _notion_rich_text_plain(block: dict) -> str:
    for key in ("paragraph", "heading_1", "heading_2", "heading_3", "bulleted_list_item"):
        node = block.get(key)
        if not isinstance(node, dict):
            continue
        texts = node.get("rich_text") or []
        chunks = []
        for t in texts:
            plain = (t or {}).get("plain_text") or ""
            if plain:
                chunks.append(plain)
        if chunks:
            return "".join(chunks).strip()
    return ""


def _notion_iter_block_children(block_id: str, page_size: int = 100) -> Iterator[dict]:
    cursor = None
    while True:
        path = f"blocks/{block_id}/children?page_size={page_size}"
        if cursor:
            path += f"&start_cursor={quote(str(cursor), safe='')}"
        data = _notion_request("GET", path)
        for item in data.get("results") or []:
            if isinstance(item, dict):
                yield item
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
        if not cursor:
            break


def _notion_append_block_children(block_id: str, children: list[dict], after: str | None = None) -> list[dict]:
    payload: dict[str, object] = {"children": children}
    if after:
        payload["after"] = after
    data = _notion_request("PATCH", f"blocks/{block_id}/children", payload)
    return list(data.get("results") or [])


def _notion_paragraph_block(text: str) -> dict:
    return {
        "object": "block",
        "type": "paragraph",
        "paragraph": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _notion_bullet_block(text: str) -> dict:
    return {
        "object": "block",
        "type": "bulleted_list_item",
        "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _notion_h3_block(text: str) -> dict:
    return {
        "object": "block",
        "type": "heading_3",
        "heading_3": {"rich_text": [{"type": "text", "text": {"content": text}}]},
    }


def _notion_external_image_block(url: str, caption: str | None = None) -> dict:
    cap = (caption or "").strip()
    caption_rich_text = []
    if cap:
        caption_rich_text = [{"type": "text", "text": {"content": cap}}]
    return {
        "object": "block",
        "type": "image",
        "image": {
            "type": "external",
            "external": {"url": url},
            "caption": caption_rich_text,
        },
    }


def _notion_day_label(dt: datetime | None) -> str:
    d = dt or datetime.now()
    return f"{d.year}/{d.month}/{d.day}"


def _notion_time_prefix(dt: datetime | None) -> str:
    if not (NOTION_CHATLOG_INCLUDE_TIME and dt):
        return ""
    return f"[{dt.strftime('%H:%M')}] "


def _notion_get_chatlog_year_page_id(dt: datetime | None) -> str | None:
    d = dt or datetime.now()
    mapping = _parse_notion_year_page_map()
    page_id = mapping.get(str(d.year))
    if page_id:
        return page_id
    if NOTION_CHATLOG_FALLBACK_PAGE_ID:
        return NOTION_CHATLOG_FALLBACK_PAGE_ID.replace("-", "")
    return None


def _notion_cache_key_date(page_id: str, day_label: str) -> str:
    return f"{page_id}:{day_label}"


def _notion_find_top_anchor_block_id(page_id: str) -> str | None:
    for block in _notion_iter_block_children(page_id, page_size=20):
        block_type = (block.get("type") or "").strip()
        if block_type == "divider":
            return str(block.get("id") or "")
        if block_type == "paragraph" and _notion_rich_text_plain(block) == "Chatlog Anchor":
            return str(block.get("id") or "")
    return None


def _notion_ensure_top_anchor(page_id: str) -> str:
    cached = _notion_top_anchor_cache.get(page_id)
    if cached:
        return cached
    found = _notion_find_top_anchor_block_id(page_id)
    if found:
        _notion_top_anchor_cache[page_id] = found
        return found
    first_block_id = None
    for block in _notion_iter_block_children(page_id, page_size=1):
        first_block_id = str(block.get("id") or "")
        break
    if first_block_id:
        print("[WARN] Notion top anchor missing; using first block as insertion anchor (not absolute top).")
        _notion_top_anchor_cache[page_id] = first_block_id
        return first_block_id
    created = _notion_append_block_children(
        page_id,
        [{"object": "block", "type": "divider", "divider": {}}],
    )
    if not created:
        raise RuntimeError("Notion anchor create returned no blocks")
    block_id = str(created[0].get("id") or "")
    if not block_id:
        raise RuntimeError("Notion anchor block id missing")
    _notion_top_anchor_cache[page_id] = block_id
    return block_id


def _notion_find_date_heading_block_id(page_id: str, day_label: str) -> str | None:
    for block in _notion_iter_block_children(page_id):
        block_type = (block.get("type") or "")
        if block_type not in {"heading_2", "heading_3"}:
            continue
        if _notion_rich_text_plain(block) == day_label:
            return str(block.get("id") or "")
    return None


def _notion_ensure_date_heading(page_id: str, day_label: str) -> str:
    key = _notion_cache_key_date(page_id, day_label)
    cached = _notion_date_heading_cache.get(key)
    if cached:
        return cached
    found = _notion_find_date_heading_block_id(page_id, day_label)
    if found:
        _notion_date_heading_cache[key] = found
        return found
    anchor_id = _notion_ensure_top_anchor(page_id)
    created = _notion_append_block_children(page_id, [_notion_h3_block(day_label)], after=anchor_id)
    if not created:
        raise RuntimeError("Notion date heading create returned no blocks")
    heading_id = str(created[0].get("id") or "")
    if not heading_id:
        raise RuntimeError("Notion date heading block id missing")
    _notion_date_heading_cache[key] = heading_id
    return heading_id


def notion_append_chitchat_text(text: str, msg_ts: datetime | None) -> None:
    if not _notion_is_chitchat_enabled():
        return
    normalized = (text or "").strip()
    if not normalized:
        return
    normalized = _notion_time_prefix(msg_ts) + normalized
    page_id = _notion_get_chatlog_year_page_id(msg_ts)
    if not page_id:
        print("[WARN] Notion chatlog page id not configured for year")
        return
    day_label = _notion_day_label(msg_ts)
    try:
        heading_id = _notion_ensure_date_heading(page_id, day_label)
        _notion_append_block_children(page_id, [_notion_bullet_block(normalized)], after=heading_id)
    except Exception as e:
        print(f"[WARN] Notion text append failed: {e}")


def _notion_extract_transcript_excerpt(transcript_path: Path, limit: int = 220) -> str | None:
    try:
        raw = transcript_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    lines: list[str] = []
    in_ai_summary = False
    for line in raw.splitlines():
        s = (line or "").strip()
        if not s:
            continue
        if s.startswith("## AI 摘要"):
            in_ai_summary = True
            continue
        if in_ai_summary:
            if s.startswith("---"):
                in_ai_summary = False
            continue
        if s.startswith("#") or s.startswith("- **") or s.startswith("---") or s.startswith("摘要："):
            continue
        lines.append(s)
        if len(lines) >= 3:
            break
    if not lines:
        return None
    excerpt = " ".join(" ".join(lines).split())
    if len(excerpt) > limit:
        excerpt = excerpt[:limit].rstrip() + "..."
    return excerpt


def notion_append_chitchat_transcript(
    *,
    title: str,
    source: str,
    transcript_path: Path,
    msg_ts: datetime | None,
) -> None:
    if not _notion_is_chitchat_enabled():
        return
    page_id = _notion_get_chatlog_year_page_id(msg_ts)
    if not page_id:
        print("[WARN] Notion chatlog page id not configured for transcript year")
        return
    day_label = _notion_day_label(msg_ts)
    normalized_title = (title or "").strip() or "untitled"
    normalized_source = (source or "").strip() or "unknown"
    excerpt = _notion_extract_transcript_excerpt(transcript_path)
    blocks = [_notion_bullet_block(f"{_notion_time_prefix(msg_ts)}[轉錄] {normalized_title}")]
    blocks.append(_notion_paragraph_block(f"來源：{normalized_source}"))
    if excerpt:
        blocks.append(_notion_paragraph_block(f"摘錄：{excerpt}"))
    try:
        heading_id = _notion_ensure_date_heading(page_id, day_label)
        _notion_append_block_children(page_id, blocks, after=heading_id)
    except Exception as e:
        print(f"[WARN] Notion transcript append failed: {e}")


def _notion_summarize_ocr_text(ocr_text: str | None, limit: int = 160) -> str | None:
    txt = (ocr_text or "").strip()
    if not txt:
        return None
    txt = " ".join(txt.split())
    if txt.startswith("[OCR failed]"):
        return None
    if len(txt) > limit:
        return txt[:limit].rstrip() + "..."
    return txt


def _infer_message_dt_from_image_path(image_path: Path) -> datetime | None:
    try:
        day_part = image_path.parent.name
        return datetime.strptime(day_part, "%Y-%m-%d")
    except Exception:
        return None


def _parse_iso_dt_or_none(value: str | None) -> datetime | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def notion_append_chitchat_image(
    *,
    image_path: Path,
    caption: str | None = None,
    ocr_text: str | None = None,
    msg_ts: datetime | None = None,
) -> None:
    if not _notion_is_chitchat_enabled():
        return
    if NOTION_CHATLOG_IMAGE_MODE not in {"link", "embed"}:
        return
    page_id = _notion_get_chatlog_year_page_id(msg_ts or _infer_message_dt_from_image_path(image_path))
    if not page_id:
        print("[WARN] Notion chatlog page id not configured for image year")
        return
    day_label = _notion_day_label(msg_ts or _infer_message_dt_from_image_path(image_path))
    blocks: list[dict] = []
    cap = (caption or "").strip()
    image_url = get_or_create_dropbox_shared_link_for_local_file(
        image_path,
        prefer_temporary=(NOTION_CHATLOG_IMAGE_MODE == "embed"),
    )
    if image_url and NOTION_CHATLOG_IMAGE_MODE == "embed":
        blocks.append(_notion_external_image_block(image_url, caption=cap or None))
    elif image_url:
        bullet = "[圖片]"
        if cap:
            bullet = f"{bullet} {cap}"
        blocks.append(_notion_bullet_block(bullet))
        blocks.append(_notion_paragraph_block(f"圖片：{image_url}"))
    else:
        if cap:
            blocks.append(_notion_bullet_block(f"[圖片] {cap}"))
        blocks.append(_notion_paragraph_block("圖片：本地已儲存（尚未取得雲端連結）"))
    ocr_summary = _notion_summarize_ocr_text(ocr_text) if NOTION_CHATLOG_OCR_MODE == "optional" else None
    if ocr_summary:
        blocks.append(_notion_paragraph_block(f"OCR 摘要：{ocr_summary}"))
    try:
        heading_id = _notion_ensure_date_heading(page_id, day_label)
        _notion_append_block_children(page_id, blocks, after=heading_id)
    except Exception as e:
        if NOTION_CHATLOG_IMAGE_MODE == "embed" and image_url:
            print(f"[WARN] Notion image embed failed, fallback to link mode: {e}")
            fallback_blocks: list[dict] = []
            bullet = "[圖片]"
            if cap:
                bullet = f"{bullet} {cap}"
            fallback_blocks.append(_notion_bullet_block(bullet))
            fallback_blocks.append(_notion_paragraph_block(f"圖片：{image_url}"))
            if ocr_summary:
                fallback_blocks.append(_notion_paragraph_block(f"OCR 摘要：{ocr_summary}"))
            try:
                heading_id = _notion_ensure_date_heading(page_id, day_label)
                _notion_append_block_children(page_id, fallback_blocks, after=heading_id)
                return
            except Exception as e2:
                print(f"[WARN] Notion image fallback append failed: {e2}")
        else:
            print(f"[WARN] Notion image append failed: {e}")


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
            remote_path = f"{root}/{category}/{rel}".replace("//", "/")
            is_markdown_note = category == "notes" and local_path.suffix.lower() in {".md", ".markdown"}

            if is_markdown_note:
                _local_changed, remote_changed = sync_markdown_with_merge(local_path, remote_path)
                fingerprint = compute_file_fingerprint(local_path)
                upsert_sync_state("dropbox", local_key, fingerprint)
                if remote_changed:
                    stats["uploaded"] += 1
                else:
                    stats["skipped"] += 1
                continue

            fingerprint = compute_file_fingerprint(local_path)
            last_fp = get_sync_state("dropbox", local_key)
            if not full_scan and last_fp == fingerprint:
                stats["skipped"] += 1
                continue
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
        if not FEATURE_NEWS_ENABLED:
            return ["新聞功能已關閉。"]
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

    news_files: list[Path] = []
    if FEATURE_NEWS_ENABLED:
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
    if scope == "news":
        if not FEATURE_NEWS_ENABLED:
            return ["新聞功能已關閉。"]
        return build_scoped_summary(day, "news")
    if scope not in {"all", "note"}:
        return ["不支援的摘要範圍，請使用 /summary、/summary note 或 /summary news。"]
    return build_note_digest(day)


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
    if not FEATURE_NEWS_ENABLED:
        return ["新聞功能已關閉。"]

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
                """,
                (cutoff_iso,),
            ).fetchall()
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
            "News commands: /news latest [YYYY-MM-DD], /news search <keywords>, /news sources, /news debug"
        ]
    return ["Unknown /news subcommand. Use /news help."]


def set_telegram_commands() -> None:
    commands = [{"command": "summary_note", "description": "3-day note digest"}]
    if FEATURE_NEWS_ENABLED:
        commands.extend(
            [
                {"command": "news_latest", "description": "Latest digest (3 days)"},
                {"command": "news_sources", "description": "List news sources"},
                {"command": "news_debug", "description": "Debug ingestion"},
                {"command": "news_help", "description": "News command help"},
            ]
        )
    if FEATURE_TRANSCRIBE_ENABLED:
        commands.append({"command": "transcribe", "description": "Transcribe url/audio to markdown"})
    commands.append({"command": "status", "description": "Bot health status"})
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
    try:
        data = resp.json() if resp.content else {}
    except Exception:
        data = {}
    print(f"sendMessage status={resp.status_code} ok={bool(data.get('ok'))}")


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


def _fmt_diag_time(ts: float) -> str:
    if not ts:
        return "never"
    try:
        dt = datetime.fromtimestamp(ts, tz=get_local_tz())
        age_sec = max(0, int(time.time() - ts))
        return f"{dt.strftime('%Y-%m-%d %H:%M:%S %Z')} ({age_sec}s ago)"
    except Exception:
        return "error"


def build_status_report() -> str:
    now = datetime.now(tz=get_local_tz())
    day = now.strftime("%Y-%m-%d")
    start_3d = (now - timedelta(days=2)).replace(hour=0, minute=0, second=0, microsecond=0)
    notes_3d = 0
    news_24h = 0

    try:
        for d in range(3):
            target_day = (start_3d + timedelta(days=d)).strftime("%Y-%m-%d")
            day_files, _ = _summary_files_for_day(target_day)
            notes_3d += len(day_files)
    except Exception:
        notes_3d = -1

    if FEATURE_NEWS_ENABLED:
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

    lines = [
        f"status time: {now.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        f"ai summary: {'on' if AI_SUMMARY_ENABLED else 'off'}",
        f"ai provider/model: {AI_SUMMARY_PROVIDER}/{_resolve_ai_model_name()}",
        f"ollama health: {_ollama_health()}",
        f"db path exists: {'yes' if DB_PATH.exists() else 'no'}",
    ]
    if APP_PROFILE == "chitchat":
        notion_ready = "yes" if _notion_is_chitchat_enabled() else "no"
        notion_year_page = _notion_get_chatlog_year_page_id(now)
        lines.append(f"notion chitchat sync: {notion_ready}")
        lines.append(f"notion target page ({now.year}): {notion_year_page or 'missing'}")
    if FEATURE_NEWS_ENABLED:
        lines.append(f"news clusters (24h): {news_24h}")
    lines.append(f"telegram mode: {'long_polling' if TELEGRAM_LONG_POLLING else 'webhook'}")
    if TELEGRAM_LONG_POLLING:
        lines.append(f"telegram poll last ok: {_fmt_diag_time(_telegram_poll_last_ok_at)}")
        lines.append(f"telegram poll last update: {_fmt_diag_time(_telegram_poll_last_update_at)}")
        lines.append(
            "telegram poll last update id: "
            + (str(_telegram_poll_last_update_id) if _telegram_poll_last_update_id is not None else "n/a")
        )
        lines.append(f"telegram poll last error: {_telegram_poll_last_error or 'none'}")
    lines.append(f"telegram send last ok: {_fmt_diag_time(_telegram_send_last_ok_at)}")
    lines.append(f"telegram send last status: {_telegram_send_last_status or 'n/a'}")
    lines.append(f"telegram send last error: {_telegram_send_last_error or 'none'}")
    lines.extend([
        f"note markdown files (3d): {notes_3d}",
        f"dropbox sync: {'on' if DROPBOX_SYNC_ENABLED else 'off'} ({DROPBOX_SYNC_TIME} {DROPBOX_SYNC_TZ_NAME})",
        f"today: {day}",
    ])
    return "\n".join(lines)


def push_news_to_subscribers() -> None:
    if not FEATURE_NEWS_ENABLED or not NEWS_PUSH_ENABLED:
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
    if not FEATURE_NEWS_ENABLED:
        print("[INFO] News worker disabled by FEATURE_NEWS_ENABLED=0")
        return

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


async def handle_ocr_choice_callback(callback: dict) -> None:
    callback_id = callback.get("id") or ""
    data = (callback.get("data") or "").strip()
    msg = callback.get("message") or {}
    chat_id = (msg.get("chat") or {}).get("id")
    m = OCR_CHOICE_CALLBACK_RE.match(data)
    if not m:
        await answer_callback_query(callback_id, "無效操作", show_alert=False)
        return
    job_id, action = m.group(1), m.group(2).lower()

    job = _get_ocr_choice_job(job_id)
    if not job:
        await answer_callback_query(callback_id, "任務不存在或已過期", show_alert=False)
        return

    (
        _job_id,
        job_chat_id,
        prompt_message_id,
        image_path_raw,
        source_msg_id,
        expires_at,
        status,
        _ocr_done,
        _ocr_md,
        caption_raw,
        message_ts_raw,
    ) = job
    job_msg_ts = _parse_iso_dt_or_none(message_ts_raw)
    job_caption = (caption_raw or "").strip() or None
    if str(chat_id or "") != str(job_chat_id):
        await answer_callback_query(callback_id, "無效操作", show_alert=False)
        return

    now = datetime.now()
    try:
        expire_dt = datetime.fromisoformat(str(expires_at))
    except Exception:
        expire_dt = now

    if status != "pending":
        await answer_callback_query(callback_id, "此圖片已處理", show_alert=False)
        return
    if now >= expire_dt:
        await answer_callback_query(callback_id, "已逾時", show_alert=False)
        _update_ocr_choice_job_status(job_id, "expired", ocr_done=0, ocr_md_saved=0)
        if prompt_message_id:
            await clear_message_inline_keyboard(int(job_chat_id), int(prompt_message_id))
        if prompt_message_id:
            await edit_message(int(job_chat_id), int(prompt_message_id), "圖片已儲存")
        if OCR_CHOICE_TIMEOUT_DEFAULT == "skip":
            asyncio.create_task(
                asyncio.to_thread(
                    notion_append_chitchat_image,
                    image_path=Path(image_path_raw),
                    caption=job_caption,
                    ocr_text=None,
                    msg_ts=job_msg_ts,
                )
            )
        return

    if action == "save":
        await answer_callback_query(callback_id, "已選擇只存圖", show_alert=False)
        _update_ocr_choice_job_status(job_id, "save", ocr_done=0, ocr_md_saved=0)
        if prompt_message_id:
            await clear_message_inline_keyboard(int(job_chat_id), int(prompt_message_id))
        if prompt_message_id:
            await edit_message(int(job_chat_id), int(prompt_message_id), "圖片已儲存")
        asyncio.create_task(
            asyncio.to_thread(
                notion_append_chitchat_image,
                image_path=Path(image_path_raw),
                caption=job_caption,
                ocr_text=None,
                msg_ts=job_msg_ts,
            )
        )
        return

    await answer_callback_query(callback_id, "已開始 OCR", show_alert=False)
    if prompt_message_id:
        await clear_message_inline_keyboard(int(job_chat_id), int(prompt_message_id))
    image_path = Path(image_path_raw)
    if not image_path.exists():
        _update_ocr_choice_job_status(job_id, "ocr_failed", ocr_done=0, ocr_md_saved=0)
        if prompt_message_id:
            await edit_message(int(job_chat_id), int(prompt_message_id), "OCR 失敗")
        return

    ocr_ok, md_saved, ocr_text = await asyncio.to_thread(
        _run_ocr_on_image,
        chat_id=int(job_chat_id),
        msg_id=source_msg_id or "",
        image_path=image_path,
        msg_ts=datetime.now(),
    )
    await asyncio.to_thread(
        notion_append_chitchat_image,
        image_path=image_path,
        caption=job_caption,
        ocr_text=ocr_text if ocr_ok else None,
        msg_ts=job_msg_ts,
    )
    if ocr_ok and md_saved:
        _update_ocr_choice_job_status(job_id, "run", ocr_done=1, ocr_md_saved=1)
        if prompt_message_id:
            await edit_message(int(job_chat_id), int(prompt_message_id), "圖片已儲存")
    else:
        _update_ocr_choice_job_status(job_id, "ocr_failed", ocr_done=int(ocr_ok), ocr_md_saved=int(md_saved))
        if prompt_message_id:
            await edit_message(int(job_chat_id), int(prompt_message_id), "OCR 失敗")


def expire_ocr_choice_jobs_once() -> None:
    for job_id, chat_id, prompt_message_id, image_path_raw, caption_raw, message_ts_raw in _list_expired_ocr_choice_jobs():
        _update_ocr_choice_job_status(job_id, "expired", ocr_done=0, ocr_md_saved=0)
        if OCR_CHOICE_TIMEOUT_DEFAULT == "skip":
            notion_append_chitchat_image(
                image_path=Path(image_path_raw),
                caption=(caption_raw or "").strip() or None,
                ocr_text=None,
                msg_ts=_parse_iso_dt_or_none(message_ts_raw),
            )
        if prompt_message_id:
            try:
                requests.post(
                    f"{TELEGRAM_API}/editMessageReplyMarkup",
                    json={
                        "chat_id": int(chat_id),
                        "message_id": int(prompt_message_id),
                        "reply_markup": {"inline_keyboard": []},
                    },
                    timeout=10,
                )
                requests.post(
                    f"{TELEGRAM_API}/editMessageText",
                    json={
                        "chat_id": int(chat_id),
                        "message_id": int(prompt_message_id),
                        "text": "圖片已儲存",
                    },
                    timeout=10,
                )
            except Exception as e:
                print(f"[WARN] OCR choice expire edit failed: {e}")


def start_ocr_choice_expire_thread() -> None:
    if not (FEATURE_OCR_ENABLED and FEATURE_OCR_CHOICE_ENABLED):
        return

    def loop():
        while True:
            try:
                expire_ocr_choice_jobs_once()
            except Exception as e:
                print(f"[WARN] OCR choice expire worker error: {e}")
            time.sleep(10)

    t = threading.Thread(target=loop, daemon=True)
    t.start()


def telegram_get_file_info(file_id: str) -> tuple[str, str]:
    last_err: Exception | None = None
    last_detail = "unknown"
    methods = ("get", "post")
    for attempt in range(TELEGRAM_FILE_FETCH_RETRIES):
        for method in methods:
            try:
                kwargs = {
                    "timeout": (
                        TELEGRAM_FILE_FETCH_CONNECT_TIMEOUT,
                        TELEGRAM_FILE_FETCH_READ_TIMEOUT,
                    ),
                }
                if method == "get":
                    kwargs["params"] = {"file_id": file_id}
                else:
                    kwargs["data"] = {"file_id": file_id}
                resp = requests.request(method, f"{TELEGRAM_API}/getFile", **kwargs)
                status = int(resp.status_code or 0)
                if status != 200:
                    preview = (resp.text or "").strip().replace("\n", " ")
                    if preview:
                        preview = preview[:160]
                    last_detail = f"HTTP {status} {preview}".strip()
                    if status == 429:
                        retry_after = 0.0
                        try:
                            retry_after = float(
                                (resp.json() or {})
                                .get("parameters", {})
                                .get("retry_after", 0)
                            )
                        except Exception:
                            retry_after = 0.0
                        if retry_after > 0:
                            time.sleep(retry_after)
                    continue

                data = resp.json()
                if not data.get("ok"):
                    last_detail = f"ok=false {str(data)[:200]}"
                    continue
                file_path = data.get("result", {}).get("file_path")
                if not file_path:
                    last_detail = "result missing file_path"
                    continue
                file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
                return file_url, file_path
            except Exception as e:
                last_err = e
                last_detail = f"{type(e).__name__}: {e}"
                continue
        if attempt < TELEGRAM_FILE_FETCH_RETRIES - 1:
            time.sleep(TELEGRAM_FILE_FETCH_RETRY_DELAY_SECONDS * (attempt + 1))
    detail = last_detail
    if last_err:
        detail = f"{detail}; last_err={type(last_err).__name__}: {last_err}"
    print(f"[WARN] telegram_get_file_info failed file_id={file_id}: {detail}")
    raise RuntimeError(f"無法取得 Telegram 檔案資訊（getFile 失敗）: {detail}")


def telegram_polling_loop() -> None:
    global _telegram_poll_last_ok_at, _telegram_poll_last_update_at, _telegram_poll_last_update_id, _telegram_poll_last_error
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
                body = (resp.text or "").strip().replace("\n", " ")
                body = body[:300]
                _telegram_poll_last_error = f"http_{resp.status_code}: {body}" if body else f"http_{resp.status_code}"
                print(f"[WARN] Telegram getUpdates http={resp.status_code} body={body}")
                time.sleep(2)
                continue
            data = resp.json()
            _telegram_poll_last_ok_at = time.time()
            if not data.get("ok"):
                desc = str(data.get("description") or "unknown_error")
                _telegram_poll_last_error = f"telegram_not_ok: {desc}"
                print(f"[WARN] Telegram getUpdates ok=false: {desc}")
                time.sleep(2)
                continue
            _telegram_poll_last_error = ""
            for update in data.get("result", []):
                update_id = update.get("update_id")
                if update_id is not None:
                    offset = update_id + 1
                    _telegram_poll_last_update_id = int(update_id)
                # Mark update as seen immediately so /status reflects the latest received update.
                _telegram_poll_last_update_at = time.time()
                try:
                    asyncio.run(process_telegram_update(update))
                except Exception as e:
                    print(f"[WARN] telegram update process error: {e}")
                    print(traceback.format_exc())
        except Exception as e:
            _telegram_poll_last_error = f"{type(e).__name__}: {e}"
            print(f"[ERROR] Telegram polling error: {e}")
            time.sleep(2)


def start_telegram_polling_thread() -> None:
    if not TELEGRAM_LONG_POLLING:
        return
    t = threading.Thread(target=telegram_polling_loop, daemon=True)
    t.start()


async def process_telegram_update(update: dict) -> None:
    callback = update.get("callback_query") or {}
    if callback:
        await handle_ocr_choice_callback(callback)
        return

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
    has_audio_attachment = FEATURE_TRANSCRIBE_ENABLED and (_extract_audio_attachment(message) is not None)

    cmd_text = text.strip()
    lower_text = cmd_text.lower()
    is_local_control_text = (
        lower_text.startswith("open ")
        or lower_text in {"notepad", "open notepad", "sort downloads", "help", "/help"}
    )
    is_slash_command = cmd_text.startswith("/")
    should_store_text = bool(cmd_text) and not is_slash_command and not is_local_control_text

    if (is_group or is_private) and chat_id and should_store_text:
        store_message("telegram", str(chat_id), chat_title or chat_type or "", user_id, user_name, text, msg_ts)
        append_markdown("telegram", chat_title or ("private" if is_private else str(chat_id)), user_name, text, msg_ts)
        # Caption text for image messages is synced through notion_append_chitchat_image;
        # skip plain text append here to avoid duplicated caption entries in Notion.
        if not has_image:
            await asyncio.to_thread(notion_append_chitchat_text, text, msg_ts)

    if is_private and chat_id and has_audio_attachment and not has_image:
        handled = await handle_transcribe_audio_message(chat_id, message, message_ts=msg_ts)
        if handled:
            return

    if is_private and chat_id and text and not has_image and not has_audio_attachment:
        try:
            if await handle_transcribe_text_command(chat_id, cmd_text, message_ts=msg_ts):
                return
            if await handle_transcribe_auto_url_message(chat_id, cmd_text, message_ts=msg_ts):
                return

            # For normal recorded text, always send a clear ACK.
            if not cmd_text.startswith("/") and not is_local_control_text:
                sent_id = await send_message(chat_id, "已成功紀錄")
                if sent_id is None:
                    print(f"[WARN] ack send failed chat_id={chat_id}")
                return

            replies, parse_mode, disable_preview = route_user_text_command(cmd_text, str(chat_id))
            for reply in replies:
                for chunk in _chunk_text_for_telegram(reply):
                    sent_id = await send_message(
                        chat_id,
                        chunk,
                        parse_mode=parse_mode,
                        disable_web_page_preview=disable_preview,
                    )
                    if sent_id is None:
                        print(f"[WARN] reply send failed chat_id={chat_id} text_preview={chunk[:80]!r}")
        except Exception as e:
            print(f"[WARN] private command handling failed: {type(e).__name__}: {e}")
            await send_message(chat_id, f"指令處理失敗：{type(e).__name__}")
        return

    if FEATURE_OCR_ENABLED and (is_private or is_group) and chat_id and has_image:
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
            try:
                out_path, _img_bytes = await asyncio.to_thread(
                    _save_telegram_image,
                    file_id=file_id,
                    file_unique_id=file_unique_id,
                    chat_id=chat_id,
                    msg_id=msg_id,
                    msg_ts=msg_ts,
                )
            except Exception as e:
                print(f"[WARN] Telegram image save failed: {e}")
                if is_private:
                    await send_message(chat_id, f"Image save failed: {type(e).__name__} {e}")
                return

            if _is_ocr_choice_enabled_for_chat(is_private=is_private, is_group=is_group):
                job_id = str(uuid.uuid4())
                prompt_text = "是否進行 OCR？"
                keyboard = _build_ocr_choice_keyboard(job_id)
                prompt_message_id = await send_message(chat_id, prompt_text, reply_markup=keyboard)
                if not prompt_message_id:
                    await asyncio.to_thread(
                        notion_append_chitchat_image,
                        image_path=out_path,
                        caption=text or None,
                        ocr_text=None,
                        msg_ts=msg_ts,
                    )
                    await send_message(chat_id, "圖片已儲存")
                    return
                created_job_id = await asyncio.to_thread(
                    _create_ocr_choice_job,
                    job_id=job_id,
                    chat_id=chat_id,
                    prompt_message_id=prompt_message_id,
                    image_path=out_path,
                    file_unique_id=file_unique_id,
                    source_msg_id=msg_id,
                    caption=text or None,
                    message_ts=msg_ts,
                )
                if created_job_id != job_id:
                    print(f"[WARN] OCR choice job id mismatch: generated={job_id} stored={created_job_id}")
                return

            ocr_ok, md_saved, ocr_text = await asyncio.to_thread(
                _run_ocr_on_image,
                chat_id=chat_id,
                msg_id=msg_id,
                image_path=out_path,
                msg_ts=msg_ts,
            )
            await asyncio.to_thread(
                notion_append_chitchat_image,
                image_path=out_path,
                caption=text or None,
                ocr_text=ocr_text if ocr_ok else None,
                msg_ts=msg_ts,
            )
            is_ocr_success = ocr_ok and md_saved
            if is_private:
                await send_message(chat_id, "圖片已儲存" if is_ocr_success else "OCR 失敗")
            else:
                group_status = "ocr succeed" if is_ocr_success else "ocr failed"
                print(f"[INFO] Telegram group image saved: {out_path} ({group_status})")
            return

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
    if not FEATURE_SLACK_ENABLED:
        print("[INFO] Slack worker disabled by FEATURE_SLACK_ENABLED=0")
        return
    t = threading.Thread(target=start_slack_socket_mode, daemon=True)
    t.start()


def start_background_workers_once() -> None:
    global _startup_done
    with _startup_lock:
        if _startup_done:
            return
        set_telegram_commands()
        start_news_thread()
        start_slack_thread()
        start_dropbox_sync_thread()
        start_ocr_choice_expire_thread()
        start_telegram_polling_thread()
        _startup_done = True


@app.on_event("startup")
def on_startup() -> None:
    start_background_workers_once()
