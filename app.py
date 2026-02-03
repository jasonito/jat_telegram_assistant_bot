import os
import re
import webbrowser
from pathlib import Path
import subprocess
import shutil
import sqlite3
from datetime import datetime
import threading
from threading import Event
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import requests
from dotenv import load_dotenv

from slack_bolt import App as SlackApp
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data")).resolve()
DB_PATH = DATA_DIR / "messages.sqlite"
MD_DIR = DATA_DIR / "markdown"
ALLOWED_GROUPS = {
    g.strip()
    for g in os.getenv("TELEGRAM_ALLOWED_GROUPS", "").split(",")
    if g.strip()
}

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")
SLACK_USER_ID = os.getenv("SLACK_USER_ID", "")

app = FastAPI()


def init_storage() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MD_DIR.mkdir(parents=True, exist_ok=True)
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

    return "Unknown command. Try: open <url> | notepad | sort downloads"


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
        if ALLOWED_GROUPS and chat_title not in ALLOWED_GROUPS:
            return JSONResponse({"ok": True})

        if text:
            store_message("telegram", str(chat_id), chat_title, user_id, user_name, text, msg_ts)
            append_markdown("telegram", chat_title, user_name, text, msg_ts)
        return JSONResponse({"ok": True})

    if chat_type == "private" and chat_id and text:
        reply = handle_command(text)
        await send_message(chat_id, reply)

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


start_slack_thread()
