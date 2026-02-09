import re
import time
import threading
from pathlib import Path
from datetime import datetime

import requests
from fastapi import Request
from fastapi.responses import JSONResponse

from config import (
    logger,
    BOT_TOKEN, TELEGRAM_API, TELEGRAM_LONG_POLLING, TELEGRAM_LOCAL_WEBHOOK_URL,
    DATA_DIR,
)
from utils import request_with_retry
from storage import store_message, append_markdown
from commands import handle_command, handle_news_command


# ---------------------------------------------------------------------------
# Command shortcuts
# ---------------------------------------------------------------------------
NEWS_SHORTCUTS = {
    "/news_latest": "/news latest",
    "/news_search": "/news search",
    "/news_sources": "/news sources",
    "/news_debug": "/news debug",
    "/news_subscribe": "/news subscribe",
    "/news_unsubscribe": "/news unsubscribe",
    "/news_help": "/news help",
}


# ---------------------------------------------------------------------------
# Telegram API helpers
# ---------------------------------------------------------------------------
async def send_message(chat_id: int, text: str) -> None:
    payload = {"chat_id": chat_id, "text": text}
    try:
        resp = request_with_retry("POST", f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)
        logger.info("sendMessage status=%d body=%s", resp.status_code, resp.text[:200])
    except (requests.ConnectionError, requests.Timeout) as exc:
        logger.error("sendMessage failed after retries: %s", exc)


def send_message_sync(chat_id: str, text: str) -> None:
    payload = {"chat_id": chat_id, "text": text}
    try:
        resp = request_with_retry("POST", f"{TELEGRAM_API}/sendMessage", json=payload, timeout=10)
        logger.info("sendMessage status=%d body=%s", resp.status_code, resp.text[:200])
    except (requests.ConnectionError, requests.Timeout) as exc:
        logger.error("sendMessage failed after retries: %s", exc)


def delete_telegram_webhook(drop_pending: bool = False) -> None:
    try:
        resp = request_with_retry(
            "POST",
            f"{TELEGRAM_API}/deleteWebhook",
            json={"drop_pending_updates": drop_pending},
            timeout=10,
        )
        logger.info("deleteWebhook status=%d body=%s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.error("deleteWebhook error: %s", exc)


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
        resp = request_with_retry(
            "POST",
            f"{TELEGRAM_API}/setMyCommands",
            json={"commands": commands},
            timeout=10,
        )
        logger.info("setMyCommands status=%d body=%s", resp.status_code, resp.text[:200])
    except Exception as exc:
        logger.error("setMyCommands error: %s", exc)


def telegram_get_file_info(file_id: str) -> tuple[str, str] | None:
    try:
        resp = request_with_retry("GET", f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=10)
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


# ---------------------------------------------------------------------------
# Webhook handler (registered on FastAPI app in app.py)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Long polling thread
# ---------------------------------------------------------------------------
def telegram_polling_loop() -> None:
    logger.info("Telegram long polling enabled.")
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
                        logger.warning(
                            "local webhook status=%d body=%s", local_resp.status_code, local_resp.text[:200]
                        )
                except Exception as e:
                    logger.warning("local webhook error: %s", e)
        except Exception as e:
            logger.error("Telegram polling error: %s", e)
            time.sleep(2)


def start_telegram_polling_thread() -> None:
    if not TELEGRAM_LONG_POLLING:
        return
    t = threading.Thread(target=telegram_polling_loop, daemon=True)
    t.start()
