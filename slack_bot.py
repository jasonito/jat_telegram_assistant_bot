import re
import threading
from datetime import datetime
from threading import Event

from slack_bolt import App as SlackApp
from slack_bolt.adapter.socket_mode import SocketModeHandler

from config import (
    logger,
    SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_USER_ID, SLACK_DEBUG,
)
from storage import (
    store_message, append_markdown, search_notes, process_slack_note,
)
from ai import merge_all, summary_ai, summary_weekly


def start_slack_socket_mode() -> None:
    if not (SLACK_BOT_TOKEN and SLACK_APP_TOKEN and SLACK_USER_ID):
        logger.info("Slack tokens/user not set; skipping Slack Socket Mode.")
        return

    slack_app = SlackApp(token=SLACK_BOT_TOKEN)
    if SLACK_DEBUG:
        logger.info("Slack Socket Mode starting...")

    @slack_app.command("/n")
    def handle_slack_note(ack, respond, command):
        ack()
        text = (command.get("text") or "").strip()
        if not text:
            respond("Usage: /n <text>")
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

    @slack_app.command("/search")
    def handle_slack_search(ack, respond, command):
        ack()
        raw = (command.get("text") or "").strip()
        if not raw:
            respond("Usage: /search <keyword> [--all]")
            return
        tokens = raw.split()
        all_channels = "--all" in tokens
        keyword = " ".join(t for t in tokens if t != "--all").strip()
        if not keyword:
            respond("Usage: /search <keyword> [--all]")
            return
        channel_id = None if all_channels else command.get("channel_id", "")
        rows = search_notes(keyword, channel_id=channel_id, limit=8)
        if not rows:
            respond("No results.")
            return
        lines = []
        for platform, ch_id, user_name, msg_text, received_ts, bucket, meeting_id in rows:
            ts = (received_ts or "")[:19]
            meta = f"{bucket}"
            if meeting_id:
                meta = f"{bucket}:{meeting_id}"
            prefix = f"({platform}) {ch_id} {meta}"
            lines.append(f"- [{ts}] {prefix} | {user_name}: {msg_text}")
        respond("\n".join(lines))

    @slack_app.command("/merge")
    def handle_slack_merge(ack, respond, command):
        ack()
        text = (command.get("text") or "").strip()
        day = text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text or "") else datetime.now().strftime("%Y-%m-%d")
        parts = merge_all(day)
        respond("\n".join(parts))

    @slack_app.command("/summary")
    def handle_slack_summary(ack, respond, command):
        ack()
        text = (command.get("text") or "").strip()
        day = text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text or "") else datetime.now().strftime("%Y-%m-%d")
        parts = summary_ai(day)
        respond("\n".join(parts))

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
            logger.debug("SLACK event received: keys=%s", list(event.keys()))
        if event.get("subtype"):
            if SLACK_DEBUG:
                logger.debug("SLACK ignored: subtype present")
            return
        if event.get("bot_id"):
            if SLACK_DEBUG:
                logger.debug("SLACK ignored: bot message")
            return
        channel_type = event.get("channel_type")
        if channel_type and channel_type != "im":
            if SLACK_DEBUG:
                logger.debug("SLACK ignored: channel_type=%s", channel_type)
            return
        if event.get("user") != SLACK_USER_ID:
            if SLACK_DEBUG:
                logger.debug("SLACK ignored: user mismatch %s", event.get("user"))
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
        if text.startswith("/n "):
            payload = text[len("/n ") :].strip()
            if payload:
                channel_id = event.get("channel", "")
                user_id = event.get("user", "")
                user_name = user_id
                ok, msg = process_slack_note(channel_id, user_id, user_name, payload, msg_ts)
                say(msg)
                return
        if text.startswith("/search "):
            raw = text[len("/search ") :].strip()
            tokens = raw.split()
            all_channels = "--all" in tokens
            keyword = " ".join(t for t in tokens if t != "--all").strip()
            if not keyword:
                say("Usage: /search <keyword> [--all]")
                return
            channel_id = None if all_channels else event.get("channel", "")
            rows = search_notes(keyword, channel_id=channel_id, limit=8)
            if not rows:
                say("No results.")
                return
            lines = []
            for platform, ch_id, user_name, msg_text, received_ts, bucket, meeting_id in rows:
                ts = (received_ts or "")[:19]
                meta = f"{bucket}"
                if meeting_id:
                    meta = f"{bucket}:{meeting_id}"
                prefix = f"({platform}) {ch_id} {meta}"
                lines.append(f"- [{ts}] {prefix} | {user_name}: {msg_text}")
            say("\n".join(lines))
            return
        if text.startswith("/merge"):
            tokens = text.split()
            day = tokens[1] if len(tokens) > 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[1]) else datetime.now().strftime("%Y-%m-%d")
            parts = merge_all(day)
            say("\n".join(parts))
            return
        if text.startswith("/summary_weekly"):
            tokens = text.split()
            day = tokens[1] if len(tokens) > 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[1]) else datetime.now().strftime("%Y-%m-%d")
            parts = summary_weekly(day)
            say("\n".join(parts))
            return
        if text.startswith("/summary"):
            tokens = text.split()
            day = tokens[1] if len(tokens) > 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[1]) else datetime.now().strftime("%Y-%m-%d")
            parts = summary_ai(day)
            say("\n".join(parts))
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
