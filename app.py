"""JAT Telegram Assistant Bot - Application entry point."""

from fastapi import FastAPI

from storage import init_storage
from telegram_bot import telegram_webhook, set_telegram_commands, start_telegram_polling_thread, send_message_sync
from slack_bot import start_slack_thread
from news import start_news_thread

app = FastAPI()

# ---------------------------------------------------------------------------
# Initialize
# ---------------------------------------------------------------------------
init_storage()

# Register Telegram webhook route
app.post("/telegram")(telegram_webhook)

# Start background services
set_telegram_commands()
start_news_thread(send_fn=send_message_sync)
start_slack_thread()
start_telegram_polling_thread()
