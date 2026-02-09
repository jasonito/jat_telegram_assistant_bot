import os
import logging
from pathlib import Path
from datetime import timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("jat_bot")

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).parent / "data")).resolve()
DB_PATH = DATA_DIR / "messages.sqlite"
NEWS_MD_DIR = DATA_DIR / "news"
NOTES_DIR = DATA_DIR / "notes"
TELEGRAM_MD_DIR = NOTES_DIR / "telegram"
SLACK_MD_DIR = NOTES_DIR / "slack"

ALLOWED_GROUPS = {
    g.strip()
    for g in os.getenv("TELEGRAM_ALLOWED_GROUPS", "").split(",")
    if g.strip()
}

# ---------------------------------------------------------------------------
# Slack
# ---------------------------------------------------------------------------
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_APP_TOKEN = os.getenv("SLACK_APP_TOKEN", "")
SLACK_USER_ID = os.getenv("SLACK_USER_ID", "")
SLACK_DEBUG = os.getenv("SLACK_DEBUG", "0").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# AI summary
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# News
# ---------------------------------------------------------------------------
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
    NEWS_TZ = timezone(timedelta(hours=8))
    logger.warning("tzdata not found; falling back to fixed UTC+08:00")


def get_local_tz():
    return NEWS_TZ
