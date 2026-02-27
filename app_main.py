import os

# Main bot fixed profile: force values to avoid accidental inheritance from other env files.
os.environ["APP_PROFILE"] = "main"
os.environ["NEWS_ENABLED"] = "1"
os.environ["FEATURE_NEWS_ENABLED"] = "1"
os.environ["FEATURE_TRANSCRIBE_ENABLED"] = "1"
os.environ["FEATURE_TRANSCRIBE_AUTO_URL"] = "1"
os.environ["FEATURE_OCR_ENABLED"] = "1"
os.environ["FEATURE_OCR_CHOICE_ENABLED"] = "1"
os.environ["OCR_CHOICE_SCOPE"] = "private"
os.environ["OCR_CHOICE_TIMEOUT_SECONDS"] = "60"
os.environ["FEATURE_SLACK_ENABLED"] = "0"

from app import app  # noqa: E402,F401
