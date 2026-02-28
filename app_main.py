import os

# Main bot default profile: keep sane defaults while allowing .env overrides.
os.environ.setdefault("APP_PROFILE", "main")
os.environ.setdefault("NEWS_ENABLED", "1")
os.environ.setdefault("FEATURE_NEWS_ENABLED", "1")
os.environ.setdefault("FEATURE_TRANSCRIBE_ENABLED", "1")
os.environ.setdefault("FEATURE_TRANSCRIBE_AUTO_URL", "1")
os.environ.setdefault("FEATURE_OCR_ENABLED", "1")
os.environ.setdefault("FEATURE_OCR_CHOICE_ENABLED", "1")
os.environ.setdefault("OCR_CHOICE_SCOPE", "private")
os.environ.setdefault("OCR_CHOICE_TIMEOUT_SECONDS", "60")
os.environ.setdefault("FEATURE_SLACK_ENABLED", "0")

from app import app  # noqa: E402,F401
