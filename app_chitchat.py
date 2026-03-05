import os

# Chitchat bot default profile: transcription on, news off.
os.environ.setdefault("APP_PROFILE", "chitchat")
os.environ.setdefault("NEWS_ENABLED", "0")
os.environ.setdefault("FEATURE_NEWS_ENABLED", "0")
os.environ.setdefault("FEATURE_TRANSCRIBE_ENABLED", "1")
os.environ.setdefault("FEATURE_TRANSCRIBE_AUTO_URL", "1")
os.environ.setdefault("FEATURE_OCR_ENABLED", "1")
os.environ.setdefault("FEATURE_OCR_CHOICE_ENABLED", "1")
os.environ.setdefault("OCR_CHOICE_SCOPE", "private")
os.environ.setdefault("OCR_CHOICE_TIMEOUT_SECONDS", "60")
os.environ.setdefault("FEATURE_SLACK_ENABLED", "0")
os.environ.setdefault("AI_SUMMARY_PROVIDER", "ollama")
os.environ.setdefault("OLLAMA_MODEL", "qwen3.5:9b")
os.environ.setdefault("AI_SUMMARY_TEMPERATURE", "0.85")
os.environ.setdefault("WEEKLY_REPORT_PUSH_ENABLED", "1")
os.environ.setdefault("WEEKLY_REPORT_PUSH_WEEKDAY", "1")
os.environ.setdefault("WEEKLY_REPORT_PUSH_TIME", "21:00")
os.environ.setdefault("WEEKLY_REPORT_PUSH_TZ", "Asia/Taipei")

from app import app  # noqa: E402,F401
