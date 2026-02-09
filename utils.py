import re
import hashlib
import subprocess
import shutil
import time
from pathlib import Path
from datetime import datetime

import requests

from config import logger


# ---------------------------------------------------------------------------
# Retry helper for external HTTP calls
# ---------------------------------------------------------------------------
def request_with_retry(
    method: str,
    url: str,
    *,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    retry_on_status: tuple[int, ...] = (429, 500, 502, 503, 504),
    **kwargs,
) -> requests.Response:
    """HTTP request with exponential backoff retry.

    Retries on network errors and specified HTTP status codes.
    Non-retryable HTTP errors (4xx except 429) are returned immediately.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.request(method, url, **kwargs)
            if resp.status_code not in retry_on_status or attempt == max_retries:
                return resp
            wait = backoff_base ** attempt
            logger.warning(
                "Retryable HTTP %d from %s (attempt %d/%d, wait %.1fs)",
                resp.status_code, url, attempt + 1, max_retries + 1, wait,
            )
            time.sleep(wait)
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_exc = exc
            if attempt == max_retries:
                raise
            wait = backoff_base ** attempt
            logger.warning(
                "Network error on %s (attempt %d/%d, wait %.1fs): %s",
                url, attempt + 1, max_retries + 1, wait, exc,
            )
            time.sleep(wait)
    raise last_exc  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def strip_markdown_noise(text: str) -> str:
    if not text:
        return text
    return text.replace("**", "").replace("__", "").replace("`", "")


def parse_day_arg(text: str) -> str:
    tokens = text.strip().split()
    if len(tokens) > 1 and re.fullmatch(r"\d{4}-\d{2}-\d{2}", tokens[1]):
        return tokens[1]
    return datetime.now().strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Desktop utilities
# ---------------------------------------------------------------------------
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
