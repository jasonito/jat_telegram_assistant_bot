"""
Media transcription helpers used by Telegram bot.

Supports:
- YouTube URLs
- Apple Podcasts URLs
- Direct audio URLs (.mp3/.m4a/.wav/.ogg/.aac/.flac/.wma/.opus)
- Uploaded local audio files
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

import whisper
import yt_dlp
import feedparser
from slugify import slugify

MAX_DURATION_SECONDS = int(os.getenv("TRANSCRIBE_MAX_DURATION_SECONDS", "10800"))
CHUNK_MINUTES = int(os.getenv("TRANSCRIBE_CHUNK_MINUTES", "25"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base").strip() or "base"
FFMPEG_LOCATION = os.getenv("FFMPEG_LOCATION", "").strip()

ALLOWED_AUDIO_EXTENSIONS = {
    ".mp3",
    ".aac",
    ".wav",
    ".flac",
    ".m4a",
    ".m4b",
    ".ogg",
    ".wma",
    ".opus",
    ".webm",
    ".mp4",
}

YOUTUBE_RE = re.compile(
    r"^https?://((www|m)\.)?(youtube\.com/(watch\?v=[\w-]+|shorts/[\w-]+)|youtu\.be/[\w-]+)(?:[/?#].*)?$"
)
APPLE_RE = re.compile(r"^https?://podcasts\.apple\.com/.+/id(\d+)")
DIRECT_AUDIO_RE = re.compile(r"\.(mp3|m4a|m4b|wav|ogg|aac|flac|wma|opus|webm|mp4)(\?|$)", re.IGNORECASE)
SPOTIFY_RE = re.compile(r"^https?://open\.spotify\.com/")

_model = None
_model_lock = threading.Lock()


def _safe_str(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, (list, tuple)):
        for item in value:
            text = _safe_str(item).strip()
            if text:
                return text
        return ""
    if isinstance(value, dict):
        for key in ("href", "url", "link", "value"):
            if key in value:
                text = _safe_str(value.get(key)).strip()
                if text:
                    return text
        return ""
    return str(value)


def _resolve_media_tool(tool: str) -> str:
    if FFMPEG_LOCATION:
        base = Path(FFMPEG_LOCATION)
        if base.is_dir():
            candidate = base / f"{tool}.exe"
            if candidate.exists():
                return str(candidate)
            candidate = base / tool
            if candidate.exists():
                return str(candidate)
        elif base.exists():
            stem = base.stem.lower()
            if stem == tool:
                return str(base)
            sibling = base.with_name(f"{tool}{base.suffix}")
            if sibling.exists():
                return str(sibling)
    found = shutil.which(tool)
    return found or tool


FFMPEG_BIN = _resolve_media_tool("ffmpeg")
FFPROBE_BIN = _resolve_media_tool("ffprobe")


def get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                _model = whisper.load_model(WHISPER_MODEL)
    return _model


def classify_url(url: str) -> tuple[str | None, str | None]:
    """Return (url_type, error)."""
    if not url:
        return None, "No URL provided."
    if YOUTUBE_RE.match(url):
        if "list=" in url:
            return None, "Playlists are not supported. Please provide a single video URL."
        return "youtube", None
    if APPLE_RE.match(url):
        return "apple", None
    if DIRECT_AUDIO_RE.search(url):
        return "direct", None
    if SPOTIFY_RE.match(url):
        return None, "Spotify is not supported. Please use an Apple Podcasts link instead."
    return None, "Unsupported URL. Supported: YouTube, Apple Podcasts, or direct audio links."


def fetch_apple_podcast_episodes(url: str) -> list[dict]:
    """Return latest episodes [{title, audio_url, show_name, publish_date}]."""
    match = re.search(r"/id(\d+)", url)
    if not match:
        return []

    podcast_id = match.group(1)
    api_url = (
        f"https://itunes.apple.com/lookup?id={podcast_id}"
        f"&country=TW&media=podcast&entity=podcastEpisode&limit=5"
    )

    try:
        req = urllib.request.Request(api_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    if not data.get("results"):
        return []

    query = urllib.parse.urlparse(url).query
    query_i = _safe_str(urllib.parse.parse_qs(query).get("i", [""])[0]).strip()
    target_track_id = int(query_i) if query_i.isdigit() else None
    show_feed_url = ""
    for entry in data["results"]:
        if entry.get("wrapperType") == "podcast":
            show_feed_url = _safe_str(entry.get("feedUrl")).strip()
            break

    def _pick_audio_url(entry: dict) -> str | None:
        # Prefer fields that are usually direct media links.
        for key in ("previewUrl", "enclosureUrl", "assetUrl", "episodeUrl"):
            val = _safe_str(entry.get(key)).strip()
            if val:
                return val
        return None

    out = []
    for entry in data["results"]:
        if entry.get("wrapperType") != "podcastEpisode":
            continue
        episode_url = _pick_audio_url(entry)
        if not episode_url:
            continue
        out.append(
            {
                "title": entry.get("trackName", "Unknown Episode"),
                "audio_url": episode_url,
                "show_name": _safe_str(entry.get("collectionName")),
                "publish_date": _safe_str(entry.get("releaseDate")),
                "track_id": entry.get("trackId"),
                "feed_url": show_feed_url,
            }
        )
    if target_track_id is not None:
        exact = []
        for ep in out:
            ep_track_raw = _safe_str(ep.get("track_id")).strip()
            ep_track_id = int(ep_track_raw) if ep_track_raw.isdigit() else 0
            if ep_track_id == target_track_id:
                exact.append(ep)
        if exact:
            return exact
    return out[:5]


def _get_rss_enclosure_url(feed_url: str, episode_title: str = "", episode_page_url: str = "") -> str | None:
    if not feed_url:
        return None
    try:
        feed = feedparser.parse(feed_url)
    except Exception:
        return None

    entries = list(getattr(feed, "entries", []) or [])
    if not entries:
        return None

    norm_title = _safe_str(episode_title).strip().lower()
    norm_page = _safe_str(episode_page_url).strip()

    def _extract_entry_audio(entry) -> str | None:
        for enc in getattr(entry, "enclosures", []) or []:
            href = _safe_str(enc.get("href")).strip()
            if href:
                return href
        for link in getattr(entry, "links", []) or []:
            href = _safe_str(link.get("href")).strip()
            rel = _safe_str(link.get("rel")).strip().lower()
            typ = _safe_str(link.get("type")).strip().lower()
            if href and (rel == "enclosure" or typ.startswith("audio/")):
                return href
        return None

    if norm_page:
        for entry in entries:
            entry_link = _safe_str(getattr(entry, "link", "")).strip()
            if entry_link and entry_link == norm_page:
                audio = _extract_entry_audio(entry)
                if audio:
                    return audio

    if norm_title:
        for entry in entries:
            entry_title = _safe_str(getattr(entry, "title", "")).strip().lower()
            if entry_title and entry_title == norm_title:
                audio = _extract_entry_audio(entry)
                if audio:
                    return audio
        for entry in entries:
            entry_title = _safe_str(getattr(entry, "title", "")).strip().lower()
            if entry_title and norm_title in entry_title:
                audio = _extract_entry_audio(entry)
                if audio:
                    return audio

    for entry in entries:
        audio = _extract_entry_audio(entry)
        if audio:
            return audio
    return None


def cleanup_files(temp_dir: Path, job_id: str) -> None:
    for pattern in [f"{job_id}.*", f"{job_id}_*"]:
        for f in glob.glob(str(temp_dir / pattern)):
            try:
                os.remove(f)
            except OSError:
                pass


def normalize_audio(input_path: Path, temp_dir: Path, job_id: str) -> Path:
    output_path = temp_dir / f"{job_id}_normalized.wav"
    try:
        result = subprocess.run(
            [FFMPEG_BIN, "-i", str(input_path), "-ar", "16000", "-ac", "1", "-y", str(output_path)],
            capture_output=True,
            timeout=180,
        )
        if result.returncode == 0 and output_path.exists():
            return output_path
    except Exception:
        pass
    return input_path


def get_audio_duration(file_path: Path) -> float:
    try:
        result = subprocess.run(
            [
                FFPROBE_BIN,
                "-v",
                "quiet",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass
    return 0


def split_audio(input_path: Path, temp_dir: Path, job_id: str, chunk_seconds: int | None = None) -> list[Path]:
    if chunk_seconds is None:
        chunk_seconds = CHUNK_MINUTES * 60

    duration = get_audio_duration(input_path)
    if duration <= 0 or duration <= chunk_seconds:
        return [input_path]

    chunks: list[Path] = []
    offset = 0
    idx = 0
    while offset < duration:
        chunk_path = temp_dir / f"{job_id}_chunk{idx}.wav"
        try:
            result = subprocess.run(
                [
                    FFMPEG_BIN,
                    "-i",
                    str(input_path),
                    "-ss",
                    str(offset),
                    "-t",
                    str(chunk_seconds),
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    "-y",
                    str(chunk_path),
                ],
                capture_output=True,
                timeout=300,
            )
            if result.returncode == 0 and chunk_path.exists():
                chunks.append(chunk_path)
            else:
                break
        except Exception:
            break
        offset += chunk_seconds
        idx += 1
    return chunks if chunks else [input_path]


def transcribe_audio(
    audio_path: Path,
    temp_dir: Path,
    job_id: str,
    on_status: Callable[[str], None] | None = None,
) -> str:
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = audio_path.stat().st_size
    if file_size < 1024:
        raise ValueError(f"Audio file is too small ({file_size} bytes).")

    model = get_model()
    duration = get_audio_duration(audio_path)
    if duration <= 0.1:
        raise ValueError(f"Audio is too short ({duration:.2f} seconds).")

    if duration <= CHUNK_MINUTES * 60:
        result = model.transcribe(str(audio_path))
        return _safe_str(result.get("text")).strip()

    chunks = split_audio(audio_path, temp_dir, job_id)
    if on_status:
        on_status(f"Transcribing {len(chunks)} segments...")

    texts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        if on_status:
            on_status(f"Transcribing segment {idx}/{len(chunks)}...")
        result = model.transcribe(str(chunk))
        txt = _safe_str(result.get("text")).strip()
        if txt:
            texts.append(txt)
    return " ".join(texts).strip()


def _safe_filename(title: str) -> str:
    date_str = datetime.date.today().isoformat()
    title_slug = slugify(title, max_length=80) or "transcript"
    return f"{date_str}_{title_slug}.md"


def save_transcript_md(
    transcript_dir: Path,
    title: str,
    source_url: str,
    source_type: str,
    duration_seconds: int | None,
    text: str,
) -> Path:
    transcript_dir.mkdir(parents=True, exist_ok=True)
    filename = _safe_filename(title)
    base_stem = Path(filename).stem
    filepath = transcript_dir / filename
    counter = 1
    while filepath.exists():
        filepath = transcript_dir / f"{base_stem}-{counter}.md"
        counter += 1

    duration_str = (
        f"{duration_seconds // 60}:{duration_seconds % 60:02d}" if duration_seconds else "Unknown"
    )
    date_str = datetime.date.today().isoformat()
    content = (
        f"# {title}\n\n"
        f"- **Source:** {source_url}\n"
        f"- **Type:** {source_type}\n"
        f"- **Date transcribed:** {date_str}\n"
        f"- **Duration:** {duration_str}\n\n"
        f"---\n\n"
        f"{text}\n"
    )
    filepath.write_text(content, encoding="utf-8")
    return filepath


def _pipeline_youtube(
    url: str,
    transcript_dir: Path,
    temp_dir: Path,
    job_id: str,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, Path]:
    try:
        if on_status:
            on_status("Checking video info...")
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True}) as ydl:
            info = ydl.extract_info(url, download=False)

        duration = int(info.get("duration", 0) or 0)
        if duration and duration > MAX_DURATION_SECONDS:
            raise ValueError(
                f"Video too long ({duration // 60} min). Max is {MAX_DURATION_SECONDS // 60} minutes."
            )

        title = info.get("title", "Unknown")
        if on_status:
            on_status(f"Downloading '{title}'...")

        output_path = temp_dir / job_id
        ydl_opts = {
            "format": "bestaudio[ext=webm]/bestaudio[ext=m4a]/bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": str(output_path),
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
        }
        if FFMPEG_LOCATION:
            ydl_opts["ffmpeg_location"] = FFMPEG_LOCATION
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        audio_file = output_path.with_suffix(".mp3")
        if not audio_file.exists():
            raise FileNotFoundError("Failed to download audio.")

        normalized = normalize_audio(audio_file, temp_dir, job_id)
        if on_status:
            on_status(f"Transcribing '{title}'...")
        text = transcribe_audio(normalized, temp_dir, job_id, on_status=on_status)
        out_path = save_transcript_md(transcript_dir, title, url, "youtube", duration or None, text)
        return title, out_path
    finally:
        cleanup_files(temp_dir, job_id)


def _pipeline_podcast(
    audio_url: str,
    title: str,
    source_url: str,
    source_type: str,
    transcript_dir: Path,
    temp_dir: Path,
    job_id: str,
    fallback_audio_urls: list[str] | None = None,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, Path]:
    def _download_via_ytdlp(target_url: str, out_base: Path) -> Path:
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": str(out_base),
            "quiet": True,
            "no_warnings": True,
            "retries": 3,
            "fragment_retries": 3,
            "noplaylist": True,
        }
        if FFMPEG_LOCATION:
            ydl_opts["ffmpeg_location"] = FFMPEG_LOCATION
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([target_url])
        files = sorted(temp_dir.glob(f"{job_id}.*"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in files:
            if f.is_file() and not str(f).endswith(".part") and f.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS:
                return f
        raise FileNotFoundError("Failed to download audio via yt-dlp.")

    def _looks_like_html(file_path: Path) -> bool:
        try:
            with file_path.open("rb") as fh:
                head = fh.read(512).lower()
            return b"<html" in head or b"<!doctype html" in head
        except OSError:
            return False

    def _download_direct(target_url: str, out_path: Path) -> Path:
        req = urllib.request.Request(target_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            with out_path.open("wb") as fh:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    fh.write(chunk)
        return out_path

    def _try_fallback_audio_urls() -> Path | None:
        for idx, candidate in enumerate(fallback_audio_urls or [], start=1):
            if not candidate or candidate == audio_url:
                continue
            ext_match2 = re.search(r"\.(mp3|m4a|m4b|wav|ogg|aac|flac|wma|opus|webm|mp4)", candidate, re.IGNORECASE)
            ext2 = ext_match2.group(1).lower() if ext_match2 else "mp3"
            candidate_file = temp_dir / f"{job_id}_fb{idx}.{ext2}"
            try:
                if on_status:
                    on_status(f"Trying fallback audio source #{idx}...")
                _download_direct(candidate, candidate_file)
                if candidate_file.exists() and candidate_file.stat().st_size >= 1024:
                    if not _looks_like_html(candidate_file) and get_audio_duration(candidate_file) > 0.1:
                        return candidate_file
            except Exception:
                continue
        return None

    try:
        if on_status:
            on_status(f"Downloading '{title}'...")

        ext_match = re.search(r"\.(mp3|m4a|wav|ogg|aac|flac|wma|opus)", audio_url, re.IGNORECASE)
        ext = ext_match.group(1).lower() if ext_match else "mp3"
        audio_file = temp_dir / f"{job_id}.{ext}"

        try:
            _download_direct(audio_url, audio_file)
        except urllib.error.HTTPError as e:
            raise ValueError(f"Failed to download audio (HTTP {e.code}).") from e

        if not audio_file.exists():
            raise FileNotFoundError("Failed to download audio.")
        if audio_file.stat().st_size < 1024:
            raise ValueError("Downloaded audio file is too small.")
        if _looks_like_html(audio_file):
            if on_status:
                on_status("Direct link is a webpage, trying RSS/fallback sources...")
            fallback_file = _try_fallback_audio_urls()
            if fallback_file is not None:
                audio_file = fallback_file
            else:
                if on_status:
                    on_status("Fallback sources failed, retrying with yt-dlp...")
                audio_file = _download_via_ytdlp(source_url, temp_dir / job_id)
        elif get_audio_duration(audio_file) <= 0.1:
            if on_status:
                on_status("Direct audio probe failed, trying RSS/fallback sources...")
            fallback_file = _try_fallback_audio_urls()
            if fallback_file is not None:
                audio_file = fallback_file
            else:
                if on_status:
                    on_status("Fallback sources failed, retrying with yt-dlp...")
                audio_file = _download_via_ytdlp(source_url, temp_dir / job_id)

        normalized = normalize_audio(audio_file, temp_dir, job_id)
        if on_status:
            on_status(f"Transcribing '{title}'...")
        text = transcribe_audio(normalized, temp_dir, job_id, on_status=on_status)
        out_path = save_transcript_md(transcript_dir, title, source_url, source_type, None, text)
        return title, out_path
    finally:
        cleanup_files(temp_dir, job_id)


def transcribe_url_to_markdown(
    url: str,
    transcript_dir: Path,
    temp_dir: Path,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    url_type, error = classify_url(url)
    if error:
        raise ValueError(error)

    if url_type == "youtube":
        return _pipeline_youtube(url, transcript_dir, temp_dir, job_id, on_status=on_status)

    if url_type == "apple":
        episodes = fetch_apple_podcast_episodes(url)
        if not episodes:
            raise ValueError("No episodes found for this Apple Podcasts URL.")
        episode = episodes[0]
        title = episode["title"]
        if episode.get("show_name"):
            title = f"{episode['show_name']} - {episode['title']}"
        rss_audio_url = _get_rss_enclosure_url(
            _safe_str(episode.get("feed_url")).strip(),
            episode_title=episode.get("title", ""),
            episode_page_url=episode.get("audio_url", ""),
        )
        fallback_audio_urls = [u for u in [rss_audio_url] if u]
        return _pipeline_podcast(
            episode["audio_url"],
            title,
            source_url=url,
            source_type="podcast",
            transcript_dir=transcript_dir,
            temp_dir=temp_dir,
            job_id=job_id,
            fallback_audio_urls=fallback_audio_urls,
            on_status=on_status,
        )

    if url_type == "direct":
        filename_part = url.split("/")[-1].split("?")[0]
        title = urllib.parse.unquote(filename_part).rsplit(".", 1)[0] or "Audio"
        return _pipeline_podcast(
            url,
            title,
            source_url=url,
            source_type="direct",
            transcript_dir=transcript_dir,
            temp_dir=temp_dir,
            job_id=job_id,
            on_status=on_status,
        )

    raise ValueError("Unsupported URL type.")


def transcribe_upload_to_markdown(
    audio_path: Path,
    original_filename: str,
    transcript_dir: Path,
    temp_dir: Path,
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    job_id = str(uuid.uuid4())
    try:
        if on_status:
            on_status(f"Preparing '{original_filename}'...")
        normalized = normalize_audio(audio_path, temp_dir, job_id)
        if on_status:
            on_status(f"Transcribing '{original_filename}'...")
        text = transcribe_audio(normalized, temp_dir, job_id, on_status=on_status)
        text = _safe_str(text)
        if not text:
            raise ValueError("Transcription produced no text.")

        first_sentence = re.split(r"[.!?]\s|\n", text.strip(), maxsplit=1)[0].strip()
        title = first_sentence[:80].strip() if first_sentence else ""
        if len(title) < 5:
            title = Path(original_filename).stem or "Audio upload"
        duration = int(get_audio_duration(audio_path)) or None
        out_path = save_transcript_md(
            transcript_dir,
            title,
            source_url=f"Local upload: {original_filename}",
            source_type="upload",
            duration_seconds=duration,
            text=text,
        )
        return title, out_path
    finally:
        cleanup_files(temp_dir, job_id)
