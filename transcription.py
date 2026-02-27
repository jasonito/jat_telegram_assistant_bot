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
from slugify import slugify

MAX_DURATION_SECONDS = int(os.getenv("TRANSCRIBE_MAX_DURATION_SECONDS", "10800"))
CHUNK_MINUTES = int(os.getenv("TRANSCRIBE_CHUNK_MINUTES", "25"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base").strip() or "base"
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "zh").strip() or "zh"
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
FFMPEG_LOCATION = os.getenv("FFMPEG_LOCATION", "").strip()

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".aac", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".opus"}

YOUTUBE_RE = re.compile(
    r"^https?://((www|m)\.)?(youtube\.com/(watch\?v=[\w-]+|shorts/[\w-]+)|youtu\.be/[\w-]+)(?:[/?#].*)?$"
)
APPLE_RE = re.compile(r"^https?://podcasts\.apple\.com/.+/id(\d+)")
DIRECT_AUDIO_RE = re.compile(r"\.(mp3|m4a|wav|ogg|aac|flac|wma|opus)(\?|$)", re.IGNORECASE)
SPOTIFY_RE = re.compile(r"^https?://open\.spotify\.com/")

_model = None
_model_lock = threading.Lock()


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

    out = []
    for entry in data["results"]:
        if entry.get("wrapperType") != "podcastEpisode":
            continue
        episode_url = entry.get("episodeUrl")
        if not episode_url:
            continue
        out.append(
            {
                "title": entry.get("trackName", "Unknown Episode"),
                "audio_url": episode_url,
                "show_name": entry.get("collectionName", ""),
                "publish_date": entry.get("releaseDate", ""),
            }
        )
    return out[:3]


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
        result = model.transcribe(
            str(audio_path),
            language=WHISPER_LANGUAGE,
            beam_size=WHISPER_BEAM_SIZE,
        )
        return (result.get("text") or "").strip()

    chunks = split_audio(audio_path, temp_dir, job_id)
    if on_status:
        on_status(f"Transcribing {len(chunks)} segments...")

    texts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        if on_status:
            on_status(f"Transcribing segment {idx}/{len(chunks)}...")
        result = model.transcribe(
            str(chunk),
            language=WHISPER_LANGUAGE,
            beam_size=WHISPER_BEAM_SIZE,
        )
        txt = (result.get("text") or "").strip()
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
    on_status: Callable[[str], None] | None = None,
) -> tuple[str, Path]:
    try:
        if on_status:
            on_status(f"Downloading '{title}'...")

        ext_match = re.search(r"\.(mp3|m4a|wav|ogg|aac|flac|wma|opus)", audio_url, re.IGNORECASE)
        ext = ext_match.group(1).lower() if ext_match else "mp3"
        audio_file = temp_dir / f"{job_id}.{ext}"

        req = urllib.request.Request(audio_url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                with audio_file.open("wb") as fh:
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        fh.write(chunk)
        except urllib.error.HTTPError as e:
            raise ValueError(f"Failed to download audio (HTTP {e.code}).") from e

        if not audio_file.exists():
            raise FileNotFoundError("Failed to download audio.")
        if audio_file.stat().st_size < 1024:
            raise ValueError("Downloaded audio file is too small.")

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
        return _pipeline_podcast(
            episode["audio_url"],
            title,
            source_url=url,
            source_type="podcast",
            transcript_dir=transcript_dir,
            temp_dir=temp_dir,
            job_id=job_id,
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
