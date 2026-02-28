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
import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Callable

import requests
import yt_dlp
from slugify import slugify

try:
    from faster_whisper import BatchedInferencePipeline, WhisperModel  # type: ignore
except Exception:
    BatchedInferencePipeline = None  # type: ignore
    WhisperModel = None  # type: ignore

try:
    import whisper  # type: ignore
except Exception:
    whisper = None  # type: ignore

MAX_DURATION_SECONDS = int(os.getenv("TRANSCRIBE_MAX_DURATION_SECONDS", "10800"))
CHUNK_MINUTES = int(os.getenv("TRANSCRIBE_CHUNK_MINUTES", "25"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base").strip() or "base"
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "auto").strip() or "auto"
WHISPER_BEAM_SIZE = int(os.getenv("WHISPER_BEAM_SIZE", "5"))
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8").strip() or "int8"
WHISPER_CPU_THREADS = int(os.getenv("WHISPER_CPU_THREADS", "0"))
WHISPER_BATCH_SIZE = int(os.getenv("WHISPER_BATCH_SIZE", "16"))
CHECKPOINT_FLUSH_INTERVAL = int(os.getenv("TRANSCRIBE_CHECKPOINT_FLUSH_SECONDS", "30"))
FFMPEG_LOCATION = os.getenv("FFMPEG_LOCATION", "").strip()

ALLOWED_AUDIO_EXTENSIONS = {".mp3", ".aac", ".wav", ".flac", ".m4a", ".ogg", ".wma", ".opus"}

YOUTUBE_RE = re.compile(
    r"^https?://((www|m)\.)?(youtube\.com/(watch\?v=[\w-]+|shorts/[\w-]+)|youtu\.be/[\w-]+)(?:[/?#].*)?$"
)
APPLE_RE = re.compile(r"^https?://podcasts\.apple\.com/.+/id(\d+)")
DIRECT_AUDIO_RE = re.compile(r"\.(mp3|m4a|wav|ogg|aac|flac|wma|opus)(\?|$)", re.IGNORECASE)
SPOTIFY_RE = re.compile(r"^https?://open\.spotify\.com/")

_model = None
_batched_model = None
_model_lock = threading.Lock()


class JobCancelled(Exception):
    """Raised when a transcription job is cancelled by caller."""


def _raise_if_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event and cancel_event.is_set():
        raise JobCancelled("Cancelled by user.")


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
                if WhisperModel is not None:
                    model_kwargs = {
                        "device": "cpu",
                        "compute_type": WHISPER_COMPUTE_TYPE,
                    }
                    if WHISPER_CPU_THREADS > 0:
                        model_kwargs["cpu_threads"] = WHISPER_CPU_THREADS
                    _model = WhisperModel(WHISPER_MODEL, **model_kwargs)
                elif whisper is not None:
                    _model = whisper.load_model(WHISPER_MODEL)
                else:
                    raise RuntimeError(
                        "No whisper backend available. Install 'faster-whisper' or 'openai-whisper'."
                    )
    return _model


def get_batched_model():
    global _batched_model
    if _batched_model is None:
        with _model_lock:
            if _batched_model is None:
                if BatchedInferencePipeline is None:
                    return None
                _batched_model = BatchedInferencePipeline(model=get_model())
    return _batched_model


def _backend_name() -> str:
    return "faster-whisper" if WhisperModel is not None else "openai-whisper"


def _checkpoint_dir(temp_dir: Path) -> Path:
    cp = temp_dir / "checkpoints"
    cp.mkdir(parents=True, exist_ok=True)
    return cp


def _compute_audio_fingerprint(file_path: Path) -> str:
    file_size = file_path.stat().st_size
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        h.update(f.read(16 * 1024 * 1024))
    h.update(str(file_size).encode("ascii"))
    return h.hexdigest()


def _checkpoint_path(temp_dir: Path, fingerprint: str) -> Path:
    return _checkpoint_dir(temp_dir) / f"{fingerprint}.json"


def _get_whisper_config() -> dict:
    return {
        "backend": _backend_name(),
        "model": WHISPER_MODEL,
        "beam_size": WHISPER_BEAM_SIZE,
        "language": WHISPER_LANGUAGE,
    }


def _resolve_whisper_language() -> str | None:
    raw = (WHISPER_LANGUAGE or "").strip().lower()
    if raw in {"", "auto", "detect", "none"}:
        return None
    return WHISPER_LANGUAGE


def _normalize_sentence_key(text: str) -> str:
    t = re.sub(r"https?://\S+", "", text or "", flags=re.IGNORECASE)
    t = re.sub(r"[\W_]+", "", t, flags=re.UNICODE).lower()
    return t


def _clean_transcript_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    no_urls = re.sub(r"https?://\S+", " ", raw, flags=re.IGNORECASE)
    units = [u.strip() for u in re.split(r"(?<=[\.\!\?。！？])\s+|\n+", no_urls) if u and u.strip()]
    if not units:
        return re.sub(r"\s+", " ", no_urls).strip()
    deduped: list[str] = []
    prev_key = ""
    for u in units:
        key = _normalize_sentence_key(u)
        if not key:
            continue
        if key == prev_key:
            continue
        deduped.append(u)
        prev_key = key
    return re.sub(r"\s+", " ", " ".join(deduped)).strip()


def _load_checkpoint(temp_dir: Path, fingerprint: str, expected_duration: float) -> dict | None:
    path = _checkpoint_path(temp_dir, fingerprint)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            path.unlink()
        except Exception:
            pass
        return None
    if data.get("version") != 1:
        try:
            path.unlink()
        except Exception:
            pass
        return None
    if abs(float(data.get("audio_duration", 0.0)) - expected_duration) > 1.0:
        try:
            path.unlink()
        except Exception:
            pass
        return None
    if data.get("whisper_config") != _get_whisper_config():
        try:
            path.unlink()
        except Exception:
            pass
        return None
    if data.get("last_completed_end", 0) <= 0 or not data.get("segments"):
        return None
    return data


def _save_checkpoint(
    temp_dir: Path,
    fingerprint: str,
    audio_duration: float,
    segments: list[dict],
    source_info: dict | None = None,
) -> None:
    path = _checkpoint_path(temp_dir, fingerprint)
    tmp = path.with_suffix(".json.tmp")
    last_end = float(segments[-1]["end"]) if segments else 0.0
    data = {
        "version": 1,
        "audio_fingerprint": fingerprint,
        "audio_duration": audio_duration,
        "last_completed_end": last_end,
        "segments": segments,
        "whisper_config": _get_whisper_config(),
        "source_info": source_info or {},
        "updated_at": time.time(),
    }
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _delete_checkpoint(temp_dir: Path, fingerprint: str) -> None:
    path = _checkpoint_path(temp_dir, fingerprint)
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def cleanup_old_checkpoints(temp_dir: Path, max_age_seconds: int = 7 * 24 * 3600) -> None:
    cp_dir = _checkpoint_dir(temp_dir)
    now = time.time()
    for item in cp_dir.glob("*.json*"):
        try:
            if now - item.stat().st_mtime > max_age_seconds:
                item.unlink()
        except Exception:
            pass


def _trim_audio_from(input_path: Path, start_seconds: float, output_path: Path) -> Path | None:
    try:
        result = subprocess.run(
            [
                FFMPEG_BIN,
                "-ss",
                str(start_seconds),
                "-i",
                str(input_path),
                "-ar",
                "16000",
                "-ac",
                "1",
                "-y",
                str(output_path),
            ],
            capture_output=True,
            timeout=300,
        )
        if result.returncode == 0 and output_path.exists():
            return output_path
    except Exception:
        pass
    return None


def _is_youtube_url(url: str) -> bool:
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    # youtu.be/<video_id>
    if host == "youtu.be":
        return bool(parsed.path and parsed.path.strip("/") and "/" not in parsed.path.strip("/"))

    if host not in {"youtube.com", "m.youtube.com"}:
        return False

    path = (parsed.path or "").rstrip("/")
    query = urllib.parse.parse_qs(parsed.query or "")
    if path == "/watch":
        return bool(query.get("v", [""])[0])
    if path.startswith("/shorts/"):
        seg = path.split("/", 2)[2] if path.count("/") >= 2 else ""
        return bool(seg)
    if path.startswith("/live/"):
        seg = path.split("/", 2)[2] if path.count("/") >= 2 else ""
        return bool(seg)
    return False


def classify_url(url: str) -> tuple[str | None, str | None]:
    """Return (url_type, error)."""
    if not url:
        return None, "No URL provided."
    if _is_youtube_url(url) or YOUTUBE_RE.match(url):
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

    # Apple episode links usually include ?i=<trackId>. Prefer that episode.
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    target_episode_id = (query.get("i") or [None])[0]
    country = "TW"
    path_parts = [p for p in parsed.path.split("/") if p]
    if path_parts:
        cc = path_parts[0].strip().upper()
        if len(cc) == 2 and cc.isalpha():
            country = cc

    podcast_id = match.group(1)
    api_url = (
        f"https://itunes.apple.com/lookup?id={podcast_id}"
        f"&country={country}&media=podcast&entity=podcastEpisode&limit=50"
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
                "episode_id": str(entry.get("trackId") or ""),
                "title": entry.get("trackName", "Unknown Episode"),
                "audio_url": episode_url,
                "show_name": entry.get("collectionName", ""),
                "publish_date": entry.get("releaseDate", ""),
            }
        )
    if target_episode_id:
        for ep in out:
            if ep.get("episode_id") == str(target_episode_id):
                return [ep]
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
    cancel_event: threading.Event | None = None,
    *,
    source_info: dict | None = None,
    fingerprint_path: Path | None = None,
) -> str:
    _raise_if_cancelled(cancel_event)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    file_size = audio_path.stat().st_size
    if file_size < 1024:
        raise ValueError(f"Audio file is too small ({file_size} bytes).")

    duration = get_audio_duration(audio_path)
    if duration <= 0.1:
        raise ValueError(f"Audio is too short ({duration:.2f} seconds).")
    _raise_if_cancelled(cancel_event)

    fp_path = fingerprint_path if fingerprint_path and fingerprint_path.exists() else audio_path
    fingerprint = _compute_audio_fingerprint(fp_path)
    checkpoint = _load_checkpoint(temp_dir, fingerprint, duration)
    resume_from = 0.0
    previous_segments: list[dict] = []
    previous_text_parts: list[str] = []
    if checkpoint:
        resume_from = float(checkpoint.get("last_completed_end", 0.0))
        previous_segments = list(checkpoint.get("segments", []))
        previous_text_parts = [str(s.get("text", "")).strip() for s in previous_segments if s.get("text")]
        if on_status and duration > 0:
            pct = min(99, int((resume_from / duration) * 100))
            on_status(f"Transcribing... {pct}% (resuming)")

    if resume_from > 0 and (duration - resume_from) < 5.0:
        _delete_checkpoint(temp_dir, fingerprint)
        return _clean_transcript_text(" ".join(previous_text_parts).strip())

    transcribe_path = audio_path
    if resume_from > 0:
        trimmed_path = temp_dir / f"{job_id}_resumed.wav"
        trimmed = _trim_audio_from(audio_path, resume_from, trimmed_path)
        if trimmed:
            transcribe_path = trimmed
        else:
            resume_from = 0.0
            previous_segments = []
            previous_text_parts = []

    model = get_model()
    accumulated_segments: list[dict] = list(previous_segments)
    new_texts: list[str] = []
    last_status_update = time.time()
    last_checkpoint_flush = time.time()

    def collect_segment(start_s: float, end_s: float, text: str) -> None:
        nonlocal last_status_update, last_checkpoint_flush
        seg_text = (text or "").strip()
        if not seg_text:
            return
        new_texts.append(seg_text)
        seg_data = {
            "start": round(start_s + resume_from, 3),
            "end": round(end_s + resume_from, 3),
            "text": seg_text,
        }
        accumulated_segments.append(seg_data)
        now = time.time()
        if on_status and duration > 0 and now - last_status_update >= 5:
            pct = min(99, int((seg_data["end"] / duration) * 100))
            on_status(f"Transcribing... {pct}%")
            last_status_update = now
        if now - last_checkpoint_flush >= CHECKPOINT_FLUSH_INTERVAL:
            _save_checkpoint(temp_dir, fingerprint, duration, accumulated_segments, source_info=source_info)
            last_checkpoint_flush = now

    try:
        _raise_if_cancelled(cancel_event)
        resolved_language = _resolve_whisper_language()
        use_batched = (
            BatchedInferencePipeline is not None and get_batched_model() is not None and duration > CHUNK_MINUTES * 60
        )
        if use_batched:
            if on_status and resume_from == 0:
                on_status("Transcribing... 0%")
            batched_model = get_batched_model()
            kwargs = {
                "beam_size": WHISPER_BEAM_SIZE,
                "batch_size": WHISPER_BATCH_SIZE,
                "condition_on_previous_text": False,
                "vad_filter": True,
                "vad_parameters": dict(min_silence_duration_ms=500),
            }
            if resolved_language:
                kwargs["language"] = resolved_language
            segments, _info = batched_model.transcribe(str(transcribe_path), **kwargs)
            for seg in segments:
                _raise_if_cancelled(cancel_event)
                collect_segment(float(seg.start), float(seg.end), str(seg.text))
        elif WhisperModel is not None:
            if on_status and resume_from == 0:
                on_status("Transcribing... 0%")
            kwargs = {
                "beam_size": WHISPER_BEAM_SIZE,
                "condition_on_previous_text": False,
                "vad_filter": True,
                "vad_parameters": dict(min_silence_duration_ms=500),
            }
            if resolved_language:
                kwargs["language"] = resolved_language
            segments, _info = model.transcribe(str(transcribe_path), **kwargs)
            for seg in segments:
                _raise_if_cancelled(cancel_event)
                collect_segment(float(seg.start), float(seg.end), str(seg.text))
        else:
            if on_status:
                on_status("Transcribing...")
            kwargs = {"beam_size": WHISPER_BEAM_SIZE}
            if resolved_language:
                kwargs["language"] = resolved_language
            result = model.transcribe(str(transcribe_path), **kwargs)
            full_text = (result.get("text") or "").strip()
            if full_text:
                new_texts.append(full_text)
                accumulated_segments.append({"start": resume_from, "end": duration, "text": full_text})
        _delete_checkpoint(temp_dir, fingerprint)
        return _clean_transcript_text(" ".join(previous_text_parts + new_texts).strip())
    except Exception:
        if len(accumulated_segments) > len(previous_segments):
            _save_checkpoint(temp_dir, fingerprint, duration, accumulated_segments, source_info=source_info)
        raise


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
    cancel_event: threading.Event | None = None,
) -> tuple[str, Path]:
    try:
        _raise_if_cancelled(cancel_event)
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
        _raise_if_cancelled(cancel_event)

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
        _raise_if_cancelled(cancel_event)

        audio_file = output_path.with_suffix(".mp3")
        if not audio_file.exists():
            raise FileNotFoundError("Failed to download audio.")

        original_audio = audio_file
        normalized = normalize_audio(audio_file, temp_dir, job_id)
        if on_status:
            on_status(f"Transcribing '{title}'...")
        text = transcribe_audio(
            normalized,
            temp_dir,
            job_id,
            on_status=on_status,
            cancel_event=cancel_event,
            source_info={"source_type": "youtube", "title": title, "source_url": url},
            fingerprint_path=original_audio,
        )
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
    cancel_event: threading.Event | None = None,
) -> tuple[str, Path]:
    try:
        _raise_if_cancelled(cancel_event)
        if on_status:
            on_status(f"Downloading '{title}'...")

        ext_match = re.search(r"\.(mp3|m4a|wav|ogg|aac|flac|wma|opus)", audio_url, re.IGNORECASE)
        is_direct_audio_url = bool(ext_match)
        ext = ext_match.group(1).lower() if ext_match else "mp3"
        audio_file = temp_dir / f"{job_id}.{ext}"

        if is_direct_audio_url:
            # Use streaming requests with explicit timeouts to avoid hanging forever.
            try:
                with requests.get(
                    audio_url,
                    headers={"User-Agent": "Mozilla/5.0"},
                    stream=True,
                    timeout=(20, 60),
                ) as resp:
                    if resp.status_code >= 400:
                        raise ValueError(f"Failed to download audio (HTTP {resp.status_code}).")
                    with audio_file.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=8192):
                            _raise_if_cancelled(cancel_event)
                            if chunk:
                                fh.write(chunk)
            except requests.RequestException as e:
                raise ValueError(f"Failed to download audio ({type(e).__name__}).") from e
        else:
            if on_status:
                on_status("Resolving media URL...")
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
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([audio_url])
            except Exception as e:
                raise ValueError(f"Failed to resolve/download podcast audio: {e}") from e
            _raise_if_cancelled(cancel_event)
            candidate = output_path.with_suffix(".mp3")
            if candidate.exists():
                audio_file = candidate
            else:
                matches = sorted(temp_dir.glob(f"{job_id}.*"))
                audio_matches = [p for p in matches if p.suffix.lower() in ALLOWED_AUDIO_EXTENSIONS]
                if audio_matches:
                    audio_file = max(audio_matches, key=lambda p: p.stat().st_size)

        if not audio_file.exists():
            raise FileNotFoundError("Failed to download audio.")
        if audio_file.stat().st_size < 1024:
            raise ValueError("Downloaded audio file is too small.")

        original_audio = audio_file
        normalized = normalize_audio(audio_file, temp_dir, job_id)
        if on_status:
            on_status(f"Transcribing '{title}'...")
        text = transcribe_audio(
            normalized,
            temp_dir,
            job_id,
            on_status=on_status,
            cancel_event=cancel_event,
            source_info={"source_type": source_type, "title": title, "source_url": source_url},
            fingerprint_path=original_audio,
        )
        out_path = save_transcript_md(transcript_dir, title, source_url, source_type, None, text)
        return title, out_path
    finally:
        cleanup_files(temp_dir, job_id)


def transcribe_url_to_markdown(
    url: str,
    transcript_dir: Path,
    temp_dir: Path,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[str, Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    cleanup_old_checkpoints(temp_dir)
    _raise_if_cancelled(cancel_event)
    job_id = str(uuid.uuid4())
    url_type, error = classify_url(url)
    if error:
        raise ValueError(error)

    if url_type == "youtube":
        return _pipeline_youtube(
            url,
            transcript_dir,
            temp_dir,
            job_id,
            on_status=on_status,
            cancel_event=cancel_event,
        )

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
            cancel_event=cancel_event,
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
            cancel_event=cancel_event,
        )

    raise ValueError("Unsupported URL type.")


def transcribe_upload_to_markdown(
    audio_path: Path,
    original_filename: str,
    transcript_dir: Path,
    temp_dir: Path,
    on_status: Callable[[str], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> tuple[str, Path]:
    temp_dir.mkdir(parents=True, exist_ok=True)
    cleanup_old_checkpoints(temp_dir)
    _raise_if_cancelled(cancel_event)
    job_id = str(uuid.uuid4())
    try:
        if on_status:
            on_status(f"Preparing '{original_filename}'...")
        _raise_if_cancelled(cancel_event)
        original_audio = audio_path
        normalized = normalize_audio(audio_path, temp_dir, job_id)
        if on_status:
            on_status(f"Transcribing '{original_filename}'...")
        text = transcribe_audio(
            normalized,
            temp_dir,
            job_id,
            on_status=on_status,
            cancel_event=cancel_event,
            source_info={"source_type": "upload", "title": original_filename},
            fingerprint_path=original_audio,
        )
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
