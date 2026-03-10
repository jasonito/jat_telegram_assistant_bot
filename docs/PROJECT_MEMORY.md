# Project Memory

## Purpose

This file is a running memory for changes that are easy to break later.
Use it to record:

- pitfalls discovered during fixes
- assumptions that must stay true
- technical components touched by a change
- validation steps that caught regressions

Keep entries short and concrete. Prefer facts over narrative.

## Update Rules

Add a new entry when a change:

- modifies message routing, background workers, or persistence
- changes time handling, filesystem layout, or sync behavior
- introduces a new env var, feature gate, or external integration
- fixes a bug that could plausibly be reintroduced

For each entry, capture:

- date
- change summary
- files touched
- technologies involved
- pitfalls
- validation

## Known Pitfalls

### Runtime and Concurrency

- Telegram long polling must not spawn unbounded threads per update. Keep bounded worker execution and backpressure in place.
- SQLite is used from multiple execution paths. New DB writes should go through the shared connection helper so `busy_timeout`, `WAL`, and `foreign_keys` remain enabled.
- Startup side effects belong in FastAPI startup hooks, not module import paths, to avoid duplicate workers and accidental execution during tests.

### Time and Data Boundaries

- Telegram and Slack timestamps must be converted to timezone-aware local datetimes before generating note dates or Notion log dates.
- RSS/pubDate-style values that are already UTC should use UTC-safe conversion. Do not use `time.mktime(...)` for UTC tuples.

### File Handling

- Telegram media downloads should stream to disk. Avoid loading full audio files into memory with `.content`.
- Runtime artifacts under `read/`, `logs/`, and local `.env*` files are operational data and should stay out of commits unless there is an explicit reason.

### Access Control

- Local machine control commands (`open`, `notepad`, `sort downloads`) must stay behind `TELEGRAM_ALLOWED_CONTROL_USERS`.
- `/whoami` exists specifically to make whitelist setup maintainable. Do not remove it unless there is a replacement admin path.

## Change Log

### 2026-03-10 - Telegram-manageable RSS feed registry

- Summary: moved news feed management into SQLite, added Telegram `/news add|remove|enable|disable` subcommands, and changed `/news sources` to show feed ids plus enabled status for operator workflows.
- Files: `app.py`, `docs/PROJECT_MEMORY.md`
- Technologies: SQLite schema migration, Telegram slash-command routing, RSS feed validation with `feedparser`
- Pitfalls:
  - Feed management commands are restricted by `TELEGRAM_ALLOWED_CONTROL_USERS`; do not bypass that check when extending `/news` subcommands.
  - News ingestion now prefers DB-managed feeds over built-in defaults when `NEWS_RSS_URLS` and `NEWS_RSS_URLS_FILE` are empty; changing precedence will change runtime behavior.
  - Operators now act on feed ids from `/news sources`; keep the output stable enough that `remove/enable/disable` remain practical in Telegram chats.
- Validation:
  - `python -m py_compile app.py`
  - `pytest` unavailable in current environment (`pytest` command missing and `.venv` lacks the module), so no automated test suite was run here.

### 2026-03-09 - Weekly digest pipeline and transcription chunking

- Summary: weekly note/report generation now pre-syncs Dropbox notes for the requested date window, preserves larger raw transcript sections for AI summarization, and transcription now chunks long audio with `small` as the default Whisper model.
- Files: `app.py`, `transcription.py`, `tests/test_smoke.py`, `.env.main`, `README.md`
- Technologies: Dropbox API sync, Markdown note ingestion, AI summarization prompt assembly, Faster Whisper, ffmpeg-based audio chunking, Python `unittest`
- Pitfalls:
  - Do not reintroduce `raw[:AI_SUMMARY_MAX_CHARS]` as the main weekly note input path; it biases the summary toward the first long transcript block.
  - `/summary_notes_weekly` is AI-backed in current behavior. Treat it as an AI feature, not a pure local rule-based formatter.
  - Weekly note/report generation depends on pre-syncing the full requested Dropbox note date range, not just locally existing files.
  - Chunking in `transcribe_audio()` must preserve absolute timestamps when merging chunk results back together.
  - Current transcription runtime is CPU-oriented by default; increasing the model size without checking host capacity will regress turnaround time quickly.
- Validation:
  - `python -m py_compile app.py transcription.py tests\\test_smoke.py`
  - `python -m unittest tests.test_smoke`

### 2026-03-09 - Reliability and control hardening

- Summary: reduced Telegram processing contention, fixed timestamp handling, streamed file downloads, and added control-command allowlisting.
- Files: `app.py`, `tests/test_smoke.py`, `README.md`, `.env*.example`
- Technologies: FastAPI startup lifecycle, `ThreadPoolExecutor`, `threading.BoundedSemaphore`, SQLite WAL, Telegram Bot API file fetch, Slack event timestamps, RSS date parsing, Python `zoneinfo`, `unittest`
- Pitfalls:
  - Replacing bounded execution with per-update threads will reintroduce resource spikes and SQLite lock errors.
  - Using naive datetimes for message timestamps will misfile notes by date on hosts outside the intended timezone.
  - Reading Telegram downloads with `requests.get(...).content` will spike memory on large audio uploads.
  - Local control commands are a host-level capability; they must always remain allowlisted.
- Validation:
  - `python -m unittest discover -s tests -p "test_*.py"`
  - `python -m py_compile app.py app_main.py app_chitchat.py transcription.py tests\\test_smoke.py`

## Recommended Workflow

Before making non-trivial changes:

1. Read this file.
2. Check whether the change touches routing, persistence, timestamp handling, or local control commands.
3. Update or add smoke tests if one of the known pitfalls is in scope.
4. Add a new entry after the change lands.
