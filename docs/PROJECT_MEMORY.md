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
### 2026-03-09 - KOL digest phase 1 scaffold

- Summary: added the first KOL digest module with watchlist loading, SQLite schema bootstrap, normalized post persistence, and markdown digest rendering, plus a starter watchlist file and tests.
- Files: `.env.digest`, `data/kol_watchlist.json`, `kol_digest.py`, `tests/test_kol_digest.py`, `README.md`
- Technologies: JSON watchlist config, SQLite WAL storage, normalized social post model, markdown digest rendering, Python `unittest`
- Pitfalls:
  - Keep source-specific scraping or API logic out of `kol_digest.py`; adapters should hand over normalized posts only.
  - Do not mix digest runtime data into existing main/chitchat `DATA_DIR` values; digest profile should keep its own storage root.
  - Preserve both source post IDs and content-hash fallback logic so reposts or unstable identifiers do not explode duplicates later.
- Validation:
  - `python -m py_compile kol_digest.py tests\\test_kol_digest.py`
  - `python -m unittest tests.test_kol_digest`

### 2026-03-09 - X adapter via optional snscrape CLI

- Summary: added an X-source adapter that shells out to the `snscrape` CLI, parses JSONL output into normalized posts, and persists enabled X watchlist items through the existing KOL digest storage path.
- Files: `kol_digest.py`, `tests/test_kol_digest.py`, `README.md`
- Technologies: subprocess-based CLI integration, JSONL parsing, adapter pattern, SQLite fetch-state tracking, Python `unittest.mock`
- Pitfalls:
  - Keep `snscrape` optional at the repo level for now; adapter failures should record fetch errors instead of crashing the digest pipeline.
  - Do not leak `snscrape` JSON shapes past the adapter boundary; normalize into `NormalizedPost` immediately.
  - Expect X scraping to break over time, so the adapter must stay replaceable with an official API implementation later.
- Validation:
  - `python -m py_compile kol_digest.py tests\\test_kol_digest.py`
  - `python -m unittest tests.test_kol_digest`

### 2026-03-09 - Telegram watchlist management for digest profile

- Summary: added `/digest_watchlist` command support so the digest bot can list, add, enable, disable, and remove tracked KOL entries directly from Telegram.
- Files: `app.py`, `kol_digest.py`, `tests/test_kol_digest.py`, `tests/test_smoke.py`, `README.md`
- Technologies: Telegram command routing, JSON watchlist mutation, file replacement writes, allowlist-based admin control, Python `unittest.mock`
- Pitfalls:
  - Treat watchlist mutation as an admin capability; keep add/enable/disable/remove behind `TELEGRAM_ALLOWED_CONTROL_USERS`.
  - Keep file mutation logic in `kol_digest.py` so future CLI or UI tools reuse the same validation rules.
  - Do not depend on display names for identity; use `kol_id` for enable/disable/remove operations.
- Validation:
  - `python -m py_compile app.py kol_digest.py tests\\test_kol_digest.py tests\\test_smoke.py`
  - `python -m unittest tests.test_kol_digest tests.test_smoke`

### 2026-03-10 - Replaceable X source provider with Apify adapter skeleton

- Summary: decoupled the KOL digest X source selection from `snscrape`, added a provider factory plus an `ApifyXAdapter` skeleton, and documented the Apify env, field mapping, and cost model needed for a managed-scraper MVP.
- Files: `kol_digest.py`, `app.py`, `.env.digest`, `.env.digest.example`, `tests/test_kol_digest.py`, `README.md`, `docs/APIFY_X_ADAPTER.md`
- Technologies: adapter factory pattern, env-driven provider selection, Apify HTTP API integration skeleton, placeholder-based JSON input templating, tolerant payload normalization
- Pitfalls:
  - Do not assume one universal Apify input schema; keep actor-specific input in `APIFY_X_INPUT_TEMPLATE_JSON` or an Apify task.
  - Keep X adapter failures isolated to fetch-state errors so digest rendering still works.
  - Treat Apify as a short-term managed scraping bridge; the long-term stable route for X is still likely the official API.
- Validation:
  - `python -m py_compile kol_digest.py app.py tests\\test_kol_digest.py`
  - `python -m unittest tests.test_kol_digest`

### 2026-03-09 - Digest bot profile scaffold

- Summary: added a third Telegram bot profile scaffold for digest-only use with its own wrapper module and env template, while keeping startup separate from the existing main + chitchat pair.
- Files: `app_digest.py`, `.env.digest.example`, `README.md`
- Technologies: profile-based app bootstrap, Telegram bot env isolation, shared FastAPI app entrypoints
- Pitfalls:
  - Keep digest profile rollout independent until feature/capability flags replace more `APP_PROFILE == ...` branches in `app.py`.
  - Do not enable transcribe, OCR, Slack, or other chat-heavy features in digest profile by default.
  - Use a separate `DATA_DIR` for digest profile so future KOL artifacts and checkpoints do not mix with main or chitchat runtime data.
- Validation:
  - `python -m py_compile app_digest.py app_main.py app_chitchat.py`

### 2026-03-09 - KOL Daily Digest source strategy

- Summary: for KOL Daily Digest, adopt a staged source strategy: short term use option C to validate the pipeline quickly with scraper-based inputs, then move toward option B as the long-term architecture with source adapters and a dedicated Facebook crawler.
- Files: `docs/PROJECT_MEMORY.md`
- Technologies: social source ingestion, X API evaluation, scraper-based collection, crawler architecture, adapter pattern, scheduled digest generation
- Pitfalls:
  - Do not couple digest logic directly to a single source implementation such as `snscrape`; keep source adapters replaceable.
  - Treat `snscrape` as a bootstrap tool for MVP and internal validation, not as a long-term reliability layer.
  - Expect Facebook external public-page tracking to require a crawler path rather than relying on official API coverage.
  - Keep X integration swappable so a scraper-based MVP can later migrate to the official API without rewriting digest generation.
- Validation:
  - Decision recorded after evaluating `snscrape`, official APIs, and custom crawler tradeoffs for recurring KOL tracking.

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

### 2026-03-09 - KOL digest scheduler defaults

- Summary: wired the digest profile to run a background KOL scheduler with 6-hour fetch slots aligned to `08:00 Asia/Taipei`, and to generate one daily digest at `08:00`.
- Files: `app.py`, `.env.digest`, `.env.digest.example`, `README.md`, `tests/test_smoke.py`
- Technologies: background worker threads, wall-clock slot scheduling, SQLite-backed KOL digest storage, markdown digest output
- Pitfalls:
  - Keep the scheduler profile-scoped behind `KOL_DIGEST_ENABLED` so main/chitchat bots do not start digest jobs accidentally.
  - Align fetch slots to the digest hour; with the current defaults that means `02:00 / 08:00 / 14:00 / 20:00 Asia/Taipei`.
  - Daily digest generation should remain independent from Telegram push so transport changes do not affect collection.

### 2026-03-09 - Manual KOL digest commands

- Summary: added `/kol_today` to read or generate today's digest on demand, `/kol_yesterday` for the previous calendar day, and `/kol_now` to force a fetch cycle and rebuild the current-day digest immediately.
- Files: `app.py`, `tests/test_smoke.py`, `README.md`
- Technologies: Telegram command routing, digest preview truncation, admin-gated manual execution
- Pitfalls:
  - Keep `/kol_now` behind `TELEGRAM_ALLOWED_CONTROL_USERS`; it triggers live fetch work and should stay admin-only.
  - `/kol_today` should remain safe for normal use; if today's file is missing it may generate from existing stored posts without forcing a network fetch.
  - Scheduled `08:00` digest generation should write the previous calendar day so the file name and digest window stay aligned.

### 2026-03-11 - Facebook provider wiring and Meta review status

- Summary: generalized KOL social-source fetching beyond X, added a Facebook provider interface with both `StubFacebookAdapter` and `MetaPagePublicContentAdapter`, updated digest config/docs/tests, and validated that the current blocker for tracking third-party public Facebook Pages is Meta review/access rather than app code wiring.
- Files: `kol_digest.py`, `app.py`, `tests/test_kol_digest.py`, `README.md`, `.env.digest`, `.env.digest.example`, `privacy-policy.html`, `docs/KOL_FACEBOOK_PAGE_CANDIDATES.md`
- Technologies: provider factory pattern, Meta Graph API skeleton integration, Facebook Page URL normalization, public-page candidate triage, GitHub Pages-hosted privacy policy, Meta App Review workflow
- Pitfalls:
  - `KOL_FACEBOOK_SOURCE_PROVIDER=meta` only proves the Graph API integration path; it does not bypass Meta review requirements for third-party public Pages.
  - Current live tests with the updated token reached Graph API but returned `(#210) A page access token is required` for several likely Pages and `Unsupported get request` for others, so do not assume current permissions cover third-party Page reading.
  - Seeing `pages_read_engagement` / `pages_show_list` as `可供測試` is not evidence that `Page Public Content Access` has been approved.
  - The app dashboard currently showed no visible `Page Public Content Access` feature entry; App Review flow also prompted for business/access verification before higher review.
  - Privacy policy must use the public GitHub Pages URL, not the GitHub `blob` URL.
  - For this project's current scope, skip Threads review fields unless Threads ingestion is actually implemented.
- Validation:
  - `python -m unittest tests.test_kol_digest`
  - `python -m py_compile app.py kol_digest.py tests\\test_kol_digest.py`
  - Live Meta Graph API checks against the candidate Facebook URLs using `.env.digest`

### 2026-03-18 - LLM batch news classification

- Summary: replaced simple 5-category keyword classification with 7-category LLM batch classification for both `/news` output and weekly report news block. Categories: AI, 半導體, 台灣產業, 政治及地緣, 金融市場, 消費電子產品, 其他. Added priority-ordered disambiguation rules (Taiwan company → policy → financial → AI → semiconductor). Also fixed email HTML formatting (newlines not rendering in Gmail).
- Files: `app.py`, `.env.example`, `.env.main.example`, `README.md`, `docs/PROJECT_MEMORY.md`
- Technologies: LLM batch prompting via `_run_ai_chat()`, regex-based output parsing, keyword fallback classifier, priority-ordered disambiguation, configurable batch size
- Pitfalls:
  - `_classify_news_titles_batch()` mirrors the `_translate_news_titles_to_zh()` pattern: batch → retry once → keyword fallback. Do not remove the keyword fallback path; it is the safety net when AI is disabled or LLM output is unparseable.
  - Disambiguation priority order matters: Taiwan company (C) > policy (D) > financial (E) > AI (A) > semiconductor (B). Changing priority will shift classification results.
  - `NEWS_CLASSIFY_BATCH_SIZE` defaults to 40. Increasing it may exceed LLM context limits for smaller models.
  - The keyword classifier uses narrowed financial terms (`大漲`/`暴跌`/`飆漲`/`重挫` instead of `漲`/`跌`) to avoid false positives on non-financial price mentions.
  - Email HTML now converts `\n` to `<br>\n` before embedding in `<html><body>`. Do not double-convert if the source HTML already contains `<br>` tags.
- Validation:
  - `python -c "import ast; ast.parse(open('app.py', encoding='utf-8-sig').read())"` — syntax OK
  - Keyword classifier tested on 403 titles from `tests/24hr_news.txt`: all titles classified (zero missing), distribution A:90 B:26 C:12 D:38 E:26 F:8 G:203

## Recommended Workflow

Before making non-trivial changes:

1. Read this file.
2. Check whether the change touches routing, persistence, timestamp handling, or local control commands.
3. Update or add smoke tests if one of the known pitfalls is in scope.
4. Add a new entry after the change lands.
