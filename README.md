# JAT Telegram Assistant Bot (PoC)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -c constraints.txt
```

Notes:
- `constraints.txt` pins packaging/build tooling used during install (currently includes `setuptools<81`).
- `start.ps1` also uses this constraints file automatically when it installs dependencies.

## Deployment Docs

- Server deployment (24/7): `docs/DEPLOY_SERVER.md`
- Preflight checklist: `docs/PREFLIGHT_CHECKLIST.md`
- systemd template: `deploy/jat-bot.service.example`
- Server operation runbook: `docs/SERVER_RUNBOOK.md`
- Project memory / recurring pitfalls: `docs/PROJECT_MEMORY.md`
- KOL Daily Digest plan: `docs/KOL_DAILY_DIGEST_PLAN.md`
- KOL watchlist seed: `data/kol_watchlist.json`

## Env

Use profile-specific env files instead of a shared `.env` whenever possible:

- main bot: `.env.main`
- chitchat bot: `.env.chitchat`
- digest bot: `.env.digest`

`start.ps1` loads env values from `-EnvFile`, and `start-both.ps1` already uses `.env.main` + `.env.chitchat`.

### Segment A: Profile / Runtime
- Purpose: select bot profile and data root.
- Keys: `APP_MODULE`, `APP_PROFILE`, `DATA_DIR`.

### Segment B: Telegram Core
- Purpose: Telegram token, webhook/polling mode, and network retry behavior.
- Keys: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_GROUPS`, `TELEGRAM_ALLOWED_CONTROL_USERS`, `TELEGRAM_LONG_POLLING`, `TELEGRAM_LOCAL_WEBHOOK_URL`.
- Optional tuning: `TELEGRAM_FILE_FETCH_*`, `TELEGRAM_POLL_*`.

### Segment C: Feature Flags
- Purpose: toggle major functions without code changes.
- Keys: `FEATURE_NEWS_ENABLED`, `FEATURE_TRANSCRIBE_ENABLED`, `FEATURE_TRANSCRIBE_AUTO_URL`, `FEATURE_OCR_ENABLED`, `FEATURE_OCR_CHOICE_ENABLED`, `FEATURE_SLACK_ENABLED`.
- OCR choice behavior: `OCR_CHOICE_SCOPE`, `OCR_CHOICE_TIMEOUT_SECONDS`, `OCR_CHOICE_TIMEOUT_DEFAULT`.

### Segment D: Transcription Engine
- Purpose: control Whisper quality/speed/memory and chunking.
- Keys: `TRANSCRIBE_MAX_DURATION_SECONDS`, `TRANSCRIBE_CHUNK_MINUTES`, `TRANSCRIBE_CHECKPOINT_FLUSH_SECONDS`, `WHISPER_MODEL`, `WHISPER_LANGUAGE`, `WHISPER_BEAM_SIZE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_CPU_THREADS`, `WHISPER_BATCH_SIZE`, `FFMPEG_LOCATION`, `TRANSCRIBE_PROGRESS_HEARTBEAT_SECONDS`.
- Current default: `WHISPER_MODEL=small`.
- Long audio is now chunked inside `transcribe_audio()` before Whisper runs, then merged back with original timestamps.
- Current code path initializes Whisper on `cpu`. GPU use requires both a working CUDA stack on the host and code changes.

### Segment E: OCR Provider
- Purpose: configure image OCR backend.
- Keys: `OCR_PROVIDER`, `OCR_LANG_HINTS`, `GOOGLE_APPLICATION_CREDENTIALS`.

### Segment F: News / Digest
- Purpose: collect, filter, and summarize news.
- Keys: `NEWS_ENABLED`, `NEWS_FETCH_INTERVAL_MINUTES`, `NEWS_PUSH_ENABLED`, `NEWS_PUSH_MAX_ITEMS`, `NEWS_GNEWS_*`, `NEWS_RSS_URLS`, `NEWS_RSS_URLS_FILE`, `NEWS_URL_FETCH_*`, `NEWS_DIGEST_*`, `NOTE_DIGEST_MAX_ITEMS`.
- Note/transcript AI input budget: `NOTE_AI_INPUT_MAX_CHARS` (current default `28000`).

### Segment G: AI Summary Providers
- Purpose: shared AI config for digest/weekly report and transcript summary blocks.
- Keys: `AI_SUMMARY_ENABLED`, `AI_SUMMARY_PROVIDER`, `AI_SUMMARY_TIMEOUT_SECONDS`, `AI_SUMMARY_MAX_CHARS`, `AI_SUMMARY_TEMPERATURE`.
- Provider keys: `OPENAI_*`, `GEMINI_*`, `ANTHROPIC_*`, `HUGGINGFACE_*`, `OLLAMA_*`.

### Segment H: Dropbox Sync
- Purpose: sync notes/images/weekly report and import transcript files.
- Keys: `DROPBOX_ACCESS_TOKEN`, `DROPBOX_REFRESH_TOKEN`, `DROPBOX_APP_KEY`, `DROPBOX_APP_SECRET`, `DROPBOX_TOKEN_REFRESH_LEEWAY_SECONDS`, `DROPBOX_ROOT_PATH`, `DROPBOX_SYNC_ENABLED`, `DROPBOX_SYNC_TIME`, `DROPBOX_SYNC_TZ`, `DROPBOX_SYNC_ON_STARTUP`, `DROPBOX_TRANSCRIPTS_PATH`, `DROPBOX_TRANSCRIPTS_SYNC_ENABLED`.

### Segment H-2: Weekly Report Push
- Purpose: auto-push weekly recap to recent Telegram chats and persist markdown report.
- Keys: `WEEKLY_REPORT_PUSH_ENABLED`, `WEEKLY_REPORT_PUSH_WEEKDAY` (1=Mon ... 7=Sun), `WEEKLY_REPORT_PUSH_TIME` (`HH:MM`), `WEEKLY_REPORT_PUSH_TZ`, `WEEKLY_REPORT_PUSH_LOOKBACK_DAYS`, `WEEKLY_REPORT_PUSH_MAX_CHATS`.

### Segment I: Notion (mainly chitchat)
- Purpose: append chitchat logs/images/transcripts to Notion pages.
- Keys: `NOTION_ENABLED`, `NOTION_TOKEN`, `NOTION_VERSION`, `NOTION_CHATLOG_YEAR_PAGES_JSON`, `NOTION_CHATLOG_FALLBACK_PAGE_ID`, `NOTION_CHATLOG_IMAGE_MODE`, `NOTION_FILE_UPLOAD_VERSION`, `NOTION_CHATLOG_OCR_MODE`, `NOTION_CHATLOG_INCLUDE_TIME`.

### Segment J: Slack (optional)
- Purpose: enable Socket Mode DM logging.
- Keys: `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_USER_ID`, `SLACK_DEBUG`.

### Templates

- Main profile template: `.env.main.example`
- Chitchat profile template: `.env.chitchat.example`
- Digest profile template: `.env.digest.example`
- Legacy generic template: `.env.example`

## Run

Direct uvicorn run:

```powershell
uvicorn app_chitchat:app --host 0.0.0.0 --port 8000
```

Digest profile direct run:

```powershell
uvicorn app_digest:app --host 0.0.0.0 --port 8002
```

Recommended startup script:

```powershell
.\start.ps1
```

`start.ps1` behavior:
- Creates `.venv` automatically if missing.
- Imports env vars from `-EnvFile`.
- Installs dependencies only when `requirements.txt` hash changes (uses `.venv\requirements.sha256`).
- If `constraints.txt` exists, installs with `-c constraints.txt`.
- Starts bot process in background hidden window by default (`-ShowWindow` to show windows).

Run with specific env file and port:

```powershell
.\start.ps1 -EnvFile .env.main -Port 8000
.\start.ps1 -EnvFile .env.chitchat -Port 8001
.\start.ps1 -EnvFile .env.digest -Port 8002
```

Start both bots:

```powershell
.\start-both.ps1
```

The digest bot is currently started separately so its rollout stays independent from the existing main + chitchat startup flow.

Phase 1 KOL digest scaffold now lives in `kol_digest.py`. It currently provides:

- watchlist loading from `data/kol_watchlist.json`
- SQLite schema bootstrap for KOL sources, posts, and digest runs
- normalized post persistence with dedupe
- markdown digest rendering to `read/digests/`-style output paths
- replaceable X adapters via `build_x_source_adapter()`
- optional `snscrape`-backed X adapter via `SnscrapeXAdapter`
- optional Apify-backed X adapter skeleton via `ApifyXAdapter`
- Telegram watchlist management via `/digest_watchlist`
- digest profile background scheduler aligned to `08:00 Asia/Taipei`
- default fetch slots at `02:00 / 08:00 / 14:00 / 20:00 Asia/Taipei`, with the `08:00` slot generating the previous calendar day's digest

Current X adapter notes:

- select the provider with `KOL_X_SOURCE_PROVIDER=snscrape|apify`
- `snscrape` remains the default bootstrap path
- `ApifyXAdapter` is wired as a generic task/actor client and usually needs `APIFY_X_INPUT_TEMPLATE_JSON` to match the chosen actor schema
- provider-specific notes live in `docs/APIFY_X_ADAPTER.md`

Digest watchlist command notes:

- list: `/list_kol`
- add: `/add_kol https://x.com/example Display Name`
- add with handle: `/add_kol @example Display Name`
- today digest: `/kol_today`
- yesterday digest: `/kol_yesterday`
- fetch now + rebuild digest: `/kol_now`
- enable: `/on_kol <kol_id>`
- disable: `/off_kol <kol_id>`
- remove: `/del_kol <kol_id>`
- platform is inferred automatically: `facebook.com` => `facebook`, `x.com`/`twitter.com`/`@handle`/plain handle => `x`
- legacy forms still work: `/digest_watchlist ...` and `add kol ...`
- mutation commands require the Telegram user to be allowlisted in `TELEGRAM_ALLOWED_CONTROL_USERS`
- optional env override: `KOL_WATCHLIST_PATH`

Troubleshoot startup/command routing (keep logs streaming in current terminal):

```powershell
powershell -ExecutionPolicy Bypass -File .\start-both.ps1 -EnableLogs -Monitor
```

Stop both bots:

```powershell
.\stop-both.ps1
```

## Startup Lifecycle

Background workers are started in FastAPI startup event (`@app.on_event("startup")`), not at module import time.
This avoids accidental side effects during import and helps prevent duplicate worker startup in non-runtime contexts.

## Long Polling (No ngrok/webhook)

Set `TELEGRAM_LONG_POLLING=1`, then run:

```powershell
.\start.ps1
```

## Expose With ngrok

```powershell
ngrok http 8000
```

Copy the https URL and set the Telegram webhook:

```powershell
$env:BOT_TOKEN = (Get-Content .env | Select-String -Pattern "TELEGRAM_BOT_TOKEN" | ForEach-Object { $_.Line.Split('=')[1] })
$env:NGROK_URL = "https://YOUR-NGROK-URL"
Invoke-RestMethod -Method Post -Uri "https://api.telegram.org/bot$env:BOT_TOKEN/setWebhook" -Body @{ url = "$env:NGROK_URL/telegram" }
```

## Test

Run smoke tests:

```powershell
python -m unittest discover -s tests -p "test_*.py"
```

Private chat commands:

```text
/whoami
open https://google.com
notepad
sort downloads
/summary_notes_daily
/summary_notes_weekly
/summary_news_daily
/summary_news_weekly
/news latest
/transcribe https://www.youtube.com/watch?v=VIDEO_ID
```

`/news` and `/transcribe` availability depends on `FEATURE_NEWS_ENABLED` and `FEATURE_TRANSCRIBE_ENABLED`.

## Weekly Note Digest And Weekly Report

- `/summary_notes_weekly` uses AI summarization when `AI_SUMMARY_ENABLED=1`.
- Before note daily/weekly summary runs, the bot syncs Dropbox note markdown for the requested date range into local `DATA_DIR\notes\...`.
- Weekly note AI input now prefers larger raw transcript sections over aggressively compressed extracted lines, to preserve more source material.
- Raw URL-only lines and note metadata lines are removed before the AI call.
- Weekly note / weekly report output rules currently aim for:
  - dynamic topic count based on available content, up to 10 points
  - at most 1 action item per point
  - merged duplicate event chains
  - same-topic bucket limits to reduce over-concentration
  - Traditional Chinese news titles in the weekly report news block
- Weekly report generation logs a Dropbox pre-sync step before summarization.
- For debugging, a pre-AI weekly note input snapshot can be written under `tests\weekly_note_ai_input_YYYY-MM-DD_YYYY-MM-DD.txt`.

Local control white list:
- Set `TELEGRAM_ALLOWED_CONTROL_USERS` to a comma-separated list of Telegram `user_id` and/or `username`.
- Example: `TELEGRAM_ALLOWED_CONTROL_USERS=123456789,my_telegram_username`
- Use `/whoami` in Telegram to see your current `user_id` and `chat_id`.
Transcription flow: bot sends `已�??��??�` right after transcript is saved, then sends AI summary afterward (if enabled).

Group logging (no reply):
- Messages in allowed groups are stored in SQLite and appended to Markdown files.
- Markdown files: `DATA_DIR\notes\telegram\YYYY-MM-DD_telegram.md`
- SQLite DB: `DATA_DIR\messages.sqlite`

Slack DM logging (no reply):
- DMs from the configured `SLACK_USER_ID` are stored in the same SQLite/Markdown.
- Run uvicorn, then send a DM to your bot.

Transcription runtime behavior:
- Long audio is transcribed chunk-by-chunk and reports `Transcribing segment n/m...` while running.
- If transcription appears stuck at `0%`, the usual bottleneck is Whisper not producing its first segment yet; model size, CPU speed, chunking, and `ffmpeg/ffprobe` availability all matter.

## Debug Recipes

Generate a pre-AI weekly note input snapshot for inspection:

```powershell
@'
from pathlib import Path
import app

end_day = "2026-03-09"
days = 7
start_day = app.shift_day(end_day, -(days - 1))
day_to_raw = {}
for day in app.day_range(start_day, end_day):
    files = app._summary_files_for_day(day)
    raw = app._load_raw_summary_files(files, clip_chars=None).strip()
    if raw:
        day_to_raw[day] = raw

text = app._compose_note_ai_input_from_raw(day_to_raw, max_chars=app.NOTE_AI_INPUT_MAX_CHARS)
out = Path("tests") / f"weekly_note_ai_input_{start_day}_{end_day}.txt"
out.write_text(text, encoding="utf-8")
print(out)
'@ | python -
```

What this gives you:
- the exact note/transcript text assembled before the first AI weekly note summary call
- useful to confirm which days and which transcript sections were actually included

Diagnose a transcription job that appears stuck:

1. Check whether the bot is still in download / normalization / Whisper stage by watching the progress message text.
2. If the message stays at `0%` for a long time, assume Whisper has not emitted its first segment yet.
3. Confirm chunking is active by looking for status updates like `Transcribing segment 1/3...`.
4. Verify media tooling:
   - `ffmpeg -version`
   - `ffprobe -version`
5. Verify the active Whisper model in the env file used by the running bot:
   - `.env.main` or `.env.chitchat`
   - current recommended baseline is `WHISPER_MODEL=small`
6. Restart the bot after env changes; model changes do not apply to an already-running process.

Useful local checks:

```powershell
python -c "import transcription; print(transcription.get_transcribe_runtime_info())"
```

```powershell
ffmpeg -version
ffprobe -version
```

Image OCR and cloud sync:
- Telegram private image uploads are saved to `DATA_DIR\\images\YYYY-MM-DD\`.
- If OCR choice is enabled, bot asks per image: `?��? OCR` or `?��??�`; timeout defaults to save-only.
- OCR output is appended to `DATA_DIR\\notes\\telegram\\YYYY-MM-DD_telegram.md`.
- A Dropbox worker syncs local `notes` and `images` to:
- `/read & chat/read/notes`
- `/read & chat/read/images`
- Weekly report markdown is saved under `DATA_DIR\\weekly report\\` and also synced to Dropbox.


## Markdown Cleanup Maintenance

- Use `python tools\cleanup_dropbox_notes_md.py` to normalize existing note markdown (deduplicate duplicated headings/blocks and convert old Telegram line format to `- [HH:MM:SS] text`).
- Main profile cleanup:
  - `python tools\cleanup_dropbox_notes_md.py --env-file .env.main --remote-root "/read & chat/read" --local-notes "read/notes"`
- Chitchat profile cleanup:
  - `python tools\cleanup_dropbox_notes_md.py --env-file .env.chitchat --remote-root "/read & chat/chitchat" --local-notes "chitchat/notes"`
