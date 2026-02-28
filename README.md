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

## Env

Use profile-specific env files instead of a shared `.env` whenever possible:

- main bot: `.env.main`
- chitchat bot: `.env.chitchat`

`start.ps1` loads env values from `-EnvFile`, and `start-both.ps1` already uses `.env.main` + `.env.chitchat`.

### Segment A: Profile / Runtime
- Purpose: select bot profile and data root.
- Keys: `APP_MODULE`, `APP_PROFILE`, `DATA_DIR`.

### Segment B: Telegram Core
- Purpose: Telegram token, webhook/polling mode, and network retry behavior.
- Keys: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ALLOWED_GROUPS`, `TELEGRAM_LONG_POLLING`, `TELEGRAM_LOCAL_WEBHOOK_URL`.
- Optional tuning: `TELEGRAM_FILE_FETCH_*`, `TELEGRAM_POLL_*`.

### Segment C: Feature Flags
- Purpose: toggle major functions without code changes.
- Keys: `FEATURE_NEWS_ENABLED`, `FEATURE_TRANSCRIBE_ENABLED`, `FEATURE_TRANSCRIBE_AUTO_URL`, `FEATURE_OCR_ENABLED`, `FEATURE_OCR_CHOICE_ENABLED`, `FEATURE_SLACK_ENABLED`.
- OCR choice behavior: `OCR_CHOICE_SCOPE`, `OCR_CHOICE_TIMEOUT_SECONDS`, `OCR_CHOICE_TIMEOUT_DEFAULT`.

### Segment D: Transcription Engine
- Purpose: control Whisper quality/speed/memory and chunking.
- Keys: `TRANSCRIBE_MAX_DURATION_SECONDS`, `TRANSCRIBE_CHUNK_MINUTES`, `TRANSCRIBE_CHECKPOINT_FLUSH_SECONDS`, `WHISPER_MODEL`, `WHISPER_LANGUAGE`, `WHISPER_BEAM_SIZE`, `WHISPER_COMPUTE_TYPE`, `WHISPER_CPU_THREADS`, `WHISPER_BATCH_SIZE`, `FFMPEG_LOCATION`, `TRANSCRIBE_PROGRESS_HEARTBEAT_SECONDS`.

### Segment E: OCR Provider
- Purpose: configure image OCR backend.
- Keys: `OCR_PROVIDER`, `OCR_LANG_HINTS`, `GOOGLE_APPLICATION_CREDENTIALS`.

### Segment F: News / Digest
- Purpose: collect, filter, and summarize news.
- Keys: `NEWS_ENABLED`, `NEWS_FETCH_INTERVAL_MINUTES`, `NEWS_PUSH_ENABLED`, `NEWS_PUSH_MAX_ITEMS`, `NEWS_GNEWS_*`, `NEWS_RSS_URLS`, `NEWS_RSS_URLS_FILE`, `NEWS_URL_FETCH_*`, `NEWS_DIGEST_*`, `NOTE_DIGEST_MAX_ITEMS`.

### Segment G: AI Summary Providers
- Purpose: shared AI config for `/summary` and transcript summary blocks.
- Keys: `AI_SUMMARY_ENABLED`, `AI_SUMMARY_PROVIDER`, `AI_SUMMARY_TIMEOUT_SECONDS`, `AI_SUMMARY_MAX_CHARS`.
- Provider keys: `OPENAI_*`, `GEMINI_*`, `ANTHROPIC_*`, `HUGGINGFACE_*`, `OLLAMA_*`.

### Segment H: Dropbox Sync
- Purpose: sync notes/images and import transcript files.
- Keys: `DROPBOX_ACCESS_TOKEN`, `DROPBOX_REFRESH_TOKEN`, `DROPBOX_APP_KEY`, `DROPBOX_APP_SECRET`, `DROPBOX_TOKEN_REFRESH_LEEWAY_SECONDS`, `DROPBOX_ROOT_PATH`, `DROPBOX_SYNC_ENABLED`, `DROPBOX_SYNC_TIME`, `DROPBOX_SYNC_TZ`, `DROPBOX_SYNC_ON_STARTUP`, `DROPBOX_TRANSCRIPTS_PATH`, `DROPBOX_TRANSCRIPTS_SYNC_ENABLED`.

### Segment I: Notion (mainly chitchat)
- Purpose: append chitchat logs/images/transcripts to Notion pages.
- Keys: `NOTION_ENABLED`, `NOTION_TOKEN`, `NOTION_VERSION`, `NOTION_CHATLOG_YEAR_PAGES_JSON`, `NOTION_CHATLOG_FALLBACK_PAGE_ID`, `NOTION_CHATLOG_IMAGE_MODE`, `NOTION_CHATLOG_OCR_MODE`, `NOTION_CHATLOG_INCLUDE_TIME`.

### Segment J: Slack (optional)
- Purpose: enable Socket Mode DM logging.
- Keys: `SLACK_BOT_TOKEN`, `SLACK_APP_TOKEN`, `SLACK_USER_ID`, `SLACK_DEBUG`.

### Templates

- Main profile template: `.env.main.example`
- Chitchat profile template: `.env.chitchat.example`
- Legacy generic template: `.env.example`

## Run

Direct uvicorn run:

```powershell
uvicorn app_chitchat:app --host 0.0.0.0 --port 8000
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
```

Start both bots:

```powershell
.\start-both.ps1
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

Private chat commands:

```text
open https://google.com
notepad
sort downloads
/summary
/summary 2026-02-12
/summary note
/summary note 2026-02-12
/news latest
/transcribe https://www.youtube.com/watch?v=VIDEO_ID
```

`/news` and `/transcribe` availability depends on `FEATURE_NEWS_ENABLED` and `FEATURE_TRANSCRIBE_ENABLED`.
Transcription flow: bot sends `已�??��??�` right after transcript is saved, then sends AI summary afterward (if enabled).

Group logging (no reply):
- Messages in allowed groups are stored in SQLite and appended to Markdown files.
- Markdown files: `DATA_DIR\notes\telegram\YYYY-MM-DD_telegram.md`
- SQLite DB: `DATA_DIR\messages.sqlite`

Slack DM logging (no reply):
- DMs from the configured `SLACK_USER_ID` are stored in the same SQLite/Markdown.
- Run uvicorn, then send a DM to your bot.

Image OCR and cloud sync:
- Telegram private image uploads are saved to `DATA_DIR\\images\YYYY-MM-DD\`.
- If OCR choice is enabled, bot asks per image: `?��? OCR` or `?��??�`; timeout defaults to save-only.
- OCR output is appended to `DATA_DIR\\notes\\telegram\\YYYY-MM-DD_telegram.md`.
- A Dropbox worker syncs local `notes` and `images` to:
- `/read & chat/read/notes`
- `/read & chat/read/images`


## Markdown Cleanup Maintenance

- Use `python tools\cleanup_dropbox_notes_md.py` to normalize existing note markdown (deduplicate duplicated headings/blocks and convert old Telegram line format to `- [HH:MM:SS] text`).
- Main profile cleanup:
  - `python tools\cleanup_dropbox_notes_md.py --env-file .env.main --remote-root "/read & chat/read" --local-notes "read/notes"`
- Chitchat profile cleanup:
  - `python tools\cleanup_dropbox_notes_md.py --env-file .env.chitchat --remote-root "/read & chat/chitchat" --local-notes "chitchat/notes"`
