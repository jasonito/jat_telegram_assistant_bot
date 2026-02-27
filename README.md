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

Copy `.env.example` to `.env` and set your tokens.

- `TELEGRAM_ALLOWED_GROUPS` comma-separated group titles to log
- `DATA_DIR` location for SQLite + Markdown
- `APP_MODULE` uvicorn module name (`app`, `app_main`, `app_chitchat`)
- `TELEGRAM_LONG_POLLING` set `1` to enable getUpdates instead of webhook
- `TELEGRAM_LOCAL_WEBHOOK_URL` local URL for polling to forward updates (default: `http://127.0.0.1:8000/telegram`)

Feature flags:
- `FEATURE_NEWS_ENABLED` enable/disable news ingest + `/news` commands
- `FEATURE_TRANSCRIBE_ENABLED` enable/disable `/transcribe` and audio transcription
- `FEATURE_TRANSCRIBE_AUTO_URL` when set `1`, private chat plain URLs auto-trigger transcription
- `TRANSCRIBE_PROGRESS_HEARTBEAT_SECONDS` progress heartbeat interval while transcription is running (default: `30`)
- `FEATURE_OCR_ENABLED` enable/disable image OCR pipeline
- `FEATURE_OCR_CHOICE_ENABLED` when set `1`, image OCR asks user to choose (`進行 OCR` or `只存圖`)
- `OCR_CHOICE_SCOPE` OCR choice scope (`private` recommended)
- `OCR_CHOICE_TIMEOUT_SECONDS` choice timeout seconds before fallback to save-only (default: `60`)
- `FEATURE_SLACK_ENABLED` enable/disable Slack worker

OCR (Google Vision):
- `OCR_PROVIDER` set `google_vision`
- `OCR_LANG_HINTS` OCR language hints (default: `zh-TW,en`)
- `GOOGLE_APPLICATION_CREDENTIALS` full path to Google service-account JSON key

Dropbox sync:
- `DROPBOX_ACCESS_TOKEN` API token (legacy/fallback mode)
- `DROPBOX_REFRESH_TOKEN` long-lived refresh token (recommended)
- `DROPBOX_APP_KEY` Dropbox app key (required with refresh token)
- `DROPBOX_APP_SECRET` Dropbox app secret (required with refresh token)
- `DROPBOX_TOKEN_REFRESH_LEEWAY_SECONDS` refresh-ahead buffer in seconds (default: `300`)
- `DROPBOX_TRANSCRIPTS_PATH` Dropbox transcripts folder to import locally (default: `/Transcripts`)
- `DROPBOX_TRANSCRIPTS_SYNC_ENABLED` set `1` to sync transcripts from Dropbox before digest (default: `1`)
- `DROPBOX_ROOT_PATH` cloud root folder (default: `/read`)
- `DROPBOX_SYNC_ENABLED` set `1` to enable sync worker
- `DROPBOX_SYNC_TIME` daily sync time in `HH:MM` (default: `00:10`)
- `DROPBOX_SYNC_TZ` sync timezone (default: `Asia/Taipei`)
- `DROPBOX_SYNC_ON_STARTUP` set `1` to run one full backfill at startup

Slack (Socket Mode):
- `SLACK_BOT_TOKEN` (xoxb-...)
- `SLACK_APP_TOKEN` (xapp-...) with `connections:write`
- `SLACK_USER_ID` your user id (U...)

AI summary for `/summary`:
- `AI_SUMMARY_ENABLED` set `1` to enable
- `AI_SUMMARY_PROVIDER` supports `openai`, `gemini`, `anthropic`, `huggingface`, `ollama` (`antropic`, `hf`, `local` aliases also work)
- `AI_SUMMARY_TIMEOUT_SECONDS` request timeout for all providers
- `AI_SUMMARY_MAX_CHARS` max context size sent to model
- OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL`
- Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL`
- Anthropic: `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`
- Hugging Face: `HUGGINGFACE_API_KEY`, `HUGGINGFACE_MODEL`, `HUGGINGFACE_BASE_URL`
- Ollama: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
- `start.ps1` will auto-pull `OLLAMA_MODEL` if missing locally.

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
Transcription flow: bot sends `已成功紀錄` right after transcript is saved, then sends AI summary afterward (if enabled).

Group logging (no reply):
- Messages in allowed groups are stored in SQLite and appended to Markdown files.
- Markdown files: `DATA_DIR\notes\telegram\YYYY-MM-DD_telegram.md`
- SQLite DB: `DATA_DIR\messages.sqlite`

Slack DM logging (no reply):
- DMs from the configured `SLACK_USER_ID` are stored in the same SQLite/Markdown.
- Run uvicorn, then send a DM to your bot.

Image OCR and cloud sync:
- Telegram private image uploads are saved to `DATA_DIR\\images\YYYY-MM-DD\`.
- If OCR choice is enabled, bot asks per image: `進行 OCR` or `只存圖`; timeout defaults to save-only.
- OCR output is appended to `DATA_DIR\\notes\\telegram\\YYYY-MM-DD_telegram.md`.
- A Dropbox worker syncs local `notes` and `images` to:
- `/read & chat/read/notes`
- `/read & chat/read/images`
