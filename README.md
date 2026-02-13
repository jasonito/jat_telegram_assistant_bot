# JAT Telegram Assistant Bot (PoC)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Deployment Docs

- Server deployment (24/7): `docs/DEPLOY_SERVER.md`
- Preflight checklist: `docs/PREFLIGHT_CHECKLIST.md`
- systemd template: `deploy/jat-bot.service.example`
- Server operation runbook: `docs/SERVER_RUNBOOK.md`


## Env

Copy `.env.example` to `.env` and set your tokens. Optional group logging:

- `TELEGRAM_ALLOWED_GROUPS` comma-separated group titles to log
- `DATA_DIR` location for SQLite + Markdown
- `TELEGRAM_LONG_POLLING` set `1` to enable getUpdates instead of webhook
- `TELEGRAM_LOCAL_WEBHOOK_URL` local URL for polling to forward updates (default: `http://127.0.0.1:8000/telegram`)
- `NEWS_ENABLED` set `0` to disable all news ingest/worker/commands for this bot instance
- OCR (Google Vision):
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
- `NEWS_URL_FETCH_MAX_ARTICLES` max news URLs to deep-read for `/summary news` (default: `3`)
- `NEWS_URL_FETCH_MAX_CHARS` max extracted text chars per URL (default: `3000`)
- `NEWS_URL_FETCH_TIMEOUT_SECONDS` timeout per URL fetch (default: `6`)
- `NEWS_DIGEST_MAX_ITEMS` max items rendered in `/summary news` digest (default: `6`)
- `NEWS_DIGEST_AI_ITEMS` number of items using AI deep-summary in `/summary news` (default: `2`)
- `NEWS_DIGEST_FETCH_ARTICLE_ITEMS` number of items fetching URL full-text in `/summary news` (default: `1`)
- OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL`
- Gemini: `GEMINI_API_KEY`, `GEMINI_MODEL`
- Anthropic: `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`
- Hugging Face: `HUGGINGFACE_API_KEY`, `HUGGINGFACE_MODEL`, `HUGGINGFACE_BASE_URL`
- Ollama: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
  - `start.ps1` will auto-pull `OLLAMA_MODEL` if missing locally.

## Run

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Long polling (no ngrok/webhook)

Set `TELEGRAM_LONG_POLLING=1`, then run:

```powershell
.\start.ps1
```

Run with a specific env file and port (useful for running multiple bots on one machine):

```powershell
.\start.ps1 -EnvFile .env.main -Port 8000
.\start.ps1 -EnvFile .env.chitchat -Port 8001
```

## Expose with ngrok

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

```
open https://google.com
notepad
sort downloads
/news latest
/news latest 2026-02-12
/summary
/summary 2026-02-12
/summary note
/summary note 2026-02-12
/summary news
/summary news 2026-02-12
```

Group logging (no reply):

- Messages in allowed groups are stored in SQLite and appended to Markdown files.
- Markdown files: `DATA_DIR\markdown\YYYY-MM-DD.md`
- SQLite DB: `DATA_DIR\messages.sqlite`

Slack DM logging (no reply):

- DMs from the configured `SLACK_USER_ID` are stored in the same SQLite/Markdown.
- Run uvicorn, then send a DM to your bot.

Image OCR and cloud sync:

- Telegram private image uploads are saved to `DATA_DIR\\inbox\\images\\YYYY-MM-DD\\`.
- OCR output is appended to `DATA_DIR\\notes\\ocr\\YYYY-MM-DD_ocr.md`.
- A Dropbox worker syncs local `news`, `notes`, and `inbox/images` to:
  - `/read/news`
  - `/read/notes`
  - `/read/images`
