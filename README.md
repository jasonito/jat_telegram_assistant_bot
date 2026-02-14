# JAT Telegram Assistant Bot (PoC)

## Setup (Windows, recommended)

Use Python 3.11 for stable `openai-whisper` install on Windows.

1. Install Python 3.11 and confirm it is available:

```powershell
py -0p
```

2. Create a clean virtual environment with Python 3.11:

```powershell
cd C:\Users\KHUser\jat_telegram_assistant_bot
Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

3. Install dependencies with constraints (important for whisper build deps):

```powershell
python -m pip install --upgrade "pip<26" "setuptools<81" wheel
$env:PIP_CONSTRAINT="$PWD\constraints.txt"
python -m pip install -r requirements.txt -c constraints.txt
```

4. Install and verify `ffmpeg` (required by Whisper runtime):

```powershell
ffmpeg -version
```

5. Optional: Install Ollama (required only when `AI_SUMMARY_PROVIDER=ollama` or `local`):

```powershell
winget install Ollama.Ollama
ollama --version
ollama serve
```

If `winget` is not available, install manually:
1. Open the official download page: https://ollama.com/download/windows
2. Download and run the Windows installer.
3. Restart PowerShell and verify:

```powershell
ollama --version
ollama serve
```

In another terminal, verify service and pull a model:

```powershell
ollama list
ollama pull qwen2.5:7b
```

Notes:
- `constraints.txt` pins packaging/build tooling used during install.
- `start.ps1` also uses this constraints file automatically when it installs dependencies.
- Ollama is not installed by `pip install -r requirements.txt`; it must be installed as a system CLI.

## Deployment Docs

- Server deployment (24/7): `docs/DEPLOY_SERVER.md`
- Preflight checklist: `docs/PREFLIGHT_CHECKLIST.md`
- systemd template: `deploy/jat-bot.service.example`
- Server operation runbook: `docs/SERVER_RUNBOOK.md`

## Env

For dual-bot startup (`start-both.ps1`), create two env files from dedicated templates:

```powershell
Copy-Item .env.main.example .env.main
Copy-Item .env.chitchat.example .env.chitchat
```

Required minimum settings:
- `.env.main`: set `TELEGRAM_BOT_TOKEN`, set `APP_MODULE=app_main`
- `.env.chitchat`: set `TELEGRAM_BOT_TOKEN`, set `APP_MODULE=app_chitchat`
- `.env.chitchat.example` intentionally excludes News/Slack config because chitchat does not use those features.

Single-bot mode is also supported with one env file (for example `.env`, based on `.env.example`).

- `TELEGRAM_ALLOWED_GROUPS` comma-separated group titles to log
- `DATA_DIR` location for SQLite + Markdown
- `APP_MODULE` uvicorn module name (`app`, `app_main`, `app_chitchat`)
- `TELEGRAM_LONG_POLLING` set `1` to enable getUpdates instead of webhook
- `TELEGRAM_LOCAL_WEBHOOK_URL` local URL for polling to forward updates (default: `http://127.0.0.1:8000/telegram`)

Feature flags:
- `FEATURE_NEWS_ENABLED` enable/disable news ingest + `/news` commands
- `FEATURE_TRANSCRIBE_ENABLED` enable/disable `/transcribe` and audio transcription
- `FEATURE_TRANSCRIBE_AUTO_URL` when set `1`, private chat plain URLs auto-trigger transcription
- `FEATURE_OCR_ENABLED` enable/disable image OCR pipeline
- `FEATURE_OCR_CHOICE_ENABLED` when set `1`, image OCR asks user to choose (`進行 OCR` or `只存圖`)
- `OCR_CHOICE_SCOPE` OCR choice scope (`private` recommended)
- `OCR_CHOICE_TIMEOUT_SECONDS` choice timeout seconds before fallback to save-only (default: `60`)
- `FEATURE_SLACK_ENABLED` enable/disable Slack worker

OCR (Google Vision):
- `OCR_PROVIDER` set `google_vision`
- `OCR_LANG_HINTS` OCR language hints (default: `zh-TW,en`)
- `GOOGLE_APPLICATION_CREDENTIALS` full path to Google service-account JSON key
- After downloading the service-account key (for example `vision.json`), place it in the project `gcp` folder and set `GOOGLE_APPLICATION_CREDENTIALS` to its absolute path, for example: `C:\Users\KHUser\jat_telegram_assistant_bot\gcp\vision.json`

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

Get `DROPBOX_REFRESH_TOKEN` (Windows CLI quick path):

0. Dropbox Console prep:
- Go to Dropbox Developer Console -> your App -> `Settings`.
- Under `OAuth 2` -> `Redirect URIs`, keep exactly one URI:
- `http://localhost:8000/dropbox/callback`
- Use the exact same string in all later steps.

1. Open this authorize URL in browser (replace `YOUR_APP_KEY`):

```text
https://www.dropbox.com/oauth2/authorize?client_id=YOUR_APP_KEY&response_type=code&token_access_type=offline&redirect_uri=http://localhost:8000/dropbox/callback
```

- After approve, browser redirects to:
- `http://localhost:8000/dropbox/callback?code=XXXX`
- `ERR_CONNECTION_REFUSED` is expected; copy the full callback URL from address bar.

2. Extract `code` from full callback URL (avoid copy pollution):

```powershell
$url = "http://localhost:8000/dropbox/callback?code=XXXXXXXX"
$code = $url -replace '^.*code=', ''
$code = $code -replace '&.*$', ''
$code = $code.Trim()

$code.Length
```

- `Length` should usually be 40+ chars.

3. Exchange `code` for `refresh_token`:

```powershell
$appKey = "YOUR_APP_KEY"
$appSecret = "YOUR_APP_SECRET"
$redirect = "http://localhost:8000/dropbox/callback"

$body = "code=$code&grant_type=authorization_code&client_id=$appKey&client_secret=$appSecret&redirect_uri=$([uri]::EscapeDataString($redirect))"

Invoke-RestMethod -Method Post `
  -Uri "https://api.dropboxapi.com/oauth2/token" `
  -ContentType "application/x-www-form-urlencoded" `
  -Body $body
```

- Success JSON includes `access_token`, `refresh_token`, `expires_in`.
- Save `refresh_token` into `.env.main` / `.env.chitchat` as needed.

4. Use `refresh_token` to get a new `access_token` later:

```powershell
$refresh = "YOUR_REFRESH_TOKEN"
$appKey = "YOUR_APP_KEY"
$appSecret = "YOUR_APP_SECRET"

$body = "grant_type=refresh_token&refresh_token=$refresh&client_id=$appKey&client_secret=$appSecret"

Invoke-RestMethod -Method Post `
  -Uri "https://api.dropboxapi.com/oauth2/token" `
  -ContentType "application/x-www-form-urlencoded" `
  -Body $body
```

Common errors:
- `redirect_uri mismatch`: `redirect_uri` in request is not exactly the same as Dropbox Console setting.
- `code expired`: authorization code is one-time and short-lived; regenerate and retry immediately.
- Bad copy/paste code: always extract from full callback URL using the PowerShell snippet above.

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

Run both bots (requires `.env.main` and `.env.chitchat`):

```powershell
.\start-both.ps1
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

Group logging (no reply):
- Messages in allowed groups are stored in SQLite and appended to Markdown files.
- Markdown files: `DATA_DIR\notes\telegram\YYYY-MM-DD_telegram.md`
- SQLite DB: `DATA_DIR\messages.sqlite`

Slack DM logging (no reply):
- DMs from the configured `SLACK_USER_ID` are stored in the same SQLite/Markdown.
- Run uvicorn, then send a DM to your bot.

Image OCR and cloud sync:
- Telegram private image uploads are saved to `DATA_DIR\inbox\images\YYYY-MM-DD\`.
- If OCR choice is enabled, bot asks per image: `進行 OCR` or `只存圖`; timeout defaults to save-only.
- OCR output is appended to `DATA_DIR\notes\ocr\YYYY-MM-DD_ocr.md`.
- A Dropbox worker syncs local `notes` and `inbox/images` to:
- `/read/notes`
- `/read/images`
