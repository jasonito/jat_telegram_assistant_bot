# JAT Telegram Assistant Bot (PoC)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Env

Copy `.env.example` to `.env` and set your tokens. Optional group logging:

- `TELEGRAM_ALLOWED_GROUPS` comma-separated group titles to log
- `DATA_DIR` location for SQLite + Markdown

Slack (Socket Mode):

- `SLACK_BOT_TOKEN` (xoxb-...)
- `SLACK_APP_TOKEN` (xapp-...) with `connections:write`
- `SLACK_USER_ID` your user id (U...)

## Run

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
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
```

Group logging (no reply):

- Messages in allowed groups are stored in SQLite and appended to Markdown files.
- Markdown files: `DATA_DIR\markdown\YYYY-MM-DD.md`
- SQLite DB: `DATA_DIR\messages.sqlite`

Slack DM logging (no reply):

- DMs from the configured `SLACK_USER_ID` are stored in the same SQLite/Markdown.
- Run uvicorn, then send a DM to your bot.
