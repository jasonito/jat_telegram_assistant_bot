# Deploy To Linux Server (24/7)

This guide deploys the bot on Ubuntu with `systemd` and `TELEGRAM_LONG_POLLING=1`.

## 1) Prepare Server

```bash
sudo apt update
sudo apt install -y python3 python3-venv git
```

## 2) Clone Project

```bash
cd /opt
sudo git clone <YOUR_REPO_URL> jat_telegram_assistant_bot
sudo chown -R $USER:$USER /opt/jat_telegram_assistant_bot
cd /opt/jat_telegram_assistant_bot
```

## 3) Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 4) Configure `.env`

Copy your working local `.env` to server and update:

- `TELEGRAM_LONG_POLLING=1`
- Dropbox refresh flow:
  - `DROPBOX_REFRESH_TOKEN`
  - `DROPBOX_APP_KEY`
  - `DROPBOX_APP_SECRET`
- Transcripts sync (if used):
  - `DROPBOX_TRANSCRIPTS_PATH=/Transcripts`
  - `DROPBOX_TRANSCRIPTS_SYNC_ENABLED=1`

Optional AI on server:

- Ollama local inference:
  - Install Ollama on server.
  - Set `AI_SUMMARY_PROVIDER=ollama`
  - Set `OLLAMA_BASE_URL=http://127.0.0.1:11434`
  - Set `OLLAMA_MODEL=qwen2.5:7b` (or your model)
- If server has no Ollama, switch provider (OpenAI/Gemini/Anthropic/HF).

## 5) Verify Before Service

```bash
source .venv/bin/activate
python healthcheck.py
python -m py_compile app.py healthcheck.py
```

## 6) Install `systemd` Service

```bash
sudo cp deploy/jat-bot.service.example /etc/systemd/system/jat-bot.service
sudo nano /etc/systemd/system/jat-bot.service
```

Edit these fields:

- `User=<LINUX_USER>`
- `WorkingDirectory=/opt/jat_telegram_assistant_bot`
- `ExecStart=/opt/jat_telegram_assistant_bot/.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8000`

Then enable/start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable jat-bot
sudo systemctl start jat-bot
sudo systemctl status jat-bot
```

## 7) Operations

```bash
# logs
journalctl -u jat-bot -f

# restart after env/code updates
sudo systemctl restart jat-bot

# stop service
sudo systemctl stop jat-bot

# disable autostart
sudo systemctl disable jat-bot
```

## 8) Upgrade Flow

```bash
cd /opt/jat_telegram_assistant_bot
git pull
source .venv/bin/activate
pip install -r requirements.txt
python healthcheck.py
sudo systemctl restart jat-bot
```

