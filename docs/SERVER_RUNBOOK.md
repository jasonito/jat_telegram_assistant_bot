# Server Runbook (Start / Use / Stop)

This runbook is for `jat-bot` running on Linux with `systemd`.

## 0) Important Answer First

If your bot is deployed on a server, your personal computer can be turned off.  
The bot will continue running as long as the server stays on and `jat-bot` service is healthy.

## 1) Start Server Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable jat-bot
sudo systemctl start jat-bot
sudo systemctl status jat-bot
```

Expected status:

- `Active: active (running)`

## 2) Daily Use / Check

Check live logs:

```bash
journalctl -u jat-bot -f
```

Exit live logs with `Ctrl+C`.

Quick status:

```bash
sudo systemctl status jat-bot
```

If `systemctl status` opens a pager and you cannot type, press `q` to exit.

## 3) Restart After Config/Code Changes

```bash
cd /opt/jat-bots/jat_telegram_assistant_bot
git pull
source .venv/bin/activate
python healthcheck.py
sudo systemctl restart jat-bot
sudo systemctl status jat-bot
```

## 4) Stop Service

Temporarily stop:

```bash
sudo systemctl stop jat-bot
```

Stop auto-start on boot:

```bash
sudo systemctl disable jat-bot
```

Start again later:

```bash
sudo systemctl enable jat-bot
sudo systemctl start jat-bot
```

## 5) Reboot Server

```bash
sudo reboot
```

After reboot, verify service is back:

```bash
sudo systemctl status jat-bot
```

## 6) Common Troubleshooting

Service not running:

```bash
sudo systemctl status jat-bot
journalctl -u jat-bot -n 200 --no-pager
```

Apply env change but no effect:

- confirm `.env` path is correct in project root
- restart service:

```bash
sudo systemctl restart jat-bot
```

Dropbox auth errors:

- verify `.env` has:
  - `DROPBOX_REFRESH_TOKEN`
  - `DROPBOX_APP_KEY`
  - `DROPBOX_APP_SECRET`

Google Vision credential path error:

- ensure `GOOGLE_APPLICATION_CREDENTIALS` points to existing server file path

## 7) One-Time Markdown Cleanup (Dropbox + Local)

Use this when historical notes contain duplicate headings/blocks or old Telegram line format.

```bash
cd /opt/jat-bots/jat_telegram_assistant_bot
source .venv/bin/activate

# Main profile
python tools/cleanup_dropbox_notes_md.py --env-file .env.main --remote-root "/read & chat/read" --local-notes "read/notes"

# Chitchat profile
python tools/cleanup_dropbox_notes_md.py --env-file .env.chitchat --remote-root "/read & chat/chitchat" --local-notes "chitchat/notes"
```
