# Preflight Checklist (Security + Stability + Backup + Monitoring)

Use this before production launch.

## A. Security

- [ ] `.env` is present only on server and excluded from git.
- [ ] `.env` permission is restricted: `chmod 600 .env`.
- [ ] Telegram token rotated if it was ever shared in logs/chat/screenshots.
- [ ] Dropbox uses refresh flow:
  - [ ] `DROPBOX_REFRESH_TOKEN` set
  - [ ] `DROPBOX_APP_KEY` set
  - [ ] `DROPBOX_APP_SECRET` set
- [ ] `DROPBOX_ACCESS_TOKEN` is empty or treated as fallback only.
- [ ] Server has firewall enabled (allow only required ports, typically SSH).
- [ ] SSH hardening:
  - [ ] key-based auth enabled
  - [ ] password auth disabled (recommended)
  - [ ] root login disabled (recommended)
- [ ] System packages updated: `sudo apt update && sudo apt upgrade -y`.

## B. Stability

- [ ] `TELEGRAM_LONG_POLLING=1` configured for server runtime.
- [ ] `python healthcheck.py` passes on server.
- [ ] `python -m py_compile app.py healthcheck.py` passes.
- [ ] `systemd` service installed and running:
  - [ ] `sudo systemctl enable jat-bot`
  - [ ] `sudo systemctl start jat-bot`
  - [ ] `sudo systemctl status jat-bot`
- [ ] Restart policy is active (`Restart=always`).
- [ ] Log stream is clean during smoke test: `journalctl -u jat-bot -f`.
- [ ] Dropbox transcript sync behaves as expected:
  - [ ] `DROPBOX_TRANSCRIPTS_PATH` points to correct folder
  - [ ] downloaded files appear in `data/transcripts`

## C. Backup & Recovery

- [ ] Backup target is defined (another disk/bucket/NAS).
- [ ] Daily backup is configured for:
  - [ ] `data/messages.sqlite`
  - [ ] `data/news`
  - [ ] `data/notes`
  - [ ] `data/transcripts` (if enabled)
  - [ ] `.env` (encrypted or secret-managed backup)
- [ ] Restore test completed at least once.
- [ ] Disk usage alert threshold set (for example 80%).

## D. Monitoring & Alerting

- [ ] Uptime monitor checks service health every 1-5 minutes.
- [ ] Failure alerts configured (email/Telegram/Slack).
- [ ] `journalctl` log retention configured (`journald` limits).
- [ ] Error budget set for API failures (Dropbox/LLM provider).
- [ ] Token-expiry/auth-error alerts defined (Dropbox auth failures).

## E. Cost Control

- [ ] AI provider and model explicitly selected for production.
- [ ] Rate limits and max tokens reviewed (`AI_SUMMARY_MAX_CHARS`, provider settings).
- [ ] News fetch/sync interval tuned to expected budget.
- [ ] Monthly spend cap and alert threshold configured.

## F. Go-Live Validation

- [ ] Bot responds to a normal message in Telegram.
- [ ] `/summary` works.
- [ ] `/summary note` includes expected note/transcript content.
- [ ] Dropbox sync log shows successful cycle.
- [ ] Optional one-time markdown cleanup completed (main/chitchat notes) if historical files used legacy format.
- [ ] Reboot test passed:
  - [ ] `sudo reboot`
  - [ ] service auto-starts and bot responds after reboot.
