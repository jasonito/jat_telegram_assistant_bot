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
- KOL watchlist seed: `read/kol_watchlist.json`

## Env

Use profile-specific env files instead of a shared `.env` whenever possible:

- main bot: `.env.main`
- chitchat bot: `.env.chitchat`
- digest bot: `.env.digest`

`start.ps1` loads env values from `-EnvFile`, and `start-both.ps1` already uses `.env.main` + `.env.chitchat`.

### Segment A: Profile / Runtime
- 用途：選擇 bot profile 與資料根目錄。
- 主要參數：`APP_MODULE`、`APP_PROFILE`、`DATA_DIR`。
- 目前預設儲存位置：
- `main` 純文字 note：`H:\我的雲端硬碟\Obsidian\Resource\note`
- `main` transcript：`H:\我的雲端硬碟\Obsidian\Resource\transcript`
- `/daily_podcast` transcript：`H:\我的雲端硬碟\Obsidian\Resource\daily-podcast`
- `chitchat` 本地資料根目錄：`DATA_DIR`（僅用於 sqlite、images、runtime 暫存）
- `chitchat` 文字／圖片／transcript chatlog：Notion

### Segment B: Telegram Core
- 用途：設定 Telegram token、webhook/polling 模式與重試行為。
- 主要參數：`TELEGRAM_BOT_TOKEN`、`TELEGRAM_ALLOWED_GROUPS`、`TELEGRAM_ALLOWED_CONTROL_USERS`、`TELEGRAM_LONG_POLLING`、`TELEGRAM_LOCAL_WEBHOOK_URL`。
- 可選調整：`TELEGRAM_FILE_FETCH_*`、`TELEGRAM_POLL_*`。

### Segment C: Feature Flags
- 用途：不改程式碼就切換主要功能。
- 主要參數：`FEATURE_NEWS_ENABLED`、`FEATURE_TRANSCRIBE_ENABLED`、`FEATURE_TRANSCRIBE_AUTO_URL`、`FEATURE_OCR_ENABLED`、`FEATURE_OCR_CHOICE_ENABLED`、`FEATURE_SLACK_ENABLED`。
- OCR 選擇行為：`OCR_CHOICE_SCOPE`、`OCR_CHOICE_TIMEOUT_SECONDS`、`OCR_CHOICE_TIMEOUT_DEFAULT`。

### Segment D: Transcription Engine
Default transcription baseline: `WHISPER_MODEL=small`, `TRANSCRIBE_CHUNK_MINUTES=10`. The shorter chunk size helps long podcast episodes produce progress and checkpoints sooner on CPU-only hosts.
- 用途：控制 Whisper 的品質、速度、記憶體使用與分段策略。
- 主要參數：`TRANSCRIBE_MAX_DURATION_SECONDS`、`TRANSCRIBE_CHUNK_MINUTES`、`TRANSCRIBE_CHECKPOINT_FLUSH_SECONDS`、`WHISPER_MODEL`、`WHISPER_LANGUAGE`、`WHISPER_BEAM_SIZE`、`WHISPER_COMPUTE_TYPE`、`WHISPER_CPU_THREADS`、`WHISPER_BATCH_SIZE`、`FFMPEG_LOCATION`、`TRANSCRIBE_PROGRESS_HEARTBEAT_SECONDS`。
- 目前預設：`WHISPER_MODEL=small`。
- 長音訊會先在 `transcribe_audio()` 內部分段，再交給 Whisper，最後合併回原始時間戳。
- 目前程式路徑預設使用 `cpu`。若要改用 GPU，除了主機 CUDA 環境可用外，還需要額外程式調整。

### Segment E: OCR Provider
- 用途：設定圖片 OCR 後端。
- 主要參數：`OCR_PROVIDER`、`OCR_LANG_HINTS`、`GOOGLE_APPLICATION_CREDENTIALS`。

### Segment F: News / Digest
- 用途：蒐集、分類與整理新聞。
- 主要參數：`NEWS_ENABLED`、`NEWS_FETCH_INTERVAL_MINUTES`、`NEWS_LOOKBACK_HOURS`、`NEWS_STARTUP_FETCH_ENABLED`、`NEWS_STARTUP_NOTIFY_ENABLED`、`NEWS_PUSH_ENABLED`、`NEWS_PUSH_MAX_ITEMS`、`NEWS_GNEWS_*`、`NEWS_RSS_URLS`、`NEWS_RSS_URLS_FILE`、`NEWS_URL_FETCH_*`、`NEWS_DIGEST_*`、`NEWS_CLASSIFY_BATCH_SIZE`、`NOTE_DIGEST_MAX_ITEMS`。
- `main` profile 目前預設：`NEWS_FETCH_INTERVAL_MINUTES=360`、`NEWS_LOOKBACK_HOURS=24`、`NEWS_STARTUP_FETCH_ENABLED=1`、`NEWS_STARTUP_NOTIFY_ENABLED=1`。
- 目前行為：bot 啟動時會先立即執行一次新聞 ingest，通知已啟用的 `news_subscriptions`，之後每 6 小時固定跑一次。抓取視窗是 24 小時，因此電腦關機後下次啟動仍可補抓。
- `/news` 會讀取 `DATA_DIR\news` 下的本地 markdown，依 `Asia/Taipei` 時區篩選最近 24 小時的 `published_at`，排除房市來源後，再用 LLM 批次分類加上關鍵字 fallback，分成 7 類後輸出可點擊的 HTML 連結。
- `/house_news` 會抓取並顯示最近 24 小時房市新聞，來源包含信義房屋每日新聞、政大不動產研究中心美洲、住展雜誌房市動態、經濟日報房市、工商時報房市、好房網 News。
- `/news` 只使用 `DATA_DIR\news`。不會寫入 Obsidian `note`，也不會寫入 Notion。
- `/news_source` 會列出目前啟用/停用的新聞 RSS 來源。
- 當 `AI_SUMMARY_ENABLED=1` 時，新聞分類會使用 LLM；否則退回關鍵字分類。批次大小可用 `NEWS_CLASSIFY_BATCH_SIZE` 調整，預設為 40。
- 若近期區間的本地 `news` markdown 缺失，bot 會先嘗試從 Dropbox 同步回本地，再重試讀取。
- note/transcript 的 AI 輸入上限由 `NOTE_AI_INPUT_MAX_CHARS` 控制，目前預設 `28000`。

### Segment G: AI Summary Providers
- 用途：提供 digest 與 transcript 摘要共用的 AI 設定。
- 主要參數：`AI_SUMMARY_ENABLED`、`AI_SUMMARY_PROVIDER`、`AI_SUMMARY_TIMEOUT_SECONDS`、`AI_SUMMARY_MAX_CHARS`、`AI_SUMMARY_TEMPERATURE`。
- 各家 provider 參數：`OPENAI_*`、`GEMINI_*`、`ANTHROPIC_*`、`HUGGINGFACE_*`、`OLLAMA_*`。
- 新聞標題翻譯可透過 `NEWS_TITLE_TRANSLATION_PROVIDER` 使用獨立 provider，例如 `ollama` 或 `deeplx`。
- DeepLX 可選參數：`DEEPLX_API_URL`、`DEEPLX_AUTH_KEY`。

### Segment H: Dropbox Sync
- 用途：同步 notes、images、news，並處理 transcript 檔案同步。
- 主要參數：`DROPBOX_ACCESS_TOKEN`、`DROPBOX_REFRESH_TOKEN`、`DROPBOX_APP_KEY`、`DROPBOX_APP_SECRET`、`DROPBOX_TOKEN_REFRESH_LEEWAY_SECONDS`、`DROPBOX_ROOT_PATH`、`DROPBOX_SYNC_ENABLED`、`DROPBOX_SYNC_TIME`、`DROPBOX_SYNC_TZ`、`DROPBOX_SYNC_ON_STARTUP`、`DROPBOX_TRANSCRIPTS_PATH`、`DROPBOX_TRANSCRIPTS_SYNC_ENABLED`。

### Segment I: Notion (mainly chitchat)
- 用途：把 chitchat 的文字、圖片、transcript 追加到 Notion 頁面。
- 主要參數：`NOTION_ENABLED`、`NOTION_TOKEN`、`NOTION_VERSION`、`NOTION_CHATLOG_YEAR_PAGES_JSON`、`NOTION_CHATLOG_FALLBACK_PAGE_ID`、`NOTION_CHATLOG_IMAGE_MODE`、`NOTION_FILE_UPLOAD_VERSION`、`NOTION_CHATLOG_OCR_MODE`、`NOTION_CHATLOG_INCLUDE_TIME`。

### Segment J: Slack (optional)
- 用途：啟用 Socket Mode DM logging。
- 主要參數：`SLACK_BOT_TOKEN`、`SLACK_APP_TOKEN`、`SLACK_USER_ID`、`SLACK_DEBUG`。

### Templates

- main profile 範本：`.env.main.example`
- chitchat profile 範本：`.env.chitchat.example`
- digest profile 範本：`.env.digest.example`
- 舊版通用範本：`.env.example`

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

Register the main bot to auto-start on Windows logon:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools\register-startup-task.ps1 -TaskName "JAT Telegram Assistant Bot Main" -EnvFile ".env.main" -Port 8000 -Trigger Logon
```

Notes:
- This creates a local Windows Task Scheduler entry, so it must be registered again on another PC.
- The helper script lives at [tools/register-startup-task.ps1](/C:/Users/tsaiy/jat_telegram_assistant_bot/tools/register-startup-task.ps1).
- The current task runs `start.ps1` with `.env.main` after user logon, which is more reliable than a pre-logon startup trigger for this bot.

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
- replaceable social-source adapters via `build_x_source_adapter()`, `build_facebook_source_adapter()`, and `build_social_source_adapter()`
- optional `snscrape`-backed X adapter via `SnscrapeXAdapter`
- optional Apify-backed X adapter skeleton via `ApifyXAdapter`
- Facebook provider interface wired via `StubFacebookAdapter` placeholder, so Meta Pages API or scraper-backed implementations can plug in without changing the digest flow
- Telegram watchlist management via `/digest_watchlist`
- digest profile background scheduler aligned to `08:00 Asia/Taipei`
- default fetch slots at `02:00 / 08:00 / 14:00 / 20:00 Asia/Taipei`, with the `08:00` slot generating the previous calendar day's digest

Current X adapter notes:

- select the provider with `KOL_X_SOURCE_PROVIDER=snscrape|apify`
- `snscrape` remains the default bootstrap path
- `ApifyXAdapter` is wired as a generic task/actor client and usually needs `APIFY_X_INPUT_TEMPLATE_JSON` to match the chosen actor schema
- provider-specific notes live in `docs/APIFY_X_ADAPTER.md`

Current Facebook adapter notes:

- select the provider with `KOL_FACEBOOK_SOURCE_PROVIDER=stub`
- `stub` is the current default and intentionally fails with a clear error until a real Facebook adapter is plugged in
- planned real implementations can target Meta `Page Public Content Access` or a scraper-backed fetcher behind the same `SocialSourceAdapter` contract
- the Meta skeleton is now wired as `KOL_FACEBOOK_SOURCE_PROVIDER=meta`
- required env for the Meta skeleton: `META_GRAPH_API_ACCESS_TOKEN` or `META_PAGE_PUBLIC_CONTENT_ACCESS_TOKEN`
- optional env for the Meta skeleton: `META_GRAPH_API_VERSION`, `META_FACEBOOK_POSTS_EDGE`, `META_FACEBOOK_FIELDS`
- this adapter only proves the Graph API integration path; actual success still depends on Meta app review / `Page Public Content Access` approval and the exact Page permissions granted to the token

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

Startup troubleshooting notes:

- If `start-both.ps1` reports `Health check did not become ready ... /healthz within 30s`, check the uvicorn startup log first. The failure may be an app startup exception, not an HTTP health route problem.
- The main profile now tolerates a missing `kol_digest.py`. KOL commands/features are marked unavailable instead of crashing the whole bot during import.
- `set_telegram_commands()` failures to `api.telegram.org` should no longer kill startup. If Telegram is blocked by local firewall / endpoint policy, the bot can still start and serve `/healthz`, but Telegram polling or command sync may log connection errors.
- If you need to isolate the issue quickly, test the app import directly:

```powershell
.\.venv\Scripts\python.exe -c "import app_main; print('app_main import ok')"
```

- Then test the local server without `start-both.ps1`:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app_main:app --host 127.0.0.1 --port 8011
Invoke-WebRequest http://127.0.0.1:8011/healthz
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

`/news` 與 `/transcribe` 是否可用，取決於 `FEATURE_NEWS_ENABLED` 與 `FEATURE_TRANSCRIBE_ENABLED`。

`/summary_news_daily` 與 `/news_latest` 目前仍保留為相容別名，但主要入口是 `/news`。新聞來源列表入口是 `/news_source`。

本地控制白名單：
- 將 `TELEGRAM_ALLOWED_CONTROL_USERS` 設為以逗號分隔的 Telegram `user_id` 與／或 `username`。
- 範例：`TELEGRAM_ALLOWED_CONTROL_USERS=123456789,my_telegram_username`
- 可在 Telegram 使用 `/whoami` 查看目前的 `user_id` 與 `chat_id`。

轉錄流程：bot 會在 transcript 存檔後先送出 `transcript saved`，若有啟用 AI 摘要，之後再補送摘要。

群組記錄（不回覆）：
- `main` bot 在允許群組中的訊息會寫入 SQLite，並追加到 Obsidian note markdown。
- Markdown 檔案：`H:\我的雲端硬碟\Obsidian\Resource\note\YYYY-MM-DD_note.md`
- SQLite 資料庫：`DATA_DIR\messages.sqlite`
- `chitchat` bot 的一般聊天文字不會追加到本地 note markdown；chatlog 會寫到 Notion。

Slack DM logging（不回覆）：
- 來自指定 `SLACK_USER_ID` 的 DM 會依照目前 profile 的同一路由寫入 SQLite/Markdown。
- 啟動 uvicorn 後，再對 bot 傳送 DM。

轉錄執行行為：
- 長音訊會逐段轉錄，執行時會回報 `Transcribing segment n/m...`。
- 如果轉錄長時間停在 `0%`，通常表示 Whisper 尚未產生第一段輸出；此時模型大小、CPU 速度、chunking 與 `ffmpeg/ffprobe` 是否可用都會影響。

## Debug Recipes

診斷看起來卡住的轉錄工作：

1. 先看進度訊息，確認 bot 目前卡在下載、正規化還是 Whisper 轉錄階段。
2. 如果訊息長時間停在 `0%`，先假設 Whisper 還沒產生第一段。
3. 確認 chunking 是否有啟動，例如是否出現 `Transcribing segment 1/3...` 之類的狀態。
4. 檢查媒體工具：
   - `ffmpeg -version`
   - `ffprobe -version`
5. 確認目前 bot 使用的 env 檔內 Whisper model 設定：
   - `.env.main` 或 `.env.chitchat`
   - 目前建議基線是 `WHISPER_MODEL=small`
6. 修改 env 後要重啟 bot；已在執行中的程序不會自動套用新 model。

常用本地檢查：

```powershell
python -c "import transcription; print(transcription.get_transcribe_runtime_info())"
```

```powershell
ffmpeg -version
ffprobe -version
```

圖片 OCR 與雲端同步：
- Telegram 私訊上傳的圖片會存到 `DATA_DIR\\images\YYYY-MM-DD\`。
- 若啟用 OCR 選擇，bot 會逐張詢問：`OCR` 或 `save only`；逾時預設為 save-only。
- `main` 的 OCR 輸出會追加到 `H:\我的雲端硬碟\Obsidian\Resource\note\YYYY-MM-DD_note.md`。
- `chitchat` 的 OCR/chatlog 會寫到 Notion，不會寫到本地 Obsidian note markdown。
- Dropbox worker 會把本地 `notes` 與 `images` 同步到：
- `/read & chat/read/notes`
- `/read & chat/read/images`

目前同步說明：
- 目前啟用的本地資料根目錄是 `read/`，不是舊的 `data/`。
- `main` 的 Dropbox sync 會涵蓋 `/read & chat/read/...` 底下的 `notes`、`news`、`images`。
- `chitchat` 不應把本地 note markdown 視為主要 chatlog 儲存位置。
- `news` 會先支援 Dropbox 遠端同步回本地，再由 `/news` 讀取本地 markdown cache。

## Storage Map

- `main` bot 純文字訊息：`H:\我的雲端硬碟\Obsidian\Resource\note`
- `main` bot OCR note 追加：`H:\我的雲端硬碟\Obsidian\Resource\note`
- `main` bot transcript 檔案：`H:\我的雲端硬碟\Obsidian\Resource\transcript`
- `chitchat` bot 文字／圖片／transcript chatlog：Notion
- `/news` 本地快取與輸出 markdown：`DATA_DIR\news`
- `/daily_podcast` transcript 檔案：`H:\我的雲端硬碟\Obsidian\Resource\daily-podcast`
- SQLite 資料庫：`DATA_DIR\messages.sqlite`
- Telegram images：`DATA_DIR\images\YYYY-MM-DD\`

## News Email To Drive Export

`/news` on the main profile can trigger a Google Apps Script Web App after the
JAT News email is sent. Configure:

- `NEWS_EXPORT_WEBHOOK_URL`: Apps Script Web App `/exec` URL.
- `NEWS_EXPORT_WEBHOOK_SECRET`: shared secret sent in the JSON body.
- `NEWS_EXPORT_WEBHOOK_TIMEOUT_SECONDS`: request timeout, default `30`.

The Apps Script Web App must expose `doPost(e)` and call `exportEmailsToDrive()`
when it receives `{ "action": "export_jat_news", "secret": "..." }`. A ready-to-
paste script is available at `docs/jat_news_gmail_to_drive.gs`.

If `NEWS_EXPORT_WEBHOOK_URL` is empty, `/news` still sends email but reports that
Drive export was not triggered. Use `/news debug` to check both email and Drive
export readiness.

## Markdown Cleanup Maintenance

- 使用 `python tools\cleanup_dropbox_notes_md.py` 正規化既有 note markdown（去除重複標題／區塊，並把舊版 Telegram 行格式轉成 `- [HH:MM:SS] text`）。
- main profile cleanup：
  - `python tools\cleanup_dropbox_notes_md.py --env-file .env.main --remote-root "/read & chat/read" --local-notes "H:\我的雲端硬碟\Obsidian\Resource\note"`
- chitchat profile cleanup：
  - `chitchat` 正常流程不應依賴本地 note cleanup，因為 chatlog 的設計目標是寫入 Notion。
