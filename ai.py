import re
from datetime import datetime, timedelta
from pathlib import Path

import requests

from config import (
    logger,
    AI_SUMMARY_ENABLED, AI_SUMMARY_PROVIDER, AI_SUMMARY_TIMEOUT_SECONDS,
    AI_SUMMARY_MAX_CHARS,
    OPENAI_API_KEY, OPENAI_MODEL,
    GEMINI_API_KEY, GEMINI_MODEL,
    ANTHROPIC_API_KEY, ANTHROPIC_MODEL,
    HUGGINGFACE_API_KEY, HUGGINGFACE_MODEL, HUGGINGFACE_BASE_URL,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    NOTES_DIR, NEWS_MD_DIR,
)
from utils import request_with_retry, strip_markdown_noise


# ---------------------------------------------------------------------------
# AI provider abstraction
# ---------------------------------------------------------------------------
def _resolve_ai_provider() -> str:
    provider = (AI_SUMMARY_PROVIDER or "").strip().lower()
    if provider == "antropic":
        return "anthropic"
    if provider == "hf":
        return "huggingface"
    if provider == "local":
        return "ollama"
    return provider


def _call_ai_provider(
    provider: str, system_prompt: str, user_prompt: str, *, verbose: bool = False,
) -> str | None:
    """Unified AI provider call with retry and structured error handling."""
    try:
        if provider == "openai":
            if not OPENAI_API_KEY:
                if verbose:
                    logger.warning("AI summary enabled but OPENAI_API_KEY is empty")
                return None
            resp = request_with_retry(
                "POST",
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": OPENAI_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                if verbose:
                    logger.warning("OpenAI API failed: status=%d body=%s", resp.status_code, resp.text[:300])
                return None
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None

        if provider == "gemini":
            if not GEMINI_API_KEY:
                if verbose:
                    logger.warning("AI summary enabled but GEMINI_API_KEY is empty")
                return None
            resp = request_with_retry(
                "POST",
                f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent",
                params={"key": GEMINI_API_KEY},
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [{"parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]}],
                    "generationConfig": {"temperature": 0.2},
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                if verbose:
                    logger.warning("Gemini API failed: status=%d body=%s", resp.status_code, resp.text[:300])
                return None
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                return None
            parts = candidates[0].get("content", {}).get("parts") or []
            return "\n".join((p.get("text") or "").strip() for p in parts if p.get("text")).strip() or None

        if provider == "anthropic":
            if not ANTHROPIC_API_KEY:
                if verbose:
                    logger.warning("AI summary enabled but ANTHROPIC_API_KEY is empty")
                return None
            resp = request_with_retry(
                "POST",
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": ANTHROPIC_API_KEY,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={
                    "model": ANTHROPIC_MODEL,
                    "max_tokens": 800,
                    "temperature": 0.2,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                if verbose:
                    logger.warning("Anthropic API failed: status=%d body=%s", resp.status_code, resp.text[:300])
                return None
            data = resp.json()
            blocks = data.get("content") or []
            return "\n".join((b.get("text") or "").strip() for b in blocks if b.get("type") == "text").strip() or None

        if provider == "huggingface":
            if not HUGGINGFACE_API_KEY:
                if verbose:
                    logger.warning("AI summary enabled but HUGGINGFACE_API_KEY is empty")
                return None
            resp = request_with_retry(
                "POST",
                f"{HUGGINGFACE_BASE_URL.rstrip('/')}/chat/completions",
                headers={
                    "Authorization": f"Bearer {HUGGINGFACE_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": HUGGINGFACE_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.2,
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                if verbose:
                    logger.warning("HuggingFace API failed: status=%d body=%s", resp.status_code, resp.text[:300])
                return None
            data = resp.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "").strip() or None

        if provider == "ollama":
            resp = request_with_retry(
                "POST",
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/chat",
                headers={"Content-Type": "application/json"},
                json={
                    "model": OLLAMA_MODEL,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {"temperature": 0.2},
                },
                timeout=AI_SUMMARY_TIMEOUT_SECONDS,
            )
            if resp.status_code >= 300:
                if verbose:
                    logger.warning("Ollama API failed: status=%d body=%s", resp.status_code, resp.text[:300])
                return None
            data = resp.json()
            return data.get("message", {}).get("content", "").strip() or None

        if verbose:
            logger.warning("Unsupported AI_SUMMARY_PROVIDER=%s", provider)
        return None

    except (requests.ConnectionError, requests.Timeout) as exc:
        logger.warning("AI provider %s network error: %s", provider, exc)
        return None
    except Exception as exc:
        logger.warning("AI provider %s unexpected error: %s", provider, exc)
        return None


def run_ai_chat(system_prompt: str, user_prompt: str) -> str | None:
    provider = _resolve_ai_provider()
    return _call_ai_provider(provider, system_prompt, user_prompt)


# ---------------------------------------------------------------------------
# Output normalizers
# ---------------------------------------------------------------------------
def _normalize_news_output(text: str) -> str:
    insufficient = "資料不足"
    if not text:
        return insufficient

    def clean_prefix(line: str) -> str:
        return re.sub(r"^\s*(?:[-*?]|\d+[.)])\s*", "", line).strip()

    lines_out: list[str] = []
    for raw in text.splitlines():
        line = strip_markdown_noise(raw).strip()
        if not line:
            continue
        line = clean_prefix(line)
        if not line:
            continue
        if line.lower().startswith(("a.", "news", "today")):
            continue

        md = re.search(r"\[([^\]]+)\]\((https?://[^)\s]+)\)", line)
        if md:
            title = md.group(1).strip()
            url = md.group(2).strip()
            lines_out.append(f"- <{url}|{title}> | URL: {url}")
            continue

        slack_link = re.search(r"<(https?://[^|>]+)\|([^>]+)>", line)
        if slack_link:
            url = slack_link.group(1).strip()
            title = slack_link.group(2).strip()
            lines_out.append(f"- <{url}|{title}> | URL: {url}")
            continue

        m = re.search(r"(https?://\S+)", line)
        url = m.group(1).rstrip(").,;") if m else ""
        title = line
        title = re.sub(r"\(https?://[^)\s]+\)", "", title).strip()
        title = re.sub(r"https?://\S+", "", title).strip(" -|:")
        if not title:
            title = "untitled"

        if url:
            lines_out.append(f"- <{url}|{title}> | URL: {url}")
        else:
            lines_out.append(f"- {title} | URL: 無")

    if not lines_out:
        return insufficient

    return "\n".join(lines_out[:12])


def _normalize_notes_output(text: str) -> str:
    insufficient = "資料不足"
    if not text:
        return insufficient

    def clean_prefix(line: str) -> str:
        return re.sub(r"^\s*(?:[-*?]|\d+[.)])\s*", "", line).strip()

    out: list[str] = []
    for raw in text.splitlines():
        line = strip_markdown_noise(raw).strip()
        if not line:
            continue
        line = clean_prefix(line)
        if not line:
            continue
        if line.lower().startswith(("b.", "notes", "highlights")):
            continue
        out.append(f"- {line}")

    if not out:
        return insufficient
    return "\n".join(out[:15])


# ---------------------------------------------------------------------------
# Mode detection heuristic
# ---------------------------------------------------------------------------
def detect_mode(context_text: str) -> str:
    text = (context_text or "").strip()
    total_len = len(text)

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    max_para_len = max((len(p) for p in paragraphs), default=0)
    avg_para_len = total_len / max(len(paragraphs), 1)

    date_marks = re.findall(r"\b\d{1,2}/\d{1,2}\b|\b\d{4}-\d{2}-\d{2}\b", text)
    if len(date_marks) >= 2:
        return "article_summary"

    if max_para_len >= 300 and total_len >= 800:
        return "article_summary"

    urls = re.findall(r"https?://", text)
    if len(urls) >= 2 and avg_para_len <= 120:
        return "slack_daily"

    article_score = 0
    if len(re.findall(r"我|我的|我們", text)) >= (total_len / 100) * 3:
        article_score += 1
    if len(re.findall(r"老實說|直到|結果|而且|但最", text)) >= 3:
        article_score += 1
    if article_score >= 2:
        return "article_summary"

    return "slack_daily"


# ---------------------------------------------------------------------------
# generate_ai_summary (legacy full-context builder)
# ---------------------------------------------------------------------------
def generate_ai_summary(
    day: str,
    msg_total: int,
    note_total: int,
    msg_by_platform: dict,
    note_by_platform: dict,
    recent_msgs: list[tuple],
    recent_notes: list[tuple],
    news_titles: list[tuple],
    news_rows: list[tuple] | None = None,
) -> str | None:
    if not AI_SUMMARY_ENABLED:
        return None

    provider = _resolve_ai_provider()

    def clip(text: str, max_len: int = 180) -> str:
        t = re.sub(r"\s+", " ", str(text or "").strip())
        return t if len(t) <= max_len else t[: max_len - 3] + "..."

    msg_platform = ", ".join(f"{k}={v}" for k, v in msg_by_platform.items()) or "none"
    note_platform = ", ".join(f"{k}={v}" for k, v in note_by_platform.items()) or "none"

    activity_lines = []
    for platform, chat_title, user_name, msg_text, received_ts in recent_msgs:
        ts = (received_ts or "")[:19]
        activity_lines.append(f"- [{ts}] ({platform}) {chat_title} | {user_name}: {clip(msg_text)}")

    note_lines = []
    for platform, ch_id, user_name, msg_text, received_ts in recent_notes:
        ts = (received_ts or "")[:19]
        note_lines.append(f"- [{ts}] ({platform}) {ch_id} | {user_name}: {clip(msg_text)}")

    normalized_news_rows = []
    source_news = news_rows if news_rows is not None else news_titles
    for row in source_news or []:
        if not row:
            continue
        if isinstance(row, (list, tuple)):
            title = row[0] if len(row) > 0 else ""
            url = row[1] if len(row) > 1 else ""
        else:
            title = str(row)
            url = ""
        title = str(title or "").strip()
        url = str(url or "").strip()
        if title:
            normalized_news_rows.append((title, url))

    news_lines = []
    for title, url in normalized_news_rows:
        url = url.strip()
        has_url = bool(re.match(r"^https?://", url, re.I))
        source = url if has_url else "來源缺失"
        news_lines.append(f"- {clip(title, 180)} | {source}")

    context_lines = [
        f"Date: {day}",
        f"Message total: {msg_total}",
        f"Note total: {note_total}",
        f"Messages by platform: {msg_platform}",
        f"Notes by platform: {note_platform}",
        "Recent activity:",
        *activity_lines,
        "Recent notes:",
        *note_lines,
        "News:",
        *news_lines,
    ]

    context_text = "\n".join(context_lines)
    if len(context_text) > AI_SUMMARY_MAX_CHARS:
        context_text = context_text[:AI_SUMMARY_MAX_CHARS]

    system_prompt = (
        "You are an operations and project decision assistant. "
        "Use only the provided input. Do not invent facts. "
        "Output must be in Traditional Chinese."
    )
    SLACK_DAILY_PROMPT = (
        f"請為 {day} 產出一份 Slack 每日摘要（僅使用輸入資料，繁體中文）。\n\n"
        "輸出格式（嚴格遵守）：\n"
        "A. 今日重點\n"
        "- 3-5 點，僅整理 data/news 中實際出現的事項，並附上 URL\n"
        "- 每點一行，避免泛化描述\n\n"
        "B. 待辦與行動\n"
        "- 3-5 點，以可執行動詞開頭\n"
        "- 若資訊不足，請寫「資料不足」，不要推測\n\n"
        "品質要求：\n"
        "- 不要使用英文標題\n"
        "- 不要輸出時間軸逐字紀錄\n"
        "- 總長度約 300-500 字\n\n"
        "輸入資料：\n"
        f"{context_text}\n"
    )

    ARTICLE_SUMMARY_PROMPT = (
        "請將以下內容整理為一篇結構化摘要（繁體中文）。\n\n"
        "輸出格式：\n"
        "1. 一句話主旨（1 句）\n"
        "2. 核心內容摘要（4-6 點，每點 2-3 句）\n"
        "   - 若原文有時間段或章節（如 2/6、3/6），請保留該結構\n"
        "   - 必須保留原文中的具體數字、工具名稱與系統設計細節\n"
        "3. 作者的核心觀點及結論（1-2 句），並給予實際可執行的方案\n\n"
        "規則：\n"
        "- 僅使用原文資訊，不得新增事件或觀點\n"
        "- 若某段資訊不足，請略過，不要補寫\n"
        "原始內容：\n"
        f"{context_text}\n"
    )

    mode = detect_mode(context_text)
    if mode == "article_summary":
        user_prompt = ARTICLE_SUMMARY_PROMPT
    else:
        user_prompt = SLACK_DAILY_PROMPT

    return _call_ai_provider(provider, system_prompt, user_prompt, verbose=True)


# ---------------------------------------------------------------------------
# merge_all: raw data fallback
# ---------------------------------------------------------------------------
def merge_all(day: str) -> list[str]:
    notes_files = sorted(NOTES_DIR.rglob(f"{day}*.md"))

    day_compact = day.replace("-", "")
    news_candidates = [
        NEWS_MD_DIR / f"{day}_news.md",
        NEWS_MD_DIR / f"{day_compact}.md",
        NEWS_MD_DIR / f"{day_compact}_news.md",
    ]
    news_files = [fp for fp in news_candidates if fp.exists()]
    if not news_files:
        news_files = sorted(NEWS_MD_DIR.glob(f"{day_compact}*.md"))

    parts: list[str] = [f"{day} merged raw data"]

    if notes_files:
        parts.append("== Notes ==")
        for fp in notes_files:
            parts.append(f"# file: {fp}")
            try:
                content = fp.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                content = f"[read error] {e}"
            parts.append(content if content else "[empty]")
    else:
        parts.append("== Notes ==")
        parts.append("[no notes files]")

    if news_files:
        parts.append("== News ==")
        for fp in news_files:
            parts.append(f"# file: {fp}")
            try:
                content = fp.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                content = f"[read error] {e}"
            parts.append(content if content else "[empty]")
    else:
        parts.append("== News ==")
        parts.append("[no news files]")

    return parts


# ---------------------------------------------------------------------------
# Two-stage daily summary
# ---------------------------------------------------------------------------
def summary_ai(day: str) -> list[str]:
    if not AI_SUMMARY_ENABLED:
        return merge_all(day)

    notes_files = sorted(NOTES_DIR.rglob(f"{day}*.md"))
    day_compact = day.replace("-", "")
    news_candidates = [
        NEWS_MD_DIR / f"{day}_news.md",
        NEWS_MD_DIR / f"{day_compact}.md",
        NEWS_MD_DIR / f"{day_compact}_news.md",
    ]
    news_files = [fp for fp in news_candidates if fp.exists()]
    if not news_files:
        news_files = sorted(NEWS_MD_DIR.glob(f"{day_compact}*.md"))

    def load_raw(files: list[Path]) -> str:
        chunks: list[str] = []
        for fp in files:
            try:
                content = fp.read_text(encoding="utf-8", errors="replace").strip()
            except Exception as e:
                content = f"[read error] {e}"
            chunks.append(f"# file: {fp}\n{content or '[empty]'}")
        raw = "\n\n".join(chunks)
        return raw[:AI_SUMMARY_MAX_CHARS] if len(raw) > AI_SUMMARY_MAX_CHARS else raw

    news_raw = load_raw(news_files)
    notes_raw = load_raw(notes_files)
    system_prompt = (
        "Use only provided content. Do not invent facts. "
        "Output must be in Traditional Chinese."
    )

    a_prompt = (
        f"Summarize news for {day}. Output 8-12 items. If source is limited, output all available items and prioritize at least 6.\n"
        "Each item must contain title and URL. If URL is missing, write URL: ?.\n"
        "Do NOT use title(url) inline parenthesis style.\n"
        "Do not invent facts.\n\n"
        f"{news_raw or '[no news files]'}"
    )
    b_prompt = (
        f"Summarize notes for {day}. Output 8-15 key points.\n"
        "Focus on decisions, TODOs, risks, and next actions.\n"
        "Do NOT use markdown emphasis symbols like **.\n"
        "Do not invent facts.\n\n"
        f"{notes_raw or '[no notes files]'}"
    )

    a = run_ai_chat(system_prompt, a_prompt)
    b = run_ai_chat(system_prompt, b_prompt)
    if not a and not b:
        return merge_all(day)

    a_fmt = _normalize_news_output(a or "")
    b_fmt = _normalize_notes_output(b or "")
    return [
        f"{day} summary (AI)",
        "A. 新聞",
        a_fmt,
        "",
        "B. 筆記",
        b_fmt,
    ]


# ---------------------------------------------------------------------------
# Weekly summary
# ---------------------------------------------------------------------------
def summary_weekly(day: str) -> list[str]:
    end_day = datetime.strptime(day, "%Y-%m-%d")
    days = [(end_day - timedelta(days=offset)).strftime("%Y-%m-%d") for offset in range(6, -1, -1)]

    daily_sections: list[str] = []
    for d in days:
        parts = summary_ai(d)
        if len(parts) >= 6 and parts[0].endswith("(AI)"):
            daily_sections.append(
                "\n".join(
                    [
                        f"## {d}",
                        parts[1],
                        parts[2],
                        parts[4],
                        parts[5],
                    ]
                )
            )
            continue

        fallback = merge_all(d)
        fallback_text = "\n".join(fallback)
        daily_sections.append(f"## {d}\n{fallback_text[:600]}")

    context_text = "\n\n".join(daily_sections)
    if len(context_text) > AI_SUMMARY_MAX_CHARS:
        context_text = context_text[:AI_SUMMARY_MAX_CHARS]

    system_prompt = (
        "Use only provided content. Do not invent facts. "
        "Output must be in Traditional Chinese."
    )
    weekly_prompt = (
        f"請針對截至 {day} 的近 7 天內容產出週摘要。\n"
        "輸出格式：\n"
        "A. 新聞\n"
        "- 5-10 點：重要趨勢、關鍵公司與事件，若有網址請保留。\n\n"
        "B. 筆記\n"
        "- 5-10 點：決策、待辦、風險與下週行動。\n\n"
        "請勿編造資訊。\n\n"
        f"{context_text}"
    )
    weekly = run_ai_chat(system_prompt, weekly_prompt)

    if not weekly:
        return [f"{day} summary_weekly", context_text]
    return [f"{day} summary_weekly (AI)", weekly]
