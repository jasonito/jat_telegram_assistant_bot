import re
import sqlite3
import webbrowser
from datetime import datetime, timedelta

from config import DB_PATH, get_local_tz
from utils import parse_day_arg, open_notepad, sort_downloads
from ai import summary_weekly, summary_ai, merge_all
from news import (
    get_news_rss_urls, fetch_and_store_news, search_clusters,
    get_latest_clusters, format_cluster_list, upsert_news_subscription,
)


def handle_command(text: str) -> str:
    text = text.strip()
    text_lower = text.lower()

    if text_lower.startswith("/summary_weekly"):
        day = parse_day_arg(text)
        parts = summary_weekly(day)
        return "\n".join(parts)

    if text_lower.startswith("/summary"):
        day = parse_day_arg(text)
        parts = summary_ai(day)
        return "\n".join(parts)

    if text_lower.startswith("open "):
        url = text[5:].strip()
        if not re.match(r"^https?://", url, re.I):
            url = "https://" + url
        webbrowser.open(url)
        return f"Opened: {url}"

    if text_lower in {"notepad", "open notepad"}:
        open_notepad()
        return "Opened Notepad."

    if text_lower == "sort downloads":
        return sort_downloads()

    if text_lower in {"help", "/help"}:
        return "Commands: open <url>, notepad, sort downloads"

    return "已成功紀錄"


def handle_news_command(text: str, chat_id: str) -> list[str]:
    tokens = text.strip().split()
    sub = tokens[1].lower() if len(tokens) > 1 else "latest"
    if sub in {"subscribe", "sub"}:
        upsert_news_subscription(chat_id, True)
        return ["已訂閱新聞推播。"]
    if sub in {"unsubscribe", "unsub"}:
        upsert_news_subscription(chat_id, False)
        return ["已取消新聞推播。"]
    if sub == "sources":
        srcs = get_news_rss_urls()
        return ["News sources:\n" + "\n".join(srcs)]
    if sub == "search":
        if len(tokens) < 3:
            return ["用法：/news search <keywords>"]
        fetch_and_store_news()
        rows = search_clusters(" ".join(tokens[2:]), 10)
        return [format_cluster_list(rows)]
    if sub == "debug":
        fetch_and_store_news()
        now = datetime.now(tz=get_local_tz())
        cutoff_iso = (now - timedelta(hours=12)).isoformat()
        with sqlite3.connect(DB_PATH) as conn:
            rows = conn.execute(
                """
                SELECT source, COUNT(*) AS cnt
                FROM news_items
                WHERE published_at >= ?
                GROUP BY source
                ORDER BY cnt DESC
                """,
                (cutoff_iso,),
            ).fetchall()
        if not rows:
            return ["No news items ingested in the last 12 hours."]
        lines = ["News items by source (last 12 hours):"]
        for source, cnt in rows:
            lines.append(f"- {source}: {cnt}")
        return ["\n".join(lines)]
    if sub == "latest":
        limit = 5
        if len(tokens) >= 3 and tokens[2].isdigit():
            limit = max(1, min(20, int(tokens[2])))
        fetch_and_store_news()
        rows = get_latest_clusters(limit)
        return [format_cluster_list(rows)]
    if sub == "help":
        return [
            "News commands: /news latest [N], /news search <keywords>, /news sources, /news debug, /news subscribe, /news unsubscribe"
        ]
    return ["未知指令。用 /news help 查看指令。"]
