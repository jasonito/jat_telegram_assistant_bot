import re
import sqlite3
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import quote

import feedparser
from rapidfuzz import fuzz

from config import (
    logger,
    DB_PATH, NEWS_MD_DIR,
    NEWS_RSS_URLS_ENV, NEWS_RSS_URLS_FILE,
    NEWS_FETCH_INTERVAL_MINUTES, NEWS_PUSH_MAX_ITEMS,
    NEWS_GNEWS_QUERY, NEWS_GNEWS_HL, NEWS_GNEWS_GL, NEWS_GNEWS_CEID,
    get_local_tz,
)
from utils import hash_text


# ---------------------------------------------------------------------------
# RSS URL helpers
# ---------------------------------------------------------------------------
def build_gnews_url(query: str, hl: str, gl: str, ceid: str) -> str:
    q = quote(query)
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"


def get_news_rss_urls() -> list[str]:
    if NEWS_RSS_URLS_FILE:
        path = Path(NEWS_RSS_URLS_FILE)
        if path.exists():
            raw = path.read_text(encoding="utf-8").splitlines()
            return [u.strip() for u in raw if u.strip()]

    if NEWS_RSS_URLS_ENV:
        raw = NEWS_RSS_URLS_ENV.splitlines()
        return [u.strip() for u in raw if u.strip()]

    gnews_url = build_gnews_url(NEWS_GNEWS_QUERY, NEWS_GNEWS_HL, NEWS_GNEWS_GL, NEWS_GNEWS_CEID)
    return [
        "https://www.reuters.com/technology/rss",
        gnews_url,
        "https://feeds.bloomberg.com/technology/news.rss",
        "https://feeds.bloomberg.com/technology/ai/news.rss",
        "https://semianalysis.com/feed/",
        "https://asia.nikkei.com/rss/feed/nar",
        "https://asia.nikkei.com/rss/feed/technology",
    ]


# ---------------------------------------------------------------------------
# Title normalization
# ---------------------------------------------------------------------------
def normalize_title(title: str) -> str:
    t = (title or "").strip().lower()
    if not t:
        return ""
    t = re.sub(
        r"^\s*(reuters|bloomberg|nikkei|nikkei asia|asia nikkei|semianalysis)\s*[-:|]\s*",
        "",
        t,
    )
    t = re.sub(
        r"\s*[-:|]\s*(reuters|bloomberg|nikkei|nikkei asia|asia nikkei|semianalysis)\s*$",
        "",
        t,
    )
    t = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Feed parsing helpers
# ---------------------------------------------------------------------------
def parse_entry_datetime(entry) -> datetime | None:
    dt_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if not dt_struct:
        return None
    ts = time.mktime(dt_struct)
    return datetime.fromtimestamp(ts, tz=get_local_tz())


def get_news_source_name(feed, url: str) -> str:
    if "news.google.com/rss/search" in (url or "") and "reuters" in NEWS_GNEWS_QUERY.lower():
        return "Reuters(GNews)"
    title = getattr(feed, "feed", {}).get("title") if feed else None
    if title:
        return title
    return url


def get_source_weight(source: str | None) -> int:
    s = (source or "").lower()
    if "reuters" in s:
        return 3
    if "bloomberg" in s:
        return 3
    if "nikkei" in s:
        return 2
    if "semianalysis" in s:
        return 2
    return 1


# ---------------------------------------------------------------------------
# Fetch RSS entries
# ---------------------------------------------------------------------------
def fetch_news_entries() -> list[dict]:
    items = []
    for url in get_news_rss_urls():
        feed = feedparser.parse(url)
        source_name = get_news_source_name(feed, url)
        for entry in getattr(feed, "entries", []):
            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            if not title or not link:
                continue
            summary = entry.get("summary") or entry.get("description") or ""
            published_at = parse_entry_datetime(entry)
            items.append(
                {
                    "source": source_name,
                    "title": title,
                    "url": link,
                    "summary": summary.strip(),
                    "published_at": published_at,
                }
            )
    items.sort(key=lambda x: x["published_at"] or datetime.now(tz=get_local_tz()))
    return items


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
def find_similar_cluster(recent_rows: list[tuple], title_norm: str) -> int | None:
    if not title_norm:
        return None
    best_cluster = None
    best_score = 0
    threshold = 95 if len(title_norm) < 40 else 92
    for cluster_id, existing_norm in recent_rows:
        if not existing_norm:
            continue
        score = fuzz.token_set_ratio(title_norm, existing_norm)
        if score >= threshold and score > best_score:
            best_score = score
            best_cluster = cluster_id
    return best_cluster


def ensure_cluster(
    conn: sqlite3.Connection,
    item: dict,
    recent_rows: list[tuple],
) -> int:
    title_norm = item["title_norm"]
    cluster_id = find_similar_cluster(recent_rows, title_norm)
    published_at = item["published_at"] or datetime.now(tz=get_local_tz())
    published_iso = published_at.isoformat()
    summary_len = len(item["summary"] or "")
    source_weight = get_source_weight(item["source"])
    if cluster_id:
        row = conn.execute(
            """
            SELECT canonical_published_at, canonical_summary, canonical_title, canonical_url, canonical_source
            FROM news_clusters
            WHERE id = ?
            """,
            (cluster_id,),
        ).fetchone()
        if row:
            canonical_published_at, canonical_summary, canonical_title, canonical_url, canonical_source = row
        else:
            canonical_published_at = None
            canonical_summary = None
            canonical_title = None
            canonical_url = None
            canonical_source = None

        update_fields = {}
        canonical_summary_len = len(canonical_summary or "")
        canonical_weight = get_source_weight(canonical_source)
        canonical_time = canonical_published_at or published_iso

        replace_canonical = False
        if summary_len > canonical_summary_len:
            replace_canonical = True
        elif summary_len == canonical_summary_len:
            if source_weight > canonical_weight:
                replace_canonical = True
            elif source_weight == canonical_weight and published_iso < canonical_time:
                replace_canonical = True

        if replace_canonical:
            update_fields["canonical_title"] = item["title"]
            update_fields["canonical_url"] = item["url"]
            update_fields["canonical_source"] = item["source"]
            update_fields["canonical_published_at"] = published_iso
            update_fields["canonical_summary"] = item["summary"] or None
            update_fields["canonical_summary_source"] = item["source"] if item["summary"] else None
        elif not canonical_summary and item["summary"]:
            update_fields["canonical_summary"] = item["summary"]
            update_fields["canonical_summary_source"] = item["source"]

        if update_fields:
            sets = ", ".join(f"{k} = ?" for k in update_fields.keys())
            params = list(update_fields.values()) + [cluster_id]
            conn.execute(f"UPDATE news_clusters SET {sets} WHERE id = ?", params)
        return cluster_id

    cluster_date = published_at.strftime("%Y%m%d")
    row = conn.execute(
        "SELECT COALESCE(MAX(cluster_seq), 0) + 1 FROM news_clusters WHERE cluster_date = ?",
        (cluster_date,),
    ).fetchone()
    next_seq = row[0] if row else 1
    conn.execute(
        """
        INSERT INTO news_clusters
        (cluster_date, cluster_seq, canonical_title, canonical_url, canonical_source, canonical_published_at, canonical_summary, canonical_summary_source)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            cluster_date,
            next_seq,
            item["title"],
            item["url"],
            item["source"],
            published_iso,
            item["summary"] or None,
            item["source"] if item["summary"] else None,
        ),
    )
    cluster_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    recent_rows.append((cluster_id, title_norm))
    return cluster_id


# ---------------------------------------------------------------------------
# Fetch + store orchestrator
# ---------------------------------------------------------------------------
def fetch_and_store_news() -> set[str]:
    changed_dates: set[str] = set()
    entries = fetch_news_entries()
    if not entries:
        return changed_dates

    now = datetime.now(tz=get_local_tz())
    cutoff_dt = now - timedelta(hours=12)
    cutoff_iso = cutoff_dt.isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        recent_rows = conn.execute(
            """
            SELECT cluster_id, title_norm
            FROM news_items
            WHERE published_at IS NOT NULL AND published_at >= ?
            ORDER BY published_at DESC
            LIMIT 500
            """,
            (cutoff_iso,),
        ).fetchall()

        for item in entries:
            published_dt = item.get("published_at")
            if not published_dt:
                continue
            published_dt = published_dt.astimezone(get_local_tz())
            if published_dt < cutoff_dt:
                continue
            item["title_norm"] = normalize_title(item["title"])
            item["hash_url"] = hash_text(item["url"])
            item["hash_title"] = hash_text(item["title_norm"] or item["title"])
            exists = conn.execute(
                "SELECT 1 FROM news_items WHERE hash_url = ?",
                (item["hash_url"],),
            ).fetchone()
            if exists:
                continue

            cluster_id = ensure_cluster(conn, item, recent_rows)
            published_iso = published_dt.isoformat()
            created_iso = now.isoformat()
            conn.execute(
                """
                INSERT INTO news_items
                (cluster_id, source, title, title_norm, url, summary, published_at, hash_url, hash_title, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cluster_id,
                    item["source"],
                    item["title"],
                    item["title_norm"],
                    item["url"],
                    item["summary"],
                    published_iso,
                    item["hash_url"],
                    item["hash_title"],
                    created_iso,
                ),
            )
            row = conn.execute(
                "SELECT cluster_date FROM news_clusters WHERE id = ?",
                (cluster_id,),
            ).fetchone()
            if row and row[0]:
                changed_dates.add(row[0])
        conn.commit()

    for cluster_date in changed_dates:
        write_news_markdown_for_date(cluster_date)
    return changed_dates


# ---------------------------------------------------------------------------
# Markdown generation
# ---------------------------------------------------------------------------
def format_cluster_id(cluster_date: str, cluster_seq: int) -> str:
    return f"{cluster_date}-{int(cluster_seq):04d}"


def yaml_escape(text: str | None) -> str:
    if not text:
        return ""
    return text.replace('"', '\\"')


def write_news_markdown_for_date(cluster_date: str) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        clusters = conn.execute(
            """
            SELECT id, cluster_seq, canonical_title, canonical_url, canonical_source,
                   canonical_published_at, canonical_summary
            FROM news_clusters
            WHERE cluster_date = ?
            ORDER BY canonical_published_at ASC
            """,
            (cluster_date,),
        ).fetchall()

        blocks = []
        for (
            cluster_id,
            cluster_seq,
            canonical_title,
            canonical_url,
            canonical_source,
            canonical_published_at,
            canonical_summary,
        ) in clusters:
            sources = conn.execute(
                """
                SELECT DISTINCT source, url
                FROM news_items
                WHERE cluster_id = ?
                ORDER BY source ASC
                """,
                (cluster_id,),
            ).fetchall()

            cluster_id_str = format_cluster_id(cluster_date, cluster_seq)
            lines = [
                "---",
                f'cluster_id: "{cluster_id_str}"',
                f'published_at: "{yaml_escape(canonical_published_at)}"',
                "canonical:",
                f'  source: "{yaml_escape(canonical_source)}"',
                f'  url: "{yaml_escape(canonical_url)}"',
                "sources:",
            ]
            for source, url in sources:
                lines.append(f'  - source: "{yaml_escape(source)}"')
                lines.append(f'    url: "{yaml_escape(url)}"')
            lines.append(f'title: "{yaml_escape(canonical_title)}"')
            lines.append("---")
            summary = canonical_summary or ""
            blocks.append("\n".join(lines) + "\n" + summary.strip() + "\n")

    md_path = NEWS_MD_DIR / f"{cluster_date}_news.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(blocks))


# ---------------------------------------------------------------------------
# Queries & display
# ---------------------------------------------------------------------------
def get_latest_clusters(limit: int) -> list[tuple]:
    now = datetime.now(tz=get_local_tz())
    cutoff_iso = (now - timedelta(hours=12)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, canonical_title, canonical_url, canonical_source, canonical_published_at,
                   cluster_date, cluster_seq
            FROM news_clusters
            WHERE canonical_published_at >= ?
            ORDER BY canonical_published_at DESC
            LIMIT ?
            """,
            (cutoff_iso, limit),
        ).fetchall()
    return rows


def search_clusters(keyword: str, limit: int) -> list[tuple]:
    kw = f"%{keyword}%"
    now = datetime.now(tz=get_local_tz())
    cutoff_iso = (now - timedelta(hours=12)).isoformat()
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT c.id, c.canonical_title, c.canonical_url, c.canonical_source,
                   c.canonical_published_at, c.cluster_date, c.cluster_seq
            FROM news_clusters c
            JOIN news_items i ON i.cluster_id = c.id
            WHERE (i.title LIKE ? OR i.summary LIKE ?)
              AND c.canonical_published_at >= ?
            ORDER BY c.canonical_published_at DESC
            LIMIT ?
            """,
            (kw, kw, cutoff_iso, limit),
        ).fetchall()
    return rows


def format_cluster_list(rows: list[tuple]) -> str:
    lines = []
    for (
        _cid,
        title,
        url,
        source,
        published_at,
        cluster_date,
        cluster_seq,
    ) in rows:
        cluster_id_str = format_cluster_id(cluster_date, cluster_seq)
        lines.append(f"- [{source}] {title}")
        lines.append(f"  {url}")
        lines.append(f"  id={cluster_id_str} time={published_at}")
    return "\n".join(lines) if lines else "No news items found."


# ---------------------------------------------------------------------------
# Subscriptions
# ---------------------------------------------------------------------------
def upsert_news_subscription(chat_id: str, enabled: bool) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT chat_id FROM news_subscriptions WHERE chat_id = ?",
            (chat_id,),
        ).fetchone()
        if row:
            conn.execute(
                "UPDATE news_subscriptions SET enabled = ?, interval_minutes = ? WHERE chat_id = ?",
                (1 if enabled else 0, NEWS_FETCH_INTERVAL_MINUTES, chat_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO news_subscriptions (chat_id, enabled, interval_minutes, last_sent_at)
                VALUES (?, ?, ?, ?)
                """,
                (chat_id, 1 if enabled else 0, NEWS_FETCH_INTERVAL_MINUTES, None),
            )
        conn.commit()


def push_news_to_subscribers(send_fn) -> None:
    """Push news updates to subscribers. send_fn(chat_id, text) sends a message."""
    with sqlite3.connect(DB_PATH) as conn:
        subs = conn.execute(
            "SELECT chat_id, enabled, last_sent_at FROM news_subscriptions WHERE enabled = 1"
        ).fetchall()

    if not subs:
        return

    now = datetime.now(tz=get_local_tz())
    cutoff_iso = (now - timedelta(hours=12)).isoformat()
    now_iso = now.isoformat()
    for chat_id, _enabled, last_sent_at in subs:
        effective_last = last_sent_at
        if last_sent_at:
            try:
                last_dt = datetime.fromisoformat(last_sent_at)
                if last_dt < datetime.fromisoformat(cutoff_iso):
                    effective_last = cutoff_iso
            except Exception:
                effective_last = cutoff_iso
        with sqlite3.connect(DB_PATH) as conn:
            if effective_last:
                rows = conn.execute(
                    """
                    SELECT id, canonical_title, canonical_url, canonical_source, canonical_published_at,
                           cluster_date, cluster_seq
                    FROM news_clusters
                    WHERE canonical_published_at > ?
                      AND canonical_published_at >= ?
                    ORDER BY canonical_published_at ASC
                    LIMIT ?
                    """,
                    (effective_last, cutoff_iso, NEWS_PUSH_MAX_ITEMS),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, canonical_title, canonical_url, canonical_source, canonical_published_at,
                           cluster_date, cluster_seq
                    FROM news_clusters
                    WHERE canonical_published_at >= ?
                    ORDER BY canonical_published_at DESC
                    LIMIT ?
                    """,
                    (cutoff_iso, NEWS_PUSH_MAX_ITEMS),
                ).fetchall()
            if rows:
                msg = "News update:\n" + format_cluster_list(rows)
                send_fn(chat_id, msg)
            conn.execute(
                "UPDATE news_subscriptions SET last_sent_at = ? WHERE chat_id = ?",
                (now_iso, chat_id),
            )
            conn.commit()


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------
def start_news_thread(send_fn) -> None:
    def loop():
        while True:
            try:
                fetch_and_store_news()
                push_news_to_subscribers(send_fn)
            except Exception as e:
                logger.error("news worker error: %s", e)
            time.sleep(max(1, NEWS_FETCH_INTERVAL_MINUTES) * 60)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
