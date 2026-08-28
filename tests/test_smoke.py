import importlib
import os
import shutil
import sys
import threading
import time
import types
import unittest
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")


def _install_test_stubs() -> None:
    if "requests" not in sys.modules:
        requests_mod = types.ModuleType("requests")

        class _DummyResponse:
            status_code = 200
            text = ""

            def json(self):
                return {}

            def raise_for_status(self):
                return None

        def _dummy_request(*args, **kwargs):
            return _DummyResponse()

        requests_mod.get = _dummy_request
        requests_mod.post = _dummy_request
        requests_mod.RequestException = Exception
        sys.modules["requests"] = requests_mod

    if "yt_dlp" not in sys.modules:
        yt_dlp_mod = types.ModuleType("yt_dlp")

        class YoutubeDL:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def download(self, *args, **kwargs):
                return 0

            def extract_info(self, *args, **kwargs):
                return {}

        yt_dlp_mod.YoutubeDL = YoutubeDL
        sys.modules["yt_dlp"] = yt_dlp_mod

    if "slugify" not in sys.modules:
        slugify_mod = types.ModuleType("slugify")

        def slugify(value, max_length=None):
            text = str(value or "").strip().lower().replace(" ", "-")
            return text[:max_length] if max_length else text

        slugify_mod.slugify = slugify
        sys.modules["slugify"] = slugify_mod

    if "faster_whisper" not in sys.modules:
        fw_mod = types.ModuleType("faster_whisper")
        fw_mod.BatchedInferencePipeline = None
        fw_mod.WhisperModel = None
        sys.modules["faster_whisper"] = fw_mod

    if "whisper" not in sys.modules:
        whisper_mod = types.ModuleType("whisper")

        def load_model(*args, **kwargs):
            class _Model:
                def transcribe(self, *args, **kwargs):
                    return {"text": ""}

            return _Model()

        whisper_mod.load_model = load_model
        sys.modules["whisper"] = whisper_mod

    if "dotenv" not in sys.modules:
        dotenv_mod = types.ModuleType("dotenv")

        def load_dotenv(*args, **kwargs):
            return False

        dotenv_mod.load_dotenv = load_dotenv
        sys.modules["dotenv"] = dotenv_mod

    if "feedparser" not in sys.modules:
        feedparser_mod = types.ModuleType("feedparser")

        def parse(*args, **kwargs):
            return {}

        feedparser_mod.parse = parse
        sys.modules["feedparser"] = feedparser_mod

    if "rapidfuzz" not in sys.modules:
        rapidfuzz_mod = types.ModuleType("rapidfuzz")

        class _Fuzz:
            @staticmethod
            def ratio(*args, **kwargs):
                return 0

            @staticmethod
            def token_set_ratio(a, b):
                sa = set(str(a).split())
                sb = set(str(b).split())
                if not sa or not sb:
                    return 0
                return 100 if sa == sb else 0

        rapidfuzz_mod.fuzz = _Fuzz()
        sys.modules["rapidfuzz"] = rapidfuzz_mod

    if "fastapi" not in sys.modules:
        fastapi_mod = types.ModuleType("fastapi")

        class FastAPI:
            def __init__(self, *args, **kwargs):
                pass

            def get(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator

            def post(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator

            def on_event(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator

        class Request:
            pass

        fastapi_mod.FastAPI = FastAPI
        fastapi_mod.Request = Request
        sys.modules["fastapi"] = fastapi_mod

    if "fastapi.responses" not in sys.modules:
        responses_mod = types.ModuleType("fastapi.responses")

        class JSONResponse:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        responses_mod.JSONResponse = JSONResponse
        sys.modules["fastapi.responses"] = responses_mod


_install_test_stubs()

app = importlib.import_module("app")
transcription = importlib.import_module("transcription")


class SmokeTests(unittest.TestCase):
    def setUp(self):
        app._recent_message_fingerprints.clear()
        app._recent_update_ids.clear()
        app._recent_transcribe_request_fingerprints.clear()

    def test_trigger_recent_news_drive_export_skips_when_unconfigured(self):
        with mock.patch.object(app, "NEWS_EXPORT_WEBHOOK_URL", ""):
            with mock.patch.object(app.requests, "post") as mocked_post:
                ok, status = app._trigger_recent_news_drive_export()

        self.assertFalse(ok)
        self.assertIn("NEWS_EXPORT_WEBHOOK_URL", status)
        mocked_post.assert_not_called()

    def test_trigger_recent_news_drive_export_posts_secret_payload(self):
        class _Resp:
            status_code = 200
            text = '{"ok":true}'

        with mock.patch.object(app, "NEWS_EXPORT_WEBHOOK_URL", "https://script.google.com/macros/s/example/exec"):
            with mock.patch.object(app, "NEWS_EXPORT_WEBHOOK_SECRET", "shared-secret"):
                with mock.patch.object(app, "NEWS_EXPORT_WEBHOOK_TIMEOUT_SECONDS", 7):
                    with mock.patch.object(app.requests, "post", return_value=_Resp()) as mocked_post:
                        ok, status = app._trigger_recent_news_drive_export()

        self.assertTrue(ok)
        self.assertEqual("已觸發 Google Drive 匯出", status)
        mocked_post.assert_called_once_with(
            "https://script.google.com/macros/s/example/exec",
            json={"secret": "shared-secret", "action": "export_jat_news"},
            timeout=7,
        )

    def test_private_plain_text_records_and_sends_ack(self):
        update = {
            "update_id": 3,
            "message": {
                "message_id": 12,
                "date": 1710391864,
                "chat": {"id": 123, "type": "private", "username": "alice"},
                "from": {"id": 456, "username": "alice"},
                "text": "今天先整理研究框架，晚上再補數字。",
            },
        }

        with mock.patch.object(app, "store_message") as mocked_store:
            with mock.patch.object(app, "append_markdown") as mocked_append:
                with mock.patch.object(app, "_spawn_background_to_thread") as mocked_bg:
                    with mock.patch.object(app, "send_ack_message", new=mock.AsyncMock(return_value=True)) as mocked_ack:
                        with mock.patch.object(app, "handle_transcribe_audio_message", new=mock.AsyncMock(return_value=False)):
                            with mock.patch.object(app, "handle_transcribe_cancel_command", new=mock.AsyncMock(return_value=False)):
                                with mock.patch.object(app, "handle_transcribe_text_command", new=mock.AsyncMock(return_value=False)):
                                    with mock.patch.object(
                                        app,
                                        "handle_transcribe_auto_url_message",
                                        new=mock.AsyncMock(return_value=False),
                                    ):
                                        asyncio.run(app.process_telegram_update(update))

        mocked_store.assert_called_once()
        mocked_append.assert_called_once()
        mocked_bg.assert_called_once_with(
            app.notion_append_chitchat_text,
            "今天先整理研究框架，晚上再補數字。",
            mock.ANY,
            label="notion text append",
        )
        mocked_ack.assert_awaited_once_with(123, "已成功紀錄")

    def test_extract_supported_transcribe_urls_keeps_multiple_supported_links_in_order(self):
        text = "\n".join(
            [
                "https://podcasts.apple.com/tr/podcast/a/id1?i=100",
                "https://podcasts.apple.com/tr/podcast/b/id2?i=200",
                "https://youtu.be/LEHlhpFTRhs?si=xTB7IJvgwRlpXkmu",
                "https://example.com/not-supported",
                "https://podcasts.apple.com/tr/podcast/b/id2?i=200",
            ]
        )

        urls = app._extract_supported_transcribe_urls(text)

        self.assertEqual(
            urls,
            [
                "https://podcasts.apple.com/tr/podcast/a/id1?i=100",
                "https://podcasts.apple.com/tr/podcast/b/id2?i=200",
                "https://youtu.be/LEHlhpFTRhs?si=xTB7IJvgwRlpXkmu",
            ],
        )

    def test_handle_transcribe_text_command_processes_multiple_urls_sequentially(self):
        text = "\n".join(
            [
                "/transcribe",
                "https://podcasts.apple.com/tr/podcast/a/id1?i=100",
                "https://podcasts.apple.com/tr/podcast/b/id2?i=200",
                "https://youtu.be/LEHlhpFTRhs?si=xTB7IJvgwRlpXkmu",
            ]
        )

        async def _run():
            with mock.patch.object(app, "FEATURE_TRANSCRIBE_ENABLED", True), mock.patch.object(
                app, "_run_transcribe_url_flow", new=mock.AsyncMock(return_value=True)
            ) as mocked_flow, mock.patch.object(
                app, "send_message", new=mock.AsyncMock(return_value=1)
            ) as mocked_send:
                handled = await app.handle_transcribe_text_command(123, text)
            return handled, mocked_flow, mocked_send

        handled, mocked_flow, mocked_send = asyncio.run(_run())

        self.assertTrue(handled)
        self.assertEqual(mocked_flow.await_count, 3)
        self.assertEqual(
            [call.args[1] for call in mocked_flow.await_args_list],
            [
                "https://podcasts.apple.com/tr/podcast/a/id1?i=100",
                "https://podcasts.apple.com/tr/podcast/b/id2?i=200",
                "https://youtu.be/LEHlhpFTRhs?si=xTB7IJvgwRlpXkmu",
            ],
        )
        mocked_send.assert_awaited_once_with(123, "偵測到 3 個可轉錄網址，將依序排隊處理。")

    def test_handle_transcribe_auto_url_message_processes_multiple_urls_sequentially(self):
        text = "\n".join(
            [
                "https://podcasts.apple.com/tr/podcast/a/id1?i=100",
                "https://podcasts.apple.com/tr/podcast/b/id2?i=200",
                "https://youtu.be/LEHlhpFTRhs?si=xTB7IJvgwRlpXkmu",
            ]
        )

        async def _run():
            with mock.patch.object(app, "FEATURE_TRANSCRIBE_ENABLED", True), mock.patch.object(
                app, "FEATURE_TRANSCRIBE_AUTO_URL", True
            ), mock.patch.object(
                app, "_run_transcribe_url_flow", new=mock.AsyncMock(return_value=True)
            ) as mocked_flow, mock.patch.object(
                app, "send_message", new=mock.AsyncMock(return_value=1)
            ) as mocked_send:
                handled = await app.handle_transcribe_auto_url_message(123, text)
            return handled, mocked_flow, mocked_send

        handled, mocked_flow, mocked_send = asyncio.run(_run())

        self.assertTrue(handled)
        self.assertEqual(mocked_flow.await_count, 3)
        self.assertEqual(
            [call.args[1] for call in mocked_flow.await_args_list],
            [
                "https://podcasts.apple.com/tr/podcast/a/id1?i=100",
                "https://podcasts.apple.com/tr/podcast/b/id2?i=200",
                "https://youtu.be/LEHlhpFTRhs?si=xTB7IJvgwRlpXkmu",
            ],
        )
        mocked_send.assert_awaited_once_with(123, "偵測到 3 個可轉錄網址，將依序排隊處理。")

    def test_edit_progress_message_resends_once_then_stops_fanning_out(self):
        async def _run():
            with mock.patch.object(
                app, "edit_message", new=mock.AsyncMock(return_value=False)
            ) as mocked_edit, mock.patch.object(
                app, "send_message", new=mock.AsyncMock(side_effect=[201, 202, 203, 204])
            ) as mocked_send:
                message_id = 100
                resends_left = app.NEWS_PROGRESS_RESEND_LIMIT
                seen: list[int] = []
                for _ in range(6):
                    message_id, resends_left = await app._edit_progress_message(
                        123, message_id, "progress", resends_left
                    )
                    seen.append(message_id)
            return seen, resends_left, mocked_edit, mocked_send

        seen, resends_left, mocked_edit, mocked_send = asyncio.run(_run())

        # Every tick still attempts an edit, but resends stop once the budget is spent.
        self.assertEqual(mocked_edit.await_count, 6)
        self.assertEqual(mocked_send.await_count, app.NEWS_PROGRESS_RESEND_LIMIT)
        self.assertEqual(resends_left, 0)
        self.assertEqual(seen, [201, 202, 202, 202, 202, 202])

    def test_edit_progress_message_keeps_budget_when_edit_succeeds(self):
        async def _run():
            with mock.patch.object(
                app, "edit_message", new=mock.AsyncMock(return_value=True)
            ), mock.patch.object(app, "send_message", new=mock.AsyncMock(return_value=999)) as mocked_send:
                message_id, resends_left = await app._edit_progress_message(
                    123, 100, "progress", app.NEWS_PROGRESS_RESEND_LIMIT
                )
            return message_id, resends_left, mocked_send

        message_id, resends_left, mocked_send = asyncio.run(_run())

        self.assertEqual(message_id, 100)
        self.assertEqual(resends_left, app.NEWS_PROGRESS_RESEND_LIMIT)
        mocked_send.assert_not_awaited()

    def test_handle_transcribe_auto_url_message_ignores_unsupported_url(self):
        text = "https://example.com/not-supported"

        async def _run():
            with mock.patch.object(app, "FEATURE_TRANSCRIBE_ENABLED", True), mock.patch.object(
                app, "FEATURE_TRANSCRIBE_AUTO_URL", True
            ), mock.patch.object(
                app, "_run_transcribe_url_flow", new=mock.AsyncMock(return_value=True)
            ) as mocked_flow, mock.patch.object(
                app, "send_message", new=mock.AsyncMock(return_value=1)
            ) as mocked_send:
                handled = await app.handle_transcribe_auto_url_message(123, text)
            return handled, mocked_flow, mocked_send

        handled, mocked_flow, mocked_send = asyncio.run(_run())

        self.assertFalse(handled)
        mocked_flow.assert_not_awaited()
        mocked_send.assert_not_awaited()

    def test_dropbox_remote_path_for_local_transcript_uses_transcript_root(self):
        tmpdir = Path("tests_runtime_transcribe") / "unit_transcript_path"
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        try:
            transcript_root = tmpdir / "_runtime" / "transcribe"
            transcript_root.mkdir(parents=True, exist_ok=True)
            transcript_path = transcript_root / "2026-03-17_sample.md"
            transcript_path.write_text("sample", encoding="utf-8")

            with mock.patch.object(app, "TRANSCRIPTS_DIR", transcript_root), mock.patch.object(
                app, "DROPBOX_TRANSCRIPTS_PATH", "/Transcripts"
            ):
                remote_path = app._dropbox_remote_path_for_local_transcript(transcript_path)

            self.assertEqual(remote_path, "/Transcripts/2026-03-17_sample.md")
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)

    def test_run_dropbox_sync_uploads_local_transcripts(self):
        tmpdir = Path("tests_runtime_transcribe") / "unit_dropbox_sync"
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        try:
            transcript_root = tmpdir / "_runtime" / "transcribe"
            transcript_root.mkdir(parents=True, exist_ok=True)
            transcript_path = transcript_root / "2026-03-17_sample.md"
            transcript_path.write_text("sample transcript", encoding="utf-8")

            zero_stats = {
                "transcripts_scanned": 0,
                "transcripts_downloaded": 0,
                "transcripts_skipped": 0,
                "transcripts_failed": 0,
            }
            uploaded_paths: list[tuple[Path, str]] = []

            def _record_upload(local_path: Path, remote_path: str) -> None:
                uploaded_paths.append((local_path, remote_path))

            with mock.patch.object(app, "TRANSCRIPTS_DIR", transcript_root), mock.patch.object(
                app, "DROPBOX_SYNC_ENABLED", True
            ), mock.patch.object(
                app, "DROPBOX_TRANSCRIPTS_SYNC_ENABLED", True
            ), mock.patch.object(
                app, "DROPBOX_TRANSCRIPTS_PATH", "/Transcripts"
            ), mock.patch.object(
                app, "_get_dropbox_client", return_value=object()
            ), mock.patch.object(
                app, "_dropbox_call_with_retry", side_effect=lambda func: func(object())
            ), mock.patch.object(
                app, "_dropbox_create_folder_if_missing"
            ), mock.patch.object(
                app, "sync_dropbox_news_to_local", return_value={}
            ), mock.patch.object(
                app, "sync_dropbox_transcripts_to_local", return_value=zero_stats
            ), mock.patch.object(
                app, "iter_sync_files", return_value=iter(())
            ), mock.patch.object(
                app, "sync_file_to_dropbox", side_effect=_record_upload
            ), mock.patch.object(
                app, "get_sync_state", return_value=None
            ), mock.patch.object(
                app, "upsert_sync_state"
            ):
                stats = app.run_dropbox_sync(full_scan=False)

            self.assertEqual(uploaded_paths, [(transcript_path, "/Transcripts/2026-03-17_sample.md")])
            self.assertEqual(stats["uploaded"], 1)
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)

    def test_postprocess_transcript_output_appends_daily_note_after_summary(self):
        calls: list[str] = []

        def _record_notion(**kwargs):
            calls.append("notion")

        def _record_build_summary(path, title):
            calls.append("build_summary")
            return "summary text"

        def _record_append(chat_id, title, source, transcript_path, message_ts, summary_text):
            calls.append("append_daily_note")

        def _record_sync(path):
            calls.append("sync_transcript")

        async def _run():
            with mock.patch.object(app, "notion_append_chitchat_transcript", side_effect=_record_notion), mock.patch.object(
                app, "_build_transcript_ai_summary", side_effect=_record_build_summary
            ), mock.patch.object(
                app, "_append_transcript_summary_to_note_markdown", side_effect=_record_append
            ), mock.patch.object(
                app, "_sync_single_transcript_file_to_dropbox", side_effect=_record_sync
            ), mock.patch.object(
                app, "send_message", new=mock.AsyncMock()
            ):
                await app._postprocess_transcript_output(
                    123,
                    title="title",
                    source="source",
                    transcript_path=Path("sample.md"),
                    message_ts=None,
                )

        asyncio.run(_run())
        self.assertEqual(
            calls,
            ["notion", "build_summary", "append_daily_note", "sync_transcript"],
        )

    def test_resolve_daily_podcast_selection_defaults_to_all(self):
        selected, error = app._resolve_daily_podcast_selection("")
        self.assertIsNone(error)
        self.assertEqual(len(selected), len(app.DAILY_PODCAST_SHOWS))

    def test_resolve_daily_podcast_selection_run_all_defaults_to_all(self):
        selected, error = app._resolve_daily_podcast_selection("run all")
        self.assertIsNone(error)
        self.assertEqual(len(selected), len(app.DAILY_PODCAST_SHOWS))

    def test_resolve_daily_podcast_selection_accepts_common_run_typo(self):
        selected, error = app._resolve_daily_podcast_selection("rull all")
        self.assertIsNone(error)
        self.assertEqual(len(selected), len(app.DAILY_PODCAST_SHOWS))

    def test_resolve_daily_podcast_selection_supports_multiple_keys(self):
        selected, error = app._resolve_daily_podcast_selection("tech_orange nyt_daily")
        self.assertIsNone(error)
        self.assertEqual(
            [item["key"] for item in selected],
            ["tech_orange", "nyt_daily"],
        )

    def test_resolve_china_podcast_selection_defaults_to_all(self):
        selected, error = app._resolve_china_podcast_selection("")
        self.assertIsNone(error)
        self.assertEqual(len(selected), len(app.CHINA_PODCAST_SHOWS))

    def test_resolve_china_podcast_selection_supports_multiple_keys(self):
        selected, error = app._resolve_china_podcast_selection("chinatalk china_update")
        self.assertIsNone(error)
        self.assertEqual(
            [item["key"] for item in selected],
            ["chinatalk", "china_update"],
        )

    def test_build_china_podcast_usage_mentions_command_and_keys(self):
        usage = app._build_china_podcast_usage()
        self.assertIn("/china_podcast", usage)
        self.assertIn("hudson_china_insider", usage)
        self.assertIn("china_desk", usage)
        self.assertNotIn("/china_podcast run all", usage)

    def test_resolve_house_podcast_selection_defaults_to_all(self):
        selected, error = app._resolve_house_podcast_selection("")
        self.assertIsNone(error)
        self.assertEqual(len(selected), len(app.HOUSE_PODCAST_SHOWS))

    def test_resolve_house_podcast_selection_supports_multiple_keys(self):
        selected, error = app._resolve_house_podcast_selection("estate_learning_voice real_estate_jango")
        self.assertIsNone(error)
        self.assertEqual(
            [item["key"] for item in selected],
            ["estate_learning_voice", "real_estate_jango"],
        )

    def test_build_house_podcast_usage_mentions_command_and_keys(self):
        usage = app._build_house_podcast_usage()
        self.assertIn("/house_podcast", usage)
        self.assertIn("estate_learning_voice", usage)
        self.assertIn("find_place_live_real_estate", usage)
        self.assertNotIn("/house_podcast run all", usage)

    def test_estimate_daily_podcast_total_seconds_uses_duration_and_model(self):
        estimate = app._estimate_daily_podcast_total_seconds(
            [
                {"duration_seconds": 1800},
                {"duration_seconds": 900},
            ],
            "small",
        )
        self.assertEqual(estimate, int((1800 + 900) * 0.85 + 180))

    def test_local_file_uri_formats_windows_drive_path(self):
        uri = app._local_file_uri(Path("H:/我的雲端硬碟/Obsidian/Resource/daily-podcast/out.md"))

        self.assertTrue(uri.startswith("file:///H:/"))
        self.assertIn("Obsidian/Resource/daily-podcast/out.md", uri)
        self.assertIn("%E6%88%91", uri)

    def test_local_file_uri_can_be_embedded_in_html_anchor(self):
        uri = app._local_file_uri(Path("H:/我的雲端硬碟/Obsidian/Resource/daily-podcast/a & b.md"))
        label = "Lex <Fridman>: a & b.md"
        anchor = f'- <a href="{app.escape(uri, quote=True)}">{app.escape(label)}</a>'

        self.assertIn('href="file:///H:/', anchor)
        self.assertIn("Lex &lt;Fridman&gt;: a &amp; b.md", anchor)

    def test_localize_transcribe_status_ignores_duration_warning(self):
        self.assertEqual(
            app._localize_transcribe_status(
                "Warning: cannot detect duration, continuing without progress percentage."
            ),
            "",
        )

    def test_daily_podcast_episode_state_key(self):
        self.assertEqual(
            app._daily_podcast_episode_state_key("nyt_daily", "123456"),
            "nyt_daily:123456",
        )

    def test_daily_podcast_episode_fallback_key(self):
        episode = {
            "show_key": "nyt_daily",
            "title": "Big Markets, Big Moves",
            "publish_date": "2026-04-05T00:00:00Z",
            "source_url": "https://podcasts.apple.com/us/podcast/the-daily/id1200361736",
        }
        key = app._daily_podcast_episode_fallback_key(episode)
        self.assertIsNotNone(key)
        self.assertTrue(key.startswith("nyt_daily:fallback:"))

    def test_is_daily_podcast_episode_processed_checks_existing_state_path(self):
        episode = {
            "show_key": "nyt_daily",
            "episode_id": "123456",
            "title": "Big Markets, Big Moves",
            "publish_date": "2026-04-05T00:00:00Z",
            "source_url": "https://podcasts.apple.com/us/podcast/the-daily/id1200361736",
        }
        with mock.patch.object(app, "get_sync_state", return_value="some/path.md") as mocked_get:
            with mock.patch.object(app, "_daily_podcast_state_path_exists", return_value=True) as mocked_exists:
                self.assertTrue(app._is_daily_podcast_episode_processed(episode))
        mocked_get.assert_called_once_with("daily_podcast_episode", "nyt_daily:123456")
        mocked_exists.assert_called_once_with("some/path.md")

    def test_is_daily_podcast_episode_processed_uses_fallback_key_when_episode_id_missing(self):
        episode = {
            "show_key": "nyt_daily",
            "title": "Big Markets, Big Moves",
            "publish_date": "2026-04-05T00:00:00Z",
            "source_url": "https://podcasts.apple.com/us/podcast/the-daily/id1200361736",
        }
        with mock.patch.object(app, "get_sync_state", return_value="some/path.md") as mocked_get:
            with mock.patch.object(app, "_daily_podcast_state_path_exists", return_value=True):
                self.assertTrue(app._is_daily_podcast_episode_processed(episode))
        called_key = mocked_get.call_args[0][1]
        self.assertTrue(called_key.startswith("nyt_daily:fallback:"))

    def test_is_china_podcast_episode_processed_checks_existing_markdown_dir(self):
        episode = {
            "show_key": "chinatalk",
            "episode_id": "123456",
            "title": "ChinaTalk Example",
            "publish_date": "2026-05-18T00:00:00Z",
            "source_url": "https://podcasts.apple.com/us/podcast/chinatalk/id1289062927",
        }
        with mock.patch.object(app, "get_sync_state", return_value=None):
            with mock.patch.object(
                app,
                "_find_existing_fixed_podcast_transcript",
                return_value=Path("existing.md"),
            ) as mocked_find:
                self.assertTrue(app._is_china_podcast_episode_processed(episode))

        mocked_find.assert_called_once_with(episode, app.CHINA_PODCAST_DIR)

    def test_is_house_podcast_episode_processed_checks_existing_markdown_dir(self):
        episode = {
            "show_key": "estate_learning_voice",
            "episode_id": "123456",
            "title": "House Podcast Example",
            "publish_date": "2026-05-18T00:00:00Z",
            "source_url": "https://podcasts.apple.com/tw/podcast/example/id123456789",
        }
        with mock.patch.object(app, "get_sync_state", return_value=None):
            with mock.patch.object(
                app,
                "_find_existing_fixed_podcast_transcript",
                return_value=Path("existing.md"),
            ) as mocked_find:
                self.assertTrue(app._is_house_podcast_episode_processed(episode))

        mocked_find.assert_called_once_with(episode, app.HOUSE_PODCAST_DIR)

    def test_format_fixed_podcast_failure_compacts_stalled_transcription(self):
        message = app._format_fixed_podcast_failure(
            "Lex Fridman Podcast",
            "Transcription appears stalled at 0% for 50m 20s. Please retry with a shorter clip or a smaller Whisper model.",
        )

        self.assertEqual(
            message,
            "Lex Fridman Podcast：轉錄逾時，長時間停在 0%；建議改用較小模型或縮短音訊",
        )

    def test_mark_daily_podcast_episode_processed_uses_all_state_keys(self):
        episode = {
            "show_key": "nyt_daily",
            "episode_id": "123456",
            "title": "Big Markets, Big Moves",
            "publish_date": "2026-04-05T00:00:00Z",
            "source_url": "https://podcasts.apple.com/us/podcast/the-daily/id1200361736",
        }
        with mock.patch.object(app, "upsert_sync_state") as mocked_upsert:
            app._mark_daily_podcast_episode_processed(episode, Path("out.md"))
        self.assertEqual(mocked_upsert.call_count, 3)
        first_key = mocked_upsert.call_args_list[0][0][1]
        second_key = mocked_upsert.call_args_list[1][0][1]
        third_key = mocked_upsert.call_args_list[2][0][1]
        self.assertEqual(first_key, "nyt_daily:123456")
        self.assertTrue(second_key.startswith("nyt_daily:fallback:"))
        self.assertTrue(third_key.startswith("nyt_daily:content:"))

    def test_find_existing_fixed_podcast_transcript_matches_legacy_markdown(self):
        tmpdir = Path("tests_runtime_transcribe") / "unit_daily_podcast_existing"
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        try:
            day_dir = tmpdir / "2026-05-18"
            day_dir.mkdir(parents=True, exist_ok=True)
            transcript_path = day_dir / "The Daily - Big Markets, Big Moves.md"
            transcript_path.write_text(
                "# The Daily - Big Markets, Big Moves\n\n"
                "- **Source:** https://podcasts.apple.com/us/podcast/the-daily/id1200361736\n"
                "- **Type:** podcast\n"
                "- **Date transcribed:** 2026-05-18\n"
                "- **Duration:** Unknown\n\n"
                "---\n\nsample",
                encoding="utf-8",
            )
            episode = {
                "show_key": "nyt_daily",
                "show_label": "The Daily",
                "episode_id": "123456",
                "title": "Big Markets, Big Moves",
                "publish_date": "2026-05-18T00:00:00Z",
                "source_url": "https://podcasts.apple.com/us/podcast/the-daily/id1200361736",
            }
            self.assertEqual(
                app._find_existing_fixed_podcast_transcript(episode, tmpdir),
                transcript_path,
            )
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)

    def test_transcribe_podcast_episode_to_markdown_uses_resolved_episode_audio(self):
        tmpdir = Path("tests_runtime_transcribe") / "unit_resolved_episode"
        if tmpdir.exists():
            shutil.rmtree(tmpdir)
        episode = {
            "show_key": "lex_fridman",
            "show_label": "Lex Fridman Podcast",
            "show_name": "Lex Fridman Podcast",
            "episode_id": "987",
            "episode_url": "https://podcasts.apple.com/tw/podcast/example/id1434243584?i=987",
            "source_url": "https://podcasts.apple.com/tw/podcast/lex-fridman-podcast/id1434243584",
            "audio_url": "https://cdn.example.com/lex.mp3",
            "title": "AI and Physics",
            "publish_date": "2026-05-18T00:00:00Z",
            "duration_seconds": 7200,
        }
        try:
            with mock.patch.object(transcription, "_pipeline_podcast", return_value=("title", Path("out.md"))) as mocked:
                result = transcription.transcribe_podcast_episode_to_markdown(
                    episode,
                    tmpdir / "out",
                    tmpdir / "tmp",
                )
            self.assertEqual(result, ("title", Path("out.md")))
            args, kwargs = mocked.call_args
            self.assertEqual(args[0], "https://cdn.example.com/lex.mp3")
            self.assertEqual(args[1], "Lex Fridman Podcast - AI and Physics")
            self.assertEqual(kwargs["source_url"], "https://podcasts.apple.com/tw/podcast/example/id1434243584?i=987")
            self.assertEqual(kwargs["duration_seconds"], 7200)
            self.assertEqual(kwargs["extra_metadata"]["Show Key"], "lex_fridman")
        finally:
            if tmpdir.exists():
                shutil.rmtree(tmpdir)

    def test_local_datetime_from_unix_uses_local_tz(self):
        dt = app._local_datetime_from_unix(0)
        self.assertIsNotNone(dt)
        self.assertIsNotNone(dt.tzinfo)
        self.assertEqual(dt.utcoffset().total_seconds(), 8 * 3600)

    def test_parse_entry_datetime_treats_feedparser_time_as_utc(self):
        entry = {"published_parsed": (1970, 1, 1, 0, 0, 0, 3, 1, 0)}
        dt = app.parse_entry_datetime(entry)
        self.assertIsNotNone(dt)
        self.assertEqual(dt.strftime("%Y-%m-%d %H:%M:%S %z"), "1970-01-01 08:00:00 +0800")

    def test_allowed_control_user_matches_id_and_username(self):
        with mock.patch.object(app, "ALLOWED_CONTROL_USERS", {"12345", "alice"}):
            self.assertTrue(app._is_allowed_control_user("12345", None))
            self.assertTrue(app._is_allowed_control_user(None, "@alice"))
            self.assertFalse(app._is_allowed_control_user("999", "bob"))

    def test_handle_command_blocks_unauthorized_local_control(self):
        with mock.patch.object(app, "ALLOWED_CONTROL_USERS", set()):
            msg = app.handle_command("open https://example.com", user_id="123", user_name="alice")
        self.assertIn("未授權", msg)

    def test_download_telegram_file_streams_to_disk(self):
        class FakeResponse:
            status_code = 200

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def iter_content(self, chunk_size=0):
                yield b"hello "
                yield b"world"

        tmpdir = Path("tests_runtime")
        tmpdir.mkdir(exist_ok=True)
        out = tmpdir / "audio.bin"
        try:
            with mock.patch.object(app, "telegram_get_file_info", return_value=("https://example.com/file", "file")):
                with mock.patch.object(app.requests, "get", return_value=FakeResponse()) as mocked_get:
                    app._download_telegram_file("file-id", out)

            self.assertEqual(out.read_bytes(), b"hello world")
            mocked_get.assert_called_once()
        finally:
            if out.exists():
                out.unlink()
            if tmpdir.exists():
                tmpdir.rmdir()

    def test_telegram_get_file_info_maps_too_large_error(self):
        class FakeResponse:
            status_code = 400
            text = '{"ok":false,"error_code":400,"description":"Bad Request: file is too big"}'

        with mock.patch.object(app.requests, "request", return_value=FakeResponse(), create=True):
            with self.assertRaises(app.TelegramFileTooLargeError) as ctx:
                app.telegram_get_file_info("file-id")

        self.assertIn("Bot API getFile", str(ctx.exception))
        self.assertIn("/transcribe <可下載的音訊 URL>", str(ctx.exception))

    def test_handle_transcribe_audio_message_rejects_oversized_upload_before_getfile(self):
        message = {
            "audio": {
                "file_id": "file-id",
                "file_unique_id": "uniq",
                "file_name": "large.m4a",
                "file_size": 75 * 1024 * 1024,
            }
        }

        with mock.patch.object(app, "FEATURE_TRANSCRIBE_ENABLED", True):
            with mock.patch.object(app, "TELEGRAM_GETFILE_MAX_BYTES", 20 * 1024 * 1024):
                with mock.patch.object(app, "send_message", new=mock.AsyncMock()) as mocked_send:
                    with mock.patch.object(app, "_register_transcribe_job") as mocked_register:
                        handled = asyncio.run(app.handle_transcribe_audio_message(123, message))

        self.assertTrue(handled)
        mocked_register.assert_not_called()
        mocked_send.assert_awaited_once()
        sent_text = mocked_send.await_args.args[1]
        self.assertIn("75.0 MiB", sent_text)
        self.assertIn("20.0 MiB", sent_text)

    def test_private_youtube_url_does_not_block_on_notion_sync(self):
        update = {
            "update_id": 1,
            "message": {
                "message_id": 10,
                "date": 1710391864,
                "chat": {"id": 123, "type": "private", "username": "alice"},
                "from": {"id": 456, "username": "alice"},
                "text": "https://youtu.be/hXC7vtZCV_4?si=Lr6qGsGFs9v0ctoq",
            },
        }

        with mock.patch.object(app, "FEATURE_TRANSCRIBE_ENABLED", True), mock.patch.object(
            app, "store_message"
        ) as mocked_store:
            with mock.patch.object(app, "append_markdown") as mocked_append:
                with mock.patch.object(app, "_spawn_background_to_thread") as mocked_bg:
                    with mock.patch.object(app, "handle_transcribe_audio_message", new=mock.AsyncMock(return_value=False)):
                        with mock.patch.object(app, "handle_transcribe_cancel_command", new=mock.AsyncMock(return_value=False)):
                            with mock.patch.object(app, "handle_transcribe_text_command", new=mock.AsyncMock(return_value=False)):
                                with mock.patch.object(
                                    app,
                                    "handle_transcribe_auto_url_message",
                                    new=mock.AsyncMock(return_value=True),
                                ) as mocked_auto:
                                    asyncio.run(app.process_telegram_update(update))

        mocked_store.assert_not_called()
        mocked_append.assert_not_called()
        mocked_bg.assert_not_called()
        mocked_auto.assert_awaited_once()

    def test_private_youtube_url_with_comment_still_records_note(self):
        update = {
            "update_id": 2,
            "message": {
                "message_id": 11,
                "date": 1710391864,
                "chat": {"id": 123, "type": "private", "username": "alice"},
                "from": {"id": 456, "username": "alice"},
                "text": "這集先轉一下 https://youtu.be/hXC7vtZCV_4?si=Lr6qGsGFs9v0ctoq",
            },
        }

        with mock.patch.object(app, "store_message") as mocked_store:
            with mock.patch.object(app, "append_markdown") as mocked_append:
                with mock.patch.object(app, "_spawn_background_to_thread") as mocked_bg:
                    with mock.patch.object(app, "handle_transcribe_audio_message", new=mock.AsyncMock(return_value=False)):
                        with mock.patch.object(app, "handle_transcribe_cancel_command", new=mock.AsyncMock(return_value=False)):
                            with mock.patch.object(app, "handle_transcribe_text_command", new=mock.AsyncMock(return_value=False)):
                                with mock.patch.object(
                                    app,
                                    "handle_transcribe_auto_url_message",
                                    new=mock.AsyncMock(return_value=True),
                                ) as mocked_auto:
                                    asyncio.run(app.process_telegram_update(update))

        mocked_store.assert_called_once()
        mocked_append.assert_called_once()
        mocked_bg.assert_called_once_with(
            app.notion_append_chitchat_text,
            "這集先轉一下 https://youtu.be/hXC7vtZCV_4?si=Lr6qGsGFs9v0ctoq",
            mock.ANY,
            label="notion text append",
        )
        mocked_auto.assert_awaited_once()

    def test_private_unsupported_url_still_records_note(self):
        update = {
            "update_id": 4,
            "message": {
                "message_id": 13,
                "date": 1710391864,
                "chat": {"id": 123, "type": "private", "username": "alice"},
                "from": {"id": 456, "username": "alice"},
                "text": "https://example.com/not-supported",
            },
        }

        with mock.patch.object(app, "store_message") as mocked_store:
            with mock.patch.object(app, "append_markdown") as mocked_append:
                with mock.patch.object(app, "_spawn_background_to_thread") as mocked_bg:
                    with mock.patch.object(app, "send_ack_message", new=mock.AsyncMock(return_value=True)) as mocked_ack:
                        with mock.patch.object(app, "handle_transcribe_audio_message", new=mock.AsyncMock(return_value=False)):
                            with mock.patch.object(app, "handle_transcribe_cancel_command", new=mock.AsyncMock(return_value=False)):
                                with mock.patch.object(app, "handle_transcribe_text_command", new=mock.AsyncMock(return_value=False)):
                                    asyncio.run(app.process_telegram_update(update))

        mocked_store.assert_called_once()
        mocked_append.assert_called_once()
        mocked_bg.assert_called_once_with(
            app.notion_append_chitchat_text,
            "https://example.com/not-supported",
            mock.ANY,
            label="notion text append",
        )
        mocked_ack.assert_awaited_once_with(123, "已成功紀錄")

    def test_duplicate_edited_message_with_same_url_is_ignored(self):
        update_message = {
            "update_id": 101,
            "message": {
                "message_id": 10,
                "date": 1710391864,
                "chat": {"id": 123, "type": "private", "username": "alice"},
                "from": {"id": 456, "username": "alice"},
                "text": "https://youtu.be/hXC7vtZCV_4?si=Lr6qGsGFs9v0ctoq",
            },
        }
        update_edited = {
            "update_id": 102,
            "edited_message": {
                "message_id": 10,
                "date": 1710391864,
                "edit_date": 1710391865,
                "chat": {"id": 123, "type": "private", "username": "alice"},
                "from": {"id": 456, "username": "alice"},
                "text": "https://youtu.be/hXC7vtZCV_4?si=Lr6qGsGFs9v0ctoq",
            },
        }

        app._recent_message_fingerprints.clear()

        with mock.patch.object(app, "FEATURE_TRANSCRIBE_ENABLED", True), mock.patch.object(
            app, "store_message"
        ) as mocked_store:
            with mock.patch.object(app, "append_markdown") as mocked_append:
                with mock.patch.object(app, "_spawn_background_to_thread") as mocked_bg:
                    with mock.patch.object(app, "handle_transcribe_audio_message", new=mock.AsyncMock(return_value=False)):
                        with mock.patch.object(app, "handle_transcribe_cancel_command", new=mock.AsyncMock(return_value=False)):
                            with mock.patch.object(app, "handle_transcribe_text_command", new=mock.AsyncMock(return_value=False)):
                                with mock.patch.object(
                                    app,
                                    "handle_transcribe_auto_url_message",
                                    new=mock.AsyncMock(return_value=True),
                                ) as mocked_auto:
                                    asyncio.run(app.process_telegram_update(update_message))
                                    asyncio.run(app.process_telegram_update(update_edited))

        mocked_store.assert_not_called()
        mocked_append.assert_not_called()
        mocked_bg.assert_not_called()
        mocked_auto.assert_awaited_once()

    def test_edited_message_with_same_url_but_different_text_skips_duplicate_transcribe(self):
        update_message = {
            "update_id": 201,
            "message": {
                "message_id": 20,
                "date": 1710391864,
                "chat": {"id": 123, "type": "private", "username": "alice"},
                "from": {"id": 456, "username": "alice"},
                "text": "這集不錯 https://youtu.be/hXC7vtZCV_4?si=Lr6qGsGFs9v0ctoq",
            },
        }
        update_edited = {
            "update_id": 202,
            "edited_message": {
                "message_id": 20,
                "date": 1710391864,
                "edit_date": 1710391866,
                "chat": {"id": 123, "type": "private", "username": "alice"},
                "from": {"id": 456, "username": "alice"},
                "text": "這集真的不錯 https://youtu.be/hXC7vtZCV_4?si=Lr6qGsGFs9v0ctoq",
            },
        }

        with mock.patch.object(app, "FEATURE_TRANSCRIBE_ENABLED", True), mock.patch.object(
            app, "store_message"
        ) as mocked_store:
            with mock.patch.object(app, "append_markdown") as mocked_append:
                with mock.patch.object(app, "_spawn_background_to_thread") as mocked_bg:
                    with mock.patch.object(app, "handle_transcribe_audio_message", new=mock.AsyncMock(return_value=False)):
                        with mock.patch.object(app, "handle_transcribe_cancel_command", new=mock.AsyncMock(return_value=False)):
                            with mock.patch.object(app, "handle_transcribe_text_command", new=mock.AsyncMock(return_value=False)):
                                with mock.patch.object(
                                    app,
                                    "handle_transcribe_auto_url_message",
                                    new=mock.AsyncMock(return_value=True),
                                ) as mocked_auto:
                                    asyncio.run(app.process_telegram_update(update_message))
                                    asyncio.run(app.process_telegram_update(update_edited))

        self.assertEqual(mocked_store.call_count, 2)
        self.assertEqual(mocked_append.call_count, 2)
        self.assertEqual(mocked_bg.call_count, 2)
        mocked_auto.assert_awaited_once()

    def test_status_report_includes_telegram_poll_thread_health(self):
        class _Thread:
            name = "telegram-poll-1"

            def is_alive(self):
                return True

        with mock.patch.object(app, "TELEGRAM_LONG_POLLING", True):
            with mock.patch.object(app, "_telegram_poll_thread", _Thread()):
                with mock.patch.object(app, "_telegram_poll_thread_started_at", 100.0):
                    with mock.patch.object(app, "_telegram_poll_thread_restart_count", 2):
                        with mock.patch.object(app, "_telegram_poll_loop_last_seen_at", 195.0):
                            with mock.patch.object(app, "_telegram_poll_last_ok_at", 190.0):
                                with mock.patch.object(app, "_telegram_poll_last_update_at", 191.0):
                                    with mock.patch.object(app, "_telegram_poll_last_update_id", 123):
                                        with mock.patch.object(app, "_telegram_poll_last_error", ""):
                                            with mock.patch.object(app, "TELEGRAM_POLL_STALE_SECONDS", 30.0):
                                                with mock.patch.object(app.time, "time", return_value=200.0):
                                                    report = app.build_status_report()

        self.assertIn("telegram poll thread: alive (telegram-poll-1)", report)
        self.assertIn("telegram poll watchdog restarts: 2", report)
        self.assertIn("telegram poll stale: no", report)

    def test_healthz_reports_not_ok_when_long_polling_thread_is_down(self):
        with mock.patch.object(app, "TELEGRAM_LONG_POLLING", True):
            with mock.patch.object(app, "_telegram_poll_thread", None):
                with mock.patch.object(app, "_telegram_poll_loop_last_seen_at", 0.0):
                    payload = app.healthz()

        self.assertFalse(payload["ok"])
        self.assertEqual("long_polling", payload["telegram_mode"])
        self.assertFalse(payload["telegram_poll"]["thread_alive"])

    def test_set_telegram_commands_request_error_does_not_crash_when_response_missing(self):
        with mock.patch.object(app.requests, "Timeout", Exception, create=True):
            with mock.patch.object(app.requests, "RequestException", Exception, create=True):
                with mock.patch.object(app.requests, "post", side_effect=Exception("boom")):
                    app.set_telegram_commands()

    def test_send_message_last_error_redacts_bot_token(self):
        token_error = Exception(f"failed https://api.telegram.org/bot{app.BOT_TOKEN}/sendMessage")
        with mock.patch.object(app.requests, "post", side_effect=token_error):
            asyncio.run(app.send_message(123, "hello"))

        self.assertNotIn(app.BOT_TOKEN, app._telegram_send_last_error)
        self.assertIn("bot<redacted>", app._telegram_send_last_error)

    def test_estimate_weekly_topic_count_scales_to_ten(self):
        self.assertEqual(app._estimate_weekly_topic_count("短摘要", 3), 2)
        self.assertEqual(app._estimate_weekly_topic_count("a" * 9000, 80), 10)

    def test_collapse_similar_weekly_topics_merges_single_event_chain(self):
        items = [
            (
                "荷姆茲海峽風險",
                [
                    "美伊衝突升溫，荷姆茲海峽通行風險上升，推升原油供應不確定性。",
                    "檢查能源成本與供應鏈風險敞口。",
                ],
            ),
            (
                "油價與通膨壓力",
                [
                    "原油供應風險可能進一步推高油價與通膨預期，屬於同一事件鏈延伸。",
                    "關注油價與通膨數據。",
                ],
            ),
            (
                "韓國槓桿市場",
                [
                    "韓國高槓桿交易產品波動擴大，需另外觀察監管與流動性風險。",
                    "檢查槓桿曝險。",
                ],
            ),
        ]

        collapsed = app._collapse_similar_weekly_topics(items, max_items=10)

        self.assertEqual(len(collapsed), 2)
        merged_points = "\n".join(collapsed[0][1])
        self.assertIn("原油供應風險", merged_points)
        self.assertIn("檢查能源成本與供應鏈風險敞口。", merged_points)

    def test_translate_news_titles_to_zh_uses_ai_output_mapping(self):
        titles = [
            "Why the AI Boom Will Make Phones More Expensive",
            "Samsung's Galaxy S26 Ultra Is a Privacy-First Powerhouse",
        ]
        ai_output = (
            "1. 為何 AI 熱潮將讓手機更昂貴\n"
            "2. 三星 Galaxy S26 Ultra 主打隱私保護與高效能"
        )
        with mock.patch.object(app, "AI_SUMMARY_ENABLED", True):
            with mock.patch.object(app, "_run_ai_chat", return_value=ai_output):
                result = app._translate_news_titles_to_zh(titles)

        self.assertEqual(result[titles[0]], "為何 AI 熱潮將讓手機更昂貴")
        self.assertEqual(result[titles[1]], "三星 Galaxy S26 Ultra 主打隱私保護與高效能")

    def test_weekly_news_block_outputs_translated_titles(self):
        raw_lines = [
            "# 2026-03-03 ~ 2026-03-09 News Digest",
            "---",
            "## 1. [Why the AI Boom Will Make Phones More Expensive](https://example.com/a)",
            "## 2. [Samsung's Galaxy S26 Ultra Is a Privacy-First Powerhouse](https://example.com/b)",
        ]
        translations = {
            "Why the AI Boom Will Make Phones More Expensive": "為何 AI 熱潮將讓手機更昂貴",
            "Samsung's Galaxy S26 Ultra Is a Privacy-First Powerhouse": "三星 Galaxy S26 Ultra 主打隱私保護與高效能",
        }
        with mock.patch.object(app, "FEATURE_NEWS_ENABLED", True):
            with mock.patch.object(app, "build_news_digest_recent", return_value=raw_lines):
                with mock.patch.object(app, "_translate_news_titles_to_zh", return_value=translations):
                    result = app._build_weekly_news_block("2026-03-09", 7)

        joined = "\n".join(result)
        self.assertIn("為何 AI 熱潮將讓手機更昂貴", joined)
        self.assertIn("三星 Galaxy S26 Ultra 主打隱私保護與高效能", joined)

    def test_extract_note_lines_filters_transcript_intro(self):
        raw = "\n".join([
            "歡迎收看財報狗Podcast 我是主持人魏宇 在我旁邊的是 SKY 哈囉 大家好",
            "AI 伺服器需求持續擴大，供應鏈開始轉向高頻寬記憶體與先進封裝。",
        ])
        result = app._extract_note_lines(raw, limit=10)

        self.assertEqual(result, ["AI 伺服器需求持續擴大，供應鏈開始轉向高頻寬記憶體與先進封裝。"])

    def test_trim_note_intro_prefix_keeps_content_after_host_intro(self):
        raw = (
            "歡迎收看財報狗Podcast 我是主持人魏宇 在我旁邊的是財報狗投資總監SKY "
            "這週的股市波動很大 美國與伊朗衝突正在推升市場對油價與通膨的擔憂"
        )

        result = app._extract_note_lines(raw, limit=5)

        self.assertEqual(len(result), 1)
        self.assertNotIn("歡迎收看", result[0])
        self.assertIn("股市波動", result[0])
        self.assertIn("油價", result[0])

    def test_split_note_candidate_segments_breaks_long_transcript_line(self):
        raw = (
            "大家早安 歡迎收聽通勤10分鐘 我是Tony 我是Ester 今天第一則新聞想要跟大家講的就是 "
            "完美財報背後的大屠殺 Block在財報亮眼後仍裁員四成。 "
            "另外一個重點是AI代理與購物助理正在成為新的平台入口。 "
            "最後市場也在關注美國與伊朗衝突是否透過油價影響通膨預期。"
        )

        result = app._extract_note_lines(raw, limit=10)

        self.assertGreaterEqual(len(result), 3)
        self.assertTrue(any("Block" in line for line in result))
        self.assertTrue(any("AI代理" in line or "AI代理與購物助理" in line for line in result))
        self.assertTrue(any("油價" in line and "通膨" in line for line in result))

    def test_compose_note_ai_input_balances_days_with_larger_budget(self):
        day_to_lines = {
            "2026-03-06": [f"3月6日重點{i}：中東局勢與油價風險。" for i in range(1, 9)],
            "2026-03-07": [f"3月7日重點{i}：AI代理與企業導入。" for i in range(1, 6)],
        }

        result = app._compose_note_ai_input(day_to_lines, max_chars=600)

        self.assertIn("date: 2026-03-06", result)
        self.assertIn("date: 2026-03-07", result)
        self.assertIn("AI代理與企業導入", result)
        self.assertIn("中東局勢與油價風險", result)

    def test_load_raw_summary_files_can_skip_clip_limit(self):
        tmp_dir = Path("tests_runtime_notes_clip")
        tmp_dir.mkdir(exist_ok=True)
        try:
            fp = tmp_dir / "2026-03-09_note.md"
            fp.write_text("A" * 8000, encoding="utf-8")

            clipped = app._load_raw_summary_files([fp], clip_chars=6000)
            full = app._load_raw_summary_files([fp], clip_chars=None)

            self.assertLess(len(clipped), len(full))
            self.assertGreater(len(full), 7000)
        finally:
            if tmp_dir.exists():
                for child in tmp_dir.iterdir():
                    if child.is_file():
                        child.unlink()
                tmp_dir.rmdir()

    def test_compose_note_ai_input_from_raw_preserves_multiple_sections(self):
        raw = "\n".join([
            "# file: a.md",
            "# 第一段標題",
            "",
            "- **Source:** https://example.com/a",
            "- **Type:** podcast",
            "- **Date transcribed:** 2026-03-06",
            "- **Duration:** Unknown",
            "",
            "---",
            "",
            "第一段逐字稿內容，談中東局勢與油價風險。",
            "",
            "# 第二段標題",
            "",
            "- **Source:** https://example.com/b",
            "- **Type:** podcast",
            "- **Date transcribed:** 2026-03-06",
            "- **Duration:** Unknown",
            "",
            "---",
            "",
            "第二段逐字稿內容，談 OpenAI 與國防合作爭議。",
        ])

        result = app._compose_note_ai_input_from_raw({"2026-03-06": raw}, max_chars=2000)

        self.assertIn("第一段標題", result)
        self.assertIn("第二段標題", result)
        self.assertIn("油價風險", result)
        self.assertIn("OpenAI 與國防合作", result)
        self.assertNotIn("**Source:**", result)

    def test_compose_note_ai_input_from_raw_drops_bare_url_lines(self):
        raw = "\n".join([
            "# file: a.md",
            "https://example.com/only-link",
            "",
            "---",
            "",
            "真正要保留的逐字稿內容。",
        ])

        result = app._compose_note_ai_input_from_raw({"2026-03-09": raw}, max_chars=2000)

        self.assertNotIn("https://example.com/only-link", result)
        self.assertIn("真正要保留的逐字稿內容", result)

    def test_compact_note_summary_line_removes_show_prefix(self):
        text = "財報狗 - 掌握台股美股時事議題 - 507.【財經時事放大鏡】光 vs 銅 x 美國 vs 伊朗"
        result = app._compact_note_summary_line(text)

        self.assertNotIn("財報狗", result)
        self.assertIn("美國", result)

    def test_limit_weekly_topic_bucket_diversity_caps_same_bucket_to_two(self):
        items = [
            ("中東局勢推升油價", ["地緣政治衝突帶動原油價格與通膨壓力上升。"]),
            ("能源成本與通膨", ["油價走高正在擴散至通膨與央行政策預期。"]),
            ("美國利率與景氣", ["宏觀環境仍受通膨與利率影響。"]),
            ("AI 代理工作流", ["AI agent 工具開始進入實際工作流。"]),
            ("三星新機隱私策略", ["手機硬體與隱私功能成為賣點。"]),
        ]

        limited = app._limit_weekly_topic_bucket_diversity(items, max_items=5, per_bucket_limit=2)

        self.assertEqual(len(limited), 4)
        kept_titles = [title for title, _ in limited]
        macro_kept = [title for title in kept_titles if title in {"中東局勢推升油價", "能源成本與通膨", "美國利率與景氣"}]
        self.assertEqual(len(macro_kept), 2)
        self.assertIn("AI 代理工作流", kept_titles)
        self.assertIn("三星新機隱私策略", kept_titles)

    def test_translate_news_titles_to_zh_falls_back_to_rule_based_translation(self):
        title = "Why the AI Boom Will Make Phones, Cars and Electronics More Expensive"
        with mock.patch.object(app, "AI_SUMMARY_ENABLED", False):
            result = app._translate_news_titles_to_zh([title])

        self.assertIn("AI 熱潮", result[title])
        self.assertIn("手機", result[title])

    def test_translate_news_titles_to_zh_retries_when_ai_keeps_english_title(self):
        title = "Why the AI Boom Will Make Phones More Expensive"
        ai_outputs = [
            f"1. {title}",
            "1) 為何 AI 熱潮將使手機更昂貴",
        ]
        with mock.patch.object(app, "AI_SUMMARY_ENABLED", True):
            with mock.patch.object(app, "_run_ai_chat", side_effect=ai_outputs):
                result = app._translate_news_titles_to_zh([title])

        self.assertEqual(result[title], "為何 AI 熱潮將使手機更昂貴")

    def test_translate_news_titles_to_zh_uses_deeplx_translation(self):
        title = "Why the AI Boom Will Make Phones More Expensive"

        class FakeResp:
            status_code = 200
            content = b"1"

            def json(self):
                return {"data": "為何 AI 熱潮將使手機更昂貴"}

        with mock.patch.object(app, "AI_SUMMARY_ENABLED", True):
            with mock.patch.object(app, "NEWS_TITLE_TRANSLATION_PROVIDER", "deeplx"):
                with mock.patch.object(app.requests, "post", return_value=FakeResp()):
                    result = app._translate_news_titles_to_zh([title])

        self.assertEqual(result[title], "為何 AI 熱潮將使手機更昂貴")

    def test_parse_news_markdown_entries_extracts_title_url_and_time(self):
        raw = (
            "---\n"
            'published_at: "2026-03-14T09:00:00+08:00"\n'
            "canonical:\n"
            '  source: "Reuters"\n'
            '  url: "https://example.com/a"\n'
            'title: "Example title"\n'
            "---\n"
            "Summary text\n"
        )

        entries = app._parse_news_markdown_entries(raw)

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["title"], "Example title")
        self.assertEqual(entries[0]["url"], "https://example.com/a")

    def test_build_recent_news_links_html_reads_local_md_and_renders_html_links(self):
        tmp_news = Path("tests_runtime_news_links")
        now = datetime.now(tz=app.get_local_tz())
        current_iso = (now - timedelta(hours=1)).isoformat()
        old_iso = (now - timedelta(hours=30)).isoformat()
        current_name = now.strftime("%Y%m%d_news.md")
        previous_name = (now - timedelta(days=1)).strftime("%Y%m%d_news.md")
        try:
            if tmp_news.exists():
                for fp in tmp_news.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(tmp_news.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()
            tmp_news.mkdir(exist_ok=True)
            (tmp_news / current_name).write_text(
                "---\n"
                f'published_at: "{current_iso}"\n'
                "canonical:\n"
                '  source: "Reuters"\n'
                '  url: "https://example.com/a"\n'
                'title: "English title"\n'
                "---\n"
                "Summary\n",
                encoding="utf-8",
            )
            (tmp_news / previous_name).write_text(
                "---\n"
                f'published_at: "{old_iso}"\n'
                "canonical:\n"
                '  source: "Reuters"\n'
                '  url: "https://example.com/old"\n'
                'title: "Old title"\n'
                "---\n"
                "Summary\n",
                encoding="utf-8",
            )
            with mock.patch.object(app, "NEWS_MD_DIR", tmp_news):
                with mock.patch.object(app, "_translate_news_titles_to_zh", return_value={"English title": "中文標題"}):
                    html = app.build_recent_news_links_html(now=now)

            self.assertIn("最近 24 小時新聞", html)
            self.assertIn('href="https://example.com/a"', html)
            self.assertIn("中文標題", html)
            self.assertNotIn("https://example.com/old", html)
        finally:
            if tmp_news.exists():
                for fp in tmp_news.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(tmp_news.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()

    def test_build_recent_news_links_html_fetches_when_local_md_is_missing(self):
        tmp_news = Path("tests_runtime_news_fetch_fallback")
        now = datetime.now(tz=app.get_local_tz())
        current_iso = (now - timedelta(hours=1)).isoformat()
        current_name = now.strftime("%Y%m%d_news.md")

        def fake_fetch(**kwargs):
            (tmp_news / current_name).write_text(
                "---\n"
                f'published_at: "{current_iso}"\n'
                "canonical:\n"
                '  source: "Reuters"\n'
                '  url: "https://example.com/fetched"\n'
                'title: "Fetched title"\n'
                "---\n"
                "Summary\n",
                encoding="utf-8",
            )
            return {current_name[:8]}

        try:
            if tmp_news.exists():
                for fp in tmp_news.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
            tmp_news.mkdir(exist_ok=True)
            with mock.patch.object(app, "NEWS_MD_DIR", tmp_news):
                with mock.patch.object(app, "DROPBOX_SYNC_ENABLED", False):
                    with mock.patch.object(app, "_translate_news_titles_to_zh", return_value={}):
                        with mock.patch.object(app, "fetch_and_store_news", side_effect=fake_fetch) as fetch_mock:
                            html = app.build_recent_news_links_html(now=now)
                            self.assertEqual(fetch_mock.call_count, 1)
                            self.assertIn("https://example.com/fetched", html)

                            # allow_fetch=False 時不應再抓一次（house 已在上游抓過）
                            (tmp_news / current_name).unlink()
                            fetch_mock.reset_mock()
                            empty_html = app.build_recent_news_links_html(now=now, allow_fetch=False)
                            self.assertEqual(fetch_mock.call_count, 0)
                            self.assertIn("指定期間無可用新聞資料", empty_html)
        finally:
            if tmp_news.exists():
                for fp in tmp_news.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                tmp_news.rmdir()

    def test_news_and_house_news_filter_local_entries_by_source(self):
        tmp_news = Path("tests_runtime_house_news_links")
        now = datetime.now(tz=app.get_local_tz())
        current_iso = (now - timedelta(hours=1)).isoformat()
        current_name = now.strftime("%Y%m%d_news.md")
        try:
            if tmp_news.exists():
                for fp in tmp_news.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(tmp_news.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()
            tmp_news.mkdir(exist_ok=True)
            (tmp_news / current_name).write_text(
                "---\n"
                f'published_at: "{current_iso}"\n'
                "canonical:\n"
                '  source: "Reuters"\n'
                '  url: "https://example.com/general"\n'
                'title: "General title"\n'
                "---\n"
                "Summary\n"
                "---\n"
                f'published_at: "{current_iso}"\n'
                "canonical:\n"
                '  source: "房市動態 | 住展雜誌"\n'
                '  url: "https://example.com/house"\n'
                'title: "House title"\n'
                "---\n"
                "Summary\n",
                encoding="utf-8",
            )
            with mock.patch.object(app, "NEWS_MD_DIR", tmp_news):
                with mock.patch.object(app, "_translate_news_titles_to_zh", return_value={}):
                    news_html = app.build_recent_news_links_html(now=now)
                    house_html = app.build_recent_news_links_html(now=now, scope="house")

            self.assertIn("https://example.com/general", news_html)
            self.assertNotIn("https://example.com/house", news_html)
            self.assertIn("https://example.com/house", house_html)
            self.assertNotIn("https://example.com/general", house_html)
        finally:
            if tmp_news.exists():
                for fp in tmp_news.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(tmp_news.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()

    def _make_temp_news_db(self, name: str) -> Path:
        db_path = Path(name)
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{db_path}{suffix}")
            if candidate.exists():
                candidate.unlink()
        self.addCleanup(self._remove_temp_news_db, db_path)
        with mock.patch.object(app, "DB_PATH", db_path):
            with mock.patch.object(app, "FEATURE_NEWS_ENABLED", True):
                app.init_storage()
        return db_path

    @staticmethod
    def _remove_temp_news_db(db_path: Path) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{db_path}{suffix}")
            if candidate.exists():
                candidate.unlink()

    def test_fetch_and_store_news_survives_concurrent_duplicate_insert(self):
        db_path = self._make_temp_news_db("tests_runtime_news_race.sqlite")
        now = datetime.now(tz=app.get_local_tz())
        url = "https://example.com/race-article"
        entry = {
            "source": "Reuters",
            "title": "Race article",
            "url": url,
            "summary": "Summary",
            "published_at": now - timedelta(hours=1),
        }

        def steal_the_url(conn, item, recent_rows):
            # 模擬另一個 writer 在 exists 檢查之後、我們寫入之前搶先 commit 同一個 hash_url。
            with app._connect_db() as other:
                other.execute(
                    "INSERT INTO news_clusters (cluster_date, cluster_seq, canonical_url) VALUES (?, ?, ?)",
                    (now.strftime("%Y%m%d"), 1, url),
                )
                other_cluster_id = other.execute("SELECT last_insert_rowid()").fetchone()[0]
                other.execute(
                    """
                    INSERT INTO news_items
                    (cluster_id, source, title, title_norm, url, summary, published_at, hash_url, hash_title, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        other_cluster_id,
                        item["source"],
                        item["title"],
                        item["title_norm"],
                        url,
                        item["summary"],
                        (now - timedelta(hours=1)).isoformat(),
                        item["hash_url"],
                        item["hash_title"],
                        now.isoformat(),
                    ),
                )
                other.commit()
            return real_ensure_cluster(conn, item, recent_rows)

        real_ensure_cluster = app.ensure_cluster
        with mock.patch.object(app, "DB_PATH", db_path):
            with mock.patch.object(app, "fetch_news_entries", return_value=[entry]):
                with mock.patch.object(app, "ensure_cluster", side_effect=steal_the_url):
                    with mock.patch.object(app, "write_news_markdown_for_date") as mocked_write:
                        changed_dates = app.fetch_and_store_news(lookback_hours=24)

            with app._connect_db() as conn:
                item_count = conn.execute("SELECT COUNT(*) FROM news_items").fetchone()[0]
                cluster_count = conn.execute("SELECT COUNT(*) FROM news_clusters").fetchone()[0]

        self.assertEqual(changed_dates, set())
        self.assertEqual(item_count, 1)
        # 我們自己剛建的 cluster 沒有 item 掛上去，應該被回收，只留下另一個 writer 的那筆。
        self.assertEqual(cluster_count, 1)
        mocked_write.assert_not_called()

    def test_fetch_and_store_news_runs_are_serialised(self):
        # fetch_news_entries 回空清單時目前會在碰 DB 前就早退，但那是巧合性的安全：
        # 早退一旦移到 DB 存取之後，沒釘住 DB_PATH 的話就會寫進正式 messages.sqlite。
        db_path = self._make_temp_news_db("tests_runtime_news_serialised.sqlite")
        active = 0
        max_active = 0
        guard = threading.Lock()

        def slow_fetch(*args, **kwargs):
            nonlocal active, max_active
            with guard:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with guard:
                active -= 1
            return []

        with mock.patch.object(app, "DB_PATH", db_path), mock.patch.object(
            app, "fetch_news_entries", side_effect=slow_fetch
        ):
            threads = [
                threading.Thread(target=app.fetch_and_store_news, kwargs={"lookback_hours": 24})
                for _ in range(3)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=10)

        self.assertFalse(any(t.is_alive() for t in threads))
        self.assertEqual(max_active, 1)

    def test_news_email_subject_uses_date_only_for_news_and_house_news(self):
        now = datetime(2026, 5, 12, 9, 30, tzinfo=app.get_local_tz())

        # Pin the prefixes: they are env-configurable, and this test is about the
        # date-only suffix, not about whatever the local .env happens to set.
        with mock.patch.object(app, "NEWS_EMAIL_SUBJECT_PREFIX", "[JAT News]"), mock.patch.object(
            app, "HOUSE_NEWS_EMAIL_SUBJECT_PREFIX", "[HOUSE News]"
        ):
            self.assertEqual(app._build_recent_news_email_subject(now=now), "[JAT News] 2026-05-12")
            self.assertEqual(
                app._build_recent_news_email_subject(now=now, scope="house"),
                "[HOUSE News] 2026-05-12",
            )

    def test_news_source_command_lists_sources(self):
        rows = [(1, "Reuters", "https://example.com/rss", 1)]
        with mock.patch.object(app, "FEATURE_NEWS_ENABLED", True):
            with mock.patch.object(app, "list_news_feeds", return_value=rows):
                replies = app.handle_news_command("/news_source", "")

        self.assertEqual(len(replies), 1)
        self.assertIn("News sources:", replies[0])
        self.assertIn("#1 [enabled] Reuters", replies[0])
        self.assertIn("https://example.com/rss", replies[0])

    def test_news_sources_subcommand_is_still_rejected(self):
        with mock.patch.object(app, "FEATURE_NEWS_ENABLED", True):
            with mock.patch.object(app, "list_news_feeds") as mocked_list:
                replies = app.handle_news_command("/news sources", "")

        mocked_list.assert_not_called()
        self.assertEqual(replies, ["Unknown /news subcommand. Use /news help."])

    def test_news_sources_plural_alias_lists_sources(self):
        rows = [(1, "Reuters", "https://example.com/rss", 1)]
        with mock.patch.object(app, "FEATURE_NEWS_ENABLED", True):
            with mock.patch.object(app, "list_news_feeds", return_value=rows):
                replies = app.handle_news_command("/news_sources", "")

        self.assertTrue(app._is_news_command_text("/news_sources"))
        self.assertIn("News sources:", replies[0])
        self.assertIn("#1 [enabled] Reuters", replies[0])

    def test_news_search_uses_tokens_after_command_parser_refactor(self):
        rows = [(1, "Title", "https://example.com/news", "Reuters", "2026-04-27T00:00:00+08:00", "2026-04-27", 1)]
        with mock.patch.object(app, "FEATURE_NEWS_ENABLED", True):
            with mock.patch.object(app, "fetch_and_store_news") as mocked_fetch:
                with mock.patch.object(app, "search_clusters", return_value=rows) as mocked_search:
                    replies = app.handle_news_command("/news search china ai", "")

        mocked_fetch.assert_called_once()
        mocked_search.assert_called_once_with("china ai", 10)
        self.assertIn("Title", replies[0])

    def test_parse_sinyi_dailynews_html_extracts_news_items(self):
        html = """
        <section>
          <span>2026.05.07 新聞來源：信義房屋</span>
          <a href="/dailynews/newsct/17278">連19年不缺席！信義房屋《遠見》ESG雙料得獎</a>
          <p>連續19年獲《遠見》ESG企業永續大獎肯定，信義房屋今年一舉奪得雙料肯定。</p>
          <span>2026.04.24 新聞來源：信義房屋</span>
          <a href="https://www.sinyinews.com.tw/dailynews/newsct/17211">圓環走入歷史、綠園道即將動工</a>
          <p>台南車站周邊正迎來全面蛻變，長期而言可望成為商圈復甦的關鍵契機。</p>
        </section>
        """

        items = app.parse_sinyi_dailynews_html(html)

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["source"], "信義房屋")
        self.assertEqual(items[0]["published_at"].strftime("%Y%m%d"), "20260507")
        self.assertEqual(items[0]["url"], "https://sinyinews.com.tw/dailynews/newsct/17278")
        self.assertIn("ESG", items[0]["title"])
        self.assertIn("永續大獎", items[0]["summary"])

    def test_parse_rer_nccu_list_html_extracts_news_items(self):
        html = """
        <div>
          <a href="/article/detail/2604307211111">2026.04.30 美洲 〖美國〗抵押貸款利率再度下滑，購屋族重返市場</a>
          <a href="https://rer.nccu.edu.tw/article/detail/2510217111111">2025.10.21 (亞洲)海外新知 〖日本〗日本土地價格連續四年上漲</a>
        </div>
        """

        items = app.parse_rer_nccu_list_html(
            html,
            source_url="https://rer.nccu.edu.tw/article/list/72",
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(items[0]["source"], "政大不動產研究中心 - 美洲")
        self.assertEqual(items[0]["published_at"].strftime("%Y%m%d"), "20260430")
        self.assertEqual(items[0]["summary"], "美洲")
        self.assertEqual(items[0]["url"], "https://rer.nccu.edu.tw/article/detail/2604307211111")
        self.assertIn("抵押貸款利率", items[0]["title"])
        self.assertEqual(items[1]["summary"], "(亞洲)海外新知")

    def test_parse_twhg_news_html_extracts_news_items(self):
        html = """
        <ul>
          <li>
            <a href="re_news_details.php?ojb=60062" target="_blank">
              <div class="wtnews-name">雙重好康 年明遺贈稅免稅額上調</div>
              <div class="wtnews-date">2026-05-03</div>
            </a>
          </li>
        </ul>
        """

        items = app.parse_twhg_news_html(html)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["source"], "台灣房屋新聞")
        self.assertEqual(items[0]["published_at"].strftime("%Y%m%d"), "20260503")
        self.assertEqual(items[0]["url"], "https://news.twhg.com.tw/re_news_details.php?ojb=60062")
        self.assertIn("遺贈稅", items[0]["title"])

    def test_sync_dropbox_notes_range_to_local_downloads_missing_remote_md(self):
        class FakeEntry:
            def __init__(self, path_lower, name, rev="r1", content_hash="h1"):
                self.path_lower = path_lower
                self.name = name
                self.rev = rev
                self.content_hash = content_hash
                self.server_modified = None
                self.size = 12

        remote_root = "/root/notes"
        entry = FakeEntry(f"{remote_root}/telegram/2026-03-09_note.md", "2026-03-09_note.md")
        tmp_notes = Path("tests_runtime_notes")
        try:
            if tmp_notes.exists():
                for fp in tmp_notes.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(tmp_notes.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()
            tmp_notes.mkdir(exist_ok=True)
            with mock.patch.object(app, "DROPBOX_SYNC_ENABLED", True):
                with mock.patch.object(app, "DROPBOX_ROOT_PATH", "/root"):
                    with mock.patch.object(app, "NOTES_DIR", tmp_notes):
                        with mock.patch.object(app, "_dropbox_list_folder_entries_recursive", return_value=[entry]):
                            with mock.patch.object(app, "_dropbox_download_file_bytes", return_value=b"# Title\n\nRemote line\n"):
                                with mock.patch.object(app, "get_sync_state", return_value=None):
                                    with mock.patch.object(app, "upsert_sync_state") as mocked_upsert:
                                        stats = app.sync_dropbox_notes_range_to_local("2026-03-03", "2026-03-09")

            local_file = tmp_notes / "telegram" / "2026-03-09_note.md"
            self.assertTrue(local_file.exists())
            self.assertIn("Remote line", local_file.read_text(encoding="utf-8"))
            self.assertEqual(stats["notes_remote_downloaded"], 1)
            mocked_upsert.assert_called_once()
        finally:
            if tmp_notes.exists():
                for fp in tmp_notes.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(tmp_notes.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()

    def test_sync_dropbox_news_to_local_downloads_missing_remote_md(self):
        class FakeEntry:
            def __init__(self, path_lower, name, rev="r1", content_hash="h1"):
                self.path_lower = path_lower
                self.path_display = path_lower
                self.name = name
                self.rev = rev
                self.content_hash = content_hash
                self.server_modified = None
                self.size = 12

        remote_root = "/root/news"
        entry = FakeEntry(f"{remote_root}/2026-03-09_news.md", "2026-03-09_news.md")
        tmp_news = Path("tests_runtime_news")
        try:
            if tmp_news.exists():
                for fp in tmp_news.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(tmp_news.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()
            tmp_news.mkdir(exist_ok=True)
            with mock.patch.object(app, "FEATURE_NEWS_ENABLED", True):
                with mock.patch.object(app, "DROPBOX_SYNC_ENABLED", True):
                    with mock.patch.object(app, "DROPBOX_ROOT_PATH", "/root"):
                        with mock.patch.object(app, "NEWS_MD_DIR", tmp_news):
                            with mock.patch.object(app, "_dropbox_list_folder_entries_recursive", return_value=[entry]):
                                with mock.patch.object(app, "_dropbox_download_file_bytes", return_value=b"# News\n\nRemote line\n"):
                                    with mock.patch.object(app, "get_sync_state", return_value=None):
                                        with mock.patch.object(app, "upsert_sync_state") as mocked_upsert:
                                            stats = app.sync_dropbox_news_to_local(full_scan=True)

            local_file = tmp_news / "2026-03-09_news.md"
            self.assertTrue(local_file.exists())
            self.assertIn("Remote line", local_file.read_text(encoding="utf-8"))
            self.assertEqual(stats["news_remote_downloaded"], 1)
            mocked_upsert.assert_called_once()
        finally:
            if tmp_news.exists():
                for fp in tmp_news.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(tmp_news.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()

    def test_sync_dropbox_news_to_local_merges_remote_and_local_without_duplicate_blocks(self):
        class FakeEntry:
            def __init__(self, path_lower, name, rev="r1", content_hash="h1"):
                self.path_lower = path_lower
                self.path_display = path_lower
                self.name = name
                self.rev = rev
                self.content_hash = content_hash
                self.server_modified = None
                self.size = 12

        remote_root = "/root/news"
        entry = FakeEntry(f"{remote_root}/2026-03-09_news.md", "2026-03-09_news.md")
        tmp_news = Path("tests_runtime_news_merge")
        local_file = tmp_news / "2026-03-09_news.md"
        remote_text = "# 2026-03-09 News\n\n## Item A\n\nRemote only\n\n## Item Shared\n\nSame block\n"
        local_text = "# 2026-03-09 News\n\n## Item Shared\n\nSame block\n\n## Item B\n\nLocal only\n"
        try:
            if tmp_news.exists():
                for fp in tmp_news.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(tmp_news.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()
            tmp_news.mkdir(exist_ok=True)
            local_file.write_text(local_text, encoding="utf-8")

            with mock.patch.object(app, "FEATURE_NEWS_ENABLED", True):
                with mock.patch.object(app, "DROPBOX_SYNC_ENABLED", True):
                    with mock.patch.object(app, "DROPBOX_ROOT_PATH", "/root"):
                        with mock.patch.object(app, "NEWS_MD_DIR", tmp_news):
                            with mock.patch.object(app, "_dropbox_list_folder_entries_recursive", return_value=[entry]):
                                with mock.patch.object(app, "_dropbox_download_file_bytes", return_value=remote_text.encode("utf-8")):
                                    with mock.patch.object(app, "get_sync_state", return_value=None):
                                        with mock.patch.object(app, "upsert_sync_state"):
                                            stats = app.sync_dropbox_news_to_local(full_scan=True)

            merged = local_file.read_text(encoding="utf-8")
            self.assertIn("## Item A", merged)
            self.assertIn("## Item B", merged)
            self.assertEqual(merged.count("## Item Shared"), 1)
            self.assertEqual(stats["news_remote_downloaded"], 1)
        finally:
            if tmp_news.exists():
                for fp in tmp_news.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(tmp_news.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()

    def test_transcribe_audio_splits_long_audio_into_chunks(self):
        temp_dir = Path("tests_runtime_transcribe")
        temp_dir.mkdir(exist_ok=True)
        audio_path = temp_dir / "sample.wav"
        chunk1 = temp_dir / "sample_chunk0.wav"
        chunk2 = temp_dir / "sample_chunk1.wav"
        audio_path.write_bytes(b"x" * 2048)
        chunk1.write_bytes(b"a" * 2048)
        chunk2.write_bytes(b"b" * 2048)

        class _Seg:
            def __init__(self, start, end, text):
                self.start = start
                self.end = end
                self.text = text

        class _Model:
            def __init__(self):
                self.calls = []

            def transcribe(self, path, **kwargs):
                self.calls.append(Path(path).name)
                return iter([_Seg(0.0, 10.0, f"text-{Path(path).name}")]), {}

        model = _Model()
        statuses = []
        duration_map = {
            audio_path.name: 3600.0,
            chunk1.name: 1500.0,
            chunk2.name: 1200.0,
        }

        try:
            with mock.patch.object(transcription, "get_audio_duration", side_effect=lambda p: duration_map[Path(p).name]):
                with mock.patch.object(transcription, "get_model", return_value=model):
                    with mock.patch.object(transcription, "split_audio", return_value=[chunk1, chunk2]) as split_mock:
                        with mock.patch.object(transcription, "WhisperModel", object()):
                            with mock.patch.object(transcription, "BatchedInferencePipeline", None):
                                with mock.patch.object(transcription, "_compute_audio_fingerprint", return_value="fp"):
                                    with mock.patch.object(transcription, "_load_checkpoint", return_value=None):
                                        with mock.patch.object(transcription, "_delete_checkpoint"):
                                            text = transcription.transcribe_audio(
                                                audio_path,
                                                temp_dir,
                                                "job1",
                                                on_status=statuses.append,
                                            )
            self.assertEqual(text, f"text-{chunk1.name} text-{chunk2.name}")
            self.assertEqual(model.calls, [chunk1.name, chunk2.name])
            split_mock.assert_called_once()
            self.assertIn("Transcribing segment 1/2...", statuses)
            self.assertIn("Transcribing segment 2/2...", statuses)
        finally:
            if temp_dir.exists():
                for fp in temp_dir.rglob("*"):
                    if fp.is_file():
                        fp.unlink()
                for fp in sorted(temp_dir.rglob("*"), reverse=True):
                    if fp.is_dir():
                        fp.rmdir()

    def test_safe_filename_preserves_readable_title(self):
        title = "《財報狗 - 掌握台股美股時事議題 - 513.【財經時事放大鏡】光油並進》"
        self.assertEqual(
            transcription._safe_filename(title),
            "《財報狗 - 掌握台股美股時事議題 - 513.【財經時事放大鏡】光油並進》.md",
        )

    def test_safe_filename_removes_windows_invalid_chars(self):
        title = 'Podcast: Q&A / Earnings <Recap>?*'
        self.assertEqual(
            transcription._safe_filename(title),
            "Podcast Q&A Earnings Recap.md",
        )


if __name__ == "__main__":
    unittest.main()
