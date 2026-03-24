import importlib
import os
import shutil
import sys
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

    if "slack_bolt" not in sys.modules:
        slack_bolt_mod = types.ModuleType("slack_bolt")

        class App:
            def __init__(self, *args, **kwargs):
                pass

            def command(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator

            def event(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator

            def message(self, *args, **kwargs):
                def decorator(func):
                    return func
                return decorator

        slack_bolt_mod.App = App
        sys.modules["slack_bolt"] = slack_bolt_mod

    if "slack_bolt.adapter.socket_mode" not in sys.modules:
        socket_mode_mod = types.ModuleType("slack_bolt.adapter.socket_mode")

        class SocketModeHandler:
            def __init__(self, *args, **kwargs):
                pass

        socket_mode_mod.SocketModeHandler = SocketModeHandler
        sys.modules["slack_bolt.adapter.socket_mode"] = socket_mode_mod


_install_test_stubs()

app = importlib.import_module("app")
transcription = importlib.import_module("transcription")


class SmokeTests(unittest.TestCase):
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

        def _record_build_summary(path):
            calls.append("build_summary")
            return "summary text"

        def _record_prepend(path, summary):
            calls.append("prepend_summary")

        def _record_append(chat_id, title, source, transcript_path, message_ts):
            calls.append("append_daily_note")

        def _record_sync(path):
            calls.append("sync_transcript")

        async def _run():
            with mock.patch.object(app, "notion_append_chitchat_transcript", side_effect=_record_notion), mock.patch.object(
                app, "_build_transcript_ai_summary", side_effect=_record_build_summary
            ), mock.patch.object(
                app, "_prepend_summary_to_transcript", side_effect=_record_prepend
            ), mock.patch.object(
                app, "_append_transcript_to_telegram_markdown", side_effect=_record_append
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
            ["notion", "build_summary", "prepend_summary", "append_daily_note", "sync_transcript"],
        )

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

    def test_digest_watchlist_command_requires_authorization_for_mutation(self):
        with mock.patch.object(app, "ALLOWED_CONTROL_USERS", set()):
            msg = app.handle_command(
                "/add_kol https://x.com/kol1 KOL One",
                user_id="123",
                user_name="alice",
            )
        self.assertIn("未授權", msg)

    def test_digest_watchlist_list_and_add_route_through_helpers(self):
        fake_item = types.SimpleNamespace(
            kol_id="x-kol1",
            enabled=True,
            platform="x",
            display_name="KOL One",
            handle_or_url="https://x.com/kol1",
        )
        with mock.patch.object(app, "KOL_WATCHLIST_PATH", Path("data/kol_watchlist.json")):
            with mock.patch.object(app, "list_watchlist_entries", return_value=[fake_item]):
                listed = app.handle_command("/list_kol")
                self.assertIn("x-kol1", listed)
            with mock.patch.object(app, "ALLOWED_CONTROL_USERS", {"123"}):
                with mock.patch.object(app, "add_watchlist_entry", return_value=(fake_item, True)):
                    added = app.handle_command(
                        "/add_kol https://x.com/kol1 KOL One",
                        user_id="123",
                        user_name="alice",
                    )
        self.assertIn("added: x-kol1", added)

    def test_short_kol_commands_route_to_watchlist_handler(self):
        with mock.patch.object(app, "_handle_digest_watchlist_slash_command", return_value="ok") as mocked:
            result = app.handle_command("add kol https://x.com/kol1 KOL One", user_id="123", user_name="alice")
        self.assertEqual("ok", result)
        mocked.assert_called_once()

    def test_add_kol_infers_x_from_handle(self):
        fake_item = types.SimpleNamespace(
            kol_id="x-elonmusk",
            enabled=True,
            platform="x",
            display_name="Elon Musk",
            handle_or_url="@elonmusk",
        )
        with mock.patch.object(app, "ALLOWED_CONTROL_USERS", {"123"}):
            with mock.patch.object(app, "add_watchlist_entry", return_value=(fake_item, True)) as mocked_add:
                msg = app.handle_command("/add_kol @elonmusk Elon Musk", user_id="123", user_name="alice")
        self.assertIn("added: x-elonmusk", msg)
        self.assertEqual("x", mocked_add.call_args.kwargs["platform"])

    def test_kol_fetch_schedule_aligns_with_8am_digest(self):
        tz = timezone(timedelta(hours=8))
        aligned = datetime(2026, 3, 10, 8, 0, tzinfo=tz)
        self.assertEqual(
            app._kol_fetch_run_key(aligned, digest_hour=8, digest_minute=0, interval_hours=6),
            "2026-03-10:08:00",
        )
        self.assertEqual(
            app._kol_fetch_run_key(aligned.replace(hour=2), digest_hour=8, digest_minute=0, interval_hours=6),
            "2026-03-10:02:00",
        )
        self.assertEqual(
            app._kol_fetch_run_key(aligned.replace(hour=14), digest_hour=8, digest_minute=0, interval_hours=6),
            "2026-03-10:14:00",
        )
        self.assertIsNone(
            app._kol_fetch_run_key(aligned.replace(hour=6), digest_hour=8, digest_minute=0, interval_hours=6)
        )

    def test_kol_digest_run_key_matches_only_exact_time(self):
        tz = timezone(timedelta(hours=8))
        exact = datetime(2026, 3, 10, 8, 0, tzinfo=tz)
        self.assertEqual(
            app._kol_digest_run_key(exact, digest_hour=8, digest_minute=0),
            "2026-03-10:08:00",
        )
        self.assertIsNone(app._kol_digest_run_key(exact.replace(minute=1), digest_hour=8, digest_minute=0))
        self.assertIsNone(app._kol_digest_run_key(exact.replace(hour=7), digest_hour=8, digest_minute=0))

    def test_kol_digest_day_string_supports_yesterday(self):
        tz = timezone(timedelta(hours=8))
        now = datetime(2026, 3, 10, 0, 5, tzinfo=tz)
        self.assertEqual(app._kol_digest_day_string(0, now=now), "2026-03-10")
        self.assertEqual(app._kol_digest_day_string(-1, now=now), "2026-03-09")

    def test_kol_today_generates_digest_if_missing(self):
        fake_path = Path("digest/20260310_kol_digest.md")
        with mock.patch.object(app, "KOL_DIGEST_OUTPUT_DIR", Path("digest")):
            with mock.patch.object(Path, "exists", return_value=False):
                with mock.patch.object(app, "generate_kol_digest_for_day", return_value=fake_path) as mocked_generate:
                    with mock.patch.object(app, "_read_kol_digest_preview", return_value="# KOL Daily Digest"):
                        result = app.handle_command("/kol_today")
        self.assertIn("KOL Daily Digest", result)
        mocked_generate.assert_called_once()

    def test_kol_yesterday_generates_previous_day_digest_if_missing(self):
        fake_path = Path("digest/20260309_kol_digest.md")
        with mock.patch.object(app, "KOL_DIGEST_OUTPUT_DIR", Path("digest")):
            with mock.patch.object(app, "_kol_digest_day_string", return_value="2026-03-09"):
                with mock.patch.object(Path, "exists", return_value=False):
                    with mock.patch.object(app, "generate_kol_digest_for_day", return_value=fake_path) as mocked_generate:
                        with mock.patch.object(app, "_read_kol_digest_preview", return_value="# KOL Daily Digest - 2026-03-09"):
                            result = app.handle_command("/kol_yesterday")
        self.assertIn("2026-03-09", result)
        mocked_generate.assert_called_once_with("2026-03-09")

    def test_kol_now_requires_authorization(self):
        with mock.patch.object(app, "ALLOWED_CONTROL_USERS", set()):
            result = app.handle_command("/kol_now", user_id="123", user_name="alice")
        self.assertIn("未授權", result)

    def test_kol_now_runs_fetch_and_generate_digest(self):
        fake_path = Path("digest/20260310_kol_digest.md")
        with mock.patch.object(app, "ALLOWED_CONTROL_USERS", {"123"}):
            with mock.patch.object(app, "run_kol_fetch_cycle", return_value={"x-kol1": 2}) as mocked_fetch:
                with mock.patch.object(app, "generate_kol_digest_for_day", return_value=fake_path) as mocked_generate:
                    with mock.patch.object(app, "_read_kol_digest_preview", return_value="# KOL Daily Digest"):
                        result = app.handle_command("/kol_now", user_id="123", user_name="alice")
        self.assertIn("KOL fetch done: 1 sources", result)
        self.assertIn("x-kol1: 2 new posts fetched", result)
        mocked_fetch.assert_called_once()
        mocked_generate.assert_called_once()

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

        with mock.patch.object(app, "store_message"):
            with mock.patch.object(app, "append_markdown"):
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

        mocked_bg.assert_called_once_with(
            app.notion_append_chitchat_text,
            "https://youtu.be/hXC7vtZCV_4?si=Lr6qGsGFs9v0ctoq",
            mock.ANY,
            label="notion text append",
        )
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

    def test_build_scoped_summary_syncs_dropbox_before_note_summary(self):
        with mock.patch.object(app, "_sync_dropbox_before_weekly_report") as mocked_presync:
            with mock.patch.object(app, "build_note_digest_recent", return_value=["weekly"]) as mocked_digest:
                result = app.build_scoped_summary("2026-03-09", "note", recent_days=7)

        self.assertEqual(result, ["weekly"])
        mocked_presync.assert_called_once_with("2026-03-09", 7)
        mocked_digest.assert_called_once_with("2026-03-09", days=7)

    def test_build_scoped_summary_skips_dropbox_sync_when_disabled(self):
        with mock.patch.object(app, "_sync_dropbox_before_weekly_report") as mocked_presync:
            with mock.patch.object(app, "build_note_digest", return_value=["daily"]) as mocked_digest:
                result = app.build_scoped_summary("2026-03-09", "note")

        self.assertEqual(result, ["daily"])
        mocked_presync.assert_called_once_with("2026-03-09", 1)
        mocked_digest.assert_called_once_with("2026-03-09")

    def test_weekly_commands_return_disabled_message_when_feature_flag_off(self):
        with mock.patch.object(app, "FEATURE_WEEKLY_REPORT_ENABLED", False):
            self.assertEqual(app.handle_command("/summary_notes_weekly"), "週報功能目前已關閉。")
            self.assertEqual(app.handle_command("/summary_news_weekly"), "週報功能目前已關閉。")

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

    def test_presync_weekly_report_syncs_requested_date_range(self):
        with mock.patch.object(app, "DROPBOX_SYNC_ENABLED", True):
            with mock.patch.object(app, "run_dropbox_sync", return_value={}) as mocked_sync:
                with mock.patch.object(app, "sync_dropbox_notes_range_to_local", return_value={
                    "notes_remote_scanned": 2,
                    "notes_remote_downloaded": 1,
                    "notes_remote_skipped": 1,
                    "notes_remote_failed": 0,
                }) as mocked_range:
                    app._sync_dropbox_before_weekly_report("2026-03-09", 7)

        mocked_sync.assert_called_once_with(full_scan=False)
        mocked_range.assert_called_once_with("2026-03-03", "2026-03-09")

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


if __name__ == "__main__":
    unittest.main()
