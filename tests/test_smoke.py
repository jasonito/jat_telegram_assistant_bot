import importlib
import os
import sys
import types
import unittest
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
