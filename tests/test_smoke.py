import importlib
import os
import unittest
from pathlib import Path
from unittest import mock


os.environ.setdefault("TELEGRAM_BOT_TOKEN", "test-token")

app = importlib.import_module("app")


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


if __name__ == "__main__":
    unittest.main()
