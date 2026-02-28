from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _split_blocks(text: str) -> list[str]:
    chunks = re.split(r"(?:\r?\n){2,}", text.replace("\r\n", "\n").strip())
    return [c.strip() for c in chunks if c and c.strip()]


def _normalize_line(line: str) -> str:
    m = re.match(r"^- \[(\d{2}:\d{2}:\d{2})\] \([^)]*\) .*?:\s*(.*)$", line)
    if m:
        return f"- [{m.group(1)}] {m.group(2)}"
    return line


def clean_markdown_text(raw: str) -> str:
    text = (raw or "").replace("\r\n", "\n")
    lines = [_normalize_line(x) for x in text.split("\n")]
    text = "\n".join(lines)

    blocks = _split_blocks(text)
    heading = ""
    if blocks and blocks[0].startswith("# "):
        heading = blocks.pop(0).strip()

    out: list[str] = []
    seen: set[str] = set()
    for b in blocks:
        if heading and b.strip() == heading:
            continue
        key = hashlib.sha1(b.encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(b)

    parts: list[str] = []
    if heading:
        parts.append(heading)
    if out:
        if parts:
            parts.append("")
        parts.append("\n\n".join(out).strip())
    merged = "\n".join(parts).strip()
    return (merged + "\n") if merged else ""


def _upload_text(app_mod, remote_path: str, text: str) -> None:
    data = text.encode("utf-8")

    def _do(dbx):
        dbx.files_upload(
            data,
            app_mod.normalize_dropbox_path(remote_path),
            mode=app_mod.dropbox.files.WriteMode.overwrite,
            mute=True,
        )

    app_mod._dropbox_call_with_retry(_do)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cleanup markdown notes both in Dropbox and local notes folder.",
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        help="dotenv file path to load before importing app module (default: .env)",
    )
    parser.add_argument(
        "--remote-root",
        default="",
        help="override Dropbox root path (example: /read & chat/chitchat)",
    )
    parser.add_argument(
        "--local-notes",
        default="",
        help="override local notes directory path (example: chitchat/notes)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env_file = Path(args.env_file)
    if env_file.exists():
        load_dotenv(dotenv_path=env_file, override=True)
    elif args.env_file != ".env":
        raise FileNotFoundError(f"env file not found: {env_file}")

    import app as app_mod

    root_base = args.remote_root.strip() or app_mod.DROPBOX_ROOT_PATH
    root = app_mod.normalize_dropbox_path(root_base).rstrip("/")
    remote_notes = f"{root}/notes"
    entries = app_mod._dropbox_list_folder_entries_recursive(remote_notes)
    local_notes_dir = Path(args.local_notes).resolve() if args.local_notes else Path(app_mod.NOTES_DIR)

    scanned = 0
    changed_remote = 0
    changed_local = 0

    for e in entries:
        if not isinstance(e, app_mod.dropbox.files.FileMetadata):
            continue
        rp = getattr(e, "path_display", "") or getattr(e, "path_lower", "")
        if not rp.lower().endswith(".md"):
            continue

        scanned += 1
        b = app_mod._dropbox_download_file_bytes(rp)
        if b is None:
            continue
        old = b.decode("utf-8", errors="replace")
        new = clean_markdown_text(old)
        if new != old:
            _upload_text(app_mod, rp, new)
            changed_remote += 1

        rel = app_mod._safe_transcript_relpath(remote_notes, rp)
        local = local_notes_dir / rel
        if local.exists():
            local_old = local.read_text(encoding="utf-8", errors="replace")
            local_new = clean_markdown_text(local_old)
            if local_new != local_old:
                local.write_text(local_new, encoding="utf-8")
                changed_local += 1

    print(
        {
            "env_file": str(env_file),
            "remote_notes": remote_notes,
            "local_notes": str(local_notes_dir),
            "scanned": scanned,
            "changed_remote": changed_remote,
            "changed_local": changed_local,
        }
    )


if __name__ == "__main__":
    main()
