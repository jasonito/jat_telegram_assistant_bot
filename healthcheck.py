import base64
import os
import sys
from pathlib import Path

from dotenv import load_dotenv


def mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def check_google_vision() -> tuple[bool, str]:
    try:
        from google.cloud import vision
    except Exception as e:
        return False, f"google-cloud-vision import failed: {e}"

    creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if creds and not os.path.exists(creds):
        return False, f"GOOGLE_APPLICATION_CREDENTIALS path not found: {creds}"

    try:
        client = vision.ImageAnnotatorClient()
        sample_path = None
        image_root = Path("data") / "inbox" / "images"
        if image_root.exists():
            for fp in image_root.rglob("*"):
                if fp.is_file() and fp.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    sample_path = fp
                    break

        if sample_path:
            image_bytes = sample_path.read_bytes()
            sample_desc = f"sample={sample_path}"
        else:
            # Fallback tiny PNG if no local sample exists.
            image_bytes = base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7+fKQAAAAASUVORK5CYII="
            )
            sample_desc = "sample=embedded_1x1_png"

        image = vision.Image(content=image_bytes)
        response = client.text_detection(image=image, timeout=20)
        if response.error and response.error.message:
            return False, f"Google Vision API error: {response.error.message}"
        return True, f"Google Vision API reachable ({sample_desc})"
    except Exception as e:
        return False, f"Google Vision check failed: {e}"


def check_dropbox() -> tuple[bool, str]:
    try:
        import dropbox
    except Exception as e:
        return False, f"dropbox SDK import failed: {e}"

    token = os.getenv("DROPBOX_ACCESS_TOKEN", "").strip()
    if not token:
        return False, "DROPBOX_ACCESS_TOKEN not set"

    try:
        dbx = dropbox.Dropbox(token, timeout=20)
        account = dbx.users_get_current_account()
        name = getattr(account.name, "display_name", "") if account else ""
        return True, f"Dropbox API reachable (account: {name or 'ok'}, token: {mask_secret(token)})"
    except Exception as e:
        return False, f"Dropbox check failed: {e}"


def main() -> int:
    load_dotenv()
    checks = [
        ("Google Vision OCR", check_google_vision),
        ("Dropbox", check_dropbox),
    ]

    failed = 0
    print("== Health Check ==")
    for label, fn in checks:
        ok, msg = fn()
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {label}: {msg}")
        if not ok:
            failed += 1

    if failed:
        print(f"Health check failed: {failed} check(s) failed.")
        return 1
    print("Health check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
