"""Watch a local folder and send webhook events for new files."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

FOLDER = Path("incoming")
WEBHOOK_URL = "http://127.0.0.1:8000/webhook/file-created"
POLL_SECONDS = 1.0
TIMEOUT_SECONDS = 5.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def list_files(folder: Path) -> set[str]:
    return {p.name for p in folder.iterdir() if p.is_file()}


def send_event(filename: str) -> int:
    payload = {
        "event_id": str(uuid.uuid4()),
        "filename": filename,
        "created_at": utc_now_iso(),
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=WEBHOOK_URL,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
        return response.status


def main() -> None:
    FOLDER.mkdir(parents=True, exist_ok=True)
    seen = list_files(FOLDER)

    print(f"[watcher] watching {FOLDER.resolve()} every {POLL_SECONDS:.1f}s")
    print(f"[watcher] posting to {WEBHOOK_URL}")
    print("[watcher] create a new file in ./incoming to trigger an event")

    try:
        while True:
            current = list_files(FOLDER)
            new_files = sorted(current - seen)

            for filename in new_files:
                try:
                    status = send_event(filename)
                    if 200 <= status < 300:
                        print(f"[watcher] sent event for {filename}")
                        seen.add(filename)
                    else:
                        print(f"[watcher] server returned status={status} for {filename}")
                except error.URLError as exc:
                    print(f"[watcher] failed to send {filename}: {exc}")

            seen.intersection_update(current)
            time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        print("\n[watcher] stopping")


if __name__ == "__main__":
    main()
