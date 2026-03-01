"""Tiny local webhook receiver for file-created events."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
ENDPOINT = "/webhook/file-created"
LOG_FILE = Path("webhook-events.jsonl")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WebhookHandler(BaseHTTPRequestHandler):
    server_version = "WebhookReceiver/0.1"

    def log_message(self, fmt: str, *args) -> None:
        return

    def send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        if self.path != ENDPOINT:
            self.send_json(404, {"ok": False, "error": "not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(length) if length else b""

        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError:
            self.send_json(400, {"ok": False, "error": "invalid json"})
            return

        required = ("event_id", "filename", "created_at")
        if any(field not in payload for field in required):
            self.send_json(400, {"ok": False, "error": "missing required fields"})
            return

        with LOG_FILE.open("a", encoding="utf-8") as f:
            record = {"received_at": utc_now_iso(), "event": payload}
            f.write(json.dumps(record) + "\n")

        print(f"[receiver] got event for {payload['filename']}")
        self.send_json(200, {"ok": True})


def main() -> None:
    server = HTTPServer((HOST, PORT), WebhookHandler)
    print(f"[receiver] listening on http://{HOST}:{PORT}{ENDPOINT}")
    print(f"[receiver] writing events to {LOG_FILE.resolve()}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[receiver] stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
