"""Token-addressed image handoff for phone downloads.

Kept transport-agnostic: the store holds the tokens and policy, and the HTTP server is a thin
adapter over it, so the kiosk web UI can later serve the same store from its own process.

URLs never map to filesystem paths. A request carries an opaque token that must already exist in
the store, which removes path traversal as a category rather than trying to sanitize it.
"""

from __future__ import annotations

import html
import secrets
import socket
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import time
from typing import Callable

DEFAULT_TTL_SECONDS = 15 * 60
DEFAULT_MAX_DOWNLOADS = 20
TOKEN_BYTES = 16


@dataclass
class HandoffEntry:
    path: Path
    caption: str | None
    created_at: float
    expires_at: float
    max_downloads: int
    downloads: int = 0


@dataclass
class ImageHandoffStore:
    """Maps unguessable tokens to files, with expiry and a download cap."""

    ttl_seconds: float = DEFAULT_TTL_SECONDS
    max_downloads: int = DEFAULT_MAX_DOWNLOADS
    delete_file_on_expiry: bool = False
    _entries: dict[str, HandoffEntry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, image_path: Path, caption: str | None = None) -> str:
        image_path = image_path.expanduser().resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Cannot hand off missing file: {image_path}")

        token = secrets.token_urlsafe(TOKEN_BYTES)
        now = time()
        with self._lock:
            self._entries[token] = HandoffEntry(
                path=image_path,
                caption=caption,
                created_at=now,
                expires_at=now + self.ttl_seconds,
                max_downloads=self.max_downloads,
            )
        return token

    def resolve(self, token: str, count_download: bool = True) -> HandoffEntry | None:
        """Return the entry for a token, or None if unknown, expired or exhausted."""
        now = time()
        with self._lock:
            entry = self._entries.get(token)
            if entry is None:
                return None
            if now >= entry.expires_at or entry.downloads >= entry.max_downloads:
                self._drop(token, entry)
                return None
            if count_download:
                entry.downloads += 1
            return entry

    def revoke(self, token: str) -> None:
        with self._lock:
            entry = self._entries.pop(token, None)
        if entry is not None and self.delete_file_on_expiry:
            entry.path.unlink(missing_ok=True)

    def purge_expired(self) -> int:
        now = time()
        with self._lock:
            stale = [t for t, e in self._entries.items() if now >= e.expires_at]
            for token in stale:
                self._drop(token, self._entries[token])
        return len(stale)

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _drop(self, token: str, entry: HandoffEntry) -> None:
        """Caller must hold the lock."""
        self._entries.pop(token, None)
        if self.delete_file_on_expiry:
            entry.path.unlink(missing_ok=True)


PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Your vPRO photo</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; padding:24px; font-family:system-ui,-apple-system,Segoe UI,sans-serif;
         background:#0b0e14; color:#e6e6e6; text-align:center; }}
  img {{ max-width:100%; height:auto; border-radius:12px; margin:16px 0; }}
  .hint {{ opacity:.75; font-size:15px; line-height:1.5; }}
  .caption {{ background:#161a23; border-radius:10px; padding:12px; margin:16px 0;
              text-align:left; font-size:15px; white-space:pre-wrap; word-break:break-word; }}
  button, a.button {{ display:inline-block; background:#2f6fed; color:#fff; border:0;
              border-radius:10px; padding:14px 22px; font-size:16px; text-decoration:none;
              margin:6px 4px; }}
</style>
</head>
<body>
  <h2>Your photo is ready</h2>
  <img src="{file_url}" alt="Your generated photo">
  <p class="hint">Press and hold the image, then choose <b>Add to Photos</b> to save it.</p>
  <a class="button" href="{file_url}" download>Download</a>
  {caption_block}
  <p class="hint">This link expires in about {minutes} minutes.</p>
</body>
</html>
"""

CAPTION_BLOCK = """
  <div class="caption" id="caption">{caption}</div>
  <button onclick="navigator.clipboard.writeText(document.getElementById('caption').innerText)">
    Copy caption
  </button>
"""


def _build_handler(store: ImageHandoffStore) -> type[BaseHTTPRequestHandler]:
    class HandoffHandler(BaseHTTPRequestHandler):
        server_version = "vPROHandoff/1.0"
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0].rstrip("/")
            if path in ("", "/health"):
                self._send_bytes(b"ok", "text/plain; charset=utf-8")
                return

            kind, _, token = path.lstrip("/").partition("/")
            if kind not in ("i", "f") or not token:
                self._send_error_page(404, "Not found")
                return

            # Viewing the page should not consume a download; fetching the file does.
            entry = store.resolve(token, count_download=(kind == "f"))
            if entry is None:
                self._send_error_page(404, "This link has expired.")
                return

            if kind == "i":
                self._send_bytes(self._render_page(token, entry), "text/html; charset=utf-8")
            else:
                self._send_file(entry)

        def _render_page(self, token: str, entry: HandoffEntry) -> bytes:
            caption_block = ""
            if entry.caption:
                caption_block = CAPTION_BLOCK.format(caption=html.escape(entry.caption))
            page = PAGE_TEMPLATE.format(
                file_url=f"/f/{token}",
                caption_block=caption_block,
                minutes=max(1, int((entry.expires_at - time()) // 60)),
            )
            return page.encode("utf-8")

        def _send_file(self, entry: HandoffEntry) -> None:
            try:
                payload = entry.path.read_bytes()
            except OSError:
                self._send_error_page(410, "The image is no longer available.")
                return
            suffix = entry.path.suffix.lower()
            content_type = "image/png" if suffix == ".png" else "image/jpeg"
            self._send_bytes(
                payload,
                content_type,
                extra={"Content-Disposition": f'attachment; filename="vpro-photo{suffix}"'},
            )

        def _send_error_page(self, code: int, message: str) -> None:
            body = (
                f"<!doctype html><meta charset=utf-8><meta name=viewport "
                f"content='width=device-width,initial-scale=1'>"
                f"<body style='font-family:system-ui;background:#0b0e14;color:#e6e6e6;"
                f"text-align:center;padding:48px'><h3>{html.escape(message)}</h3>"
                f"<p>Please ask a staff member for help.</p></body>"
            ).encode("utf-8")
            self._send_bytes(body, "text/html; charset=utf-8", status=code)

        def _send_bytes(
            self,
            payload: bytes,
            content_type: str,
            status: int = 200,
            extra: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; img-src 'self'; style-src 'unsafe-inline'; "
                "script-src 'unsafe-inline'",
            )
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args: object) -> None:
            if self.server.log_requests:  # type: ignore[attr-defined]
                super().log_message(fmt, *args)

    return HandoffHandler


class HandoffServer:
    """Serves the handoff store over HTTP on the local network."""

    def __init__(
        self,
        store: ImageHandoffStore,
        host: str = "0.0.0.0",
        port: int = 8765,
        advertise_host: str | None = None,
        log_requests: bool = False,
    ) -> None:
        self.store = store
        self._httpd = ThreadingHTTPServer((host, port), _build_handler(store))
        self._httpd.log_requests = log_requests  # type: ignore[attr-defined]
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None
        self.port = self._httpd.server_address[1]
        self.advertise_host = advertise_host or detect_lan_ip()

    @property
    def base_url(self) -> str:
        return f"http://{self.advertise_host}:{self.port}"

    def start(self) -> HandoffServer:
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._httpd.serve_forever, name="vpro-handoff", daemon=True
            )
            self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def __enter__(self) -> HandoffServer:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


def detect_lan_ip(fallback: str = "127.0.0.1") -> str:
    """Best-effort outbound interface address; no packets are sent."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("192.0.2.1", 9))  # TEST-NET-1, guaranteed unroutable
        return probe.getsockname()[0]
    except OSError:
        try:
            return socket.gethostbyname(socket.gethostname())
        except OSError:
            return fallback
    finally:
        probe.close()


ProgressCallback = Callable[[str], None]
