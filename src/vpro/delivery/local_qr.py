"""QR-code handoff over the local network.

The QR encodes a URL, not the image: QR tops out near 2.9 KB of binary payload while a delivered
JPEG is 0.5-2 MB.

Requires the guest's phone to reach the kiosk. Venue guest WiFi often enables client isolation,
which blocks that; test the actual network before relying on this channel.
"""

from __future__ import annotations

from pathlib import Path

from .base import DeliveryRequest, DeliveryResult
from .handoff import HandoffServer, ImageHandoffStore


class LocalQrDelivery:
    name = "local-qr"

    def __init__(
        self,
        store: ImageHandoffStore | None = None,
        server: HandoffServer | None = None,
        host: str = "0.0.0.0",
        port: int = 8765,
        advertise_host: str | None = None,
    ) -> None:
        self.store = store or (server.store if server is not None else ImageHandoffStore())
        self._owns_server = server is None
        self.server = server or HandoffServer(
            self.store, host=host, port=port, advertise_host=advertise_host
        )
        self.server.start()

    def deliver(self, request: DeliveryRequest) -> DeliveryResult:
        try:
            token = self.store.add(Path(request.image_path), caption=request.caption)
        except OSError as exc:
            return DeliveryResult(ok=False, channel=self.name, error=f"{type(exc).__name__}: {exc}")

        url = f"{self.server.base_url}/i/{token}"
        return DeliveryResult(
            ok=True,
            channel=self.name,
            url=url,
            qr_svg=render_qr_svg(url),
            expires_in_seconds=self.store.ttl_seconds,
        )

    def close(self) -> None:
        if self._owns_server:
            self.server.stop()


def render_qr_svg(data: str, scale: int = 8, border: int = 2) -> str:
    """Inline SVG so the kiosk page needs no image round-trip."""
    import io

    import segno

    buffer = io.BytesIO()
    segno.make(data, error="m").save(
        buffer, kind="svg", scale=scale, border=border, xmldecl=False, svgns=True
    )
    return buffer.getvalue().decode("utf-8")


def render_qr_terminal(data: str, compact: bool = True) -> str:
    """Scannable QR for a console, so delivery can be tested before any UI exists.

    Compact mode uses half-block characters, which a cp1252 Windows console cannot encode; fall
    back to the wider ASCII-safe form when the active stdout encoding cannot represent them.
    """
    import io
    import sys

    import segno

    if compact:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        try:
            "\u2580".encode(encoding)
        except (LookupError, UnicodeEncodeError):
            compact = False

    buffer = io.StringIO()
    segno.make(data, error="m").terminal(buffer, compact=compact)
    return buffer.getvalue()
