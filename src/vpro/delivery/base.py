"""Delivery channel interface.

Delivery is the one part of the kiosk we expect to change: local QR now, possibly Twilio or a
hosted link later. Everything upstream should depend on this interface, never on a channel.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class DeliveryRequest:
    image_path: Path
    caption: str | None = None
    # Only channels that address a person use this; QR handoff ignores it.
    recipient: str | None = None


@dataclass(frozen=True)
class DeliveryResult:
    ok: bool
    channel: str
    url: str | None = None
    qr_svg: str | None = None
    expires_in_seconds: float | None = None
    error: str | None = None


@runtime_checkable
class DeliveryChannel(Protocol):
    name: str

    def deliver(self, request: DeliveryRequest) -> DeliveryResult:
        """Hand the finished image to the guest. Must not raise for expected failures."""

    def close(self) -> None:
        """Release any resources (servers, sessions)."""


class DeliveryError(RuntimeError):
    pass
