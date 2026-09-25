from __future__ import annotations

from .base import DeliveryChannel
from .local_qr import LocalQrDelivery
from .s3 import S3QrDelivery
from .twilio import TwilioDelivery

CHANNELS = ("local-qr", "s3-qr", "twilio", "none")


class NoDelivery:
    """Saves nothing and sends nothing; the image is already on disk."""

    name = "none"

    def deliver(self, request):  # type: ignore[no-untyped-def]
        from .base import DeliveryResult

        return DeliveryResult(ok=True, channel=self.name, url=str(request.image_path))

    def close(self) -> None:
        return None


def build_delivery(channel: str, **kwargs: object) -> DeliveryChannel:
    if channel == "local-qr":
        return LocalQrDelivery(**kwargs)  # type: ignore[arg-type]
    if channel == "s3-qr":
        return S3QrDelivery(**kwargs)  # type: ignore[arg-type]
    if channel == "twilio":
        return TwilioDelivery(**kwargs)  # type: ignore[arg-type]
    if channel == "none":
        return NoDelivery()
    raise ValueError(f"Unknown delivery channel {channel!r}; expected one of {CHANNELS}")
