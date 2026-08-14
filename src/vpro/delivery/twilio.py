"""Twilio MMS delivery.

Deliberately unimplemented: the service choice is not settled, and the local QR channel is the
default. This exists so the interface shape is proven against a second, very different channel
rather than fitted to the first one.

To implement:
  - add `twilio` to the kiosk extra, construct `Client(account_sid, auth_token)`
  - the image must be reachable by Twilio's fetchers, so a LAN URL will not do; this needs a
    publicly reachable media URL, which is the main design cost of this channel
  - map failures onto DeliveryResult rather than raising, so the kiosk can fall back
"""

from __future__ import annotations

import os

from .base import DeliveryRequest, DeliveryResult


class TwilioDelivery:
    name = "twilio"

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
        public_base_url: str | None = None,
    ) -> None:
        # Credentials come from the environment so they stay out of argv and logs.
        self.account_sid = account_sid or os.environ.get("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or os.environ.get("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or os.environ.get("TWILIO_FROM_NUMBER")
        self.public_base_url = public_base_url or os.environ.get("VPRO_PUBLIC_BASE_URL")

    def deliver(self, request: DeliveryRequest) -> DeliveryResult:
        return DeliveryResult(
            ok=False,
            channel=self.name,
            error=(
                "Twilio delivery is not implemented. It needs a publicly reachable media URL, "
                "which the local-only kiosk does not have; see the module docstring."
            ),
        )

    def close(self) -> None:
        return None
