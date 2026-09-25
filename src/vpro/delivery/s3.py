"""Temporary image delivery through an S3-compatible object store."""

from __future__ import annotations

import os
import secrets
from pathlib import Path
from typing import Any

from .base import DeliveryRequest, DeliveryResult
from .local_qr import render_qr_svg


class S3QrDelivery:
    """Upload a private image and return a short-lived presigned download URL."""

    name = "s3-qr"

    def __init__(
        self,
        bucket: str | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        url_ttl_seconds: int | None = None,
        key_prefix: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.bucket = bucket or os.environ.get("VPRO_S3_BUCKET", "vpro-photo-delivery")
        self.endpoint_url = endpoint_url or os.environ.get("VPRO_S3_ENDPOINT_URL")
        self.region_name = region_name or os.environ.get("VPRO_S3_REGION")
        self.url_ttl_seconds = url_ttl_seconds or int(
            os.environ.get("VPRO_S3_URL_TTL_SECONDS", "900")
        )
        self.key_prefix = (key_prefix or os.environ.get("VPRO_S3_KEY_PREFIX", "sessions")).strip("/")
        self.client = client or self._build_client()

    def _build_client(self) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "S3 delivery requires the optional 'delivery' dependencies; run 'uv sync --extra delivery'."
            ) from exc

        kwargs: dict[str, str] = {}
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.region_name:
            kwargs["region_name"] = self.region_name
        return boto3.client("s3", **kwargs)

    def deliver(self, request: DeliveryRequest) -> DeliveryResult:
        image_path = Path(request.image_path)
        if not image_path.is_file():
            return DeliveryResult(
                ok=False,
                channel=self.name,
                error=f"FileNotFoundError: Cannot hand off missing file: {image_path}",
            )

        key = f"{self.key_prefix}/{secrets.token_urlsafe(16)}{image_path.suffix.lower()}"
        content_type = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        try:
            self.client.upload_file(
                str(image_path),
                self.bucket,
                key,
                ExtraArgs={
                    "ContentType": content_type,
                    "ContentDisposition": f'attachment; filename="vpro-photo{image_path.suffix.lower()}"',
                },
            )
            url = self.client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket, "Key": key},
                ExpiresIn=self.url_ttl_seconds,
            )
        except Exception as exc:  # S3 providers expose different exception classes.
            return DeliveryResult(ok=False, channel=self.name, error=f"{type(exc).__name__}: {exc}")

        return DeliveryResult(
            ok=True,
            channel=self.name,
            url=url,
            qr_svg=render_qr_svg(url),
            expires_in_seconds=self.url_ttl_seconds,
        )

    def close(self) -> None:
        return None