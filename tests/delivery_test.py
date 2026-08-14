"""Delivery layer tests: handoff store policy, HTTP behavior, and QR generation.

    python tests/delivery_test.py

Uses a real loopback server; no camera, GPU or model needed.
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path
from time import sleep

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpro.delivery import (  # noqa: E402
    DeliveryRequest,
    HandoffServer,
    ImageHandoffStore,
    LocalQrDelivery,
    TwilioDelivery,
    build_delivery,
    render_qr_svg,
)

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed. {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


def sample_image(tmp: Path, name: str = "photo.jpg") -> Path:
    from PIL import Image

    path = tmp / name
    Image.new("RGB", (64, 80), (90, 140, 200)).save(path)
    return path


def fetch(url: str) -> tuple[int, bytes, dict[str, str]]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(), dict(exc.headers)


def test_store_expiry_and_download_cap(tmp: Path) -> None:
    image = sample_image(tmp)
    store = ImageHandoffStore(ttl_seconds=0.3, max_downloads=2)

    token = store.add(image)
    check("token is unguessable length", len(token) >= 20, token)
    check("resolves once", store.resolve(token) is not None)
    check("resolves twice", store.resolve(token) is not None)
    check("third resolve exceeds cap", store.resolve(token) is None)

    token2 = store.add(image)
    sleep(0.35)
    check("expired token does not resolve", store.resolve(token2) is None)

    check("unknown token does not resolve", store.resolve("not-a-real-token") is None)
    check("store drops dead entries", len(store) == 0, str(len(store)))


def test_store_rejects_missing_file(tmp: Path) -> None:
    store = ImageHandoffStore()
    raised = False
    try:
        store.add(tmp / "does-not-exist.jpg")
    except FileNotFoundError:
        raised = True
    check("missing file is rejected", raised)


def test_http_serves_only_known_tokens(tmp: Path) -> None:
    image = sample_image(tmp)
    store = ImageHandoffStore(ttl_seconds=60)
    with HandoffServer(store, host="127.0.0.1", port=0, advertise_host="127.0.0.1") as server:
        base = server.base_url
        token = store.add(image, caption="Hello #vPRO")

        status, body, headers = fetch(f"{base}/i/{token}")
        check("page returns 200", status == 200, str(status))
        check("page embeds the file link", f"/f/{token}".encode() in body)
        check("caption is shown", b"Hello #vPRO" in body)
        check("no-store is set", headers.get("Cache-Control") == "no-store")

        status, payload, headers = fetch(f"{base}/f/{token}")
        check("file returns 200", status == 200, str(status))
        check("file bytes match", payload == image.read_bytes())
        check("content type is jpeg", headers.get("Content-Type") == "image/jpeg")
        check("download disposition set", "attachment" in headers.get("Content-Disposition", ""))

        status, _, _ = fetch(f"{base}/i/bogus-token")
        check("unknown token 404s", status == 404, str(status))

        # A traversal attempt must be treated as an unknown token, never as a path.
        for probe in ("/f/..%2f..%2fpyproject.toml", "/i/../../pyproject.toml", "/pyproject.toml"):
            status, _, _ = fetch(f"{base}{probe}")
            check(f"traversal blocked: {probe}", status == 404, str(status))

        status, body, _ = fetch(f"{base}/health")
        check("health endpoint responds", status == 200 and body == b"ok")


def test_page_view_does_not_consume_downloads(tmp: Path) -> None:
    image = sample_image(tmp)
    store = ImageHandoffStore(ttl_seconds=60, max_downloads=1)
    with HandoffServer(store, host="127.0.0.1", port=0, advertise_host="127.0.0.1") as server:
        token = store.add(image)
        fetch(f"{server.base_url}/i/{token}")
        fetch(f"{server.base_url}/i/{token}")
        status, _, _ = fetch(f"{server.base_url}/f/{token}")
        check("file still available after page views", status == 200, str(status))
        status, _, _ = fetch(f"{server.base_url}/f/{token}")
        check("download cap applies to the file", status == 404, str(status))


def test_local_qr_channel(tmp: Path) -> None:
    image = sample_image(tmp)
    channel = LocalQrDelivery(host="127.0.0.1", port=0, advertise_host="127.0.0.1")
    try:
        result = channel.deliver(DeliveryRequest(image_path=image, caption="Posted from #vPRO"))
        check("delivery succeeds", result.ok, str(result.error))
        check("returns a url", (result.url or "").startswith("http://127.0.0.1:"), str(result.url))
        check("returns inline svg qr", "<svg" in (result.qr_svg or ""))
        check("reports expiry", (result.expires_in_seconds or 0) > 0)

        status, body, _ = fetch(result.url or "")
        check("url is reachable", status == 200 and b"Your photo is ready" in body)

        missing = channel.deliver(DeliveryRequest(image_path=tmp / "gone.jpg"))
        check("missing image reports failure", not missing.ok)
        check("failure does not raise", missing.error is not None)
    finally:
        channel.close()


def test_twilio_stub_is_honest() -> None:
    result = TwilioDelivery().deliver(DeliveryRequest(image_path=Path("x.jpg")))
    check("twilio reports not implemented", not result.ok)
    check("twilio explains why", "not implemented" in (result.error or "").lower())


def test_factory(tmp: Path) -> None:
    none_channel = build_delivery("none")
    result = none_channel.deliver(DeliveryRequest(image_path=tmp / "any.jpg"))
    check("none channel succeeds", result.ok)

    rejected = False
    try:
        build_delivery("carrier-pigeon")
    except ValueError:
        rejected = True
    check("unknown channel rejected", rejected)


def test_qr_encodes_url() -> None:
    svg = render_qr_svg("http://192.168.1.50:8765/i/abc")
    check("qr renders svg", svg.startswith("<svg") or "<svg" in svg)
    check("qr is non-trivial", len(svg) > 500, str(len(svg)))


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="vpro-delivery-test-") as raw:
        tmp = Path(raw)
        for test in (
            test_store_expiry_and_download_cap,
            test_store_rejects_missing_file,
            test_http_serves_only_known_tokens,
            test_page_view_does_not_consume_downloads,
            test_local_qr_channel,
            test_factory,
        ):
            print(f"\n{test.__name__}")
            test(tmp)
        for test in (test_twilio_stub_is_honest, test_qr_encodes_url):
            print(f"\n{test.__name__}")
            test()

    print(f"\n{len(PASSED)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
