"""Kiosk web app and camera tests with hardware stubbed out.

    python tests/kiosk_app_test.py

No camera, GPU, model or network required.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from time import sleep

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from vpro.kiosk.app import create_app  # noqa: E402
from vpro.kiosk.camera import CameraStream  # noqa: E402
from vpro.kiosk.service import KioskService  # noqa: E402
from vpro.kiosk.session import State  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed. {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


class FakeCamera:
    """Stands in for CameraStream without touching a device."""

    def __init__(self, index: int = 0, rotate: int = 0) -> None:
        self.index = index
        self.rotate = rotate
        self.is_running = True
        self.stats = type("S", (), {"fps": 30.0, "inference_seconds": 0.01, "last_error": None})()
        self.frame: np.ndarray | None = np.zeros((80, 64, 3), dtype=np.uint8)
        self.stopped = False

    def latest_frame(self, max_age=None):
        return self.frame

    def mjpeg_frames(self, boundary: str = "frame"):
        yield b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\n\r\nx\r\n"

    def start(self):
        self.is_running = True
        return self

    def stop(self) -> None:
        self.is_running = False
        self.stopped = True


class StubService(KioskService):
    """Real session and routing; fake camera, renderer and delivery."""

    def __init__(self, tmp: Path) -> None:
        super().__init__()
        self.config.output_dir = tmp
        self.camera = FakeCamera()
        self.delivery = None
        self.runner = None
        self.pipeline_calls = 0
        self.tmp = tmp

    def _capture_now(self) -> None:
        path = self.tmp / "capture.jpg"
        path.write_bytes(b"\xff\xd8\xff\xd9")
        self.session.capture(path)

    def _start_pipeline(self) -> None:
        self.pipeline_calls += 1  # do not spawn threads in tests

    def _build_camera_stream(self) -> FakeCamera:
        return FakeCamera(self.config.camera_index, self.config.capture_rotate)


def client_for(tmp: Path) -> tuple[TestClient, StubService]:
    service = StubService(tmp)
    return TestClient(create_app(service)), service


def test_index_and_static(tmp: Path) -> None:
    client, _ = client_for(tmp)
    response = client.get("/")
    check("index serves html", response.status_code == 200 and "<html" in response.text.lower())
    check("index has all screens", response.text.count('class="screen') >= 11, response.text.count('class="screen'))
    check("ready screen has no caption toggle", "caption-btn" not in response.text)
    check("registered Intel vPro mark is present", "Intel vPro®" in response.text)
    check("kiosk assets use current cache version", response.text.count("?v=event-8") == 2)
    # "your phone" appears in the QR copy; what must not exist is a field asking for a number.
    check("no phone number input", 'type="tel"' not in response.text)
    check("no consent text about text messages", "text message" not in response.text.lower())

    for asset in ("/static/app.js", "/static/styles.css"):
        check(f"serves {asset}", client.get(asset).status_code == 200)

    check("health ok", client.get("/health").json() == {"ok": True})


def test_state_endpoint(tmp: Path) -> None:
    client, _ = client_for(tmp)
    body = client.get("/api/state").json()
    check("starts idle", body["state"] == "idle")
    check("exposes scenes", len(body["scenes"]) >= 6)
    check("scene has label", "label" in body["scenes"][0])
    check("idle has no timer", body["seconds_remaining"] is None)
    check("reports status", "camera_open" in body["status"])
    check("reports camera index", body["status"]["camera_index"] == 0)
    check("reports camera rotation", body["status"]["camera_rotate"] == 0)
    check("reports queue rejections", "queue_rejections" in body["status"])
    check("reports pipeline timings", "pipeline_timings" in body["status"])


def test_staff_can_switch_camera_index(tmp: Path) -> None:
    client, service = client_for(tmp)
    service.config.capture_rotate = 90
    old_camera = service.camera
    body = client.post("/api/action/next_camera").json()
    check("next camera increments index", body["status"]["camera_index"] == 1)
    check("next camera preserves rotation", body["status"]["camera_rotate"] == 90)
    check("old camera stopped", old_camera.stopped)
    check("new camera opened", service.camera.index == 1 and service.camera.rotate == 90 and service.camera.is_running)

    body = client.post("/api/action/previous_camera").json()
    check("previous camera decrements index", body["status"]["camera_index"] == 0)
    check("previous camera preserves rotation", service.camera.rotate == 90)


def test_full_flow_through_http(tmp: Path) -> None:
    client, service = client_for(tmp)

    check("start", client.post("/api/action/start").json()["state"] == "consent")
    check("consent", client.post("/api/action/accept_consent").json()["state"] == "select_scene")

    body = client.post("/api/action/choose_scene", json={"scene": "fuji"}).json()
    check("scene chosen", body["state"] == "pose" and body["scene"] == "fuji")

    check("capture", client.post("/api/action/capture").json()["state"] == "review")
    check("capture served", client.get("/api/capture.jpg").status_code == 200)

    check("accept", client.post("/api/action/accept_capture").json()["state"] == "generating")
    check("pipeline started once", service.pipeline_calls == 1)

    service.session.generation_succeeded(
        tmp / "final.jpg", delivery_url="http://kiosk/i/tok", delivery_qr_svg='<svg data-url="http://kiosk/i/tok"/>'
    )
    body = client.get("/api/state").json()
    check("ready state", body["state"] == "ready")
    check("qr exposed", body["qr_svg"] == '<svg data-url="http://kiosk/i/tok"/>')
    check("url exposed", body["delivery_url"] == "http://kiosk/i/tok")

    check("finish", client.post("/api/action/finish").json()["state"] == "done")
    check("reset", client.post("/api/action/reset").json()["state"] == "idle")


def test_invalid_transition_returns_409(tmp: Path) -> None:
    client, _ = client_for(tmp)
    response = client.post("/api/action/finish")
    check("out-of-order action is 409", response.status_code == 409, str(response.status_code))
    check("explains the conflict", "requires state" in response.json()["detail"])


def test_unknown_action_and_bad_scene(tmp: Path) -> None:
    client, _ = client_for(tmp)
    check("unknown action is 400", client.post("/api/action/teleport").status_code == 400)

    client.post("/api/action/start")
    client.post("/api/action/accept_consent")
    response = client.post("/api/action/choose_scene", json={"scene": ""})
    check("empty scene is 400", response.status_code == 400, str(response.status_code))


def test_missing_images_404(tmp: Path) -> None:
    client, _ = client_for(tmp)
    check("no final image yet", client.get("/api/final.jpg").status_code == 404)
    check("no capture yet", client.get("/api/capture.jpg").status_code == 404)


def test_custom_location(tmp: Path) -> None:
    from vpro.kiosk.scenes import get_scene

    client, service = client_for(tmp)
    client.post("/api/action/start")
    client.post("/api/action/accept_consent")
    for location in ("", "x", "x" * 101, "Kyoto\nJapan", "<script>", ["Kyoto"]):
        response = client.post("/api/action/choose_scene", json={"scene": "custom", "custom_location": location})
        check("invalid custom location rejected", response.status_code == 400)
        check("invalid place stays on selection", service.session.state is State.SELECT_SCENE)
    response = client.post("/api/action/choose_scene", json={"scene": "custom", "custom_location": "  Kyoto,   Japan "})
    check("custom destination enters pose", response.json()["state"] == "pose")
    check("custom place normalized", response.json()["custom_location"] == "Kyoto, Japan")
    scene = get_scene("custom", service.session.data.custom_location)
    check("custom location reaches generated prompt", "Kyoto, Japan" in scene.background_prompt())
    check("prompt guides perspective and ground", "ground plane" in scene.background_prompt() and "eye-level" in scene.background_prompt())
    client.post("/api/action/choose_another_scene")
    check("changing scenes clears custom place", service.session.data.custom_location is None)
    client.post("/api/action/choose_scene", json={"scene": "fuji"})
    check("preset still works after custom", service.session.data.scene == "fuji")
    client.post("/api/action/reset")
    check("reset clears custom place", client.get("/api/state").json()["custom_location"] is None)


def test_preview_stream(tmp: Path) -> None:
    client, _ = client_for(tmp)
    with client.stream("GET", "/api/preview.mjpg") as response:
        check("preview streams", response.status_code == 200)
        check(
            "multipart content type",
            "multipart/x-mixed-replace" in response.headers["content-type"],
            response.headers["content-type"],
        )


def test_camera_stream_without_device() -> None:
    """A missing camera must fail loudly in stats, not hang or crash the thread."""
    camera = CameraStream(index=999)
    camera.start()
    opened = camera.wait_until_open(timeout=15)
    camera.stop()
    check("open attempt completes", opened)
    check("no frame from a missing camera", camera.latest_frame() is None)
    check("error recorded", (camera.stats.last_error or "") != "")


def test_camera_latest_frame_is_a_copy() -> None:
    camera = CameraStream(index=999)
    frame = np.ones((4, 4, 3), dtype=np.uint8)
    camera._frame = frame
    copy = camera.latest_frame()
    copy[0, 0, 0] = 99
    check("caller cannot mutate the buffer", frame[0, 0, 0] == 1)


def test_camera_capture_frame_matches_mirrored_preview() -> None:
    camera = CameraStream(index=999)
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0, 0] = 10
    frame[0, 2, 0] = 20
    camera._frame = frame

    captured = camera.latest_capture_frame()
    check("capture frame mirrors preview orientation", captured[0, 0, 0] == 20 and captured[0, 2, 0] == 10)
    check("capture mirror does not mutate camera frame", frame[0, 0, 0] == 10 and frame[0, 2, 0] == 20)


def test_session_timeout_visible_over_http(tmp: Path) -> None:
    client, service = client_for(tmp)
    client.post("/api/action/start")
    service.session.timeouts[State.CONSENT] = 0.05
    sleep(0.1)
    check("expired session resets on poll", client.get("/api/state").json()["state"] == "idle")


def test_camera_rotation() -> None:
    """A camera mounted on its side must be corrected before pose runs, so shapes swap."""
    from vpro.kiosk.camera import rotate_frame

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    frame[0, 0] = (255, 255, 255)  # top-left marker

    check("0 degrees is a no-op", rotate_frame(frame, 0).shape == (100, 200, 3))
    check("90 swaps axes", rotate_frame(frame, 90).shape == (200, 100, 3))
    check("270 swaps axes", rotate_frame(frame, 270).shape == (200, 100, 3))
    check("180 keeps shape", rotate_frame(frame, 180).shape == (100, 200, 3))

    # Clockwise: the top-left corner ends up top-right.
    rotated = rotate_frame(frame, 90)
    check("90 rotates clockwise", tuple(rotated[0, -1]) == (255, 255, 255), str(rotated[0, -1]))

    rejected = False
    try:
        rotate_frame(frame, 45)
    except ValueError:
        rejected = True
    check("invalid rotation rejected", rejected)

    bad_config = False
    try:
        CameraStream(index=999, rotate=45)
    except ValueError:
        bad_config = True
    check("camera rejects invalid rotation", bad_config)


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="vpro-kiosk-test-") as raw:
        tmp = Path(raw)
        for test in (
            test_index_and_static,
            test_state_endpoint,
            test_full_flow_through_http,
            test_invalid_transition_returns_409,
            test_unknown_action_and_bad_scene,
            test_missing_images_404,
            test_custom_location,
            test_preview_stream,
            test_session_timeout_visible_over_http,
        ):
            print(f"\n{test.__name__}")
            test(tmp)
        for test in (
            test_camera_stream_without_device,
            test_camera_latest_frame_is_a_copy,
            test_camera_capture_frame_matches_mirrored_preview,
            test_camera_rotation,
        ):
            print(f"\n{test.__name__}")
            test()

    print(f"\n{len(PASSED)} checks passed.")
    return 0


def serve_preview() -> None:
    import tempfile
    import cv2
    import uvicorn
    from vpro.delivery.local_qr import render_qr_svg

    class PreviewService(StubService):
        def _capture_now(self) -> None:
            path = self.tmp / "capture.jpg"
            cv2.imwrite(str(path), np.full((1350, 1080, 3), 110, dtype=np.uint8))
            self.session.capture(path)

        def _start_pipeline(self) -> None:
            self.status.stage = "generating"
            session_id = self.session.session_id
            def complete():
                self._apply(session_id, lambda: self.session.generation_succeeded(
                    self.session.data.capture_path, delivery_url="http://127.0.0.1:8879/api/final.jpg",
                    delivery_qr_svg=render_qr_svg("http://127.0.0.1:8879/api/final.jpg"), caption="Preview fixture"
                ))
            threading.Timer(2, complete).start()

    class PreviewCamera(FakeCamera):
        def mjpeg_frames(self, boundary="frame"):
            payload = cv2.imencode(".jpg", np.full((1350, 1080, 3), 110, dtype=np.uint8))[1].tobytes()
            yield b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"

    with tempfile.TemporaryDirectory(prefix="vpro-ui-preview-") as raw:
        service = PreviewService(Path(raw))
        service.camera = PreviewCamera()
        service.status.renderer_state = "ready"
        print("Hardware-free UI preview: http://127.0.0.1:8879")
        uvicorn.run(create_app(service), host="127.0.0.1", port=8879, log_level="warning")


if __name__ == "__main__":
    if "--serve-preview" in sys.argv:
        serve_preview()
        raise SystemExit(0)
    raise SystemExit(main())
