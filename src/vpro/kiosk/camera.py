"""Persistent camera with a latest-frame buffer and optional pose overlay.

Opening the camera per guest costs warmup frames and shutter lag, so the kiosk keeps it open for
the life of the app. A single worker thread owns the device; readers always get the most recent
frame rather than a queued backlog, which is what a live preview wants.
"""

from __future__ import annotations

import threading
import math
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any

import numpy as np

from .framing import FramingFeedback, evaluate_framing, target_box

JPEG_QUALITY = 80
GUIDE_OK = (120, 220, 120)
GUIDE_BAD = (120, 170, 245)
VALID_ROTATIONS = (0, 90, 180, 270)


def rotate_frame(frame: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate clockwise by `degrees`, correcting for a physically rotated camera.

    Mounting the camera on its side puts the sensor's long axis vertical, which is what a 4:5
    portrait output wants; without it a landscape frame is cropped and most of the sensor is
    wasted on width we throw away.
    """
    if degrees == 0:
        return frame

    import cv2

    codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    if degrees not in codes:
        raise ValueError(f"rotation must be one of {VALID_ROTATIONS}, got {degrees}")
    return cv2.rotate(frame, codes[degrees])


@dataclass
class CameraStats:
    frames: int = 0
    inference_seconds: float = 0.0
    last_error: str | None = None
    fps: float = 0.0


class CameraStream:
    """Owns the capture device; annotates frames with pose when a backend is supplied."""

    def __init__(
        self,
        index: int = 0,
        width: int = 1920,
        height: int = 1080,
        pose_backend: Any = None,
        pose_device: str = "intel:gpu",
        annotate: bool = True,
        jpeg_quality: int = JPEG_QUALITY,
        mirror_preview: bool = True,
        rotate: int = 0,
        pose_fps: float = 15.0,
    ) -> None:
        if rotate not in VALID_ROTATIONS:
            raise ValueError(f"rotate must be one of {VALID_ROTATIONS}, got {rotate}")
        if not math.isfinite(pose_fps) or pose_fps < 0:
            raise ValueError("pose_fps must be >= 0")
        self.index = index
        self.width = width
        self.height = height
        self.pose_backend = pose_backend
        self.pose_device = pose_device
        self.annotate = annotate
        self.jpeg_quality = jpeg_quality
        self.mirror_preview = mirror_preview
        self.rotate = rotate
        self.pose_interval = 0.0 if pose_fps == 0 else 1.0 / pose_fps

        self.stats = CameraStats()
        self.framing: FramingFeedback = evaluate_framing(None, 0, 0)
        self.show_guide = True
        self._capture: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._frame_ready = threading.Condition(self._lock)
        self._opened = threading.Event()
        self._last_pose_at = 0.0
        self._last_bbox: tuple[int, int, int, int] | None = None
        self._last_result: Any = None
        self._preview_thread: threading.Thread | None = None
        self._frame_time = 0.0
        self._frame_sequence = 0
        self._subscribers = 0
        self.inference_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> CameraStream:
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._opened.clear()
        self._last_result = None
        with self._lock:
            self._frame = None
            self._jpeg = None
            self._frame_time = 0
        self._thread = threading.Thread(target=self._run, name="vpro-camera", daemon=True)
        self._thread.start()
        self._preview_thread = threading.Thread(target=self._preview_loop, name="vpro-preview", daemon=True)
        self._preview_thread.start()
        return self

    def wait_until_open(self, timeout: float = 10.0) -> bool:
        return self._opened.wait(timeout)

    def stop(self) -> None:
        self._stop.set()
        with self._frame_ready:
            self._frame_ready.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5)
        if self._preview_thread is not None:
            self._preview_thread.join(timeout=5)

    def __enter__(self) -> CameraStream:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- readers -------------------------------------------------------------------

    def latest_frame(self, max_age: float | None = None) -> np.ndarray | None:
        """Most recent raw BGR frame, copied so callers cannot mutate the buffer.

        Deliberately unmirrored: the preview is flipped so guests can adjust naturally, but the
        delivered photo should not be, or any text on their clothing comes out reversed.
        """
        with self._lock:
            if max_age is not None and perf_counter() - self._frame_time > max_age:
                return None
            return None if self._frame is None else self._frame.copy()

    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def wait_for_jpeg(self, timeout: float = 1.0) -> bytes | None:
        with self._frame_ready:
            self._frame_ready.wait(timeout)
            return self._jpeg

    def mjpeg_frames(self, boundary: str = "frame"):
        """Yield multipart chunks for an MJPEG response."""
        with self._lock:
            self._subscribers += 1
        try:
            while not self._stop.is_set():
                payload = self.wait_for_jpeg(timeout=1.0)
                if payload is None:
                    continue
                yield (
                    f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(payload)}\r\n\r\n"
                ).encode("ascii") + payload + b"\r\n"
        finally:
            with self._lock:
                self._subscribers -= 1

    def _preview_loop(self) -> None:
        import cv2

        sequence = -1
        while not self._stop.is_set():
            with self._frame_ready:
                self._frame_ready.wait_for(
                    lambda: self._stop.is_set() or self._frame_sequence != sequence, timeout=0.5
                )
                sequence = self._frame_sequence
                frame = None if self._frame is None else self._frame.copy()
                visible = self._subscribers > 0
            if frame is None or not visible:
                self._stop.wait(0.02)
                continue
            try:
                if self.annotate and self.pose_backend is not None:
                    display = self._annotate(frame)
                else:
                    self._last_result = None
                    display = cv2.flip(frame, 1) if self.mirror_preview else frame
                height, width = display.shape[:2]
                if max(height, width) > 1280:
                    scale = 1280 / max(height, width)
                    display = cv2.resize(display, (int(width * scale), int(height * scale)))
                ok, encoded = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
                if ok:
                    with self._frame_ready:
                        self._jpeg = encoded.tobytes()
                        self._frame_ready.notify_all()
            except Exception as exc:
                self.stats.last_error = f"preview failed: {exc}"

    # -- worker --------------------------------------------------------------------

    def _run(self) -> None:
        import cv2

        capture = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(self.index)
        if not capture.isOpened():
            self.stats.last_error = f"Could not open camera index {self.index}"
            self._opened.set()
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._capture = capture
        self._opened.set()

        window_start = perf_counter()
        window_frames = 0
        try:
            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    self.stats.last_error = "camera read failed"
                    sleep(0.05)
                    continue

                # Rotate first: pose, framing and the stored capture all work in this orientation.
                frame = rotate_frame(frame, self.rotate)

                with self._frame_ready:
                    self._frame = frame
                    self._frame_time = perf_counter()
                    self._frame_sequence += 1
                    self._frame_ready.notify_all()

                self.stats.frames += 1
                window_frames += 1
                elapsed = perf_counter() - window_start
                if elapsed >= 1.0:
                    self.stats.fps = window_frames / elapsed
                    window_start = perf_counter()
                    window_frames = 0
        finally:
            capture.release()
            self._capture = None

    def _annotate(self, frame: np.ndarray) -> np.ndarray:
        start = perf_counter()
        try:
            if start - self._last_pose_at >= self.pose_interval:
                with self.inference_lock:
                    results = self.pose_backend.predict(
                        {"source": frame, "device": self.pose_device, "verbose": False}
                    )
                self._last_result = results[0] if isinstance(results, list) and results else None
                self._last_pose_at = start
                self.stats.inference_seconds = perf_counter() - start
        except Exception as exc:
            # A preview without skeletons beats a dead camera thread.
            self.stats.last_error = f"pose overlay failed: {exc}"
            self._last_result = None

        display = frame.copy()
        bbox = None
        has_lower_body: bool | None = None
        if self._last_result is not None:
            # No class labels: they are drawn before the mirror flip, so they render backwards,
            # and "person 0.96" means nothing to a guest anyway.
            display = self._last_result.plot(img=display, labels=False, conf=False)
            bbox, index = _largest_person_bbox(self._last_result)
            if index is not None:
                has_lower_body = _has_lower_body(self._last_result, index)
        self._last_bbox = bbox

        height, width = frame.shape[:2]
        self.framing = evaluate_framing(bbox, width, height, has_lower_body)

        if self.mirror_preview:
            import cv2

            # Flip before the overlay so guide text is not drawn backwards.
            display = cv2.flip(display, 1)
        if self.show_guide:
            display = self._draw_guide(display, bbox)
        return display

    def _draw_guide(self, frame: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
        import cv2

        out = frame if frame.flags.writeable else frame.copy()
        height, width = out.shape[:2]
        box = target_box(width, height)
        colour = GUIDE_OK if self.framing.ok else GUIDE_BAD

        # Dashed rectangle so it reads as a guide rather than a detection.
        x1, y1, x2, y2 = box.as_tuple
        dash = 26
        for x in range(x1, x2, dash * 2):
            cv2.line(out, (x, y1), (min(x + dash, x2), y1), colour, 3)
            cv2.line(out, (x, y2), (min(x + dash, x2), y2), colour, 3)
        for y in range(y1, y2, dash * 2):
            cv2.line(out, (x1, y), (x1, min(y + dash, y2)), colour, 3)
            cv2.line(out, (x2, y), (x2, min(y + dash, y2)), colour, 3)

        return out


def _largest_person_bbox(result: Any) -> tuple[tuple[int, int, int, int] | None, int | None]:
    """Biggest detection, which is the guest standing closest to the booth."""
    boxes = getattr(result, "boxes", None)
    xyxy = getattr(boxes, "xyxy", None) if boxes is not None else None
    if xyxy is None or len(xyxy) == 0:
        return None, None

    best = None
    best_index = None
    best_area = 0.0
    for index, row in enumerate(xyxy):
        values = row.tolist() if hasattr(row, "tolist") else list(row)
        x1, y1, x2, y2 = (int(v) for v in values[:4])
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area > best_area:
            best_area = area
            best = (x1, y1, x2, y2)
            best_index = index
    return best, best_index


#: COCO pose indices for knees and ankles.
LOWER_BODY_KEYPOINTS = (13, 14, 15, 16)
KEYPOINT_CONFIDENCE = 0.5


def _has_lower_body(result: Any, index: int) -> bool | None:
    """True when knees or ankles are visible, so the subject is not cut off at the waist."""
    keypoints = getattr(result, "keypoints", None)
    data = getattr(keypoints, "data", None) if keypoints is not None else None
    if data is None or len(data) <= index:
        return None

    person = data[index]
    person = person.tolist() if hasattr(person, "tolist") else person
    if len(person) <= max(LOWER_BODY_KEYPOINTS):
        return None

    for keypoint_index in LOWER_BODY_KEYPOINTS:
        point = person[keypoint_index]
        if len(point) >= 3 and float(point[2]) >= KEYPOINT_CONFIDENCE:
            return True
    return False
