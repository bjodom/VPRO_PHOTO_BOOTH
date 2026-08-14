"""Persistent camera with a latest-frame buffer and optional pose overlay.

Opening the camera per guest costs warmup frames and shutter lag, so the kiosk keeps it open for
the life of the app. A single worker thread owns the device; readers always get the most recent
frame rather than a queued backlog, which is what a live preview wants.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from time import perf_counter, sleep
from typing import Any

import numpy as np

JPEG_QUALITY = 80


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
    ) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.pose_backend = pose_backend
        self.pose_device = pose_device
        self.annotate = annotate
        self.jpeg_quality = jpeg_quality

        self.stats = CameraStats()
        self._capture: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._jpeg: bytes | None = None
        self._frame_ready = threading.Condition(self._lock)
        self._opened = threading.Event()

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> CameraStream:
        if self._thread is not None:
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="vpro-camera", daemon=True)
        self._thread.start()
        return self

    def wait_until_open(self, timeout: float = 10.0) -> bool:
        return self._opened.wait(timeout)

    def stop(self) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> CameraStream:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- readers -------------------------------------------------------------------

    def latest_frame(self) -> np.ndarray | None:
        """Most recent raw BGR frame, copied so callers cannot mutate the buffer."""
        with self._lock:
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
        while not self._stop.is_set():
            payload = self.wait_for_jpeg(timeout=1.0)
            if payload is None:
                continue
            yield (
                f"--{boundary}\r\nContent-Type: image/jpeg\r\n"
                f"Content-Length: {len(payload)}\r\n\r\n"
            ).encode("ascii") + payload + b"\r\n"

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

                display = frame
                if self.annotate and self.pose_backend is not None:
                    display = self._annotate(frame)

                encoded = cv2.imencode(
                    ".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                )[1].tobytes()

                with self._frame_ready:
                    self._frame = frame
                    self._jpeg = encoded
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
            results = self.pose_backend.predict(
                {"source": frame, "device": self.pose_device, "verbose": False}
            )
        except Exception as exc:
            # A preview without skeletons beats a dead camera thread.
            self.stats.last_error = f"pose overlay failed: {exc}"
            return frame
        self.stats.inference_seconds = perf_counter() - start
        if isinstance(results, list) and results:
            return results[0].plot()
        return frame
