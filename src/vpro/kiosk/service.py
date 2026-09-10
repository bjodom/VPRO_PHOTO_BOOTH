"""Kiosk service: owns the long-lived resources and drives the guest pipeline.

The whole point of the persistent process is here. The camera stays open, the Juggernaut pipeline
stays compiled, and the delivery server keeps running, so a guest pays only for their own render.
"""

from __future__ import annotations

import threading
import traceback
import json
import math
from contextlib import nullcontext
from statistics import median
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from ..delivery import DeliveryRequest, build_delivery
from ..vision.juggernaut_types import RenderRequest
from .camera import CameraStream
from .scenes import SCENES, get_scene
from .session import KioskSession, State

CAPTION = "Made at the Intel vPro Photo Booth #vPro #IntelAI"


@dataclass
class KioskConfig:
    output_dir: Path = Path("outputs/kiosk")
    camera_index: int = 0
    capture_width: int = 1920
    capture_height: int = 1080
    pose_fps: float = 15.0
    #: Clockwise correction for a camera mounted on its side; 90 or 270 gives a portrait frame.
    capture_rotate: int = 0

    yolo_model_path: Path = Path("models/yolo26/yolo26x-pose_openvino_model")
    yolo_device: str = "intel:gpu"
    rmbg_model_dir: Path = Path("models/rmbg/rmbg-1.4")
    rmbg_device: str = "AUTO"
    mask_quality: str = "high"

    scene_image: Path = Path("assets/scenes/portrait_scene_1080x1350.jpg")
    prop_image: Path = Path("assets/props/vpro_laptop.png")

    juggernaut_model_id: str = "OpenVINO/Juggernaut-XL-v9-fp16-ov"
    juggernaut_device: str = "GPU"
    local_files_only: bool = True
    juggernaut_cache_dir: Path = Path("outputs/openvino_cache/juggernaut")
    steps: int = 30
    guidance_scale: float = 5.0
    #: Near 1.0: the masked region is generated outright, not nudged.
    strength: float = 0.99
    width: int = 1080
    height: int = 1350
    #: Alpha at or above this is preserved. Lower values are blended edge pixels carrying the
    #: backdrop colour, so keeping them bakes a halo into the result.
    opaque_threshold: int = 250
    mask_erode_px: int = 2
    mask_feather_px: int = 8
    output_retention_hours: float = 24.0

    delivery_channel: str = "local-qr"
    delivery_host: str = "0.0.0.0"
    delivery_port: int = 8765
    delivery_advertise_host: str | None = None

    countdown_seconds: int = 3
    enable_generation: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.output_retention_hours) or self.output_retention_hours < 0:
            raise ValueError("Retention must be finite and nonnegative")
        if not 0 <= self.countdown_seconds <= 30:
            raise ValueError("Countdown must be between 0 and 30 seconds")


@dataclass
class ServiceStatus:
    camera_open: bool = False
    camera_fps: float = 0.0
    renderer_state: str = "idle"
    renders_served: int = 0
    last_error: str | None = None
    startup_seconds: float | None = None
    queue_rejections: int = 0
    last_pipeline_timings: dict[str, float] = field(default_factory=dict)
    stage: str = "idle"
    queue_depth: int = 0


class KioskService:
    """Wires the session state machine to the camera, renderer and delivery channel."""

    def __init__(self, config: KioskConfig | None = None) -> None:
        self.config = config or KioskConfig()
        self.session = KioskSession()
        self.status = ServiceStatus()
        self.camera: CameraStream | None = None
        self.runner: Any = None
        self.delivery: Any = None
        self._backend: Any = None
        self._rmbg_runtime: Any = None
        self._lock = threading.RLock()
        self._worker: threading.Thread | None = None
        self._warmed = False
        self._stop = threading.Event()
        self._maintenance: threading.Thread | None = None
        self._active_paths: set[Path] = set()
        self._pipeline_started = 0.0
        self._render_samples: list[float] = []

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> KioskService:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        from openvino import Core
        from ..vision.rmbg_runtime import validate_rmbg_assets

        available = {device.split(".")[0] for device in Core().available_devices}
        for configured in (self.config.yolo_device, self.config.rmbg_device, self.config.juggernaut_device):
            target = configured.removeprefix("intel:").upper().split(".")[0]
            if target not in available and target not in {"AUTO", "MULTI", "HETERO"}:
                raise RuntimeError(f"Configured device {configured} is unavailable; detected {sorted(available)}")
        validate_rmbg_assets(self.config.rmbg_model_dir)
        if not self.config.prop_image.is_file():
            raise FileNotFoundError(f"Laptop prop not found: {self.config.prop_image}")

        from ..backends.factory import build_backend

        self._backend = build_backend("openvino")
        self._backend.load_model(self.config.yolo_model_path)

        from ..vision.rmbg_runtime import load_rmbg_runtime

        self._rmbg_runtime = load_rmbg_runtime(
            self.config.rmbg_model_dir, device=self.config.rmbg_device
        )
        self._rmbg_runtime.warmup(runs=1)
        self._cleanup_old_outputs()

        self.camera = CameraStream(
            index=self.config.camera_index,
            width=self.config.capture_width,
            height=self.config.capture_height,
            pose_backend=self._backend,
            pose_device=self.config.yolo_device,
            rotate=self.config.capture_rotate,
            pose_fps=self.config.pose_fps,
        ).start()

        self.delivery = build_delivery(
            self.config.delivery_channel,
            **(
                {
                    "host": self.config.delivery_host,
                    "port": self.config.delivery_port,
                    "advertise_host": self.config.delivery_advertise_host,
                }
                if self.config.delivery_channel == "local-qr"
                else {}
            ),
        )

        if self.config.enable_generation:
            from ..vision.juggernaut_runner import JuggernautRunner

            self.runner = JuggernautRunner(
                model_id=self.config.juggernaut_model_id,
                device=self.config.juggernaut_device,
                local_files_only=self.config.local_files_only,
                openvino_cache_dir=self.config.juggernaut_cache_dir,
                task="inpaint",
                warmup=True,
                warmup_width=self.config.width,
                warmup_height=self.config.height,
            ).start()
            self._warmed = True

        self._maintenance = threading.Thread(target=self._maintain, name="vpro-retention", daemon=True)
        self._maintenance.start()
        return self

    def shutdown(self) -> None:
        self._stop.set()
        if self.camera is not None:
            self.camera.stop()
        if self._worker is not None:
            self._worker.join(timeout=5)
        if self._maintenance is not None:
            self._maintenance.join(timeout=5)
        if self.runner is not None:
            self.runner.shutdown()
        if self.delivery is not None:
            self.delivery.close()

    def __enter__(self) -> KioskService:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()

    # -- state for the UI ----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        session = self.session
        session.tick()
        if self.camera is not None:
            # Pose overlay is only useful on the pose screen; skip the inference otherwise so it
            # does not compete with a render for the GPU.
            self.camera.annotate = session.state is State.POSE
            self.status.camera_open = self.camera.is_running
            self.status.camera_fps = round(self.camera.stats.fps, 1)
        if self.runner is not None:
            status = self.runner.status()
            self.status.renderer_state = status.state
            self.status.renders_served = status.renders_served
            self.status.startup_seconds = status.startup_seconds
            self.status.queue_rejections = status.queue_rejections
            self.status.queue_depth = status.queue_depth

        return {
            "state": session.state.value,
            "session_id": session.session_id,
            "seconds_remaining": session.seconds_remaining,
            "scene": session.data.scene,
            "custom_location": session.data.custom_location,
            "error": session.data.error,
            "delivery_url": session.data.delivery_url,
            "qr_svg": session.data.delivery_qr_svg,
            "caption": session.data.caption,
            "has_final_image": session.data.final_path is not None,
            "generation_enabled": self.config.enable_generation,
            "countdown_seconds": self.config.countdown_seconds,
            "retention_hours": self.config.output_retention_hours,
            "stage": self.status.stage,
            "elapsed_seconds": round(perf_counter() - self._pipeline_started, 1)
            if session.state is State.GENERATING else 0,
            "estimated_seconds": round(median(self._render_samples)) if len(self._render_samples) >= 3 else None,
            "busy": self._worker is not None and self._worker.is_alive(),
            "framing": self._framing_snapshot(),
            "scenes": [
                {"key": s.key, "label": s.label, "description": s.description} for s in SCENES
            ],
            "status": {
                "camera_open": self.status.camera_open,
                "camera_fps": self.status.camera_fps,
                "renderer_state": self.status.renderer_state,
                "renders_served": self.status.renders_served,
                "startup_seconds": self.status.startup_seconds,
                "queue_rejections": self.status.queue_rejections,
                "queue_depth": self.status.queue_depth,
                "pipeline_timings": dict(self.status.last_pipeline_timings),
                "last_error": self.status.last_error,
            },
        }

    def _framing_snapshot(self) -> dict[str, Any]:
        if self.camera is None or not hasattr(self.camera, "framing"):
            return {"ok": True, "status": "unknown", "message": ""}
        framing = self.camera.framing
        return {
            "ok": framing.ok,
            "status": framing.status,
            "message": framing.message,
            "height_ratio": round(framing.height_ratio, 3),
        }

    # -- guest actions -------------------------------------------------------------

    def act(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            return self._act(action, payload)

    def _act(self, action: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        payload = payload or {}
        session = self.session
        if "session_id" in payload and payload["session_id"] != session.session_id:
            raise ValueError("This session has ended. Please start again.")

        if action == "start":
            session.start()
        elif action == "accept_consent":
            session.accept_consent()
        elif action == "choose_scene":
            scene = get_scene(str(payload.get("scene", "")), payload.get("custom_location"))
            session.choose_scene(scene.key, scene.label if scene.key == "custom" else None)
        elif action == "capture":
            self._capture_now()
        elif action == "retake":
            session.retake()
        elif action == "accept_capture":
            self._require_worker_idle()
            session.accept_capture()
            self._start_pipeline()
        elif action == "finish":
            session.finish()
        elif action == "retry":
            self._require_worker_idle()
            if session.state is State.CAMERA_ERROR and self.camera is not None:
                self.camera.stop()
                self.camera.start()
            resumed = session.retry()
            if resumed is State.GENERATING:
                self._start_pipeline()
        elif action == "choose_another_scene":
            session.choose_another_scene()
        elif action == "reset":
            session.reset(reason="guest")
        else:
            raise ValueError(f"Unknown action {action!r}")

        return self.snapshot()

    def _require_worker_idle(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            raise ValueError("The previous image is still finishing. Please try again shortly.")
        if self.runner is not None and self.runner.status().state == "rendering":
            raise ValueError("The renderer is still finishing. Please try again shortly.")

    def _capture_now(self) -> None:
        self.session._require(State.POSE)
        if self.camera is None:
            self.session.camera_failed("camera is not running")
            return
        frame = self.camera.latest_frame(max_age=1.0)
        if frame is None:
            self.session.camera_failed("no frame available from the camera")
            return

        import cv2

        stamp = uuid4().hex
        path = self.config.output_dir / f"capture_{stamp}.jpg"
        if not cv2.imwrite(str(path), frame):
            self.session.camera_failed("Could not save the photo. Ask staff to check disk space.")
            return
        self.session.capture(path)
        self.camera.annotate = False
        self._warm_renderer()

    def _warm_renderer(self) -> None:
        """Run the deferred warmup once the guest is past the live preview.

        Warmup is a full-resolution diffusion pass, which starves the preview if it runs during
        posing; here it overlaps the review screen instead, so the first guest's render is as fast
        as every later one.
        """
        if self._warmed or self.runner is None:
            return
        self._warmed = True
        try:
            self.runner.submit_warmup()
        except Exception as exc:
            print(f"[kiosk] warmup could not be queued: {exc}")

    # -- pipeline ------------------------------------------------------------------

    def _start_pipeline(self) -> None:
        session_id = self.session.session_id
        self._pipeline_started = perf_counter()
        self.status.last_pipeline_timings = {}
        self.status.last_error = None
        self.status.stage = "preparing"
        self._worker = threading.Thread(
            target=self._run_pipeline, args=(session_id,), name="vpro-kiosk-pipeline", daemon=True
        )
        self._worker.start()

    def _run_pipeline(self, session_id: int) -> None:
        try:
            final_path, delivery = self._produce(session_id)
        except Exception as exc:
            detail = f"{type(exc).__name__}: {exc}"
            self.status.last_error = detail
            traceback.print_exc()
            self._apply(session_id, lambda: self.session.generation_failed(
                "Your image could not be completed. Please try again or ask staff for help."
            ))
            self._record_metrics(False)
            return
        finally:
            with self._lock:
                self._active_paths.clear()

        if final_path is None:
            return  # the guest left; _produce already bailed out

        if delivery is None or not delivery.ok:
            reason = "delivery unavailable" if delivery is None else (delivery.error or "unknown")
            self.status.last_error = reason
            self._apply(session_id, lambda: self.session.delivery_failed(
                "Your portrait is saved. Please retry the download or ask staff."
            ))
            return

        self._apply(
            session_id,
            lambda: self.session.generation_succeeded(
                final_path,
                delivery_url=delivery.url,
                delivery_qr_svg=delivery.qr_svg,
                caption=CAPTION,
            ),
        )

    def _produce(self, session_id: int) -> tuple[Path | None, Any]:
        import cv2
        import numpy as np

        from ..vision.pipeline import compose_portrait_from_image

        config = self.config
        with self._lock:
            if self._abandoned(session_id):
                return None, None
            capture_path = self.session.data.capture_path
            scene_key = self.session.data.scene
            custom_location = self.session.data.custom_location
            saved_final = self.session.data.final_path
            if capture_path is not None:
                self._active_paths.add(capture_path)
        if saved_final is not None and saved_final.is_file():
            return saved_final, self._deliver(session_id, saved_final)
        if capture_path is None or scene_key is None:
            raise RuntimeError("capture or scene missing when the pipeline started")

        scene = get_scene(scene_key, custom_location)
        stamp = uuid4().hex
        composed_path = config.output_dir / f"composed_{stamp}.jpg"
        coverage_path = config.output_dir / f"coverage_{stamp}.png"

        # The destination is generated around the guest, so the backdrop only has to be neutral.
        canvas_path = config.output_dir / "neutral_canvas.png"
        if not canvas_path.exists():
            if not cv2.imwrite(
                str(canvas_path),
                np.full((config.height, config.width, 3), 128, dtype=np.uint8),
            ):
                raise RuntimeError("Could not write composition canvas")

        timings: dict[str, float] = {}
        self.status.stage = "composing"
        started = perf_counter()
        with self.camera.inference_lock if self.camera is not None else nullcontext():
            composed = compose_portrait_from_image(
                backend=self._backend,
                input_image_path=capture_path,
                rmbg_model_dir=config.rmbg_model_dir,
                rmbg_device=config.rmbg_device,
                yolo_device=config.yolo_device,
                scene_image_path=canvas_path,
                prop_image_path=config.prop_image,
                output_image_path=composed_path,
                mask_quality=config.mask_quality,
                verbose=False,
                coverage_output_path=coverage_path,
                rmbg_runtime=self._rmbg_runtime,
                timings=timings,
            )
        timings["compose_total_seconds"] = perf_counter() - started
        self.status.last_pipeline_timings = dict(timings)
        print(f"[kiosk] compose {perf_counter() - started:.2f}s -> {composed}")

        if self._abandoned(session_id):
            return None, None

        final_path = composed
        if self.runner is not None:
            self.status.stage = "generating"
            mask_path = self._build_inpaint_mask(coverage_path, stamp)
            render_path = config.output_dir / f"final_{stamp}.jpg"
            started = perf_counter()
            future = self.runner.submit(
                RenderRequest(
                    mode="inpaint",
                    prompt=scene.background_prompt(),
                    negative_prompt=scene.background_negative_prompt(),
                    output_path=render_path,
                    steps=config.steps,
                    guidance_scale=config.guidance_scale,
                    strength=config.strength,
                    width=config.width,
                    height=config.height,
                    input_image_path=composed,
                    mask_image_path=mask_path,
                )
            )
            while True:
                if self._abandoned(session_id):
                    future.cancel()
                    return None, None
                try:
                    result = future.result(timeout=0.25)
                    break
                except FutureTimeout:
                    if perf_counter() - started > 170:
                        future.cancel()
                        raise RuntimeError("Rendering exceeded the event time limit")
            timings["juggernaut_seconds"] = perf_counter() - started
            timings["queue_wait_seconds"] = result.queue_wait_seconds
            timings["render_seconds"] = result.render_seconds
            self.status.last_pipeline_timings = dict(timings)
            print(f"[kiosk] render {perf_counter() - started:.2f}s ok={result.ok}")
            if not result.ok or result.output_path is None:
                raise RuntimeError(result.error or "Generation returned no image")
            else:
                final_path = result.output_path

        if self._abandoned(session_id):
            return None, None

        with self._lock:
            if self._abandoned(session_id):
                return None, None
            self.session.data.final_path = final_path
        delivery = self._deliver(session_id, final_path)
        timings["total_seconds"] = perf_counter() - self._pipeline_started
        self.status.last_pipeline_timings.update(timings)
        self._record_metrics(bool(delivery and delivery.ok))
        if delivery and delivery.ok:
            self._render_samples = (self._render_samples + [timings["total_seconds"]])[-20:]
        return final_path, delivery

    def _record_metrics(self, success: bool) -> None:
        record = {"time": datetime.now().isoformat(), "success": success,
                  "generation_enabled": self.runner is not None,
                  "task": "inpaint", "steps": self.config.steps, "strength": self.config.strength,
                  "yolo_device": self.config.yolo_device, "rmbg_device": self.config.rmbg_device,
                  "render_device": self.config.juggernaut_device, **self.status.last_pipeline_timings}
        try:
            with (self.config.output_dir / "pipeline_metrics.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError as exc:
            print(f"[kiosk] could not record metrics: {exc}")

    def _deliver(self, session_id: int, final_path: Path) -> Any:
        from ..delivery.base import DeliveryResult

        with self._lock:
            if self._abandoned(session_id):
                return None
            self.status.stage = "delivery"
        started = perf_counter()
        try:
            return self.delivery.deliver(DeliveryRequest(image_path=final_path, caption=CAPTION))
        except Exception as exc:
            return DeliveryResult(ok=False, channel=self.config.delivery_channel, error=str(exc))
        finally:
            self.status.last_pipeline_timings["delivery_seconds"] = perf_counter() - started

    def _maintain(self) -> None:
        while not self._stop.wait(60):
            with self._lock:
                self.session.tick()
            self._cleanup_old_outputs()
            store = getattr(self.delivery, "store", None)
            if store is not None:
                store.purge_expired()

    def _cleanup_old_outputs(self) -> None:
        if self.config.output_retention_hours <= 0:
            return
        cutoff = datetime.now().timestamp() - (self.config.output_retention_hours * 3600.0)
        with self._lock:
            protected = self._active_paths | {
                self.session.data.capture_path, self.session.data.final_path
            }
        store = getattr(self.delivery, "store", None)
        if store is not None:
            protected |= store.active_paths()
        for path in self.config.output_dir.iterdir():
            try:
                if (path not in protected and path.resolve() not in protected
                    and path.name.startswith(("capture_", "composed_", "coverage_", "mask_", "final_"))
                    and path.is_file() and path.stat().st_mtime < cutoff):
                    path.unlink(missing_ok=True)
            except OSError as exc:
                print(f"[kiosk] could not remove stale output {path}: {exc}")

    def _build_inpaint_mask(self, coverage_path: Path, stamp: str) -> Path:
        """White is repainted, so invert the guest's opaque coverage.

        Only fully opaque pixels are preserved. A partially transparent pixel was blended with the
        canvas behind it, so keeping it bakes that colour in as a halo around hair.
        """
        import cv2
        import numpy as np

        from ..vision.juggernaut_runtime import feather_mask

        coverage = cv2.imread(str(coverage_path), cv2.IMREAD_GRAYSCALE)
        if coverage is None:
            raise RuntimeError(f"Could not read coverage mask: {coverage_path}")

        opaque = ((coverage >= self.config.opaque_threshold).astype(np.uint8)) * 255
        locked = feather_mask(
            opaque,
            feather_px=self.config.mask_feather_px,
            expand_px=-abs(self.config.mask_erode_px),
        )
        mask_path = self.config.output_dir / f"mask_{stamp}.png"
        if not cv2.imwrite(str(mask_path), 255 - locked):
            raise RuntimeError("Could not save inpaint mask")
        return mask_path

    def _abandoned(self, session_id: int) -> bool:
        """True once the guest's session has been replaced, so results must be discarded."""
        return self._stop.is_set() or self.session.session_id != session_id

    def _apply(self, session_id: int, action) -> None:
        with self._lock:
            if self._abandoned(session_id):
                print("[kiosk] discarding result for an abandoned session")
                return
            try:
                action()
            except Exception as exc:
                print(f"[kiosk] could not apply pipeline result: {exc}")
