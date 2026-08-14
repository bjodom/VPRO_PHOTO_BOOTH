"""Kiosk service: owns the long-lived resources and drives the guest pipeline.

The whole point of the persistent process is here. The camera stays open, the Juggernaut pipeline
stays compiled, and the delivery server keeps running, so a guest pays only for their own render.
"""

from __future__ import annotations

import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from ..delivery import DeliveryRequest, build_delivery
from ..vision.juggernaut_types import RenderRequest
from .camera import CameraStream
from .scenes import DEFAULT_NEGATIVE_PROMPT, SCENES, get_scene
from .session import KioskSession, State

CAPTION = "Made at the Intel vPro Photo Booth #vPro #IntelAI"


@dataclass
class KioskConfig:
    output_dir: Path = Path("outputs/kiosk")
    camera_index: int = 0
    capture_width: int = 1920
    capture_height: int = 1080

    yolo_model_path: Path = Path("models/yolo26/yolo26x-pose_openvino_model")
    yolo_device: str = "intel:gpu"
    rmbg_model_dir: Path = Path("models/rmbg/rmbg-1.4")
    rmbg_device: str = "AUTO"
    mask_quality: str = "high"

    scene_image: Path = Path("assets/scenes/portrait_scene_1080x1350.jpg")
    prop_image: Path = Path("assets/props/vpro_laptop.png")

    juggernaut_model_id: str = "OpenVINO/Juggernaut-XL-v9-fp16-ov"
    juggernaut_device: str = "GPU"
    juggernaut_cache_dir: Path = Path("outputs/openvino_cache/juggernaut")
    steps: int = 28
    guidance_scale: float = 4.2
    strength: float = 0.16
    width: int = 1080
    height: int = 1350

    delivery_channel: str = "local-qr"
    delivery_host: str = "0.0.0.0"
    delivery_port: int = 8765
    delivery_advertise_host: str | None = None

    countdown_seconds: int = 3
    enable_generation: bool = True


@dataclass
class ServiceStatus:
    camera_open: bool = False
    camera_fps: float = 0.0
    renderer_state: str = "idle"
    renders_served: int = 0
    last_error: str | None = None
    startup_seconds: float | None = None


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
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._warmed = False

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> KioskService:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        from ..backends.factory import build_backend

        self._backend = build_backend("openvino")
        self._backend.load_model(self.config.yolo_model_path)

        self.camera = CameraStream(
            index=self.config.camera_index,
            width=self.config.capture_width,
            height=self.config.capture_height,
            pose_backend=self._backend,
            pose_device=self.config.yolo_device,
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

            # warmup=False: diffusion inference starves the live preview, so it is deferred
            # until a guest has finished at the camera (see docs/optimization.md 2.6).
            self.runner = JuggernautRunner(
                model_id=self.config.juggernaut_model_id,
                device=self.config.juggernaut_device,
                openvino_cache_dir=self.config.juggernaut_cache_dir,
                task="img2img",
                warmup=False,
                warmup_width=self.config.width,
                warmup_height=self.config.height,
            ).start()

        return self

    def shutdown(self) -> None:
        if self.camera is not None:
            self.camera.stop()
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

        return {
            "state": session.state.value,
            "session_id": session.session_id,
            "seconds_remaining": session.seconds_remaining,
            "scene": session.data.scene,
            "error": session.data.error,
            "delivery_url": session.data.delivery_url,
            "qr_svg": session.data.delivery_qr_svg,
            "caption": session.data.caption,
            "has_final_image": session.data.final_path is not None,
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
        payload = payload or {}
        session = self.session

        if action == "start":
            session.start()
        elif action == "accept_consent":
            session.accept_consent()
        elif action == "choose_scene":
            session.choose_scene(str(payload.get("scene", "")))
        elif action == "capture":
            self._capture_now()
        elif action == "retake":
            session.retake()
        elif action == "accept_capture":
            session.accept_capture()
            self._start_pipeline()
        elif action == "finish":
            session.finish()
        elif action == "retry":
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

    def _capture_now(self) -> None:
        if self.camera is None:
            self.session.camera_failed("camera is not running")
            return
        frame = self.camera.latest_frame()
        if frame is None:
            self.session.camera_failed("no frame available from the camera")
            return

        import cv2

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.config.output_dir / f"capture_{stamp}.jpg"
        cv2.imwrite(str(path), frame)
        self.session.capture(path)
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
            self._apply(session_id, lambda: self.session.generation_failed(detail))
            return

        if final_path is None:
            return  # the guest left; _produce already bailed out

        if delivery is None or not delivery.ok:
            reason = "delivery unavailable" if delivery is None else (delivery.error or "unknown")
            self._apply(session_id, lambda: self.session.delivery_failed(reason))
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
        from ..vision.pipeline import compose_portrait_from_image

        config = self.config
        capture_path = self.session.data.capture_path
        scene_key = self.session.data.scene
        if capture_path is None or scene_key is None:
            raise RuntimeError("capture or scene missing when the pipeline started")

        scene = get_scene(scene_key)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        composed_path = config.output_dir / f"composed_{stamp}.jpg"

        background = scene.pick_background()
        if background is None:
            # Falls back to the studio backdrop; at strength 0.16 the render cannot invent a
            # landmark, so run scripts/generate_scene_backgrounds.py to make destinations appear.
            print(f"[kiosk] no background for {scene.key}; using {config.scene_image}")
            background = config.scene_image

        started = perf_counter()
        composed = compose_portrait_from_image(
            backend=self._backend,
            input_image_path=capture_path,
            rmbg_model_dir=config.rmbg_model_dir,
            rmbg_device=config.rmbg_device,
            yolo_device=config.yolo_device,
            scene_image_path=background,
            prop_image_path=config.prop_image,
            output_image_path=composed_path,
            mask_quality=config.mask_quality,
            verbose=False,
        )
        print(f"[kiosk] compose {perf_counter() - started:.2f}s -> {composed}")

        if self._abandoned(session_id):
            return None, None

        final_path = composed
        if self.runner is not None:
            render_path = config.output_dir / f"final_{stamp}.jpg"
            started = perf_counter()
            result = self.runner.render(
                RenderRequest(
                    mode="img2img",
                    prompt=scene.prompt(),
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    output_path=render_path,
                    steps=config.steps,
                    guidance_scale=config.guidance_scale,
                    strength=config.strength,
                    width=config.width,
                    height=config.height,
                    input_image_path=composed,
                )
            )
            print(f"[kiosk] render {perf_counter() - started:.2f}s ok={result.ok}")
            if not result.ok or result.output_path is None:
                # The deterministic compose is a real image, so hand that over instead of failing.
                self.status.last_error = result.error
                print(f"[kiosk] render failed, using deterministic compose: {result.error}")
            else:
                final_path = result.output_path

        if self._abandoned(session_id):
            return None, None

        delivery = self.delivery.deliver(
            DeliveryRequest(image_path=final_path, caption=CAPTION)
        )
        return final_path, delivery

    def _abandoned(self, session_id: int) -> bool:
        """True once the guest's session has been replaced, so results must be discarded."""
        return self.session.session_id != session_id

    def _apply(self, session_id: int, action) -> None:
        with self._lock:
            if self._abandoned(session_id):
                print("[kiosk] discarding result for an abandoned session")
                return
            try:
                action()
            except Exception as exc:
                print(f"[kiosk] could not apply pipeline result: {exc}")
