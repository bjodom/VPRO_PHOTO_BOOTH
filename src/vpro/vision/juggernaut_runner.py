"""Long-lived Juggernaut render worker.

Loading and compiling the pipeline costs ~16s per process launch and cannot be cached away
(see docs/optimization.md section 10). This keeps one compiled pipeline resident on a worker
thread so only the first guest pays that cost.

Renders are strictly serialized: there is one GPU and one compiled pipeline, and concurrent
calls into the same OpenVINO compiled model are not safe to assume.
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .juggernaut_runtime import (
    DEFAULT_RENDER_ATTEMPTS,
    load_juggernaut_pipeline,
    render_img2img,
    render_inpaint,
    render_text2img,
)
from .juggernaut_types import (
    TASK_IMG2IMG,
    TASK_INPAINT,
    TASK_TEXT2IMG,
    TASKS,
    RenderRequest,
    RenderResult,
)

STATE_IDLE = "idle"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_RENDERING = "rendering"
STATE_FAILED = "failed"
STATE_STOPPED = "stopped"

_SHUTDOWN = object()


@dataclass(frozen=True)
class RunnerStatus:
    state: str
    model_id: str
    device: str
    task: str
    renders_served: int
    failures: int
    queue_depth: int
    startup_seconds: float | None
    error: str | None
    queue_rejections: int


class JuggernautRunner:
    """Owns a compiled pipeline and serves render requests from a queue."""

    def __init__(
        self,
        # FP16 alternative: OpenVINO/Juggernaut-XL-v9-fp16-ov
        model_id: str = "OpenVINO/Juggernaut-XL-v9-int8-ov",
        device: str = "GPU",
        local_files_only: bool = True,
        openvino_cache_dir: Path | None = None,
        task: str = TASK_TEXT2IMG,
        queue_size: int = 8,
        warmup: bool = True,
        warmup_width: int = 1080,
        warmup_height: int = 1350,
        attempts: int = DEFAULT_RENDER_ATTEMPTS,
        vae_precision_hint: str | None = None,
    ) -> None:
        if task not in TASKS:
            raise ValueError(f"task must be one of {TASKS}, got {task!r}")

        self.model_id = model_id
        self.device = device
        self.task = task
        self._local_files_only = local_files_only
        self._openvino_cache_dir = openvino_cache_dir
        self._warmup = warmup
        self._warmup_size = (int(warmup_width), int(warmup_height))
        self._attempts = attempts
        self._vae_precision_hint = vae_precision_hint

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._pipeline: Any = None
        self._warmup_dir: Path | None = None

        self._state = STATE_IDLE
        self._error: str | None = None
        self._startup_seconds: float | None = None
        self._renders_served = 0
        self._failures = 0
        self._queue_rejections = 0

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> JuggernautRunner:
        """Begin loading on a background thread and return immediately."""
        with self._lock:
            if self._thread is not None:
                return self
            self._ready.clear()
            self._error = None
            self._state = STATE_LOADING
            self._thread = threading.Thread(
                target=self._run, name="juggernaut-runner", daemon=True
            )
            self._thread.start()
        return self

    def wait_until_ready(self, timeout: float | None = None) -> bool:
        """Block until the pipeline is loaded. Returns False on timeout, raises if loading failed."""
        if self._thread is None:
            raise RuntimeError("Runner not started; call start() first.")
        if not self._ready.wait(timeout):
            return False
        if self._state == STATE_FAILED:
            raise RuntimeError(f"Juggernaut runner failed to load: {self._error}")
        return True

    def shutdown(self, wait: bool = True, timeout: float | None = 60.0) -> None:
        """Stop the worker after the in-flight render, releasing the compiled model."""
        with self._lock:
            thread = self._thread
            if thread is None:
                return
            self._thread = None
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is not _SHUTDOWN:
                    item[1].cancel()
            self._queue.put_nowait(_SHUTDOWN)
        if wait:
            thread.join(timeout)

    def __enter__(self) -> JuggernautRunner:
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.shutdown()

    # -- submission ----------------------------------------------------------------

    def submit(self, request: RenderRequest) -> Future[RenderResult]:
        """Queue a render. Raises immediately on a bad request, a failed runner, or a full queue."""
        request = request.validated()
        if request.mode != self.task:
            raise ValueError(
                f"Runner was loaded for task '{self.task}' but request mode is '{request.mode}'. "
                "Construct a separate runner for the other task."
            )
        if self._thread is None and self._state != STATE_LOADING:
            raise RuntimeError("Runner not started; call start() first.")
        if self._state == STATE_FAILED:
            raise RuntimeError(f"Juggernaut runner failed to load: {self._error}")

        future: Future[RenderResult] = Future()
        try:
            with self._lock:
                if self._thread is None or self._state == STATE_FAILED:
                    raise RuntimeError("Renderer is stopped or unavailable")
                self._queue.put_nowait((request, future, perf_counter()))
        except queue.Full as exc:
            self._queue_rejections += 1
            raise RuntimeError(
                f"Juggernaut render queue is full ({self._queue.maxsize}); try again shortly."
            ) from exc
        return future

    def render(self, request: RenderRequest, timeout: float | None = None) -> RenderResult:
        """Submit and wait. Convenience for callers that have nothing else to do."""
        return self.submit(request).result(timeout)

    def submit_warmup(self) -> Future[RenderResult]:
        """Queue the warmup as a normal request.

        Diffusion inference starves a concurrent YOLO preview (measured: 30 fps -> 8 fps, one
        580ms frame), while loading and compiling do not. Callers showing a live preview should
        construct with warmup=False and call this once the camera is done.
        """
        return self.submit(self._warmup_request())

    # -- status --------------------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    def status(self) -> RunnerStatus:
        return RunnerStatus(
            state=self._state,
            model_id=self.model_id,
            device=self.device,
            task=self.task,
            renders_served=self._renders_served,
            failures=self._failures,
            queue_depth=self._queue.qsize(),
            startup_seconds=self._startup_seconds,
            error=self._error,
            queue_rejections=self._queue_rejections,
        )

    # -- worker --------------------------------------------------------------------

    def _run(self) -> None:
        try:
            start = perf_counter()
            self._pipeline = load_juggernaut_pipeline(
                model_id=self.model_id,
                device=self.device,
                local_files_only=self._local_files_only,
                openvino_cache_dir=self._openvino_cache_dir,
                task=self.task,
                vae_precision_hint=self._vae_precision_hint,
            )
            if self._warmup:
                self._run_warmup()
            self._startup_seconds = perf_counter() - start
            self._state = STATE_READY
        except BaseException as exc:  # a silently dead worker would hang every caller
            self._error = f"{type(exc).__name__}: {exc}"
            self._state = STATE_FAILED
            self._ready.set()
            self._drain_pending()
            return

        self._ready.set()
        self._serve()

    def _serve(self) -> None:
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                break

            request, future, queued_at = item
            queue_wait = perf_counter() - queued_at
            if not future.set_running_or_notify_cancel():
                continue  # caller abandoned the request

            self._state = STATE_RENDERING
            start = perf_counter()
            try:
                output_path = self._execute(request)
            except BaseException as exc:
                self._failures += 1
                self._state = STATE_READY
                future.set_result(
                    RenderResult(
                        ok=False,
                        output_path=None,
                        render_seconds=perf_counter() - start,
                        queue_wait_seconds=queue_wait,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                continue

            self._renders_served += 1
            self._state = STATE_READY
            future.set_result(
                RenderResult(
                    ok=True,
                    output_path=output_path,
                    render_seconds=perf_counter() - start,
                    queue_wait_seconds=queue_wait,
                )
            )

        self._pipeline = None
        self._cleanup_warmup_dir()
        self._state = STATE_STOPPED

    def _cleanup_warmup_dir(self) -> None:
        if self._warmup_dir is None:
            return
        import shutil

        shutil.rmtree(self._warmup_dir, ignore_errors=True)
        self._warmup_dir = None

    def _execute(self, request: RenderRequest) -> Path:
        if request.mode == TASK_INPAINT:
            assert request.input_image_path is not None and request.mask_image_path is not None
            assert request.strength is not None
            return render_inpaint(
                pipeline=self._pipeline,
                input_image_path=request.input_image_path,
                mask_image_path=request.mask_image_path,
                output_path=request.output_path,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                strength=request.strength,
                width=request.width,
                height=request.height,
                seed=request.seed,
                attempts=self._attempts,
            )
        if request.mode == TASK_IMG2IMG:
            assert request.input_image_path is not None and request.strength is not None
            return render_img2img(
                pipeline=self._pipeline,
                input_image_path=request.input_image_path,
                output_path=request.output_path,
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                steps=request.steps,
                guidance_scale=request.guidance_scale,
                strength=request.strength,
                width=request.width,
                height=request.height,
                seed=request.seed,
                attempts=self._attempts,
            )
        return render_text2img(
            pipeline=self._pipeline,
            output_path=request.output_path,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            steps=request.steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            seed=request.seed,
            attempts=self._attempts,
        )

    def _warmup_request(self) -> RenderRequest:
        """Build a cheap render at production resolution.

        The size must match real output: a 512x512 warmup left the first 1080x1350 render twice as
        slow as the second, because the shape-specific setup had not been paid.
        """
        import tempfile

        from PIL import Image

        width, height = self._warmup_size
        if self._warmup_dir is None:
            self._warmup_dir = Path(tempfile.mkdtemp(prefix="vpro-warmup-"))

        guide_path: Path | None = None
        mask_path: Path | None = None
        if self.task in (TASK_IMG2IMG, TASK_INPAINT):
            guide_path = self._warmup_dir / "guide.jpg"
            if not guide_path.exists():
                Image.new("RGB", (width, height), (127, 127, 127)).save(guide_path)
        if self.task == TASK_INPAINT:
            mask_path = self._warmup_dir / "mask.png"
            if not mask_path.exists():
                Image.new("L", (width, height), 255).save(mask_path)

        return RenderRequest(
            mode=self.task,
            prompt="warmup",
            output_path=self._warmup_dir / "warmup.jpg",
            steps=2,
            guidance_scale=1.0,
            width=width,
            height=height,
            seed=0,
            strength=0.5 if self.task in (TASK_IMG2IMG, TASK_INPAINT) else None,
            input_image_path=guide_path,
            mask_image_path=mask_path,
        )

    def _run_warmup(self) -> None:
        """Absorb first-render cost at boot so no guest pays it."""
        self._execute(self._warmup_request())

    def _drain_pending(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is _SHUTDOWN:
                continue
            _, future, _ = item
            if future.set_running_or_notify_cancel():
                future.set_result(
                    RenderResult(
                        ok=False,
                        output_path=None,
                        render_seconds=0.0,
                        queue_wait_seconds=0.0,
                        error=self._error,
                    )
                )
