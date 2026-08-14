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
    render_text2img,
)
from .juggernaut_types import TASK_IMG2IMG, TASK_TEXT2IMG, RenderRequest, RenderResult

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


class JuggernautRunner:
    """Owns a compiled pipeline and serves render requests from a queue."""

    def __init__(
        self,
        model_id: str = "OpenVINO/Juggernaut-XL-v9-fp16-ov",
        device: str = "GPU",
        local_files_only: bool = True,
        openvino_cache_dir: Path | None = None,
        task: str = TASK_TEXT2IMG,
        queue_size: int = 8,
        warmup: bool = True,
        warmup_width: int = 1080,
        warmup_height: int = 1350,
        attempts: int = DEFAULT_RENDER_ATTEMPTS,
    ) -> None:
        if task not in (TASK_TEXT2IMG, TASK_IMG2IMG):
            raise ValueError(f"task must be '{TASK_TEXT2IMG}' or '{TASK_IMG2IMG}', got {task!r}")

        self.model_id = model_id
        self.device = device
        self.task = task
        self._local_files_only = local_files_only
        self._openvino_cache_dir = openvino_cache_dir
        self._warmup = warmup
        self._warmup_size = (int(warmup_width), int(warmup_height))
        self._attempts = attempts

        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max(1, int(queue_size)))
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._pipeline: Any = None

        self._state = STATE_IDLE
        self._error: str | None = None
        self._startup_seconds: float | None = None
        self._renders_served = 0
        self._failures = 0

    # -- lifecycle -----------------------------------------------------------------

    def start(self) -> JuggernautRunner:
        """Begin loading on a background thread and return immediately."""
        with self._lock:
            if self._thread is not None:
                return self
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
        self._queue.put(_SHUTDOWN)
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
            self._queue.put_nowait((request, future, perf_counter()))
        except queue.Full as exc:
            raise RuntimeError(
                f"Juggernaut render queue is full ({self._queue.maxsize}); try again shortly."
            ) from exc
        return future

    def render(self, request: RenderRequest, timeout: float | None = None) -> RenderResult:
        """Submit and wait. Convenience for callers that have nothing else to do."""
        return self.submit(request).result(timeout)

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
        self._state = STATE_STOPPED

    def _execute(self, request: RenderRequest) -> Path:
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

    def _run_warmup(self) -> None:
        """Absorb first-render cost at boot so no guest pays it.

        Must run at production resolution: a 512x512 warmup left a 1080x1350 first render 2x
        slower than the second, because the shape-specific setup had not been paid.
        """
        import tempfile

        from PIL import Image

        width, height = self._warmup_size
        with tempfile.TemporaryDirectory(prefix="vpro-warmup-") as tmp:
            tmp_dir = Path(tmp)
            output_path = tmp_dir / "warmup.jpg"
            common = {
                "pipeline": self._pipeline,
                "output_path": output_path,
                "prompt": "warmup",
                "negative_prompt": None,
                "steps": 2,
                "guidance_scale": 1.0,
                "width": width,
                "height": height,
                "seed": 0,
                "attempts": 1,
            }
            try:
                if self.task == TASK_IMG2IMG:
                    guide_path = tmp_dir / "guide.jpg"
                    Image.new("RGB", (width, height), (127, 127, 127)).save(guide_path)
                    render_img2img(input_image_path=guide_path, strength=0.5, **common)
                else:
                    render_text2img(**common)
            except Exception as exc:
                # A failed warmup must not stop the runner from serving real requests.
                print(f"Juggernaut warmup render failed (continuing): {exc}", flush=True)

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
