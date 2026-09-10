"""Lifecycle and concurrency tests for JuggernautRunner, with the pipeline stubbed out.

Runs in about a second; no GPU or model required.

    python tests/juggernaut_runner_test.py
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from time import sleep

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpro.vision import juggernaut_runner as jr  # noqa: E402
from vpro.vision.juggernaut_runner import (  # noqa: E402
    STATE_FAILED,
    STATE_READY,
    STATE_STOPPED,
    JuggernautRunner,
)
from vpro.vision.juggernaut_types import RenderRequest  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed. {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


class Recorder:
    """Stands in for the pipeline, tracking overlap and call order."""

    def __init__(self) -> None:
        self.concurrent = 0
        self.max_concurrent = 0
        self.calls: list[str] = []
        self.lock = threading.Lock()
        self.fail_on: set[str] = set()
        self.delay = 0.01

    def render(self, **kwargs: object) -> Path:
        output_path = Path(str(kwargs["output_path"]))
        with self.lock:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
            self.calls.append(output_path.name)
        try:
            sleep(self.delay)
            if output_path.name in self.fail_on:
                raise RuntimeError("simulated render failure")
            return output_path
        finally:
            with self.lock:
                self.concurrent -= 1


def install(monkey: Recorder, load_delay: float = 0.05, load_error: Exception | None = None) -> None:
    def fake_load(**kwargs: object) -> object:
        sleep(load_delay)
        if load_error is not None:
            raise load_error
        return object()

    jr.load_juggernaut_pipeline = fake_load  # type: ignore[assignment]
    jr.render_text2img = lambda **kw: monkey.render(**kw)  # type: ignore[assignment]
    jr.render_img2img = lambda **kw: monkey.render(**kw)  # type: ignore[assignment]


def request(name: str, **overrides: object) -> RenderRequest:
    base = {
        "mode": "text2img",
        "prompt": "a test prompt",
        "output_path": Path(f"/tmp/{name}"),
    }
    base.update(overrides)
    return RenderRequest(**base)  # type: ignore[arg-type]


def test_serializes_renders() -> None:
    rec = Recorder()
    install(rec)
    with JuggernautRunner(warmup=False, queue_size=16) as runner:
        runner.wait_until_ready(timeout=5)
        futures = [runner.submit(request(f"r{i}.jpg")) for i in range(8)]
        results = [f.result(timeout=10) for f in futures]

    check("all renders succeed", all(r.ok for r in results))
    check("renders never overlap", rec.max_concurrent == 1, f"max_concurrent={rec.max_concurrent}")
    check("FIFO order preserved", rec.calls == [f"r{i}.jpg" for i in range(8)], str(rec.calls))


def test_submit_before_ready_is_queued() -> None:
    rec = Recorder()
    install(rec, load_delay=0.3)
    runner = JuggernautRunner(warmup=False).start()
    future = runner.submit(request("early.jpg"))  # queued while still loading
    result = future.result(timeout=10)
    check("request submitted during load completes", result.ok)
    check("queue wait is recorded", result.queue_wait_seconds > 0.2, str(result.queue_wait_seconds))
    runner.shutdown()


def test_render_error_keeps_worker_alive() -> None:
    rec = Recorder()
    rec.fail_on = {"bad.jpg"}
    install(rec)
    with JuggernautRunner(warmup=False) as runner:
        runner.wait_until_ready(timeout=5)
        bad = runner.submit(request("bad.jpg")).result(timeout=10)
        good = runner.submit(request("good.jpg")).result(timeout=10)
        state_after = runner.state
        status = runner.status()

    check("failed render reports ok=False", not bad.ok)
    check("failure carries an error string", "simulated render failure" in (bad.error or ""))
    check("worker survives and serves next request", good.ok)
    check("state returns to ready", state_after == STATE_READY, state_after)
    check("counters track both outcomes", status.renders_served == 1 and status.failures == 1)


def test_load_failure_is_surfaced() -> None:
    rec = Recorder()
    install(rec, load_error=RuntimeError("no model"))
    runner = JuggernautRunner(warmup=False).start()

    queued = runner.submit(request("doomed.jpg"))
    raised = False
    try:
        runner.wait_until_ready(timeout=5)
    except RuntimeError as exc:
        raised = "no model" in str(exc)

    check("wait_until_ready raises on load failure", raised)
    check("state is failed", runner.state == STATE_FAILED, runner.state)
    check("queued request is failed, not left hanging", not queued.result(timeout=5).ok)

    rejected = False
    try:
        runner.submit(request("later.jpg"))
    except RuntimeError:
        rejected = True
    check("further submits are rejected", rejected)


def test_validation_rejects_bad_requests() -> None:
    rec = Recorder()
    install(rec)
    with JuggernautRunner(warmup=False) as runner:
        runner.wait_until_ready(timeout=5)
        for label, kwargs in (
            ("zero steps", {"steps": 0}),
            ("excessive steps", {"steps": 10_000}),
            ("empty prompt", {"prompt": "   "}),
            ("huge width", {"width": 99_999}),
        ):
            rejected = False
            try:
                runner.submit(request("x.jpg", **kwargs))
            except ValueError:
                rejected = True
            check(f"rejects {label}", rejected)

        mismatched = False
        try:
            runner.submit(
                request("y.jpg", mode="img2img", strength=0.2, input_image_path=Path("/tmp/g.jpg"))
            )
        except ValueError:
            mismatched = True
        check("rejects task mismatch", mismatched)


def test_queue_full_is_rejected_fast() -> None:
    rec = Recorder()
    rec.delay = 0.4
    install(rec)
    with JuggernautRunner(warmup=False, queue_size=2) as runner:
        runner.wait_until_ready(timeout=5)
        for i in range(3):
            try:
                runner.submit(request(f"q{i}.jpg"))
            except RuntimeError:
                pass
        overflowed = False
        try:
            for i in range(10):
                runner.submit(request(f"overflow{i}.jpg"))
        except RuntimeError as exc:
            overflowed = "queue is full" in str(exc)
        check("full queue rejects rather than blocking", overflowed)
        check("queue rejection is counted", runner.status().queue_rejections >= 1)


def test_cancelled_request_is_skipped() -> None:
    rec = Recorder()
    rec.delay = 0.3
    install(rec)
    with JuggernautRunner(warmup=False, queue_size=8) as runner:
        runner.wait_until_ready(timeout=5)
        first = runner.submit(request("busy.jpg"))
        second = runner.submit(request("cancelled.jpg"))
        third = runner.submit(request("after.jpg"))
        check("pending request can be cancelled", second.cancel())
        first.result(timeout=10)
        third.result(timeout=10)

    check("cancelled render never ran", "cancelled.jpg" not in rec.calls, str(rec.calls))
    check("later request still ran", "after.jpg" in rec.calls)


def test_shutdown_stops_worker() -> None:
    rec = Recorder()
    install(rec)
    runner = JuggernautRunner(warmup=False).start()
    runner.wait_until_ready(timeout=5)
    runner.submit(request("last.jpg")).result(timeout=10)
    runner.shutdown()
    sleep(0.1)
    check("state is stopped after shutdown", runner.state == STATE_STOPPED, runner.state)
    check("shutdown is idempotent", (runner.shutdown() or True))


def test_warmup_failure_does_not_block() -> None:
    rec = Recorder()
    rec.fail_on = {"warmup.jpg"}
    install(rec)
    with JuggernautRunner(warmup=True) as runner:
        runner.wait_until_ready(timeout=5)
        result = runner.submit(request("real.jpg")).result(timeout=10)
    check("runner serves requests after a failed warmup", result.ok)


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        print(f"\n{test.__name__}")
        test()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
