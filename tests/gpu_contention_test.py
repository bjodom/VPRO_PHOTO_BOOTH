"""Measure whether preloading Juggernaut degrades the live YOLO pose preview.

_run_social_pipeline starts the Juggernaut load during camera capture so the ~29s startup is
hidden. Both workloads target the same GPU, so this checks what that costs the preview.

Worth re-running on each target system: the answer depends on VRAM and GPU class. Measured on an
Arc 390 (16GB) laptop; the target booth is an Arc B70 (32GB), which should fare better.

Run the baseline first, then the contended case:

    python tests/gpu_contention_test.py --seconds 45
    python tests/gpu_contention_test.py --seconds 45 --preload
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--capture-width", type=int, default=2560)
    parser.add_argument("--capture-height", type=int, default=1440)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=REPO_ROOT / "models" / "yolo26" / "yolo26x-pose_openvino_model",
    )
    parser.add_argument("--yolo-device", default="intel:gpu")
    parser.add_argument("--juggernaut-device", default="GPU")
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Start a JuggernautRunner partway through to reproduce the kiosk preload.",
    )
    parser.add_argument(
        "--preload-after",
        type=float,
        default=5.0,
        help="Seconds of clean preview before the Juggernaut load starts.",
    )
    parser.add_argument("--show", action="store_true", help="Display the annotated preview.")
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run the warmup render during the preview. Off by default because that is what the "
            "kiosk does; turn it on to reproduce the contention it causes."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import cv2

    from vpro.backends.factory import build_backend

    print(f"loading YOLO pose backend on {args.yolo_device}")
    backend = build_backend("openvino")
    backend.load_model(args.model_path)

    capture = cv2.VideoCapture(args.camera_index, cv2.CAP_DSHOW)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.capture_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.capture_height)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera index {args.camera_index}")
    actual = (
        int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print(f"camera open at {actual[0]}x{actual[1]}")

    # Warm the preview so first-inference cost does not land in the measurement.
    for _ in range(10):
        ok, frame = capture.read()
        if ok:
            backend.predict({"source": frame, "device": args.yolo_device, "verbose": False})

    runner = None
    runner_started_at: float | None = None
    runner_ready_at: float | None = None

    frame_times: list[tuple[float, float]] = []  # (elapsed_at_end, inference_seconds)
    start = perf_counter()
    try:
        while True:
            elapsed = perf_counter() - start
            if elapsed >= args.seconds:
                break

            if (
                args.preload
                and runner is None
                and elapsed >= args.preload_after
            ):
                from vpro.vision.juggernaut_runner import JuggernautRunner

                print(f"[{elapsed:5.1f}s] starting Juggernaut load")
                runner = JuggernautRunner(
                    device=args.juggernaut_device,
                    openvino_cache_dir=REPO_ROOT / "outputs" / "openvino_cache" / "juggernaut",
                    task="img2img",
                    warmup=args.warmup,
                    warmup_width=1080,
                    warmup_height=1350,
                ).start()
                runner_started_at = elapsed

            ok, frame = capture.read()
            if not ok:
                continue

            t0 = perf_counter()
            results = backend.predict(
                {"source": frame, "device": args.yolo_device, "verbose": False}
            )
            inference = perf_counter() - t0
            frame_times.append((perf_counter() - start, inference))

            if runner is not None and runner_ready_at is None and runner.state == "ready":
                runner_ready_at = perf_counter() - start
                print(f"[{runner_ready_at:5.1f}s] Juggernaut ready")

            if args.show and isinstance(results, list) and results:
                cv2.imshow("contention probe", results[0].plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        if args.show:
            cv2.destroyAllWindows()
        if runner is not None:
            runner.shutdown()

    report(frame_times, runner_started_at, runner_ready_at)
    return 0


def summarize(label: str, samples: list[float]) -> None:
    if not samples:
        print(f"{label:24s} (no frames)")
        return
    ordered = sorted(samples)
    p50 = statistics.median(ordered)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    print(
        f"{label:24s} frames={len(samples):4d}  fps={1 / statistics.fmean(samples):5.2f}  "
        f"p50={p50 * 1000:6.1f}ms  p95={p95 * 1000:6.1f}ms  max={max(ordered) * 1000:7.1f}ms"
    )


def report(
    frame_times: list[tuple[float, float]],
    started_at: float | None,
    ready_at: float | None,
) -> None:
    print("\n--- per-second preview FPS ---")
    if frame_times:
        last_second = int(frame_times[-1][0])
        for second in range(last_second + 1):
            bucket = [d for (t, d) in frame_times if second <= t < second + 1]
            marker = ""
            if started_at is not None and second == int(started_at):
                marker = "  <-- Juggernaut load starts"
            elif ready_at is not None and second == int(ready_at):
                marker = "  <-- Juggernaut ready"
            elif started_at is not None and ready_at is not None and started_at < second < ready_at:
                marker = "  (loading)"
            bar = "#" * len(bucket)
            print(f"  {second:3d}s {len(bucket):3d} fps {bar}{marker}")

    print("\n--- summary ---")
    if started_at is None:
        summarize("baseline", [d for _, d in frame_times])
    else:
        summarize("before load", [d for t, d in frame_times if t < started_at])
        end = ready_at if ready_at is not None else frame_times[-1][0]
        summarize("during load", [d for t, d in frame_times if started_at <= t < end])
        if ready_at is not None:
            summarize("after ready", [d for t, d in frame_times if t >= ready_at])


if __name__ == "__main__":
    raise SystemExit(main())
