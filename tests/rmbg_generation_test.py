"""Opt-in RMBG/Juggernaut coexistence regression using a saved photo, never a camera.

    python tests/rmbg_generation_test.py --image outputs/kiosk/capture_example.jpg

Run with the kiosk stopped. Use --rmbg-device CPU in a separate process for a control.
This isolates segmentation and inpainting; it does not exercise YOLO or delivery.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--rmbg-device", default="GPU")
    parser.add_argument("--render-device", default="GPU")
    parser.add_argument("--rmbg-model-dir", type=Path,
                        default=REPO_ROOT / "models/rmbg/rmbg-1.4")
    parser.add_argument("--model-id", default="OpenVINO/Juggernaut-XL-v9-fp16-ov")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="New directory for photos and JSONL diagnostics; must not exist.")
    return parser


def check_segmentation(runtime, image, phase, record):
    import numpy as np

    started = perf_counter()
    details = {}
    try:
        tensor, original_size = runtime.preprocess_image(image)
        raw = np.asarray(runtime.infer(tensor)[runtime._output])
        finite = np.isfinite(raw)
        valid = raw[finite]
        details = {
            "nonfinite": int(raw.size - np.count_nonzero(finite)),
            "raw_min": float(valid.min()) if valid.size else None,
            "raw_max": float(valid.max()) if valid.size else None,
        }
        mask = runtime.postprocess_mask(raw, original_size)
        if mask.shape != image.shape[:2] or mask.min() == mask.max():
            raise RuntimeError("RMBG returned an empty, constant, or incorrectly sized mask")
    except Exception as error:
        record(phase=phase, kind="segmentation", ok=False,
               error=f"{type(error).__name__}: {error}",
               seconds=perf_counter() - started, **details)
        return None
    record(phase=phase, kind="segmentation", ok=True,
           seconds=perf_counter() - started, **details)
    return mask


def run_sequence(runtime, runner, image, request, runs, timeout, record) -> bool:
    import cv2

    runtime.warmup(runs=1)
    baseline = check_segmentation(runtime, image, "before_load", record)
    if baseline is None:
        return False
    guide = cv2.resize(image, (request.width, request.height))
    mask = cv2.resize(255 - baseline, (request.width, request.height))
    for path, pixels in ((request.input_image_path, guide), (request.mask_image_path, mask)):
        if not cv2.imwrite(str(path), pixels):
            raise RuntimeError(f"Could not write fixture: {path}")

    phase = "load"
    passed = True
    try:
        runner.start()
        if not runner.wait_until_ready(timeout=timeout):
            raise TimeoutError("Juggernaut loading exceeded the test timeout")
        passed &= check_segmentation(runtime, image, "after_load", record) is not None

        phase = "warmup"
        result = runner.submit_warmup().result(timeout=timeout)
        record(phase=phase, kind="render", ok=result.ok, error=result.error)
        passed &= result.ok
        passed &= check_segmentation(runtime, image, "after_warmup", record) is not None
        for index in range(1, runs + 1):
            phase = f"render_{index}"
            current = replace(request, output_path=request.output_path.with_name(f"{phase}.jpg"))
            result = runner.render(current, timeout=timeout)
            record(phase=phase, kind="render", ok=result.ok, error=result.error,
                   seconds=result.render_seconds)
            passed &= result.ok
            passed &= check_segmentation(runtime, image, f"after_{phase}", record) is not None
    except Exception as error:
        record(phase=phase, kind="execution", ok=False,
               error=f"{type(error).__name__}: {error}")
        return False
    finally:
        runner.shutdown(timeout=10)
    return bool(passed)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.runs < 2:
        parser.error("--runs must be at least 2 to exercise repeated renders")
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be positive and finite")
    if not args.image.is_file():
        parser.error(f"Saved input image not found: {args.image}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["OPENVINO_TELEMETRY_OPT_OUT"] = "1"

    import cv2
    from vpro.kiosk.scenes import get_scene
    from vpro.vision.juggernaut_runner import JuggernautRunner
    from vpro.vision.juggernaut_types import RenderRequest
    from vpro.vision.rmbg_runtime import load_rmbg_runtime

    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Could not decode image: {args.image}")
    output_dir = args.output_dir or (
        REPO_ROOT / "outputs" / f"rmbg_generation_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = output_dir / "metrics.jsonl"

    def record(**entry):
        line = json.dumps(entry, allow_nan=False)
        with metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    record(kind="settings", rmbg_device=args.rmbg_device, render_device=args.render_device,
           image=str(args.image.resolve()), model_id=args.model_id,
           rmbg_model_dir=str(args.rmbg_model_dir), runs=args.runs, seed=args.seed,
           steps=30, strength=0.99, guidance_scale=5.0, width=1080, height=1350,
           attempts=1, local_files_only=True)
    try:
        scene = get_scene("fuji")
        request = RenderRequest(
            mode="inpaint", prompt=scene.background_prompt(),
            negative_prompt=scene.background_negative_prompt(),
            output_path=output_dir / "render.jpg", input_image_path=output_dir / "guide.png",
            mask_image_path=output_dir / "mask.png", steps=30, guidance_scale=5.0,
            strength=0.99, width=1080, height=1350, seed=args.seed,
        ).validated()
        runtime = load_rmbg_runtime(args.rmbg_model_dir, device=args.rmbg_device)
        runner = JuggernautRunner(
            model_id=args.model_id, device=args.render_device, local_files_only=True,
            openvino_cache_dir=REPO_ROOT / "outputs/openvino_cache/juggernaut",
            task="inpaint", warmup=False, warmup_width=request.width,
            warmup_height=request.height, attempts=1,
        )
        passed = run_sequence(runtime, runner, image, request, args.runs, args.timeout, record)
    except Exception as error:
        record(phase="setup", kind="execution", ok=False,
               error=f"{type(error).__name__}: {error}")
        passed = False
    record(kind="summary", ok=passed)
    print(f"{'PASS' if passed else 'FAIL'}: diagnostics in {metrics_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())