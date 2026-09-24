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
    parser.add_argument("--model-id", default="OpenVINO/dreamshaper-8-inpainting-int8-ov")
    parser.add_argument("--scene", default="fuji")
    parser.add_argument("--scene-image", type=Path, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1350)
    parser.add_argument("--attempts", type=int, default=1)
    parser.add_argument("--disable-safety-checker", action="store_true")
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


def build_scene_aware_guide(image, foreground_mask, scene, scene_image_path, output_size):
    import cv2
    import numpy as np

    width, height = output_size
    scene_image = cv2.imread(str(scene_image_path))
    if scene_image is None:
        raise RuntimeError("Could not read the portrait scene image")
    source_h, source_w = scene_image.shape[:2]
    target_w, target_h = output_size
    target_ratio = target_w / float(target_h)
    source_ratio = source_w / float(source_h)
    if source_ratio > target_ratio:
        crop_w = max(1, int(round(source_h * target_ratio)))
        center_x = int(round(source_w * scene.scene_crop_center_x))
        left = max(0, min(source_w - crop_w, center_x - crop_w // 2))
        scene_image = scene_image[:, left:left + crop_w]
    elif source_ratio < target_ratio:
        crop_h = max(1, int(round(source_w / target_ratio)))
        top = max(0, (source_h - crop_h) // 2)
        scene_image = scene_image[top:top + crop_h, :]
    canvas = cv2.resize(scene_image, (width, height), interpolation=cv2.INTER_AREA)

    ys, xs = np.where(foreground_mask > 32)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("RMBG mask has no foreground pixels")
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    subject = image[y1:y2, x1:x2]
    alpha = foreground_mask[y1:y2, x1:x2]
    target_h = int(height * scene.subject_scale)
    scale = target_h / float(max(1, subject.shape[0]))
    out_w = max(2, int(round(subject.shape[1] * scale)))
    out_h = max(2, int(round(subject.shape[0] * scale)))
    subject = cv2.resize(subject, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.resize(alpha, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    paste_x = int(width * scene.subject_center_x) - out_w // 2
    paste_y = int(height * scene.feet_y) - out_h

    x_start, y_start = max(0, paste_x), max(0, paste_y)
    x_end, y_end = min(width, paste_x + out_w), min(height, paste_y + out_h)
    sx_start, sy_start = x_start - paste_x, y_start - paste_y
    sx_end, sy_end = sx_start + (x_end - x_start), sy_start + (y_end - y_start)
    subject = subject[sy_start:sy_end, sx_start:sx_end]
    alpha = alpha[sy_start:sy_end, sx_start:sx_end].astype(np.float32) / 255.0
    roi = canvas[y_start:y_end, x_start:x_end].astype(np.float32)
    canvas[y_start:y_end, x_start:x_end] = (
        subject.astype(np.float32) * alpha[..., None] + roi * (1.0 - alpha[..., None])
    ).astype(np.uint8)

    locked = np.zeros((height, width), dtype=np.uint8)
    locked[y_start:y_end, x_start:x_end] = (alpha * 255).astype(np.uint8)
    return canvas, 255 - locked


def restore_subject(output_path, guide_path, mask_path):
    import cv2
    import numpy as np

    output = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    guide = cv2.imread(str(guide_path), cv2.IMREAD_COLOR)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if output is None or guide is None or mask is None:
        raise RuntimeError("Could not restore subject in diagnostic output")
    locked = cv2.GaussianBlur(255 - mask, (5, 5), 0).astype(np.float32) / 255.0
    restored = (
        guide.astype(np.float32) * locked[..., None]
        + output.astype(np.float32) * (1.0 - locked[..., None])
    ).clip(0, 255).astype(np.uint8)
    if not cv2.imwrite(str(output_path), restored):
        raise RuntimeError(f"Could not write restored diagnostic output: {output_path}")


def run_sequence(runtime, runner, image, request, runs, timeout, record, scene=None, scene_image_path=None) -> bool:
    import cv2

    if scene is None:
        from vpro.kiosk.scenes import get_scene
        scene = get_scene("fuji")
    if scene_image_path is None:
        scene_image_path = REPO_ROOT / "assets/scenes/portrait_scene_1080x1350.jpg"

    runtime.warmup(runs=1)
    baseline = check_segmentation(runtime, image, "before_load", record)
    if baseline is None:
        return False
    guide, mask = build_scene_aware_guide(
        image, baseline, scene, scene_image_path, (request.width, request.height)
    )
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
            if result.ok and result.output_path is not None and result.output_path.exists():
                restore_subject(result.output_path, request.input_image_path, request.mask_image_path)
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
    scene_image_path = args.scene_image
    if scene_image_path is None:
        candidate = REPO_ROOT / "src/vpro/kiosk/static" / f"{args.scene}.jpg"
        scene_image_path = candidate if candidate.exists() else REPO_ROOT / "assets/scenes/portrait_scene_1080x1350.jpg"
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.steps < 1 or args.width < 64 or args.height < 64 or args.attempts < 1:
        parser.error("steps, width, height, and attempts must be positive")
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
           scene=args.scene, scene_image=str(scene_image_path), steps=args.steps, strength=0.99, guidance_scale=5.0,
           width=args.width, height=args.height, attempts=args.attempts,
           disable_safety_checker=args.disable_safety_checker, local_files_only=True)
    try:
        scene = get_scene(args.scene)
        request = RenderRequest(
            mode="inpaint", prompt=scene.background_prompt(),
            negative_prompt=scene.background_negative_prompt(),
            output_path=output_dir / "render.jpg", input_image_path=output_dir / "guide.png",
            mask_image_path=output_dir / "mask.png", steps=30, guidance_scale=5.0,
            strength=0.99, width=args.width, height=args.height, seed=args.seed,
        ).validated()
        runtime = load_rmbg_runtime(args.rmbg_model_dir, device=args.rmbg_device)
        runner = JuggernautRunner(
            model_id=args.model_id, device=args.render_device, local_files_only=True,
            openvino_cache_dir=REPO_ROOT / "outputs/openvino_cache/juggernaut",
            task="inpaint", warmup=False, warmup_width=request.width,
            warmup_height=request.height, attempts=args.attempts,
            disable_safety_checker=args.disable_safety_checker,
        )
        passed = run_sequence(
            runtime, runner, image, request, args.runs, args.timeout, record, scene, scene_image_path
        )
    except Exception as error:
        record(phase="setup", kind="execution", ok=False,
               error=f"{type(error).__name__}: {error}")
        passed = False
    record(kind="summary", ok=passed)
    print(f"{'PASS' if passed else 'FAIL'}: diagnostics in {metrics_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())