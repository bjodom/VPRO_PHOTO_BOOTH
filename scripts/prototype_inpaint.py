"""Inpainting prototype: generate the destination around a preserved guest.

img2img applies uniform denoise to the whole frame, so it cannot both invent a landmark and keep
a face. Inpainting can: the subject is locked, everything else is generated at full strength, and
the model resolves the boundary itself.

The mask is feathered so the silhouette blends rather than reading as a cut-out.

    python scripts/prototype_inpaint.py --capture outputs/kiosk/capture_*.jpg --scene eiffel
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpro.backends.factory import build_backend  # noqa: E402
from vpro.kiosk.scenes import DEFAULT_NEGATIVE_PROMPT, get_scene  # noqa: E402

# A second person is the classic inpaint failure here: the guest is masked out, so any mention of
# people in the prompt gets painted into the background.
BACKGROUND_NEGATIVE = (
    f"{DEFAULT_NEGATIVE_PROMPT}, another person, second person, people, crowd, faces, "
    "tourists, mannequin, statue of a person, indoor, table, desk, furniture"
)
from vpro.vision.juggernaut_runtime import (  # noqa: E402
    feather_mask,
    load_juggernaut_pipeline,
    render_img2img,
    render_inpaint,
)
from vpro.vision.portrait_compositor import (  # noqa: E402
    cleanup_mask_for_primary_subject,
    compose_portrait,
    select_primary_subject,
)
from vpro.vision.rmbg_runtime import load_rmbg_runtime  # noqa: E402

OUT = REPO_ROOT / "outputs" / "inpaint_prototype"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, default=None, help="Guest photo; default newest.")
    parser.add_argument("--scene", default="eiffel")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument(
        "--strength",
        type=float,
        default=0.99,
        help="Inpaint denoise. Near 1.0 because the masked region should be fully generated.",
    )
    parser.add_argument("--feather", type=int, default=8, help="Mask feather radius in pixels.")
    parser.add_argument(
        "--expand",
        type=int,
        default=-6,
        help="Dilate (positive) or erode (negative) the lock. Negative discards matte edge pixels "
        "contaminated by the backdrop colour, which is what causes a halo.",
    )
    parser.add_argument(
        "--neutral-canvas",
        action="store_true",
        help="Composite onto flat grey instead of a pre-generated background. Grey bleeds into "
        "semi-transparent hair and shows up as a halo.",
    )
    parser.add_argument(
        "--subject-prompt",
        action="store_true",
        help="Describe a person in the prompt. Off by default: the guest already exists, so "
        "mentioning a person makes the model paint another one into the background.",
    )
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--compare-img2img",
        action="store_true",
        help="Also run the current guided pass for a side-by-side.",
    )
    return parser


def newest_capture() -> Path:
    candidates = sorted((REPO_ROOT / "outputs" / "kiosk").glob("capture_*.jpg"))
    if not candidates:
        raise SystemExit("No captures found; run the kiosk once or pass --capture.")
    return candidates[-1]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)

    capture = (args.capture or newest_capture()).expanduser().resolve()
    scene = get_scene(args.scene)
    print(f"capture: {capture.name}")
    print(f"scene:   {scene.label} - {scene.description}\n")

    source = cv2.imread(str(capture), cv2.IMREAD_COLOR)
    if source is None:
        raise SystemExit(f"Could not read {capture}")

    backend = build_backend("openvino")
    backend.load_model(REPO_ROOT / "models" / "yolo26" / "yolo26x-pose_openvino_model")

    started = perf_counter()
    rmbg = load_rmbg_runtime(REPO_ROOT / "models" / "rmbg" / "rmbg-1.4", device="AUTO")
    raw_mask, foreground = rmbg.segment(source)
    results = backend.predict({"source": source, "device": "intel:gpu", "verbose": False})
    primary = select_primary_subject(results[0], source.shape[:2])
    clean = cleanup_mask_for_primary_subject(raw_mask, primary.bbox_xyxy, quality="high")
    foreground[:, :, 3] = clean

    # Compositing over a plausible background keeps matte edges from picking up a flat colour.
    canvas_path = None if args.neutral_canvas else scene.pick_background(REPO_ROOT / "assets" / "scenes")
    if canvas_path is None:
        canvas_path = OUT / "neutral.png"
        cv2.imwrite(str(canvas_path), np.full((1350, 1080, 3), 128, dtype=np.uint8))
        print("canvas:  neutral grey (no pre-generated background for this scene)")
    else:
        print(f"canvas:  {canvas_path.name}")

    composition = compose_portrait(
        source_bgr=source,
        foreground_bgra=foreground,
        cleaned_mask=clean,
        yolo_result=results[0],
        primary=primary,
        scene_path=canvas_path,
        prop_path=REPO_ROOT / "assets" / "props" / "vpro_laptop.png",
        output_size=(1080, 1350),
        mask_quality="high",
    )
    print(f"compose: {perf_counter() - started:.2f}s")

    guide_path = OUT / "guide.jpg"
    cv2.imwrite(str(guide_path), composition.image_bgr)

    coverage = composition.coverage_mask
    if coverage is None:
        raise SystemExit("Compositor did not return a coverage mask.")

    locked = feather_mask(coverage, feather_px=args.feather, expand_px=args.expand)
    # diffusers repaints white, so invert: keep the guest, generate the rest.
    inpaint_mask = 255 - locked
    mask_path = OUT / "mask.png"
    cv2.imwrite(str(mask_path), inpaint_mask)
    cv2.imwrite(str(OUT / "coverage.png"), coverage)
    print(f"mask:    feather={args.feather}px expand={args.expand}px -> {mask_path.name}")

    pipeline = load_juggernaut_pipeline(
        model_id="OpenVINO/Juggernaut-XL-v9-fp16-ov",
        device=args.device,
        openvino_cache_dir=REPO_ROOT / "outputs" / "openvino_cache" / "juggernaut",
        task="inpaint",
    )

    prompt = scene.prompt() if args.subject_prompt else scene.background_prompt()
    negative = DEFAULT_NEGATIVE_PROMPT if args.subject_prompt else BACKGROUND_NEGATIVE
    print(f"prompt:  {prompt[:90]}...")

    started = perf_counter()
    out_path = render_inpaint(
        pipeline=pipeline,
        input_image_path=guide_path,
        mask_image_path=mask_path,
        output_path=OUT / f"inpaint_{scene.key}.jpg",
        prompt=prompt,
        negative_prompt=negative,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        width=1080,
        height=1350,
        seed=args.seed,
    )
    print(f"\ninpaint: {perf_counter() - started:.2f}s -> {out_path}")

    if args.compare_img2img:
        img2img_pipeline = load_juggernaut_pipeline(
            model_id="OpenVINO/Juggernaut-XL-v9-fp16-ov",
            device=args.device,
            openvino_cache_dir=REPO_ROOT / "outputs" / "openvino_cache" / "juggernaut",
            task="img2img",
        )
        started = perf_counter()
        current = render_img2img(
            pipeline=img2img_pipeline,
            input_image_path=guide_path,
            output_path=OUT / f"img2img_{scene.key}.jpg",
            prompt=scene.prompt(),
            negative_prompt=DEFAULT_NEGATIVE_PROMPT,
            steps=28,
            guidance_scale=4.2,
            strength=0.16,
            width=1080,
            height=1350,
            seed=args.seed,
        )
        print(f"img2img: {perf_counter() - started:.2f}s -> {current}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
