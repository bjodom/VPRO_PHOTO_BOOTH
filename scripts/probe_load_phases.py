"""Phase 0 probe: attribute Juggernaut startup time to specific OpenVINO calls.

The instrumented harness showed ~337s inside OVDiffusionPipeline.from_pretrained(compile=False)
versus only ~12s of explicit component compilation, which is the opposite of the expected split.
This script times the underlying Core operations directly so the cost can be attributed to
read_model (IR parse/weight load) or compile_model (cache import / GPU upload).

Usage:
    python scripts/probe_load_phases.py --component unet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

DEFAULT_SNAPSHOT = Path(
    Path.home()
    / ".cache/huggingface/hub/models--OpenVINO--Juggernaut-XL-v9-fp16-ov"
    / "snapshots/fd77382a22b578485816d03d7e1e03ebcf6d4373"
)
REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--component", default="unet")
    parser.add_argument("--device", default="GPU")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=REPO_ROOT / "outputs" / "openvino_cache" / "juggernaut",
        help="CACHE_DIR to use; pass an empty directory to measure an uncached compile.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Compile without CACHE_DIR to isolate cache-import cost.",
    )
    parser.add_argument(
        "--mode",
        choices=["core", "pipeline", "img2img"],
        default="core",
        help=(
            "core: time raw OpenVINO calls. pipeline: cProfile the full optimum load. "
            "img2img: time the guided render used by the real social pipeline."
        ),
    )
    parser.add_argument("--top", type=int, default=35, help="Rows of profile output to show.")
    parser.add_argument("--guide-image", type=Path, default=None, help="Guide image for img2img mode.")
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--guidance-scale", type=float, default=4.2)
    parser.add_argument(
        "--strength",
        type=float,
        default=0.16,
        help="img2img strength; the identity-lock preset uses 0.16, balanced 0.20, stylized 0.35.",
    )
    parser.add_argument("--runs", type=int, default=3)
    return parser


def probe_img2img(args: argparse.Namespace) -> int:
    """Time the guided render actually used by the kiosk, which the text2img bench does not cover."""
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from vpro.vision.juggernaut_runtime import load_juggernaut_pipeline, render_img2img

    guide = args.guide_image
    if guide is None:
        candidates = sorted((REPO_ROOT / "outputs" / "juggernaut_prompt_tests").glob("*.jpg"))
        if not candidates:
            raise SystemExit("No guide image found; pass --guide-image.")
        guide = candidates[-1]
    print(f"guide image: {guide}")

    start = perf_counter()
    pipeline = load_juggernaut_pipeline(
        model_id="OpenVINO/Juggernaut-XL-v9-fp16-ov",
        device=args.device,
        local_files_only=True,
        openvino_cache_dir=args.cache_dir,
    )
    print(f"\nstartup: {perf_counter() - start:.2f}s")

    out_dir = REPO_ROOT / "outputs" / "img2img_probe"
    for index in range(args.runs):
        start = perf_counter()
        render_img2img(
            pipeline=pipeline,
            input_image_path=guide,
            output_path=out_dir / f"probe_{index + 1}.jpg",
            prompt="a person standing in front of the Great Pyramids of Giza, photorealistic travel photograph",
            negative_prompt="illustration, cartoon, blurry, deformed",
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            strength=args.strength,
            width=1080,
            height=1350,
            seed=1234,
        )
        elapsed = perf_counter() - start
        effective = max(1, round(args.steps * args.strength))
        print(
            f"img2img run {index + 1}: {elapsed:.2f}s "
            f"(steps={args.steps} strength={args.strength} -> ~{effective} effective steps)"
        )
    return 0


def probe_pipeline(args: argparse.Namespace) -> int:
    """cProfile OVDiffusionPipeline.from_pretrained to attribute the unexplained startup time."""
    import cProfile
    import pstats
    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from vpro.vision.juggernaut_runtime import load_juggernaut_pipeline

    timings: dict[str, float] = {}
    profiler = cProfile.Profile()
    profiler.enable()
    load_juggernaut_pipeline(
        model_id="OpenVINO/Juggernaut-XL-v9-fp16-ov",
        device=args.device,
        local_files_only=True,
        openvino_cache_dir=None if args.no_cache else args.cache_dir,
        timings=timings,
    )
    profiler.disable()

    print("\n--- phase timings ---")
    for phase, seconds in sorted(timings.items(), key=lambda item: -item[1]):
        print(f"{phase:>34}: {seconds:7.2f}s")

    print("\n--- top by cumulative time ---")
    stats = pstats.Stats(profiler)
    stats.sort_stats("cumulative").print_stats(args.top)

    print("\n--- top by internal time ---")
    stats.sort_stats("tottime").print_stats(args.top)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.mode == "pipeline":
        return probe_pipeline(args)
    if args.mode == "img2img":
        return probe_img2img(args)

    model_xml = args.snapshot_dir / args.component / "openvino_model.xml"
    if not model_xml.exists():
        raise SystemExit(f"IR not found: {model_xml}")
    weights = model_xml.with_suffix(".bin")
    print(f"component: {args.component}")
    print(f"weights:   {weights.stat().st_size / (1024 ** 3):.2f} GB")

    start = perf_counter()
    import openvino as ov

    print(f"import openvino:          {perf_counter() - start:7.2f}s")

    start = perf_counter()
    core = ov.Core()
    print(f"ov.Core():                {perf_counter() - start:7.2f}s")

    start = perf_counter()
    core.get_property(args.device, "FULL_DEVICE_NAME")
    print(f"device plugin init:       {perf_counter() - start:7.2f}s")

    start = perf_counter()
    model = core.read_model(model_xml)
    print(f"core.read_model():        {perf_counter() - start:7.2f}s")

    config: dict[str, str] = {}
    if not args.no_cache:
        cache_dir = args.cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        config["CACHE_DIR"] = str(cache_dir)
        print(f"CACHE_DIR: {cache_dir}")
    else:
        print("CACHE_DIR: disabled")

    start = perf_counter()
    core.compile_model(model, args.device, config)
    print(f"core.compile_model():     {perf_counter() - start:7.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
