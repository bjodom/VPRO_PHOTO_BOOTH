"""Text-to-image only bench for the OpenVINO Juggernaut pipeline.

Focused on prompt and quality-setting iteration: no captured/guide image is used.
Every run appends its settings to a log file so runs can be compared later.

Example:
    python tests/juggernaut_test.py --preset stylized --steps 32 --seed 1234
    python tests/juggernaut_test.py --scene pyramids --runs 3
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpro.vision.juggernaut_runtime import load_juggernaut_pipeline, render_text2img  # noqa: E402

PHOTOREAL_STYLE = (
    "photorealistic travel photograph, shot on Canon EOS R5, 35mm lens, f/8, "
    "natural golden hour light, ultra detailed textures, high dynamic range, "
    "sharp focus, realistic colors, 8k"
)

SCENES: dict[str, str] = {
    "pyramids": "the Great Pyramids of Giza rising over the desert, camels and dunes in the foreground, clear sky",
    "eiffel": "the Eiffel Tower in Paris seen from the Trocadero, wet cobblestones, soft morning haze over the city",
    "colosseum": "the Roman Colosseum at sunrise, weathered stone arches, empty piazza, long shadows",
    "tajmahal": "the Taj Mahal reflected in its long water pool, white marble glowing at dawn, thin mist",
    "machu": "Machu Picchu terraces with Huayna Picchu behind, drifting clouds in the valley, lush green ridges",
    "santorini": "white-washed Santorini cliffside village with blue domes above the Aegean Sea at sunset",
    "fuji": "Mount Fuji behind Chureito Pagoda, cherry blossoms in bloom, crisp snow-capped peak",
    "goldengate": "the Golden Gate Bridge emerging from low fog, calm bay water, warm afternoon light",
}

DEFAULT_SCENE = "pyramids"

DEFAULT_NEGATIVE_PROMPT = (
    "illustration, painting, drawing, cartoon, anime, 3d render, cgi, video game, "
    "blurry, lowres, jpeg artifacts, oversaturated, overexposed, distorted perspective, "
    "warped architecture, watermark, text, signature, logo, people staring at camera, deformed"
)

PRESETS: dict[str, dict[str, float]] = {
    "fast": {"steps": 16, "guidance_scale": 3.5},
    "balanced": {"steps": 24, "guidance_scale": 4.5},
    "quality": {"steps": 36, "guidance_scale": 5.5},
    "stylized": {"steps": 30, "guidance_scale": 6.0},
}


def build_prompt(scene_key: str, custom_prompt: str | None) -> str:
    if custom_prompt:
        return custom_prompt
    return f"{SCENES[scene_key]}, {PHOTOREAL_STYLE}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-id", default="OpenVINO/Juggernaut-XL-v9-fp16-ov")
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--local-only", action="store_true", help="Use cached/local model files only.")
    parser.add_argument("--scene", default=DEFAULT_SCENE, choices=sorted(SCENES), help="Built-in landmark scene.")
    parser.add_argument("--prompt", default=None, help="Full prompt override (bypasses --scene and style suffix).")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--preset", default="balanced", choices=sorted(PRESETS))
    parser.add_argument("--steps", type=int, default=None, help="Override preset step count.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Override preset guidance scale.")
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1350)
    parser.add_argument("--seed", type=int, default=None, help="Seed for run 1; later runs increment by 1.")
    parser.add_argument("--runs", type=int, default=1, help="Number of renders in this invocation.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "juggernaut_prompt_tests")
    parser.add_argument("--log-file", type=Path, default=REPO_ROOT / "outputs" / "juggernaut_test_log.txt")
    return parser


def validate(args: argparse.Namespace, steps: int) -> None:
    if steps < 1:
        raise ValueError("--steps must be >= 1")
    if args.width < 64 or args.height < 64:
        raise ValueError("--width and --height must be >= 64")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")


def next_run_number(log_path: Path) -> int:
    if not log_path.exists():
        return 1
    count = 0
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("run "):
            count += 1
    return count + 1


def append_log(log_path: Path, run_number: int, settings: dict[str, object]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"run {run_number} - {datetime.now().isoformat(timespec='seconds')}"]
    for key, value in settings.items():
        lines.append(f"    {key}: {value}")
    lines.append("")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    profile = PRESETS[args.preset]
    steps = args.steps if args.steps is not None else int(profile["steps"])
    guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else float(profile["guidance_scale"])
    )
    validate(args, steps)

    prompt = build_prompt(args.scene, args.prompt)
    log_path = args.log_file.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Juggernaut pipeline: {args.model_id} (device={args.device}, local_only={args.local_only})")
    load_start = perf_counter()
    pipeline = load_juggernaut_pipeline(
        model_id=args.model_id,
        device=args.device,
        local_files_only=args.local_only,
    )
    load_sec = perf_counter() - load_start
    print(f"Pipeline ready in {load_sec:.2f}s")

    for index in range(args.runs):
        run_number = next_run_number(log_path)
        seed = None if args.seed is None else args.seed + index
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"run{run_number:04d}_{args.scene}_{stamp}.jpg"

        print(f"\n[run {run_number}] {steps} steps, cfg {guidance_scale}, seed {seed}")
        render_start = perf_counter()
        saved_path = render_text2img(
            pipeline=pipeline,
            output_path=output_path,
            prompt=prompt,
            negative_prompt=args.negative_prompt,
            steps=steps,
            guidance_scale=guidance_scale,
            width=args.width,
            height=args.height,
            seed=seed,
        )
        render_sec = perf_counter() - render_start
        print(f"[run {run_number}] saved {saved_path} in {render_sec:.2f}s")

        append_log(
            log_path,
            run_number,
            {
                "model_id": args.model_id,
                "device": args.device,
                "local_only": args.local_only,
                "scene": "custom" if args.prompt else args.scene,
                "preset": args.preset,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": args.width,
                "height": args.height,
                "seed": seed,
                "render_seconds": round(render_sec, 2),
                "output": saved_path,
                "prompt": json.dumps(prompt),
                "negative_prompt": json.dumps(args.negative_prompt),
            },
        )

    print(f"\nSettings log appended: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
