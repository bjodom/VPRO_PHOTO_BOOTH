"""Pre-generate landmark backgrounds for each kiosk destination.

The guest is composited onto one of these, rather than the diffusion pass being asked to invent a
landmark from a studio backdrop. At the guided strength the kiosk uses (0.16) it cannot do that,
which is why every destination previously looked the same.

Pre-generating also removes the background render from the guest's wait entirely.

    python scripts/generate_scene_backgrounds.py --variants 3
    python scripts/generate_scene_backgrounds.py --scene eiffel --variants 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpro.kiosk.scenes import BACKGROUND_ROOT, DEFAULT_NEGATIVE_PROMPT, SCENES  # noqa: E402
from vpro.vision.juggernaut_types import RenderRequest  # noqa: E402

# People in a background would fight the composited guest.
BACKGROUND_NEGATIVE = f"{DEFAULT_NEGATIVE_PROMPT}, people, person, crowd, faces, tourists"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default=None, help="Only generate this scene key.")
    parser.add_argument("--variants", type=int, default=3, help="Images per scene.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1350)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--seed", type=int, default=7000)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / BACKGROUND_ROOT)
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing variants.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from vpro.vision.juggernaut_runner import JuggernautRunner

    scenes = [s for s in SCENES if args.scene is None or s.key == args.scene]
    if not scenes:
        raise SystemExit(f"Unknown scene {args.scene!r}; expected one of {[s.key for s in SCENES]}")

    runner = JuggernautRunner(
        device=args.device,
        openvino_cache_dir=REPO_ROOT / "outputs" / "openvino_cache" / "juggernaut",
        task="text2img",
        warmup=False,
        queue_size=len(scenes) * args.variants + 4,
    ).start()

    total_start = perf_counter()
    written = 0
    try:
        for scene in scenes:
            directory = args.output_root / scene.key
            directory.mkdir(parents=True, exist_ok=True)
            print(f"\n{scene.key}: {scene.description}")

            for index in range(args.variants):
                path = directory / f"bg_{index + 1:02d}.jpg"
                if path.exists() and not args.overwrite:
                    print(f"  skip {path.name} (exists)")
                    continue

                started = perf_counter()
                result = runner.render(
                    RenderRequest(
                        mode="text2img",
                        prompt=scene.background_prompt(),
                        negative_prompt=BACKGROUND_NEGATIVE,
                        output_path=path,
                        steps=args.steps,
                        guidance_scale=args.guidance_scale,
                        width=args.width,
                        height=args.height,
                        seed=args.seed + index,
                    )
                )
                if not result.ok:
                    print(f"  FAILED {path.name}: {result.error}")
                    continue
                written += 1
                print(f"  {path.name} in {perf_counter() - started:.1f}s")
    finally:
        runner.shutdown()

    print(f"\n{written} background(s) written in {perf_counter() - total_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
