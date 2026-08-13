from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any


def _resolve_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _make_generator(seed: int | None) -> Any:
    if seed is None:
        return None
    try:
        import torch
    except Exception:
        return None
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _aligned_dim(value: int) -> int:
    # SDXL pipelines require dimensions divisible by 8.
    return max(64, int(value) - (int(value) % 8))


def load_juggernaut_pipeline(
    model_id: str,
    device: str = "AUTO",
    local_files_only: bool = True,
    openvino_cache_dir: Path | None = None,
) -> Any:
    try:
        from optimum.intel import OVDiffusionPipeline
    except Exception as exc:
        raise RuntimeError(
            "Juggernaut runtime requires optimum[openvino]. "
            "Install with: pip install \"optimum[openvino]\""
        ) from exc

    source: str = model_id
    candidate = Path(model_id).expanduser()
    if candidate.exists():
        source = str(candidate.resolve())

    ov_config: dict[str, str] = {}
    if openvino_cache_dir is not None:
        cache_dir = openvino_cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        ov_config["CACHE_DIR"] = str(cache_dir)

    print(
        f"Loading and compiling OpenVINO pipeline components on {device}; "
        "this can take several minutes on first use.",
        flush=True,
    )
    if ov_config:
        print(f"OpenVINO compiled-model cache: {ov_config['CACHE_DIR']}", flush=True)
    load_start = perf_counter()
    pipeline = OVDiffusionPipeline.from_pretrained(
        source,
        local_files_only=local_files_only,
        export=False,
        device=device,
        ov_config=ov_config,
        compile=False,
    )
    print(f"OpenVINO pipeline metadata loaded in {perf_counter() - load_start:.2f}s.", flush=True)

    components = getattr(pipeline, "components", {})
    for component_name, component in components.items():
        compile_method = getattr(component, "compile", None)
        if not callable(compile_method):
            continue
        print(f"Compiling OpenVINO component on {device}: {component_name}", flush=True)
        component_start = perf_counter()
        compile_method()
        print(
            f"OpenVINO component ready: {component_name} "
            f"({perf_counter() - component_start:.2f}s)",
            flush=True,
        )

    print(f"OpenVINO pipeline components ready in {perf_counter() - load_start:.2f}s.", flush=True)
    return pipeline


def render_text2img(
    pipeline: Any,
    output_path: Path,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    seed: int | None,
) -> Path:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Juggernaut text2img output processing requires Pillow. Install with: pip install pillow"
        ) from exc

    output_path = _resolve_path(output_path)
    generator = _make_generator(seed)

    requested_w = int(width)
    requested_h = int(height)
    model_w = _aligned_dim(requested_w)
    model_h = _aligned_dim(requested_h)

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "width": model_w,
        "height": model_h,
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if generator is not None:
        kwargs["generator"] = generator

    result = pipeline(**kwargs)
    if not hasattr(result, "images") or not result.images:
        raise RuntimeError("Juggernaut text2img returned no images.")

    image = result.images[0]
    if image.size != (requested_w, requested_h):
        image = image.resize((requested_w, requested_h), Image.Resampling.LANCZOS)
    image.save(output_path)
    return output_path


def render_img2img(
    pipeline: Any,
    input_image_path: Path,
    output_path: Path,
    prompt: str,
    negative_prompt: str | None,
    steps: int,
    guidance_scale: float,
    strength: float,
    width: int,
    height: int,
    seed: int | None,
) -> Path:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Juggernaut img2img requires Pillow. Install with: pip install pillow"
        ) from exc

    input_image_path = input_image_path.expanduser().resolve()
    if not input_image_path.exists():
        raise RuntimeError(f"Juggernaut guide image not found: {input_image_path}")

    output_path = _resolve_path(output_path)
    generator = _make_generator(seed)

    requested_w = int(width)
    requested_h = int(height)
    model_w = _aligned_dim(requested_w)
    model_h = _aligned_dim(requested_h)

    guide_image = Image.open(input_image_path).convert("RGB")
    guide_image = guide_image.resize((model_w, model_h), Image.Resampling.LANCZOS)

    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "image": guide_image,
        "num_inference_steps": int(steps),
        "guidance_scale": float(guidance_scale),
        "strength": float(strength),
    }
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if generator is not None:
        kwargs["generator"] = generator

    result = pipeline(**kwargs)
    if not hasattr(result, "images") or not result.images:
        raise RuntimeError("Juggernaut img2img returned no images.")

    image = result.images[0]
    if image.size != (requested_w, requested_h):
        image = image.resize((requested_w, requested_h), Image.Resampling.LANCZOS)
    image.save(output_path)
    return output_path
