from __future__ import annotations

import os
from pathlib import Path
from time import perf_counter
from typing import Any


def _configure_offline_defaults(local_files_only: bool) -> None:
    """Avoid startup network calls that block ~337s on socket timeouts when the Hub is unreachable.

    `local_files_only` is not honored by optimum's library inference, which still queries the Hub;
    OpenVINO telemetry separately blocks on its own analytics endpoint.
    """
    os.environ.setdefault("OPENVINO_TELEMETRY_OPT_OUT", "1")
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")


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


TASK_TEXT2IMG = "text2img"
TASK_IMG2IMG = "img2img"


def _require_task(pipeline: Any, task: str) -> None:
    """Guard against a text2img pipeline silently ignoring `image`/`strength` kwargs."""
    name = type(pipeline).__name__
    is_img2img = "Img2Img" in name
    if task == TASK_IMG2IMG and not is_img2img:
        raise RuntimeError(
            f"render_img2img requires an image-to-image pipeline, got {name}. "
            f"Load it with load_juggernaut_pipeline(..., task='{TASK_IMG2IMG}'). "
            "A text2img pipeline accepts and discards 'image' and 'strength', producing output "
            "that ignores the guide image entirely."
        )
    if task == TASK_TEXT2IMG and is_img2img:
        raise RuntimeError(
            f"render_text2img requires a text-to-image pipeline, got {name}. "
            f"Load it with load_juggernaut_pipeline(..., task='{TASK_TEXT2IMG}')."
        )


def load_juggernaut_pipeline(
    model_id: str,
    device: str = "AUTO",
    local_files_only: bool = True,
    openvino_cache_dir: Path | None = None,
    timings: dict[str, float] | None = None,
    task: str = TASK_TEXT2IMG,
) -> Any:
    """Load and compile the pipeline; `timings`, if given, is filled with per-phase seconds."""
    if task not in (TASK_TEXT2IMG, TASK_IMG2IMG):
        raise ValueError(f"task must be '{TASK_TEXT2IMG}' or '{TASK_IMG2IMG}', got {task!r}")
    _configure_offline_defaults(local_files_only)

    import_start = perf_counter()
    try:
        from optimum.intel import OVPipelineForImage2Image, OVPipelineForText2Image
    except Exception as exc:
        raise RuntimeError(
            "Juggernaut runtime requires optimum[openvino]. "
            "Install with: pip install \"optimum[openvino]\""
        ) from exc
    pipeline_class = (
        OVPipelineForImage2Image if task == TASK_IMG2IMG else OVPipelineForText2Image
    )
    import_sec = perf_counter() - import_start
    if timings is not None:
        timings["optimum_import_seconds"] = import_sec
    print(f"optimum.intel import: {import_sec:.2f}s", flush=True)

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
        f"Loading and compiling OpenVINO pipeline components on {device} for task '{task}'; "
        "this can take several minutes on first use.",
        flush=True,
    )
    if ov_config:
        print(f"OpenVINO compiled-model cache: {ov_config['CACHE_DIR']}", flush=True)
    load_start = perf_counter()
    pipeline = pipeline_class.from_pretrained(
        source,
        local_files_only=local_files_only,
        export=False,
        device=device,
        ov_config=ov_config,
        compile=False,
    )
    metadata_sec = perf_counter() - load_start
    if timings is not None:
        timings["metadata_seconds"] = metadata_sec
    print(
        f"OpenVINO pipeline metadata loaded in {metadata_sec:.2f}s "
        f"({type(pipeline).__name__}).",
        flush=True,
    )

    components = getattr(pipeline, "components", {})
    compile_total = 0.0
    for component_name, component in components.items():
        compile_method = getattr(component, "compile", None)
        if not callable(compile_method):
            continue
        print(f"Compiling OpenVINO component on {device}: {component_name}", flush=True)
        component_start = perf_counter()
        compile_method()
        component_sec = perf_counter() - component_start
        compile_total += component_sec
        if timings is not None:
            timings[f"compile_{component_name}_seconds"] = component_sec
        print(
            f"OpenVINO component ready: {component_name} "
            f"({component_sec:.2f}s)",
            flush=True,
        )

    total_sec = perf_counter() - load_start
    if timings is not None:
        timings["compile_total_seconds"] = compile_total
        timings["load_total_seconds"] = total_sec
        timings["startup_total_seconds"] = import_sec + total_sec
    print(f"OpenVINO pipeline components ready in {total_sec:.2f}s.", flush=True)
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

    _require_task(pipeline, TASK_TEXT2IMG)
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

    _require_task(pipeline, TASK_IMG2IMG)

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
