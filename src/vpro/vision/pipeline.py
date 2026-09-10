"""Deterministic portrait composition, shared by the CLI and the kiosk service.

Lives here rather than in cli.py so both entry points use one implementation.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from .portrait_compositor import (
    SubjectSelection,
    cleanup_mask_for_primary_subject,
    compose_portrait,
    save_composed_image,
    select_primary_subject,
)
from .rmbg_runtime import load_rmbg_runtime

OUTPUT_SIZE = (1080, 1350)


def compose_portrait_from_image(
    backend: object,
    input_image_path: Path,
    rmbg_model_dir: Path,
    rmbg_device: str,
    yolo_device: str,
    scene_image_path: Path,
    prop_image_path: Path,
    output_image_path: Path,
    mask_quality: str,
    verbose: bool = True,
    coverage_output_path: Path | None = None,
    rmbg_runtime: Any | None = None,
    timings: dict[str, float] | None = None,
) -> Path:
    import cv2

    input_image_path = input_image_path.expanduser().resolve()
    scene_image_path = scene_image_path.expanduser().resolve()
    prop_image_path = prop_image_path.expanduser().resolve()

    if not input_image_path.exists():
        raise RuntimeError(f"Input image not found: {input_image_path}")
    if not scene_image_path.exists():
        raise RuntimeError(
            "Scene image not found. Provide a valid portrait asset path. "
            f"Missing: {scene_image_path}"
        )
    if not prop_image_path.exists():
        raise RuntimeError(
            f"Laptop prop image not found. Provide a valid alpha PNG. Missing: {prop_image_path}"
        )

    source = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Could not read input image: {input_image_path}")

    rmbg_start = perf_counter()
    if rmbg_runtime is None:
        rmbg_runtime = load_rmbg_runtime(rmbg_model_dir, device=rmbg_device)
    raw_mask, foreground = rmbg_runtime.segment(source)
    if timings is not None:
        timings["rmbg_seconds"] = perf_counter() - rmbg_start

    yolo_start = perf_counter()
    yolo_results = backend.predict({"source": source, "device": yolo_device, "verbose": False})
    if not isinstance(yolo_results, list) or not yolo_results:
        raise RuntimeError("YOLO returned no results for portrait composition.")

    result = yolo_results[0]
    if timings is not None:
        timings["yolo_seconds"] = perf_counter() - yolo_start
    try:
        primary = select_primary_subject(result, source.shape[:2])
    except RuntimeError:
        # Fall back to the RMBG mask extent when pose selection finds no usable subject.
        ys, xs = np.where(raw_mask > 16)
        if len(xs) == 0 or len(ys) == 0:
            raise RuntimeError("Could not determine a primary subject from YOLO or RMBG mask.")
        primary = SubjectSelection(
            index=-1,
            bbox_xyxy=(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
            score=0.0,
        )

    clean_mask = cleanup_mask_for_primary_subject(
        raw_mask,
        primary.bbox_xyxy,
        quality=mask_quality,
    )
    foreground[:, :, 3] = clean_mask

    compose_start = perf_counter()
    composed = compose_portrait(
        source_bgr=source,
        foreground_bgra=foreground,
        cleaned_mask=clean_mask,
        yolo_result=result,
        primary=primary,
        scene_path=scene_image_path,
        prop_path=prop_image_path,
        output_size=OUTPUT_SIZE,
        mask_quality=mask_quality,
    )

    saved_path = save_composed_image(composed.image_bgr, output_image_path)
    if coverage_output_path is not None and composed.coverage_mask is not None:
        coverage_output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(coverage_output_path), composed.coverage_mask)
    if timings is not None:
        timings["composition_seconds"] = perf_counter() - compose_start
    if verbose:
        anchor = composed.anchor_xy
        anchor_text = "none" if anchor is None else f"{anchor[0]},{anchor[1]}"
        print(
            "Portrait composition saved: "
            f"{saved_path} (primary_subject_index={composed.primary_index}, "
            f"prop_anchor={anchor_text})"
        )
    return saved_path
