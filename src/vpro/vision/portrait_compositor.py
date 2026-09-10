from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class SubjectSelection:
    index: int
    bbox_xyxy: tuple[int, int, int, int]
    score: float


@dataclass
class CompositionResult:
    image_bgr: np.ndarray
    primary_index: int
    anchor_xy: tuple[int, int] | None
    #: Where subject and prop pixels landed on the canvas; inpainting locks these.
    coverage_mask: np.ndarray | None = None


@lru_cache(maxsize=16)
def _load_scene_canvas(scene_path: str, output_size: tuple[int, int]) -> np.ndarray:
    scene = cv2.imread(scene_path, cv2.IMREAD_COLOR)
    if scene is None:
        raise RuntimeError(f"Could not read scene image: {scene_path}")
    return cv2.resize(scene, output_size, interpolation=cv2.INTER_AREA)


@lru_cache(maxsize=16)
def _load_prop_image(prop_path: str) -> np.ndarray:
    prop = cv2.imread(prop_path, cv2.IMREAD_UNCHANGED)
    if prop is None:
        raise RuntimeError(f"Could not read laptop prop image: {prop_path}")
    if prop.ndim != 3 or prop.shape[2] != 4:
        raise RuntimeError("Laptop prop image must be a BGRA/PNG with alpha channel.")
    return prop


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.array([])
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return value.numpy()
    return np.array(value)


def select_primary_subject(
    yolo_result: Any,
    image_shape: tuple[int, int],
    min_area_ratio: float = 0.03,
) -> SubjectSelection:
    """Pick one person using confidence, center bias, and minimum size threshold."""
    h, w = image_shape
    boxes = _to_numpy(getattr(yolo_result.boxes, "xyxy", None))
    confs = _to_numpy(getattr(yolo_result.boxes, "conf", None)).reshape(-1)
    classes = _to_numpy(getattr(yolo_result.boxes, "cls", None)).reshape(-1)

    if boxes.size == 0:
        raise RuntimeError("No detections found for portrait composition.")

    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    max_dist = float(np.linalg.norm(center))

    best_idx = -1
    best_score = -1.0
    best_box = (0, 0, w, h)

    for i in range(len(boxes)):
        if classes.size and int(classes[i]) != 0:
            continue

        x1, y1, x2, y2 = boxes[i]
        x1_i = max(0, min(int(round(x1)), w - 1))
        y1_i = max(0, min(int(round(y1)), h - 1))
        x2_i = max(x1_i + 1, min(int(round(x2)), w))
        y2_i = max(y1_i + 1, min(int(round(y2)), h))

        area = float((x2_i - x1_i) * (y2_i - y1_i))
        area_ratio = area / float(w * h)
        if area_ratio < min_area_ratio:
            continue

        box_center = np.array([(x1_i + x2_i) / 2.0, (y1_i + y2_i) / 2.0], dtype=np.float32)
        dist = float(np.linalg.norm(box_center - center))
        center_bias = 1.0 - min(1.0, dist / max_dist)
        conf = float(confs[i]) if confs.size else 0.5

        score = (0.65 * conf) + (0.25 * center_bias) + (0.10 * min(1.0, area_ratio / 0.30))
        if score > best_score:
            best_score = score
            best_idx = i
            best_box = (x1_i, y1_i, x2_i, y2_i)

    if best_idx < 0:
        # Fallback to largest person-class box even if below threshold.
        person_idx = [i for i in range(len(boxes)) if not classes.size or int(classes[i]) == 0]
        if not person_idx:
            person_idx = list(range(len(boxes)))

        largest_i = max(
            person_idx,
            key=lambda idx: float((boxes[idx][2] - boxes[idx][0]) * (boxes[idx][3] - boxes[idx][1])),
        )
        x1, y1, x2, y2 = boxes[largest_i]
        best_idx = largest_i
        best_box = (
            max(0, min(int(round(x1)), w - 1)),
            max(0, min(int(round(y1)), h - 1)),
            max(1, min(int(round(x2)), w)),
            max(1, min(int(round(y2)), h)),
        )
        best_score = 0.0

    return SubjectSelection(index=int(best_idx), bbox_xyxy=best_box, score=float(best_score))


def cleanup_mask_for_primary_subject(
    mask: np.ndarray,
    primary_bbox: tuple[int, int, int, int],
    quality: str = "balanced",
) -> np.ndarray:
    """Remove non-primary people by keeping the component around the primary subject."""
    if mask.ndim != 2:
        raise ValueError("mask must be single-channel uint8")

    normalized_quality = quality.strip().lower()
    if normalized_quality not in {"fast", "balanced", "high"}:
        raise ValueError("quality must be one of: fast, balanced, high")

    threshold_map = {
        "fast": 20,
        "balanced": 28,
        "high": 36,
    }
    open_kernel_map = {
        "fast": 3,
        "balanced": 5,
        "high": 7,
    }
    close_kernel_map = {
        "fast": 3,
        "balanced": 7,
        "high": 11,
    }
    bbox_margin_map = {
        "fast": 0.22,
        "balanced": 0.16,
        "high": 0.10,
    }

    x1, y1, x2, y2 = primary_bbox
    h, w = mask.shape[:2]

    threshold = threshold_map[normalized_quality]
    open_k = open_kernel_map[normalized_quality]
    close_k = close_kernel_map[normalized_quality]
    bbox_margin = bbox_margin_map[normalized_quality]

    bin_mask = (mask > threshold).astype(np.uint8)
    open_kernel = np.ones((open_k, open_k), dtype=np.uint8)
    close_kernel = np.ones((close_k, close_k), dtype=np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, open_kernel)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, close_kernel)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    mx = int(round(bw * bbox_margin))
    my = int(round(bh * bbox_margin))
    gx1 = max(0, x1 - mx)
    gy1 = max(0, y1 - my)
    gx2 = min(w, x2 + mx)
    gy2 = min(h, y2 + my)

    gate = np.zeros_like(bin_mask, dtype=np.uint8)
    gate[gy1:gy2, gx1:gx2] = 1
    bin_mask = (bin_mask * gate).astype(np.uint8)

    num_labels, labels = cv2.connectedComponents(bin_mask)
    if num_labels <= 1:
        return mask

    cx = max(0, min((x1 + x2) // 2, mask.shape[1] - 1))
    cy = max(0, min((y1 + y2) // 2, mask.shape[0] - 1))
    chosen = int(labels[cy, cx])

    if chosen == 0:
        # pick component with max overlap with primary bbox
        best_overlap = 0
        for label in range(1, num_labels):
            component = (labels == label).astype(np.uint8)
            overlap = int(component[y1:y2, x1:x2].sum())
            if overlap > best_overlap:
                best_overlap = overlap
                chosen = label

    if chosen == 0:
        return mask

    keep = (labels == chosen).astype(np.uint8)
    cleaned = (mask * keep).astype(np.uint8)
    cleaned = cv2.GaussianBlur(cleaned, (5, 5), 0)

    if normalized_quality == "high":
        # Stronger edge smoothing reduces haloing and small noisy regions.
        cleaned = cv2.GaussianBlur(cleaned, (7, 7), 0)

    return cleaned


def _compute_prop_anchor(
    yolo_result: Any,
    primary_index: int,
) -> tuple[tuple[float, float], float] | None:
    """Return anchor (x, y) and rotation angle in degrees from wrist/elbow keypoints."""
    if primary_index < 0:
        return None

    keypoints = getattr(yolo_result, "keypoints", None)
    if keypoints is None:
        return None

    xy = _to_numpy(getattr(keypoints, "xy", None))
    kp_conf = _to_numpy(getattr(keypoints, "conf", None))
    if xy.size == 0 or primary_index >= len(xy):
        return None

    pts = xy[primary_index]
    conf = kp_conf[primary_index] if kp_conf.size else np.ones((pts.shape[0],), dtype=np.float32)

    # COCO keypoint indices: left elbow=7, right elbow=8, left wrist=9, right wrist=10
    candidates = [
        (9, 7, float(conf[9]) if len(conf) > 9 else 0.0),
        (10, 8, float(conf[10]) if len(conf) > 10 else 0.0),
    ]
    wrist_idx, elbow_idx, wrist_conf = max(candidates, key=lambda item: item[2])
    if wrist_conf < 0.15:
        return None

    wrist = pts[wrist_idx]
    elbow = pts[elbow_idx] if elbow_idx < len(pts) else wrist
    vec = wrist - elbow
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    return (float(wrist[0]), float(wrist[1])), angle


def _overlay_alpha(base_mask: np.ndarray, overlay_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
    """Accumulate overlay coverage using the same placement maths as _overlay_bgra."""
    out = base_mask.copy()
    h, w = out.shape[:2]
    oh, ow = overlay_bgra.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + ow)
    y2 = min(h, y + oh)
    if x1 >= x2 or y1 >= y2:
        return out

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    patch_alpha = overlay_bgra[oy1:oy2, ox1:ox2, 3]
    out[y1:y2, x1:x2] = np.maximum(out[y1:y2, x1:x2], patch_alpha)
    return out


def _overlay_bgra(base_bgr: np.ndarray, overlay_bgra: np.ndarray, x: int, y: int) -> np.ndarray:
    out = base_bgr.copy()
    h, w = out.shape[:2]
    oh, ow = overlay_bgra.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + ow)
    y2 = min(h, y + oh)
    if x1 >= x2 or y1 >= y2:
        return out

    ox1 = x1 - x
    oy1 = y1 - y
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    patch = overlay_bgra[oy1:oy2, ox1:ox2]
    alpha = (patch[:, :, 3:4].astype(np.float32)) / 255.0
    out_roi = out[y1:y2, x1:x2].astype(np.float32)
    fg = patch[:, :, :3].astype(np.float32)
    out[y1:y2, x1:x2] = (fg * alpha + out_roi * (1.0 - alpha)).astype(np.uint8)
    return out


def compose_portrait(
    source_bgr: np.ndarray,
    foreground_bgra: np.ndarray,
    cleaned_mask: np.ndarray,
    yolo_result: Any,
    primary: SubjectSelection,
    scene_path: Path,
    prop_path: Path,
    output_size: tuple[int, int] = (1080, 1350),
    mask_quality: str = "high",
) -> CompositionResult:
    """Compose one primary subject into a portrait scene with deterministic placement."""
    canvas_w, canvas_h = output_size

    canvas = _load_scene_canvas(str(scene_path.expanduser().resolve()), (canvas_w, canvas_h)).copy()

    alpha = cleaned_mask
    ys, xs = np.where(alpha > 12)
    if len(xs) == 0 or len(ys) == 0:
        raise RuntimeError("RMBG mask is empty after primary-subject cleanup.")

    sx1, sx2 = int(xs.min()), int(xs.max()) + 1
    sy1, sy2 = int(ys.min()), int(ys.max()) + 1
    subj_crop = foreground_bgra[sy1:sy2, sx1:sx2]
    subj_h, subj_w = subj_crop.shape[:2]
    if subj_h < 2 or subj_w < 2:
        raise RuntimeError("Subject crop is too small for composition.")

    target_h = int(canvas_h * 0.78)
    scale = max(0.1, target_h / float(subj_h))
    out_h = max(2, int(round(subj_h * scale)))
    out_w = max(2, int(round(subj_w * scale)))

    subj_scaled = cv2.resize(subj_crop, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # Feet-close composition for portrait social framing.
    feet_y = int(canvas_h * 0.92)
    paste_y = feet_y - out_h
    paste_x = int((canvas_w - out_w) / 2)

    composed = _overlay_bgra(canvas, subj_scaled, paste_x, paste_y)
    coverage = _overlay_alpha(
        np.zeros(canvas.shape[:2], dtype=np.uint8), subj_scaled, paste_x, paste_y
    )

    # Deterministic laptop prop anchor from keypoints.
    anchor_xy: tuple[int, int] | None = None
    prop_anchor = _compute_prop_anchor(yolo_result, primary.index)
    if prop_anchor is not None:
        (ax, ay), angle = prop_anchor
        tx = int(round((ax - sx1) * scale + paste_x))
        ty = int(round((ay - sy1) * scale + paste_y))
    else:
        # Deterministic fallback when keypoints are unavailable.
        tx = int(paste_x + out_w * 0.68)
        ty = int(paste_y + out_h * 0.53)
        angle = 0.0

    anchor_xy = (tx, ty)

    prop = _load_prop_image(str(prop_path.expanduser().resolve()))

    # Size prop relative to visible subject width.
    target_prop_w = max(60, int(out_w * 0.30))
    prop_scale = target_prop_w / float(prop.shape[1])
    prop_h = max(20, int(round(prop.shape[0] * prop_scale)))
    prop_w = max(20, int(round(prop.shape[1] * prop_scale)))
    prop_resized = cv2.resize(prop, (prop_w, prop_h), interpolation=cv2.INTER_LINEAR)

    center = (prop_w / 2.0, prop_h / 2.0)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    prop_rot = cv2.warpAffine(
        prop_resized,
        rot,
        (prop_w, prop_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )

    # Place prop near wrist with a small vertical offset.
    px = tx - prop_w // 2
    py = ty - int(prop_h * 0.35)
    composed = _overlay_bgra(composed, prop_rot, px, py)
    coverage = _overlay_alpha(coverage, prop_rot, px, py)

    return CompositionResult(
        image_bgr=composed,
        primary_index=primary.index,
        anchor_xy=anchor_xy,
        coverage_mask=coverage,
    )


def save_composed_image(image_bgr: np.ndarray, output_path: Path) -> Path:
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise RuntimeError(f"Could not write composed image to {output_path}")
    return output_path
