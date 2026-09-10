from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class RMBGRuntime:
    """Minimal OpenVINO runtime wrapper for RMBG-style single-image inference."""

    def __init__(self, model_dir: Path, device: str = "AUTO") -> None:
        import openvino as ov

        self.model_dir = model_dir
        self.device = device
        self._core = ov.Core()
        self._compiled_model = self._core.compile_model(
            model=str(model_dir / "model.xml"),
            device_name=device,
        )
        self._input = self._compiled_model.input(0)
        self._output = self._compiled_model.output(0)
        self._input_shape = [int(dim) for dim in self._input.shape]

    def warmup(self, runs: int = 1) -> None:
        import numpy as np

        if runs < 1:
            raise ValueError("runs must be >= 1")

        shape = [int(dim) for dim in self._input.shape]
        dummy = np.zeros(shape, dtype=np.float32)

        for _ in range(runs):
            _ = self._compiled_model([dummy])

    def infer(self, input_tensor):
        """Run raw inference; caller is responsible for preprocessing/postprocessing."""
        return self._compiled_model([input_tensor])

    def preprocess_image(self, image_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        """Convert BGR image into normalized NCHW tensor expected by RMBG."""
        import cv2

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("image_bgr must be a non-empty numpy array")

        if image_bgr.ndim == 2:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 4:
            image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGRA2BGR)

        orig_h, orig_w = image_bgr.shape[:2]
        _, channels, input_h, input_w = self._input_shape
        if channels != 3:
            raise ValueError(f"Expected RMBG input channels=3, got {channels}")

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        image = resized.astype(np.float32) / 255.0
        # RMBG reference preprocessing uses mean=[0.5,0.5,0.5], std=[1,1,1]
        image = image - 0.5
        tensor = np.transpose(image, (2, 0, 1))[None, ...].astype(np.float32)
        return tensor, (orig_h, orig_w)

    def postprocess_mask(
        self,
        raw_output: Any,
        original_size: tuple[int, int],
    ) -> np.ndarray:
        """Resize and normalize model output into uint8 foreground mask."""
        import cv2

        out = np.array(raw_output)
        if not np.isfinite(out).all():
            raise RuntimeError("RMBG inference returned non-finite values; subject segmentation failed.")
        if out.ndim == 4:
            out = out[0, 0]
        elif out.ndim == 3:
            out = out[0]
        elif out.ndim != 2:
            raise ValueError(f"Unexpected RMBG output shape: {out.shape}")

        orig_h, orig_w = original_size
        resized = cv2.resize(out.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        min_val = float(np.min(resized))
        max_val = float(np.max(resized))
        if max_val > min_val:
            norm = (resized - min_val) / (max_val - min_val)
        else:
            norm = np.zeros_like(resized, dtype=np.float32)

        mask = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
        return mask

    def apply_mask(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Return BGRA foreground image using RMBG alpha mask."""
        import cv2

        if mask.ndim != 2:
            raise ValueError("mask must be single-channel")
        b, g, r = cv2.split(image_bgr)
        return cv2.merge((b, g, r, mask))

    def segment(self, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run full RMBG pipeline and return (mask, foreground_bgra)."""
        tensor, orig_size = self.preprocess_image(image_bgr)
        result = self.infer(tensor)
        raw_output = result[self._output]
        mask = self.postprocess_mask(raw_output, orig_size)
        foreground = self.apply_mask(image_bgr, mask)
        return mask, foreground


def is_rmbg_model_dir(path: Path) -> bool:
    """Return True when path contains expected RMBG OpenVINO artifacts."""
    return (
        path.is_dir()
        and (path / "model.xml").exists()
        and (path / "model.bin").exists()
    )


def validate_rmbg_assets(path: Path) -> Path:
    """Fail-fast validation for local RMBG model assets."""
    model_dir = path.expanduser().resolve()

    if not model_dir.exists():
        raise ValueError(f"RMBG model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise ValueError(f"RMBG model path is not a directory: {model_dir}")
    if not (model_dir / "model.xml").exists():
        raise ValueError(f"Missing RMBG model.xml in {model_dir}")
    if not (model_dir / "model.bin").exists():
        raise ValueError(f"Missing RMBG model.bin in {model_dir}")

    return model_dir


def load_rmbg_runtime(path: Path, device: str = "AUTO") -> RMBGRuntime:
    model_dir = validate_rmbg_assets(path)
    runtime = RMBGRuntime(model_dir=model_dir, device=device)
    return runtime
