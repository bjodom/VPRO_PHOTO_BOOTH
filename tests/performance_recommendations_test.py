"""Regression checks for resource reuse and kiosk performance controls.

    python tests/performance_recommendations_test.py

These checks use image files and service configuration only; no camera, GPU, or model is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from time import time

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpro.kiosk.camera import CameraStream  # noqa: E402
from vpro.kiosk.service import KioskConfig, KioskService  # noqa: E402
from vpro.vision.portrait_compositor import (  # noqa: E402
    _load_prop_image,
    _load_scene_canvas,
)
from vpro.vision.pipeline import compose_portrait_from_image  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed. {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


def test_asset_caches() -> None:
    with TemporaryDirectory(prefix="vpro-assets-") as raw:
        root = Path(raw)
        scene_path = root / "scene.jpg"
        prop_path = root / "prop.png"
        cv2.imwrite(str(scene_path), np.full((40, 50, 3), 128, dtype=np.uint8))
        prop = np.zeros((12, 16, 4), dtype=np.uint8)
        prop[:, :, 3] = 255
        cv2.imwrite(str(prop_path), prop)

        _load_scene_canvas.cache_clear()
        _load_prop_image.cache_clear()
        first_scene = _load_scene_canvas(str(scene_path), (20, 30))
        second_scene = _load_scene_canvas(str(scene_path), (20, 30))
        first_prop = _load_prop_image(str(prop_path))
        second_prop = _load_prop_image(str(prop_path))

        check("scene asset is cached", first_scene is second_scene)
        check("prop asset is cached", first_prop is second_prop)
        check("scene cache has output size", first_scene.shape[:2] == (30, 20))


def test_pose_rate_validation() -> None:
    camera = CameraStream(index=999, pose_fps=10.0)
    check("pose interval is configured", abs(camera.pose_interval - 0.1) < 1e-9)

    rejected = False
    try:
        CameraStream(index=999, pose_fps=-1.0)
    except ValueError:
        rejected = True
    check("negative pose rate rejected", rejected)


def test_output_retention() -> None:
    with TemporaryDirectory(prefix="vpro-output-") as raw:
        root = Path(raw)
        old_path = root / "old.jpg"
        new_path = root / "new.jpg"
        old_path.write_bytes(b"old")
        new_path.write_bytes(b"new")
        old_time = time() - 7200
        import os

        os.utime(old_path, (old_time, old_time))

        service = KioskService(KioskConfig(output_dir=root, output_retention_hours=1))
        service._cleanup_old_outputs()
        check("old output is removed", not old_path.exists())
        check("recent output is retained", new_path.exists())


def test_composition_accepts_reusable_rmbg_runtime() -> None:
    class FakeBoxes:
        xyxy = np.array([[5, 5, 45, 35]], dtype=np.float32)
        conf = np.array([0.99], dtype=np.float32)
        cls = np.array([0], dtype=np.float32)

    class FakeKeypoints:
        xy = np.zeros((1, 17, 2), dtype=np.float32)
        conf = np.ones((1, 17), dtype=np.float32)

    class FakeResult:
        boxes = FakeBoxes()
        keypoints = FakeKeypoints()

    class FakeBackend:
        def predict(self, _input):
            return [FakeResult()]

    class FakeRmbg:
        def __init__(self) -> None:
            self.calls = 0

        def segment(self, image):
            self.calls += 1
            mask = np.full(image.shape[:2], 255, dtype=np.uint8)
            foreground = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
            foreground[:, :, 3] = mask
            return mask, foreground

    with TemporaryDirectory(prefix="vpro-compose-") as raw:
        root = Path(raw)
        input_path = root / "input.jpg"
        scene_path = root / "scene.jpg"
        prop_path = root / "prop.png"
        output_path = root / "output.jpg"
        image = np.full((40, 50, 3), 128, dtype=np.uint8)
        cv2.imwrite(str(input_path), image)
        cv2.imwrite(str(scene_path), image)
        prop = np.zeros((12, 16, 4), dtype=np.uint8)
        prop[:, :, 3] = 255
        cv2.imwrite(str(prop_path), prop)
        runtime = FakeRmbg()
        timings: dict[str, float] = {}

        compose_portrait_from_image(
            backend=FakeBackend(),
            input_image_path=input_path,
            rmbg_model_dir=root,
            rmbg_device="CPU",
            yolo_device="CPU",
            scene_image_path=scene_path,
            prop_image_path=prop_path,
            output_image_path=output_path,
            mask_quality="fast",
            rmbg_runtime=runtime,
            timings=timings,
            verbose=False,
        )

        check("injected RMBG runtime is used", runtime.calls == 1)
        check("composition records RMBG timing", "rmbg_seconds" in timings)
        check("composition records YOLO timing", "yolo_seconds" in timings)
        check("composition output is written", output_path.exists())


def main() -> int:
    for test in (
        test_asset_caches,
        test_pose_rate_validation,
        test_output_retention,
        test_composition_accepts_reusable_rmbg_runtime,
    ):
        print(f"\n{test.__name__}")
        test()
    print(f"\n{len(PASSED)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())