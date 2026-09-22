"""Checks that the configured YOLO device (NPU or GPU) actually reaches pose inference.

    python tests/pose_device_test.py

No camera, GPU, or NPU is required; the pose backend is a mock.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpro.kiosk.camera import CameraStream  # noqa: E402
from vpro.kiosk.service import KioskConfig, KioskService  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed. {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


def test_camera_forwards_configured_device_to_predict() -> None:
    """`_annotate` must pass the exact device string it was given, not a hardcoded one."""
    for device in ("intel:npu", "intel:gpu"):
        result = Mock()
        result.boxes = None
        result.plot.side_effect = lambda **kwargs: kwargs["img"]
        backend = Mock()
        backend.predict.return_value = [result]

        camera = CameraStream(index=999, pose_backend=backend, pose_device=device)
        camera._annotate(np.zeros((100, 100, 3), dtype=np.uint8))

        check(
            f"predict receives {device}",
            backend.predict.call_args.args[0]["device"] == device,
            backend.predict.call_args,
        )


def test_service_forwards_yolo_device_to_camera() -> None:
    """The kiosk must wire its configured yolo_device into the camera's pose_device, not GPU always."""
    for device in ("intel:npu", "intel:gpu"):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "model.xml").write_text("<model/>")
            (root / "model.bin").write_bytes(b"model")
            (root / "prop.png").write_bytes(b"prop")
            config = KioskConfig(
                output_dir=root, rmbg_model_dir=root, prop_image=root / "prop.png", yolo_device=device
            )
            with patch("openvino.Core") as core, \
                 patch("vpro.backends.factory.build_backend", return_value=Mock()), \
                 patch("vpro.vision.rmbg_runtime.load_rmbg_runtime"), \
                 patch("vpro.kiosk.service.CameraStream") as camera_cls, \
                 patch("vpro.kiosk.service.build_delivery"), \
                 patch("vpro.vision.juggernaut_runner.JuggernautRunner"):
                core.return_value.available_devices = ["CPU", "GPU", "NPU"]
                service = KioskService(config).start()
                try:
                    check(
                        f"camera constructed with pose_device={device}",
                        camera_cls.call_args.kwargs["pose_device"] == device,
                        camera_cls.call_args,
                    )
                finally:
                    service.shutdown()


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        print(f"\n{test.__name__}")
        test()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
