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


def test_vae_precision_contract() -> None:
    from types import SimpleNamespace
    from unittest.mock import Mock, patch
    from vpro.vision.juggernaut_runtime import load_juggernaut_pipeline

    shared_config = {"CACHE_DIR": "test-cache"}
    components = {
        name: SimpleNamespace(ov_config=shared_config, compile=Mock())
        for name in ("unet", "vae_encoder", "vae_decoder")
    }
    factory = Mock()
    factory.from_pretrained.return_value = SimpleNamespace(components=components)
    module = SimpleNamespace(OVPipelineForImage2Image=factory,
                             OVPipelineForInpainting=factory, OVPipelineForText2Image=factory)
    with patch.dict(sys.modules, {"optimum.intel": module}):
        load_juggernaut_pipeline("test-model", device="GPU", task="inpaint")
    for name in components:
        check(f"{name} preserves model precision", "INFERENCE_PRECISION_HINT" not in components[name].ov_config)
        check(f"{name} preserves cache", components[name].ov_config["CACHE_DIR"] == "test-cache")
        components[name].compile.assert_called_once_with()
    check("shared config remains unchanged", shared_config == {"CACHE_DIR": "test-cache"})


def test_rmbg_nonfinite_output() -> None:
    from unittest.mock import Mock, patch
    from vpro.vision.rmbg_runtime import RMBGRuntime

    core = Mock()
    core.compile_model.return_value.input.return_value.shape = (1, 3, 8, 8)
    with patch("openvino.Core", return_value=core):
        runtime = RMBGRuntime(Path("test-model"), device="GPU")
    for invalid in (np.nan, np.inf):
        try:
            runtime.postprocess_mask(np.full((1, 1, 8, 8), invalid), (8, 8))
        except RuntimeError as error:
            check("RMBG rejects non-finite inference", "non-finite" in str(error))
        else:
            raise AssertionError("Non-finite RMBG output was silently accepted")
    valid = np.linspace(0, 1, 64).reshape(1, 1, 8, 8)
    mask = runtime.postprocess_mask(valid, (8, 8))
    check("valid RMBG output still normalizes", mask.min() == 0 and mask.max() == 255)


def test_cli_capture_contract() -> None:
    from unittest.mock import Mock, create_autospec, patch
    from vpro import cli

    args = cli.build_parser().parse_args([
        "--run-social-pipeline", "--juggernaut-skip-guided", "--capture-skip-rmbg"
    ])
    capture = create_autospec(cli._capture_single_image, return_value=Path("capture.jpg"))
    with patch.object(cli, "build_backend", return_value=Mock()), \
         patch.object(cli, "_capture_single_image", capture), \
         patch.object(cli, "_compose_portrait_from_image", return_value=Path("composed.jpg")), \
         patch.object(cli.shutil, "copy2"):
        cli._run_social_pipeline(args)
    check("social capture matches real signature", capture.call_count == 1)

    args = cli.build_parser().parse_args(["--kiosk", "--kiosk-pose-fps", "9"])
    with patch("vpro.kiosk.service.KioskService") as service, patch("vpro.kiosk.app.run"):
        cli._run_kiosk(args)
    check("kiosk receives pose rate", service.call_args.args[0].pose_fps == 9)


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
    from unittest.mock import Mock
    result = Mock()
    result.boxes = None
    result.plot.side_effect = lambda **kwargs: kwargs["img"]
    camera.pose_backend = Mock()
    camera.pose_backend.predict.return_value = [result]
    camera.pose_interval = 100
    camera.mirror_preview = False
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    camera._annotate(frame)
    camera._annotate(frame)
    check("pose result reused between updates", camera.pose_backend.predict.call_count == 1)
    check("overlay drawn on every preview", result.plot.call_count == 2)
    check("preview never modifies raw capture", not frame.any())
    camera._frame = frame
    check("stale camera frame rejected", camera.latest_frame(max_age=1) is None)


def test_output_retention() -> None:
    with TemporaryDirectory(prefix="vpro-output-") as raw:
        root = Path(raw)
        old_path = root / "capture_old.jpg"
        new_path = root / "new.jpg"
        old_path.write_bytes(b"old")
        new_path.write_bytes(b"new")
        old_time = time() - 7200
        import os

        os.utime(old_path, (old_time, old_time))

        service = KioskService(KioskConfig(output_dir=root, output_retention_hours=1))
        from vpro.delivery.handoff import ImageHandoffStore
        from types import SimpleNamespace
        store = ImageHandoffStore()
        token = store.add(old_path)
        service.delivery = SimpleNamespace(store=store)
        service._cleanup_old_outputs()
        check("active download is retained", old_path.exists())
        store.revoke(token)
        service._cleanup_old_outputs()
        check("old output is removed", not old_path.exists())
        check("recent output is retained", new_path.exists())


def test_camera_acquires_during_slow_pose() -> None:
    import threading
    from unittest.mock import Mock, patch

    inference_started = threading.Event()
    release_inference = threading.Event()
    frames_advanced = threading.Event()
    pacing = threading.Event()

    class Capture:
        count = 0

        def isOpened(self):
            return True

        def set(self, *args):
            return True

        def read(self):
            pacing.wait(0.005)
            self.count += 1
            if inference_started.is_set() and self.count >= 10:
                frames_advanced.set()
            return True, np.full((100, 80, 3), self.count % 255, dtype=np.uint8)

        def release(self):
            pass

    def predict(_payload):
        inference_started.set()
        release_inference.wait(3)
        return []

    camera = CameraStream(pose_backend=Mock(predict=predict))
    camera._subscribers = 1
    with patch("cv2.VideoCapture", return_value=Capture()):
        camera.start()
        try:
            check("slow pose inference started", inference_started.wait(2))
            check("capture advances during blocked inference", frames_advanced.wait(2))
            check("fresh raw capture available", camera.latest_frame(max_age=1) is not None)
        finally:
            release_inference.set()
            camera.stop()


def test_service_delivery_retry() -> None:
    from unittest.mock import Mock
    from vpro.kiosk.session import State
    from vpro.delivery.base import DeliveryResult

    with TemporaryDirectory() as raw:
        root = Path(raw)
        final_path = root / "final_retry.jpg"
        final_path.write_bytes(b"photo")
        service = KioskService(KioskConfig(output_dir=root))
        service.session.state = State.GENERATING
        service.session.data.final_path = final_path
        service.delivery = Mock()
        service.delivery.deliver.side_effect = [
            DeliveryResult(ok=False, channel="local-qr", error="offline"),
            DeliveryResult(ok=True, channel="local-qr", url="http://booth/i/test", qr_svg="<svg/>")
        ]
        service._run_pipeline(service.session.session_id)
        check("handoff failure preserves image", service.session.data.final_path == final_path)
        check("handoff failure has recovery state", service.session.state is State.DELIVERY_ERROR)
        service.act("retry")
        service._worker.join(timeout=5)
        check("delivery-only retry succeeds", service.session.state is State.READY)
        check("retry uses identical image", all(call.args[0].image_path == final_path
              for call in service.delivery.deliver.call_args_list))


def test_service_startup_contract() -> None:
    from unittest.mock import Mock, patch
    with TemporaryDirectory() as raw:
        root = Path(raw)
        (root / "model.xml").write_text("<model/>")
        (root / "model.bin").write_bytes(b"model")
        (root / "prop.png").write_bytes(b"prop")
        config = KioskConfig(output_dir=root, rmbg_model_dir=root, prop_image=root / "prop.png")
        with patch("openvino.Core") as core, \
             patch("vpro.backends.factory.build_backend", return_value=Mock()), \
             patch("vpro.vision.rmbg_runtime.load_rmbg_runtime") as rmbg, \
             patch("vpro.kiosk.service.CameraStream"), \
             patch("vpro.kiosk.service.build_delivery"), \
             patch("vpro.vision.juggernaut_runner.JuggernautRunner") as renderer:
            core.return_value.available_devices = ["CPU", "GPU", "NPU"]
            service = KioskService(config).start()
            try:
                check("resident RMBG loaded once", rmbg.call_count == 1)
                check("kiosk renderer is offline inpaint", renderer.call_args.kwargs["local_files_only"]
                      and renderer.call_args.kwargs["task"] == "inpaint")
                check("renderer warms before guests", renderer.call_args.kwargs["warmup"])
                check("retention worker started", service._maintenance.is_alive())
            finally:
                service.shutdown()
            check("retention worker stops", not service._maintenance.is_alive())


def test_render_failure_is_not_success() -> None:
    from concurrent.futures import Future
    from unittest.mock import Mock, patch
    from vpro.kiosk.session import State
    from vpro.vision.juggernaut_types import RenderResult

    with TemporaryDirectory() as raw:
        root = Path(raw)
        source = root / "capture_test.jpg"
        cv2.imwrite(str(source), np.full((80, 64, 3), 127, dtype=np.uint8))
        service = KioskService(KioskConfig(output_dir=root))
        service.session.state = State.GENERATING
        service.session.data.capture_path = source
        service.session.data.scene = "fuji"
        future = Future()
        future.set_result(RenderResult(False, None, 0.1, 0.0, "black output"))
        service.runner = Mock()
        service.runner.submit.return_value = future
        service.delivery = Mock()

        def compose(**kwargs):
            cv2.imwrite(str(kwargs["coverage_output_path"]), np.full((80, 64), 255, dtype=np.uint8))
            return source

        with patch("vpro.vision.pipeline.compose_portrait_from_image", side_effect=compose):
            service._run_pipeline(service.session.session_id)
        check("failed render has error state", service.session.state is State.GENERATION_ERROR)
        check("gray fallback is never delivered", not service.delivery.deliver.called)
        check("failed render recorded", '"success": false' in (root / "pipeline_metrics.jsonl").read_text())
        service.session.reset()
        try:
            service.act("start", {"session_id": service.session.session_id - 1})
        except ValueError:
            pass
        else:
            raise AssertionError("stale action accepted")
        check("stale browser action rejected", service.session.state is State.IDLE)


def test_custom_location_uses_inpainting() -> None:
    from concurrent.futures import Future
    from unittest.mock import Mock, patch
    from vpro.kiosk.session import State
    from vpro.vision.juggernaut_types import RenderResult
    from vpro.delivery.base import DeliveryResult

    with TemporaryDirectory() as raw:
        root = Path(raw)
        source = root / "capture_custom.jpg"
        rendered = root / "final_custom.jpg"
        pixels = np.full((80, 64, 3), 127, dtype=np.uint8)
        cv2.imwrite(str(source), pixels)
        cv2.imwrite(str(rendered), pixels)
        service = KioskService(KioskConfig(output_dir=root))
        service.session.state = State.SELECT_SCENE
        service.act("choose_scene", {"scene": "custom", "custom_location": "Kyoto, Japan"})
        service.session.capture(source)
        service.session.accept_capture()
        future = Future()
        future.set_result(RenderResult(True, rendered, 0.1, 0.0))
        service.runner = Mock()
        service.runner.submit.return_value = future
        service.delivery = Mock()
        service.delivery.deliver.return_value = DeliveryResult(ok=True, channel="local-qr", url="http://booth/i/custom")

        def compose(**kwargs):
            canvas = cv2.imread(str(kwargs["scene_image_path"]))
            check("custom composition uses neutral canvas", np.all(canvas == 128))
            mask = np.zeros((80, 64), dtype=np.uint8)
            mask[20:60, 20:44] = 255
            cv2.imwrite(str(kwargs["coverage_output_path"]), mask)
            return source

        with patch("vpro.vision.pipeline.compose_portrait_from_image", side_effect=compose):
            service._run_pipeline(service.session.session_id)
        request = service.runner.submit.call_args.args[0]
        check("custom scene renders with inpainting", request.mode == "inpaint")
        check("custom place passed to renderer", "Kyoto, Japan" in request.prompt)
        check("custom locations do not forbid interiors", "indoor" not in request.negative_prompt)
        check("captured portrait conditions rendering", request.input_image_path == source)
        mask = cv2.imread(str(request.mask_image_path), cv2.IMREAD_GRAYSCALE)
        check("person protected while background generated", mask[40, 32] < 10 and mask[0, 0] > 245)
        check("delivers generated result not composition", service.session.data.final_path == rendered
              and service.session.state is State.READY)


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
        test_cli_capture_contract,
        test_asset_caches,
        test_vae_precision_contract,
        test_rmbg_nonfinite_output,
        test_pose_rate_validation,
        test_camera_acquires_during_slow_pose,
        test_output_retention,
        test_service_delivery_retry,
        test_service_startup_contract,
        test_render_failure_is_not_success,
        test_custom_location_uses_inpainting,
        test_composition_accepts_reusable_rmbg_runtime,
    ):
        print(f"\n{test.__name__}")
        test()
    print(f"\n{len(PASSED)} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())