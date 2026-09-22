from argparse import ArgumentParser
import math
from pathlib import Path
import shutil
from time import perf_counter, sleep

import numpy as np

from .backends.factory import build_backend
from .vision import (
    SubjectSelection,
    compose_portrait,
    cleanup_mask_for_primary_subject,
    load_rmbg_runtime,
    save_composed_image,
    select_primary_subject,
    validate_rmbg_assets,
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="vPRO inference runner")
    parser.add_argument(
        "--backend",
        default="openvino",
        choices=["openvino", "torch"],
        help="Inference backend to use (OpenVINO is the default).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models"),
        help="Path to model artifact directory or file.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short webcam inference smoke test.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index used for smoke test mode.",
    )
    parser.add_argument(
        "--device",
        default="intel:gpu",
        help="Inference device used in smoke test mode.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=30,
        help="Number of frames to process in smoke test mode.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show annotated frames during smoke test (press 'q' to quit early).",
    )
    parser.add_argument(
        "--capture-single-image",
        action="store_true",
        help="Capture one still image from the live camera feed and save it for rendering.",
    )
    parser.add_argument(
        "--capture-output",
        type=Path,
        default=Path("outputs/visitor_capture.jpg"),
        help="Output image path used with --capture-single-image.",
    )
    parser.add_argument(
        "--capture-rmbg-foreground-output",
        type=Path,
        default=Path("outputs/visitor_capture_no_bg.png"),
        help="Output PNG path for RMBG foreground with alpha channel.",
    )
    parser.add_argument(
        "--capture-rmbg-mask-output",
        type=Path,
        default=Path("outputs/visitor_capture_mask.png"),
        help="Output PNG path for RMBG foreground mask.",
    )
    parser.add_argument(
        "--capture-skip-rmbg",
        action="store_true",
        help="Skip RMBG processing during --capture-single-image.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=12,
        help="Number of camera warmup frames to discard before still capture.",
    )
    parser.add_argument(
        "--capture-delay-seconds",
        type=int,
        default=3,
        help="Countdown delay before still capture so the subject can pose and hold.",
    )
    parser.add_argument(
        "--capture-width",
        type=int,
        default=1920,
        help="Requested camera capture width for kiosk preview and still capture.",
    )
    parser.add_argument(
        "--capture-height",
        type=int,
        default=1080,
        help="Requested camera capture height for kiosk preview and still capture.",
    )
    parser.add_argument(
        "--capture-rotate",
        type=int,
        default=0,
        choices=[0, 90, 180, 270],
        help=(
            "Clockwise rotation applied to camera frames. Use 90 or 270 for a camera mounted on "
            "its side, which puts the sensor's long axis vertical for the 4:5 portrait output."
        ),
    )
    parser.add_argument(
        "--capture-auto-start",
        action="store_true",
        help="Start countdown automatically without clicking the on-screen Start button.",
    )
    parser.add_argument(
        "--capture-wrist-stable-seconds",
        type=float,
        default=0.7,
        help="Continuous wrist-detection hold time required before countdown can begin.",
    )
    parser.add_argument(
        "--validate-rmbg",
        action="store_true",
        help="Validate local RMBG model assets and run warmup inference.",
    )
    parser.add_argument(
        "--rmbg-model-dir",
        type=Path,
        default=Path("models/rmbg/rmbg-1.4"),
        help="RMBG model directory containing model.xml and model.bin.",
    )
    parser.add_argument(
        "--kiosk",
        action="store_true",
        help="Run the persistent kiosk app (web UI on localhost, QR handoff on the LAN).",
    )
    parser.add_argument(
        "--kiosk-host",
        default="127.0.0.1",
        help="Interface for the kiosk UI. Keep on localhost; only delivery needs the LAN.",
    )
    parser.add_argument("--kiosk-port", type=int, default=8000, help="Kiosk UI port.")
    parser.add_argument("--kiosk-output-dir", type=Path, default=Path("outputs/kiosk"))
    parser.add_argument("--open-browser", action="store_true", help="Open the local kiosk browser once the server is listening.")
    parser.add_argument(
        "--kiosk-pose-fps",
        type=float,
        default=15.0,
        help="Maximum YOLO pose inference rate during kiosk preview; 0 means every frame.",
    )
    parser.add_argument(
        "--kiosk-output-retention-hours",
        type=float,
        default=24.0,
        help="Delete kiosk output files older than this at startup; 0 disables cleanup.",
    )
    parser.add_argument(
        "--kiosk-no-generation",
        action="store_true",
        help="Skip Juggernaut and deliver the deterministic compose; useful for UI work.",
    )
    parser.add_argument(
        "--test-delivery",
        action="store_true",
        help="Serve one image over the local network and print a scannable QR code, then wait.",
    )
    parser.add_argument(
        "--delivery-channel",
        default="local-qr",
        choices=["local-qr", "twilio", "none"],
        help="Delivery channel used to hand the finished image to the guest.",
    )
    parser.add_argument(
        "--delivery-image",
        type=Path,
        default=Path("outputs/social_final.jpg"),
        help="Image to hand off in --test-delivery mode.",
    )
    parser.add_argument(
        "--delivery-host",
        default="0.0.0.0",
        help="Interface the handoff server binds to; must be LAN-reachable for phones.",
    )
    parser.add_argument(
        "--delivery-port",
        type=int,
        default=8765,
        help="Handoff server port.",
    )
    parser.add_argument(
        "--delivery-advertise-host",
        default=None,
        help="Address put in the QR code. Defaults to the detected LAN address.",
    )
    parser.add_argument(
        "--delivery-caption",
        default="Made at the Intel vPro Photo Booth #vPro #IntelAI",
        help="Caption offered to the guest for their social post.",
    )
    parser.add_argument(
        "--npu",
        action="store_true",
        help=(
            "Run YOLO pose on the Intel NPU while leaving RMBG on its configured device "
            "(CPU by default). Juggernaut stays on the GPU; explicit "
            "--device / --rmbg-device values still win."
        ),
    )
    parser.add_argument(
        "--rmbg-device",
        default="CPU",
        help="RMBG device (default: CPU to avoid sharing the generation GPU).",
    )
    parser.add_argument(
        "--mask-quality",
        default="high",
        choices=["fast", "balanced", "high"],
        help="Mask cleanup quality preset used for composition (high gives tighter masks at higher cost).",
    )
    parser.add_argument(
        "--compose-portrait",
        action="store_true",
        help="Compose a final 1080x1350 portrait image using RMBG + YOLO keypoints.",
    )
    parser.add_argument(
        "--compose-input-image",
        type=Path,
        default=Path("outputs/visitor_capture.jpg"),
        help="Input visitor image used for portrait composition.",
    )
    parser.add_argument(
        "--compose-scene-image",
        type=Path,
        default=Path("assets/scenes/portrait_scene_1080x1350.jpg"),
        help="Portrait background scene image path.",
    )
    parser.add_argument(
        "--compose-prop-image",
        type=Path,
        default=Path("assets/props/vpro_laptop.png"),
        help="Laptop prop PNG path with alpha channel.",
    )
    parser.add_argument(
        "--compose-output-image",
        type=Path,
        default=Path("outputs/final_portrait_1080x1350.jpg"),
        help="Final portrait output image path.",
    )
    parser.add_argument(
        "--run-social-pipeline",
        action="store_true",
        help="Run one-shot social pipeline: capture -> deterministic compose -> Juggernaut guided render with fallback.",
    )
    parser.add_argument(
        "--final-output-image",
        type=Path,
        default=Path("outputs/final_portrait_1080x1350_final.jpg"),
        help="Final deliverable output path for one-shot social pipeline mode.",
    )
    parser.add_argument(
        "--test-juggernaut",
        action="store_true",
        help="Run local OpenVINO Juggernaut test renders (smoke + optional guided).",
    )
    parser.add_argument(
        "--juggernaut-model-id",
        # FP16 alternative: OpenVINO/Juggernaut-XL-v9-fp16-ov
        default="OpenVINO/Juggernaut-XL-v9-int8-ov",
        help="Hugging Face model id or local model directory for OpenVINO Juggernaut pipeline.",
    )
    parser.add_argument(
        "--juggernaut-device",
        default="GPU",
        help="OpenVINO device used to compile and run the Juggernaut pipeline.",
    )
    parser.add_argument(
        "--juggernaut-local-only",
        action="store_true",
        help="Use cache/local files only (no network download).",
    )
    parser.add_argument(
        "--juggernaut-vae-precision",
        default="",
        help="OpenVINO INFERENCE_PRECISION_HINT applied only to the VAE encoder/decoder, working "
        "around the fp16 VAE overflow that produces all-black frames. Empty (default) leaves the "
        "pipeline default precision; on this hardware 'f32' caused heavy paging and renders past "
        "the 170s event timeout, so only set this after re-measuring the tradeoff.",
    )
    parser.add_argument(
        "--juggernaut-openvino-cache-dir",
        type=Path,
        default=Path("outputs/openvino_cache/juggernaut"),
        help="Persistent OpenVINO compiled-model cache for Juggernaut.",
    )
    parser.add_argument(
        "--juggernaut-prompt",
        default="Hyperdetailed photography, person presenting a Lenovo laptop in a modern expo booth, cinematic portrait lighting",
        help="Prompt used for Juggernaut test renders.",
    )
    parser.add_argument(
        "--juggernaut-negative-prompt",
        default="blurry, lowres, extra fingers, extra hands, watermark, text",
        help="Negative prompt used for Juggernaut test renders.",
    )
    parser.add_argument(
        "--juggernaut-preset",
        default="balanced",
        choices=["identity-lock", "balanced", "stylized"],
        help="Quality/identity preset for Juggernaut (balanced default).",
    )
    parser.add_argument(
        "--juggernaut-steps",
        type=int,
        default=None,
        help="Inference step count override for Juggernaut renders.",
    )
    parser.add_argument(
        "--juggernaut-guidance-scale",
        type=float,
        default=None,
        help="Classifier-free guidance scale override for Juggernaut renders.",
    )
    parser.add_argument(
        "--juggernaut-seed",
        type=int,
        default=None,
        help="Optional random seed for deterministic Juggernaut output.",
    )
    parser.add_argument(
        "--juggernaut-width",
        type=int,
        default=1080,
        help="Output width for Juggernaut test render.",
    )
    parser.add_argument(
        "--juggernaut-height",
        type=int,
        default=1350,
        help="Output height for Juggernaut test render.",
    )
    parser.add_argument(
        "--juggernaut-smoke-output",
        type=Path,
        default=Path("outputs/juggernaut_smoke.jpg"),
        help="Output path for Juggernaut text-to-image smoke render.",
    )
    parser.add_argument(
        "--juggernaut-guided-input-image",
        type=Path,
        default=Path("outputs/final_portrait_1080x1350_lenovo_test.jpg"),
        help="Guide image for Juggernaut img2img test (typically composed portrait output).",
    )
    parser.add_argument(
        "--juggernaut-guided-output",
        type=Path,
        default=Path("outputs/juggernaut_guided.jpg"),
        help="Output path for Juggernaut image-guided render.",
    )
    parser.add_argument(
        "--juggernaut-guided-strength",
        type=float,
        default=None,
        help="Img2img strength override for Juggernaut guided render.",
    )
    parser.add_argument(
        "--juggernaut-skip-guided",
        action="store_true",
        help="Skip image-guided Juggernaut test even if guide image exists.",
    )
    parser.add_argument(
        "--juggernaut-no-preload",
        action="store_true",
        help=(
            "Load Juggernaut after capture instead of during it. Preloading hides ~29s of startup "
            "behind the capture flow but shares the GPU with the live pose preview."
        ),
    )
    return parser


def _open_camera(
    camera_index: int,
    capture_width: int = 1920,
    capture_height: int = 1080,
):
    import cv2

    if capture_width < 1 or capture_height < 1:
        raise ValueError("--capture-width and --capture-height must be >= 1")

    def _configure_capture_resolution(cap_obj) -> None:
        cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, float(capture_width))
        cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, float(capture_height))

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if cap.isOpened():
        _configure_capture_resolution(cap)
        return cap

    cap.release()
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        _configure_capture_resolution(cap)
        return cap

    cap.release()
    raise RuntimeError(f"Could not open camera index {camera_index}.")


def _capture_single_image(
    camera_index: int,
    output_path: Path,
    warmup_frames: int,
    delay_seconds: int,
    capture_width: int,
    capture_height: int,
    yolo_preview_backend: object | None,
    yolo_device: str,
    auto_start: bool,
    wrist_stable_seconds: float,
) -> Path:
    import cv2

    preview_window = "vPRO Capture Preview"

    def _to_numpy(array_like) -> np.ndarray | None:
        if array_like is None:
            return None
        value = array_like
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        try:
            return np.asarray(value)
        except Exception:
            return None

    def _wrist_detected_from_result(result: object, min_confidence: float = 0.30) -> bool:
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None:
            return False
        data = getattr(keypoints, "data", None)
        arr = _to_numpy(data)
        if arr is None or arr.ndim != 3 or arr.shape[1] <= 10:
            return False

        # COCO keypoints: left wrist=9, right wrist=10.
        for person in arr:
            for wrist_idx in (9, 10):
                x = float(person[wrist_idx][0])
                y = float(person[wrist_idx][1])
                conf = float(person[wrist_idx][2]) if person.shape[1] > 2 else 1.0
                if np.isfinite(x) and np.isfinite(y) and x > 1.0 and y > 1.0 and conf >= min_confidence:
                    return True
        return False

    def _draw_countdown_overlay(frame: np.ndarray, seconds_left: int) -> np.ndarray:
        overlay = frame.copy()
        h, _ = overlay.shape[:2]
        text = f"Capturing in {seconds_left}s"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.9
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        margin = 16
        x = margin
        y = max(th + margin, h - margin)

        # Lower-left dark backing for readability across bright frames.
        pad = 10
        rx1 = max(0, x - pad)
        ry1 = max(0, y - th - baseline - pad)
        rx2 = min(overlay.shape[1], x + tw + pad)
        ry2 = min(h, y + baseline + pad)
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
        cv2.putText(overlay, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return overlay

    def _draw_start_button(frame: np.ndarray, mode: str) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        overlay = frame.copy()
        h, w = overlay.shape[:2]
        bw = max(220, int(w * 0.18))
        bh = max(72, int(h * 0.08))
        margin = 24
        x1 = max(0, w - bw - margin)
        y1 = max(0, h - bh - margin)
        x2 = min(w, x1 + bw)
        y2 = min(h, y1 + bh)

        if mode == "countdown":
            bg = (38, 127, 55)
            label = "Capturing..."
        elif mode == "waiting_wrist":
            bg = (32, 110, 160)
            label = "Show Wrist"
        else:
            bg = (52, 52, 52)
            label = "Start Capture"
        cv2.rectangle(overlay, (x1, y1), (x2, y2), bg, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.8
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thickness)
        tx = x1 + max(8, int((bw - tw) / 2))
        ty = y1 + max(th + 8, int((bh + th) / 2))
        cv2.putText(overlay, label, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return overlay, (x1, y1, x2, y2)

    def _draw_wrist_status(
        frame: np.ndarray,
        wrist_detected: bool,
        capture_requested: bool,
        steady_elapsed: float,
        steady_required: float,
    ) -> np.ndarray:
        overlay = frame.copy()
        if capture_requested and not wrist_detected:
            text = "Waiting for wrist..."
            color = (32, 110, 160)
        elif capture_requested and steady_required > 0 and steady_elapsed < steady_required:
            text = f"Hold wrist steady {steady_elapsed:.1f}/{steady_required:.1f}s"
            color = (32, 110, 160)
        elif wrist_detected:
            text = "Wrist detected"
            color = (46, 170, 76)
        else:
            text = "Show hand/wrist"
            color = (66, 66, 66)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        margin = 16
        x1 = margin
        y1 = margin
        x2 = x1 + tw + 20
        y2 = y1 + th + baseline + 20
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)
        tx = x1 + 10
        ty = y2 - baseline - 8
        cv2.putText(overlay, text, (tx, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        return overlay

    def _render_preview(frame: np.ndarray) -> tuple[np.ndarray, bool]:
        if yolo_preview_backend is None:
            return frame, False
        try:
            results = yolo_preview_backend.predict(
                {
                    "source": frame,
                    "device": yolo_device,
                    "verbose": False,
                }
            )
            if isinstance(results, list) and results:
                result = results[0]
                return result.plot(), _wrist_detected_from_result(result)
        except Exception:
            pass
        return frame, False

    if warmup_frames < 0:
        raise ValueError("--warmup-frames must be >= 0")
    if delay_seconds < 0:
        raise ValueError("--capture-delay-seconds must be >= 0")
    if capture_width < 1 or capture_height < 1:
        raise ValueError("--capture-width and --capture-height must be >= 1")
    if wrist_stable_seconds < 0:
        raise ValueError("--capture-wrist-stable-seconds must be >= 0")

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = _open_camera(camera_index, capture_width, capture_height)
    try:
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window, capture_width, capture_height)

        button_state: dict[str, bool] = {"clicked": auto_start}
        button_rect: dict[str, tuple[int, int, int, int] | None] = {"rect": None}

        def _on_mouse(event, mx, my, _flags, _userdata) -> None:
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            rect = button_rect["rect"]
            if rect is None:
                return
            x1, y1, x2, y2 = rect
            if x1 <= mx <= x2 and y1 <= my <= y2:
                button_state["clicked"] = True

        cv2.setMouseCallback(preview_window, _on_mouse)

        for _ in range(warmup_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Failed during camera warmup on index {camera_index}."
                )
            preview_source = cv2.flip(frame, 1)
            preview_render, wrist_detected = _render_preview(preview_source)
            preview_render = _draw_wrist_status(
                preview_render,
                wrist_detected=wrist_detected,
                capture_requested=button_state["clicked"],
                steady_elapsed=0.0,
                steady_required=wrist_stable_seconds,
            )
            preview, rect = _draw_start_button(preview_render, mode="idle")
            button_rect["rect"] = rect
            cv2.imshow(preview_window, preview)
            cv2.waitKey(1)

        print("Preview ready. Click 'Start Capture' or press SPACE/ENTER to begin countdown.")
        capture_at: float | None = None
        last_printed_second: int | None = None
        waiting_for_wrist_printed = False
        waiting_for_stable_printed = False
        wrist_seen_since: float | None = None

        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Failed to read frame from camera index {camera_index}."
                )

            now = perf_counter()
            preview_source = cv2.flip(frame, 1)
            render, wrist_detected = _render_preview(preview_source)

            if wrist_detected:
                if wrist_seen_since is None:
                    wrist_seen_since = now
            else:
                wrist_seen_since = None

            steady_elapsed = 0.0 if wrist_seen_since is None else (now - wrist_seen_since)
            wrist_is_stable = steady_elapsed >= wrist_stable_seconds

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                raise RuntimeError("Capture cancelled by operator.")
            if key in (ord(" "), 13):
                button_state["clicked"] = True

            if button_state["clicked"] and capture_at is None:
                if wrist_is_stable:
                    capture_at = now + float(delay_seconds)
                    waiting_for_wrist_printed = False
                    waiting_for_stable_printed = False
                    if delay_seconds > 0:
                        print(f"Hold your pose. Capturing in {delay_seconds} second(s)...")
                elif not wrist_detected and not waiting_for_wrist_printed:
                    print("Capture requested. Waiting for wrist detection...")
                    waiting_for_wrist_printed = True
                elif wrist_detected and not waiting_for_stable_printed and wrist_stable_seconds > 0:
                    print(
                        "Wrist detected. Hold steady "
                        f"for {wrist_stable_seconds:.1f}s to start countdown..."
                    )
                    waiting_for_stable_printed = True

            if not button_state["clicked"]:
                waiting_for_wrist_printed = False
                waiting_for_stable_printed = False

            active = capture_at is not None
            render = _draw_wrist_status(
                render,
                wrist_detected=wrist_detected,
                capture_requested=button_state["clicked"],
                steady_elapsed=steady_elapsed,
                steady_required=wrist_stable_seconds,
            )
            if active:
                button_mode = "countdown"
            elif button_state["clicked"]:
                button_mode = "waiting_wrist"
            else:
                button_mode = "idle"
            render, rect = _draw_start_button(render, mode=button_mode)
            button_rect["rect"] = rect

            if capture_at is not None and delay_seconds > 0:
                seconds_left = max(0, int(math.ceil(capture_at - now)))
                render = _draw_countdown_overlay(render, seconds_left)
                if seconds_left != last_printed_second and seconds_left > 0:
                    print(f"  {seconds_left}...")
                    last_printed_second = seconds_left

            cv2.imshow(preview_window, render)

            if capture_at is None:
                continue

            if now >= capture_at:
                frame_to_save = frame.copy()
                cv2.imshow(preview_window, render)
                cv2.waitKey(120)
                break

        if not cv2.imwrite(str(output_path), frame_to_save):
            raise RuntimeError(f"Could not write captured image to {output_path}.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"Captured still image: {output_path}")
    return output_path


def _run_rmbg_on_captured_image(
    captured_image_path: Path,
    model_dir: Path,
    device: str,
    mask_output_path: Path,
    foreground_output_path: Path,
    mask_quality: str,
) -> tuple[Path, Path]:
    import cv2

    captured_image_path = captured_image_path.expanduser().resolve()
    mask_output_path = mask_output_path.expanduser().resolve()
    foreground_output_path = foreground_output_path.expanduser().resolve()
    mask_output_path.parent.mkdir(parents=True, exist_ok=True)
    foreground_output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(captured_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read captured image at {captured_image_path}.")

    runtime = load_rmbg_runtime(model_dir, device=device)
    mask, foreground = runtime.segment(image)

    # Optional generic quality cleanup even before primary-subject composition.
    if mask_quality in {"balanced", "high"}:
        kernel_size = 3 if mask_quality == "balanced" else 5
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        bin_mask = (mask > (24 if mask_quality == "balanced" else 32)).astype(np.uint8)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
        bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
        mask = (mask * bin_mask).astype(np.uint8)
        foreground[:, :, 3] = mask

    if not cv2.imwrite(str(mask_output_path), mask):
        raise RuntimeError(f"Could not write RMBG mask to {mask_output_path}.")
    if not cv2.imwrite(str(foreground_output_path), foreground):
        raise RuntimeError(
            f"Could not write RMBG foreground to {foreground_output_path}."
        )

    print(f"RMBG mask saved: {mask_output_path}")
    print(f"RMBG foreground saved: {foreground_output_path}")
    return mask_output_path, foreground_output_path


def _run_juggernaut_tests(args) -> None:
    from .vision.juggernaut_runtime import (
        load_juggernaut_pipeline,
        render_img2img,
        render_text2img,
    )

    preset_map = {
        "identity-lock": {
            "steps": 28,
            "guidance_scale": 4.2,
            "guided_strength": 0.16,
        },
        "balanced": {
            "steps": 24,
            "guidance_scale": 4.5,
            "guided_strength": 0.20,
        },
        "stylized": {
            "steps": 30,
            "guidance_scale": 6.0,
            "guided_strength": 0.35,
        },
    }

    steps, guidance_scale, guided_strength = _resolve_juggernaut_profile(args, preset_map)

    _validate_juggernaut_dimensions_and_strength(
        width=args.juggernaut_width,
        height=args.juggernaut_height,
        guided_strength=guided_strength,
        steps=steps,
    )

    print(
        "Loading Juggernaut pipeline: "
        f"{args.juggernaut_model_id} (device={args.juggernaut_device}, "
        f"local_only={args.juggernaut_local_only})"
    )
    print(
        "Juggernaut preset: "
        f"{args.juggernaut_preset} "
        f"(steps={steps}, guidance={guidance_scale}, guided_strength={guided_strength})"
    )
    pipeline = load_juggernaut_pipeline(
        model_id=args.juggernaut_model_id,
        device=args.juggernaut_device,
        local_files_only=args.juggernaut_local_only,
        openvino_cache_dir=args.juggernaut_openvino_cache_dir,
        task="text2img",
    )

    smoke_path = render_text2img(
        pipeline=pipeline,
        output_path=args.juggernaut_smoke_output,
        prompt=args.juggernaut_prompt,
        negative_prompt=args.juggernaut_negative_prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        width=args.juggernaut_width,
        height=args.juggernaut_height,
        seed=args.juggernaut_seed,
    )
    print(f"Juggernaut smoke render saved: {smoke_path}")

    guided_input = args.juggernaut_guided_input_image.expanduser().resolve()
    if args.juggernaut_skip_guided:
        print("Juggernaut guided render skipped by --juggernaut-skip-guided.")
        return
    if not guided_input.exists():
        print(
            "Juggernaut guided render skipped because guide image is missing: "
            f"{guided_input}"
        )
        return

    guided_pipeline = load_juggernaut_pipeline(
        model_id=args.juggernaut_model_id,
        device=args.juggernaut_device,
        local_files_only=args.juggernaut_local_only,
        openvino_cache_dir=args.juggernaut_openvino_cache_dir,
        task="img2img",
    )
    guided_path = render_img2img(
        pipeline=guided_pipeline,
        input_image_path=guided_input,
        output_path=args.juggernaut_guided_output,
        prompt=args.juggernaut_prompt,
        negative_prompt=args.juggernaut_negative_prompt,
        steps=steps,
        guidance_scale=guidance_scale,
        strength=guided_strength,
        width=args.juggernaut_width,
        height=args.juggernaut_height,
        seed=args.juggernaut_seed,
    )
    print(f"Juggernaut guided render saved: {guided_path}")


def _resolve_juggernaut_profile(args, preset_map: dict[str, dict[str, float]]) -> tuple[int, float, float]:
    profile = preset_map[args.juggernaut_preset]
    steps = args.juggernaut_steps if args.juggernaut_steps is not None else profile["steps"]
    guidance_scale = (
        args.juggernaut_guidance_scale
        if args.juggernaut_guidance_scale is not None
        else profile["guidance_scale"]
    )
    guided_strength = (
        args.juggernaut_guided_strength
        if args.juggernaut_guided_strength is not None
        else profile["guided_strength"]
    )

    return int(steps), float(guidance_scale), float(guided_strength)


def _validate_juggernaut_dimensions_and_strength(
    width: int,
    height: int,
    guided_strength: float,
    steps: int,
) -> None:

    if steps < 1:
        raise ValueError("--juggernaut-steps must be >= 1")
    if width < 64 or height < 64:
        raise ValueError("--juggernaut-width and --juggernaut-height must be >= 64")
    if not 0.0 <= guided_strength <= 1.0:
        raise ValueError("--juggernaut-guided-strength must be within [0.0, 1.0]")


def _run_social_pipeline(args) -> None:
    from .vision.juggernaut_runner import JuggernautRunner
    from .vision.juggernaut_types import RenderRequest

    preset_map = {
        "identity-lock": {"steps": 28, "guidance_scale": 4.2, "guided_strength": 0.16},
        "balanced": {"steps": 24, "guidance_scale": 4.5, "guided_strength": 0.20},
        "stylized": {"steps": 30, "guidance_scale": 6.0, "guided_strength": 0.35},
    }
    steps, guidance_scale, guided_strength = _resolve_juggernaut_profile(args, preset_map)

    _validate_juggernaut_dimensions_and_strength(
        width=args.juggernaut_width,
        height=args.juggernaut_height,
        guided_strength=guided_strength,
        steps=steps,
    )

    t_total = perf_counter()

    # Loading takes ~29s, so start it before the camera work and let it overlap posing and compose.
    # Warmup is deferred: compiling does not disturb the preview, but diffusion inference does
    # (measured 30fps -> 8fps), so it runs after the camera is released.
    runner: JuggernautRunner | None = None
    if not args.juggernaut_skip_guided and not args.juggernaut_no_preload:
        runner = JuggernautRunner(
            model_id=args.juggernaut_model_id,
            device=args.juggernaut_device,
            local_files_only=args.juggernaut_local_only,
            openvino_cache_dir=args.juggernaut_openvino_cache_dir,
            task="img2img",
            warmup=False,
            warmup_width=args.juggernaut_width,
            warmup_height=args.juggernaut_height,
        ).start()
        print("Juggernaut pipeline loading in the background during capture.")

    backend = build_backend(args.backend)
    backend.load_model(args.model_path)

    t_capture = perf_counter()
    captured_path = _capture_single_image(
        camera_index=args.camera_index,
        output_path=args.capture_output,
        warmup_frames=args.warmup_frames,
        delay_seconds=args.capture_delay_seconds,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        yolo_preview_backend=backend,
        yolo_device=args.device,
        auto_start=args.capture_auto_start,
        wrist_stable_seconds=args.capture_wrist_stable_seconds,
    )
    capture_sec = perf_counter() - t_capture

    if runner is not None:
        runner.submit_warmup()  # camera is released; overlap warmup with RMBG and compose

    rmbg_sec = 0.0

    t_compose = perf_counter()
    deterministic_path = _compose_portrait_from_image(
        backend=backend,
        input_image_path=captured_path,
        rmbg_model_dir=args.rmbg_model_dir,
        rmbg_device=args.rmbg_device,
        yolo_device=args.device,
        scene_image_path=args.compose_scene_image,
        prop_image_path=args.compose_prop_image,
        output_image_path=args.compose_output_image,
        mask_quality=args.mask_quality,
        mask_output_path=None if args.capture_skip_rmbg else args.capture_rmbg_mask_output,
        foreground_output_path=None if args.capture_skip_rmbg else args.capture_rmbg_foreground_output,
    )
    compose_sec = perf_counter() - t_compose

    final_output = args.final_output_image.expanduser().resolve()
    final_output.parent.mkdir(parents=True, exist_ok=True)

    juggernaut_sec = 0.0
    used_fallback = False
    if args.juggernaut_skip_guided:
        shutil.copy2(deterministic_path, final_output)
        used_fallback = True
        print(
            "One-shot pipeline final output uses deterministic compose because "
            "--juggernaut-skip-guided was set."
        )
    else:
        t_juggernaut = perf_counter()
        try:
            print(
                "Juggernaut preset: "
                f"{args.juggernaut_preset} "
                f"(steps={steps}, guidance={guidance_scale}, guided_strength={guided_strength})"
            )
            if runner is None:
                runner = JuggernautRunner(
                    model_id=args.juggernaut_model_id,
                    device=args.juggernaut_device,
                    local_files_only=args.juggernaut_local_only,
                    openvino_cache_dir=args.juggernaut_openvino_cache_dir,
                    task="img2img",
                    warmup_width=args.juggernaut_width,
                    warmup_height=args.juggernaut_height,
                ).start()

            result = runner.render(
                RenderRequest(
                    mode="img2img",
                    prompt=args.juggernaut_prompt,
                    negative_prompt=args.juggernaut_negative_prompt,
                    output_path=args.juggernaut_guided_output,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    strength=guided_strength,
                    width=args.juggernaut_width,
                    height=args.juggernaut_height,
                    seed=args.juggernaut_seed,
                    input_image_path=deterministic_path,
                )
            )
            if not result.ok or result.output_path is None:
                raise RuntimeError(result.error or "guided render returned no output")
            print(
                f"Juggernaut guided render: {result.render_seconds:.2f}s "
                f"(waited {result.queue_wait_seconds:.2f}s for the pipeline)"
            )
            shutil.copy2(result.output_path, final_output)
            print(f"One-shot pipeline final output saved from Juggernaut: {final_output}")
        except Exception as exc:
            used_fallback = True
            shutil.copy2(deterministic_path, final_output)
            print(
                "Juggernaut guided render failed; used deterministic fallback. "
                f"Reason: {exc}"
            )
        juggernaut_sec = perf_counter() - t_juggernaut

    if runner is not None:
        runner.shutdown()

    total_sec = perf_counter() - t_total
    print(
        "One-shot pipeline timings (seconds): "
        f"capture={capture_sec:.2f}, rmbg={rmbg_sec:.2f}, compose={compose_sec:.2f}, "
        f"juggernaut={juggernaut_sec:.2f}, total={total_sec:.2f}"
    )
    print(
        "One-shot pipeline summary: "
        f"captured={captured_path}, deterministic={deterministic_path}, "
        f"final={final_output}, fallback_used={used_fallback}"
    )


def _compose_portrait_from_image(
    backend: object,
    input_image_path: Path,
    rmbg_model_dir: Path,
    rmbg_device: str,
    yolo_device: str,
    scene_image_path: Path,
    prop_image_path: Path,
    output_image_path: Path,
    mask_quality: str,
    mask_output_path: Path | None = None,
    foreground_output_path: Path | None = None,
) -> Path:
    from .vision.pipeline import compose_portrait_from_image

    return compose_portrait_from_image(
        backend=backend,
        input_image_path=input_image_path,
        rmbg_model_dir=rmbg_model_dir,
        rmbg_device=rmbg_device,
        yolo_device=yolo_device,
        scene_image_path=scene_image_path,
        prop_image_path=prop_image_path,
        output_image_path=output_image_path,
        mask_quality=mask_quality,
        mask_output_path=mask_output_path,
        foreground_output_path=foreground_output_path,
    )


def _run_smoke_test(
    backend: object,
    camera_index: int,
    device: str,
    max_frames: int,
    show: bool,
) -> None:
    import cv2

    if max_frames < 1:
        raise ValueError("--max-frames must be >= 1")

    cap = _open_camera(camera_index)
    print(
        f"Starting smoke test: camera={camera_index}, device={device}, frames={max_frames}"
    )

    processed = 0
    start = perf_counter()

    try:
        while processed < max_frames:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"Failed to read frame from camera index {camera_index}."
                )

            # Send one frame at a time through the active backend.
            results = backend.predict(
                {
                    "source": frame,
                    "device": device,
                    "verbose": False,
                }
            )

            if show:
                annotated = frame
                if isinstance(results, list) and results:
                    try:
                        annotated = results[0].plot()
                    except Exception:
                        annotated = frame

                cv2.imshow("vPRO Smoke Test", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
    finally:
        cap.release()
        if show:
            cv2.destroyAllWindows()

    elapsed = perf_counter() - start
    fps = processed / elapsed if elapsed > 0 else 0.0
    print(
        "Smoke test passed: "
        f"processed={processed}, elapsed={elapsed:.2f}s, avg_fps={fps:.2f}"
    )


def _run_kiosk(args) -> None:
    from .kiosk.app import run
    from .kiosk.service import KioskConfig, KioskService

    config = KioskConfig(
        output_dir=args.kiosk_output_dir,
        camera_index=args.camera_index,
        capture_width=args.capture_width,
        capture_height=args.capture_height,
        capture_rotate=args.capture_rotate,
        pose_fps=args.kiosk_pose_fps,
        yolo_device=args.device,
        rmbg_model_dir=args.rmbg_model_dir,
        rmbg_device=args.rmbg_device,
        mask_quality=args.mask_quality,
        scene_image=args.compose_scene_image,
        prop_image=args.compose_prop_image,
        juggernaut_model_id=args.juggernaut_model_id,
        juggernaut_device=args.juggernaut_device,
        juggernaut_vae_precision_hint=args.juggernaut_vae_precision or None,
        juggernaut_cache_dir=args.juggernaut_openvino_cache_dir,
        local_files_only=args.juggernaut_local_only,
        steps=args.juggernaut_steps if args.juggernaut_steps is not None else 30,
        guidance_scale=args.juggernaut_guidance_scale if args.juggernaut_guidance_scale is not None else 5.0,
        strength=args.juggernaut_guided_strength if args.juggernaut_guided_strength is not None else 0.99,
        countdown_seconds=args.capture_delay_seconds,
        delivery_channel=args.delivery_channel,
        delivery_host=args.delivery_host,
        delivery_port=args.delivery_port,
        delivery_advertise_host=args.delivery_advertise_host,
        enable_generation=not args.kiosk_no_generation,
        output_retention_hours=args.kiosk_output_retention_hours,
    )
    # --model-path defaults to the generic "models" dir, which is not a YOLO export directory.
    if args.model_path != Path("models"):
        config.yolo_model_path = args.model_path

    _validate_juggernaut_dimensions_and_strength(
        width=config.width, height=config.height, guided_strength=config.strength, steps=config.steps
    )
    service = KioskService(config)
    try:
        service.start()
    except BaseException:
        service.shutdown()
        raise
    print(f"\nKiosk UI:  http://{args.kiosk_host}:{args.kiosk_port}")
    if service.delivery is not None and hasattr(service.delivery, "server"):
        print(f"Handoff:   {service.delivery.server.base_url} (guest phones)")
    print("Ctrl+C to stop.\n")
    try:
        run(service, host=args.kiosk_host, port=args.kiosk_port, open_browser=args.open_browser)
    except KeyboardInterrupt:
        pass
    finally:
        service.shutdown()


def _run_delivery_test(args) -> None:
    import sys

    from .delivery import DeliveryRequest, build_delivery, render_qr_terminal

    # The QR uses block characters that a default Windows console encoding cannot represent.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    image_path = args.delivery_image.expanduser().resolve()
    if not image_path.exists():
        raise SystemExit(
            f"Image not found: {image_path}. Pass --delivery-image with a rendered output."
        )

    kwargs: dict[str, object] = {}
    if args.delivery_channel == "local-qr":
        kwargs = {
            "host": args.delivery_host,
            "port": args.delivery_port,
            "advertise_host": args.delivery_advertise_host,
        }

    channel = build_delivery(args.delivery_channel, **kwargs)
    try:
        result = channel.deliver(
            DeliveryRequest(image_path=image_path, caption=args.delivery_caption)
        )
        if not result.ok:
            raise SystemExit(f"Delivery failed on {result.channel}: {result.error}")

        print(f"\nchannel: {result.channel}")
        print(f"url:     {result.url}")
        if result.expires_in_seconds:
            print(f"expires: {result.expires_in_seconds / 60:.0f} min")
        if result.url and args.delivery_channel == "local-qr":
            print(render_qr_terminal(result.url))
            print(
                "Scan with a phone camera. If nothing loads, the phone likely cannot reach this\n"
                "machine: guest WiFi often blocks device-to-device traffic (client isolation).\n"
                "Try a phone on the same network first, then a hotspot.\n"
                "Ctrl+C to stop."
            )
        try:
            while True:
                sleep(1)
        except KeyboardInterrupt:
            print("\nstopping handoff server")
    finally:
        channel.close()


def _apply_npu_preference(parser: ArgumentParser, args) -> None:
    """Route the vision models to the NPU, leaving anything the caller set explicitly alone."""
    if not args.npu:
        return

    import openvino

    available = openvino.Core().available_devices
    if "NPU" not in available:
        raise SystemExit(f"--npu requested but no NPU device found. Available: {available}")

    if args.device == parser.get_default("device"):
        args.device = "intel:npu"
    print(f"NPU mode: yolo={args.device}, rmbg={args.rmbg_device}, juggernaut={args.juggernaut_device}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _apply_npu_preference(parser, args)

    if args.test_delivery:
        _run_delivery_test(args)
        return

    if args.kiosk:
        _run_kiosk(args)
        return

    if args.test_juggernaut:
        _run_juggernaut_tests(args)
        return

    if args.run_social_pipeline:
        _run_social_pipeline(args)
        return

    if args.validate_rmbg:
        model_dir = validate_rmbg_assets(args.rmbg_model_dir)
        runtime = load_rmbg_runtime(model_dir, device=args.rmbg_device)
        runtime.warmup(runs=1)
        print(f"RMBG validation passed: {model_dir} on device={args.rmbg_device}")
        return

    if args.capture_single_image:
        preview_backend = None
        try:
            preview_backend = build_backend(args.backend)
            preview_backend.load_model(args.model_path)
        except Exception as exc:
            print(
                "Warning: Could not load YOLO preview backend for capture mode; "
                f"falling back to raw camera preview. Reason: {exc}"
            )

        captured_path = _capture_single_image(
            camera_index=args.camera_index,
            output_path=args.capture_output,
            warmup_frames=args.warmup_frames,
            delay_seconds=args.capture_delay_seconds,
            capture_width=args.capture_width,
            capture_height=args.capture_height,
            yolo_preview_backend=preview_backend,
            yolo_device=args.device,
            auto_start=args.capture_auto_start,
            wrist_stable_seconds=args.capture_wrist_stable_seconds,
        )

        if not args.capture_skip_rmbg:
            _run_rmbg_on_captured_image(
                captured_image_path=captured_path,
                model_dir=args.rmbg_model_dir,
                device=args.rmbg_device,
                mask_output_path=args.capture_rmbg_mask_output,
                foreground_output_path=args.capture_rmbg_foreground_output,
                mask_quality=args.mask_quality,
            )
        return

    backend = build_backend(args.backend)
    backend.load_model(args.model_path)

    if args.compose_portrait:
        _compose_portrait_from_image(
            backend=backend,
            input_image_path=args.compose_input_image,
            rmbg_model_dir=args.rmbg_model_dir,
            rmbg_device=args.rmbg_device,
            yolo_device=args.device,
            scene_image_path=args.compose_scene_image,
            prop_image_path=args.compose_prop_image,
            output_image_path=args.compose_output_image,
            mask_quality=args.mask_quality,
        )
        return

    if args.smoke_test:
        _run_smoke_test(
            backend=backend,
            camera_index=args.camera_index,
            device=args.device,
            max_frames=args.max_frames,
            show=args.show,
        )


if __name__ == "__main__":
    main()
