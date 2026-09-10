# vPRO

uv-based Python project scaffold designed to support two inference backends:

- OpenVINO (`openvino`) as the primary backend
- PyTorch (`torch`) as optional secondary backend

Target OpenVINO version line: `2026.3.x`

The architecture is OpenVINO-first for Intel hardware and keeps PyTorch optional when needed.

YOLO26 runtime behavior mirrors the local demo at `C:\Users\bjodom\ai_projects\ultralytics-demo`:
- Accept either `.pt` checkpoints or pre-exported `_openvino_model` directories.
- Auto-export to OpenVINO once when needed.
- Reuse existing OpenVINO artifacts on subsequent runs.

## Quick Start

```powershell
uv sync

# Get Juggernaut XL for the final guided render. This downloads the OpenVINO
# Juggernaut XL model into the local Hugging Face cache for reuse.
uvx --from huggingface_hub hf download OpenVINO/Juggernaut-XL-v9-fp16-ov

# Run the complete app flow using the cloned repository layout.
uv run vpro --run-social-pipeline --backend openvino --model-path models/yolo26/yolo26x-pose_openvino_model --device intel:gpu --camera-index 0 --compose-scene-image assets/scenes/portrait_scene_1080x1350.jpg --compose-prop-image assets/props/lenovo.laptop.png --juggernaut-model-id OpenVINO/Juggernaut-XL-v9-fp16-ov
```

## Install Optional Backend Extras

Install PyTorch extra:

```powershell
uv sync --extra torch
```

## CLI

```powershell
uv run vpro --backend openvino --model-path models
uv run vpro --backend torch --model-path models
```

## Kiosk Performance Controls

The kiosk keeps YOLO, RMBG, and Juggernaut resources resident for the life of the process. RMBG is
loaded and warmed once at startup, while static scene and prop assets are cached between guests.
The preview camera continues capturing at the configured camera rate, but YOLO pose inference is
limited to 15 FPS by default so the live preview leaves GPU headroom for other work.

Tune the pose rate and output retention window for a target booth:

```powershell
uv run vpro --kiosk --kiosk-pose-fps 10 --kiosk-output-retention-hours 48
```

The kiosk state endpoint reports the latest deterministic-pipeline timings and Juggernaut queue
rejections. Re-run `tests/gpu_contention_test.py` on each target hardware configuration; integrated
and discrete Intel GPUs can behave differently under concurrent camera and diffusion workloads.

## OpenVINO Smoke Test

Run a short webcam inference smoke test (camera open + model inference loop):

```powershell
uv run vpro --backend openvino --model-path models/yolo26/yolo26x-pose.pt --smoke-test --device intel:gpu --camera-index 0 --max-frames 30
```

Show live annotated frames during the smoke test:

```powershell
uv run vpro --backend openvino --model-path models/yolo26/yolo26x-pose.pt --smoke-test --show --device intel:gpu --camera-index 0 --max-frames 30
```

If you already have exported OpenVINO artifacts, pass the export folder instead of a `.pt` file:

```powershell
uv run vpro --backend openvino --model-path models/yolo26/yolo26x-pose_openvino_model --smoke-test --device intel:npu --camera-index 0 --max-frames 30
```

## Capture Single Render Input Image

Extract one still image from the live camera feed and run RMBG end-to-end (preprocess -> OpenVINO inference -> mask postprocess):

```powershell
uv run vpro --backend openvino --model-path models/yolo26/yolo26x-pose_openvino_model --device intel:gpu --capture-single-image --camera-index 0 --capture-width 2560 --capture-height 1440 --capture-output outputs/visitor_capture.jpg --capture-rmbg-mask-output outputs/visitor_capture_mask.png --capture-rmbg-foreground-output outputs/visitor_capture_no_bg.png --warmup-frames 12 --capture-delay-seconds 3
```

This command saves:
- captured frame (`outputs/visitor_capture.jpg`)
- RMBG mask (`outputs/visitor_capture_mask.png`)
- RMBG foreground with alpha (`outputs/visitor_capture_no_bg.png`)

To capture only and skip RMBG processing:

```powershell
uv run vpro --capture-single-image --capture-skip-rmbg
```

Mask quality presets:
- fast: minimal cleanup, lowest latency
- balanced: medium cleanup
- high: strongest cleanup for harder backgrounds

Capture mode now runs as a kiosk flow:
- live preview with YOLO26 annotations at requested camera resolution (default 2560x1440)
- on-screen Start Capture button (click to begin countdown)
- lower-left countdown timer while the subject holds pose
- capture gate: countdown does not start until a wrist keypoint is detected and held steadily

By default, capture waits 3 seconds before taking the still after Start Capture is pressed and a wrist is detected steadily for 0.7 seconds. Set `--capture-delay-seconds 0` for immediate capture once the wrist stability gate is satisfied.

Adjust wrist stability gate if needed:

```powershell
uv run vpro --capture-single-image --capture-wrist-stable-seconds 1.0
```

For unattended testing, skip click-to-start with:

```powershell
uv run vpro --capture-single-image --capture-auto-start
```

Example:

```powershell
uv run vpro --capture-single-image --mask-quality high
```

## Validate RMBG Runtime Assets

Validate RMBG model files and run one OpenVINO warmup inference:

```powershell
uv run vpro --validate-rmbg --rmbg-model-dir models/rmbg/rmbg-1.4 --rmbg-device AUTO
```

## Compose Final Portrait (1080x1350)

Run deterministic portrait composition with all four behaviors:
- primary-subject auto-selection (multi-person tolerant)
- subject auto-scale and portrait placement
- deterministic laptop prop anchor from YOLO keypoints
- final export as 1080x1350 image

```powershell
uv run vpro --backend openvino --model-path models/yolo26/yolo26x-pose_openvino_model --compose-portrait --compose-input-image outputs/visitor_capture.jpg --compose-scene-image assets/scenes/portrait_scene_1080x1350.jpg --compose-prop-image assets/props/vpro_laptop.png --compose-output-image outputs/final_portrait_1080x1350.jpg --device intel:gpu --rmbg-model-dir models/rmbg/rmbg-1.4 --rmbg-device AUTO
```

For tighter foreground masking in cluttered scenes, add:

```powershell
--mask-quality high
```

Notes:
- `--compose-scene-image` must point to a portrait background image.
- `--compose-prop-image` must point to an alpha PNG laptop render.

## One-Shot Social Pipeline (Capture -> Compose -> Juggernaut)

Run a production-style single command flow with automatic fallback:
1. Capture image from camera
2. Deterministic portrait compose
3. Juggernaut guided render
4. If Juggernaut fails, fallback to deterministic output

```powershell
uv run vpro --run-social-pipeline --backend openvino --model-path models/yolo26/yolo26x-pose_openvino_model --device intel:gpu --camera-index 0 --compose-scene-image assets/scenes/portrait_scene_1080x1350.jpg --compose-prop-image assets/props/lenovo.laptop.png
```

Useful outputs:
- deterministic compose: `--compose-output-image` (default `outputs/final_portrait_1080x1350.jpg`)
- final deliverable: `--final-output-image` (default `outputs/final_portrait_1080x1350_final.jpg`)
- guided render artifact: `--juggernaut-guided-output` (default `outputs/juggernaut_guided.jpg`)

The command prints stage timings for capture, RMBG, compose, Juggernaut, and total runtime.

## Juggernaut Local OpenVINO Test

Install test dependencies for local OpenVINO diffusion rendering:

```powershell
uv sync --extra gen
```

Run both test stages (single pipeline load):
- stage 1: text-to-image smoke render
- stage 2: image-guided render using the composed portrait as guide

```powershell
uv run vpro --test-juggernaut --juggernaut-model-id OpenVINO/Juggernaut-XL-v9-fp16-ov --juggernaut-guided-input-image outputs/final_portrait_1080x1350_lenovo_test.jpg
```

The standalone prompt test stores reusable compiled OpenVINO artifacts in `outputs/openvino_cache/juggernaut` by default. This cache survives normal process exits and avoids rebuilding GPU kernels when the model, target device or driver, OpenVINO version, and compilation-relevant shapes/settings are unchanged. Optimum still reloads the multi-gigabyte SDXL OpenVINO model files in each new Python process, so use a long-lived application process when low latency across separate requests matters. Override the cache location with `--openvino-cache-dir PATH`.

By default this allows first-run download and then reuses the local Hugging Face cache on subsequent runs. For offline/cache-only mode, add `--juggernaut-local-only`.

Juggernaut presets:
- `identity-lock`: stronger identity retention (lower guided strength)
- `balanced`: default profile for kiosk testing
- `stylized`: more creative/stylized output with higher drift risk

Text-to-image prompt-test presets:

| Preset | Steps | CFG (`guidance_scale`) |
|---|---:|---:|
| `fast` | 16 | 3.5 |
| `balanced` | 24 | 4.5 |
| `quality` | 36 | 5.5 |
| `stylized` | 30 | 6.0 |

CFG means classifier-free guidance: it controls how strongly the image follows the prompt. Lower values allow more natural variation; higher values follow the prompt more strictly but can look harsher or oversaturated.

Preset flag:

```powershell
uv run vpro --test-juggernaut --juggernaut-preset balanced
```

Default outputs:
- `outputs/juggernaut_smoke.jpg`
- `outputs/juggernaut_guided.jpg`

Useful flags:
- `--juggernaut-device GPU`
- `--juggernaut-openvino-cache-dir outputs/openvino_cache/juggernaut` (persistent compiled-model cache)
- `--juggernaut-preset balanced`
- `--juggernaut-steps 24` (manual override)
- `--juggernaut-width 1080 --juggernaut-height 1350`
- `--juggernaut-guidance-scale 4.5` (manual override)
- `--juggernaut-guided-strength 0.20` (manual override)
- `--juggernaut-seed 42`

## Project Layout

```text
src/vpro/
	cli.py
	backends/
		base.py
		factory.py
		torch_backend.py
		openvino_backend.py
	vision/
		portrait_compositor.py
		rmbg_runtime.py
		yolo26_runtime.py
```

OpenVINO backend now resolves `.pt` checkpoints to OpenVINO exports automatically and loads them with Ultralytics YOLO.
