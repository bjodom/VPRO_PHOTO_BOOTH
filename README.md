# vPRO

Local photo-booth kiosk with consent, destination selection, camera preview, portrait generation,
and QR delivery. OpenVINO models remain resident between guests. Two inference backends are available:

- OpenVINO (`openvino`) as the primary backend
- PyTorch (`torch`) as optional secondary backend

Target OpenVINO version line: `2026.4.x`

The architecture is OpenVINO-first for Intel hardware and keeps PyTorch optional when needed.

YOLO26 runtime behavior mirrors the local demo at `C:\Users\bjodom\ai_projects\ultralytics-demo`:
- Accept either `.pt` checkpoints or pre-exported `_openvino_model` directories.
- Auto-export to OpenVINO once when needed.
- Reuse existing OpenVINO artifacts on subsequent runs.

## Quick Start

```powershell
# One-time setup with network access. Add -UseIntelProxy on the Intel network.
.\run.ps1 -Sync -DownloadJuggernaut

# Daily event launch. Opens the browser after the local server starts.
.\run.ps1
```

The launcher uses `.venv` directly, defaults to offline model loading, and installs both `gen`
and `kiosk` extras when `-Sync` is requested. Local YOLO and RMBG XML/BIN artifacts and the laptop
prop must already be present. The browser is at http://127.0.0.1:8000 by default.

Your selected device defaults are YOLO26 `intel:npu`, RMBG `CPU`, and Juggernaut `GPU`.
RMBG stays off the generation GPU to avoid shared-GPU contention. Use `-RmbgDevice GPU`
only as an explicit override. This is a deployment profile, not a claim that it is optimal
on every Intel system.
The renderer loads and warms before the first guest can start. The normal kiosk uses inpainting;
historical 5-6 second img2img measurements are not an established kiosk latency guarantee.

Useful launcher options:

```powershell
# Set the LAN address that guest phones can reach.
.\run.ps1 -DeliveryAdvertiseHost 192.168.1.100

# Isolate acceptance-run photos and select a different UI port.
.\run.ps1 -OutputDir outputs/event_validation -KioskPort 8001

# Diagnostic one-shot flow only, not the guest kiosk.
.\run.ps1 -Mode social -CaptureAutoStart -JuggernautPreset identity-lock

# Verify emitted arguments without starting camera or models.
.\run.ps1 -DryRun
```

Run `.\run.ps1 -?` to see all launcher parameters. The lower-level `uv run vpro` commands below
remain available for individual smoke tests and development workflows; prefer `uv run --no-sync`
after setup so a plain sync does not remove optional packages.

## Event Acceptance

Destination selection also includes **Somewhere else**: enter a place name (2-100 characters),
then choose **Use this place**. Custom locations remain local to the guest session and clear on reset.
Both preset and custom destinations use masked inpainting on a neutral canvas conditioned on the
captured portrait; the selection thumbnails are never used as guest-photo backgrounds. The person
and laptop are positioned by the compositor and protected by the mask while the model generates
the surroundings. Perspective, ground-plane, and contact-shadow prompts encourage a coherent scene,
but this is not automatic reposing or full relighting of the preserved person. Unfamiliar or very
specific places may be approximate because no online geographic lookup is performed.

Run the hardware-free regression gate with `.\tests\cli_pipeline_test.ps1`.
Then follow [the event checklist](docs/event_readiness.md) on the actual booth and guest network.
Passing unit tests does not certify camera quality, generated likeness, or phone connectivity.

### RMBG GPU Regression

Stop the kiosk and other inference workloads first. Using a saved photo with a visible person,
run the opt-in shared-GPU regression (replace `saved-photo.jpg` with the actual path):

```powershell
.\.venv\Scripts\python.exe tests/rmbg_generation_test.py --image saved-photo.jpg

# Separate-process control: same photo, seed, and renders; only RMBG moves to CPU.
.\.venv\Scripts\python.exe tests/rmbg_generation_test.py --image saved-photo.jpg --rmbg-device CPU
```

The test keeps RMBG resident, checks its raw output before Juggernaut loading, after loading,
after warmup, and after each of three inpainting renders (including the last). It uses the
production runtimes, fixed seed, 30 steps, and 1080x1350 output, without retries masking failures.
It uses cached models only and never opens the camera. Allow several minutes per invocation.
It isolates RMBG/Juggernaut coexistence, not the full kiosk's YOLO, composition, or delivery flow.

Each invocation creates a new `outputs/rmbg_generation_*` directory containing a saved-image
guide, a baseline-derived inpainting mask, rendered photos, and `metrics.jsonl`. Records include
the failure stage, raw non-finite counts, finite min/max, and render errors. Any invalid mask,
render failure, or timeout returns exit code 1. A failed baseline stops before Juggernaut loads;
later segmentation failures remain recorded while the remaining renders use the fixed fixture.
These diagnostic photos are not covered by kiosk retention; remove them after investigation.
The standard hardware-free gate tests the probe's logic with mocks, but does not run GPU inference.

The kiosk does not deliver a gray intermediate image as a successful AI portrait. Rendering
failures offer retry; delivery failures retain the finished image and retry only the handoff.
Hold the Intel vPro header for 1.8 seconds (or focus it and press Ctrl+Enter) for staff controls.
This deliberate gesture prevents casual activation; it is not authentication. Keep the kiosk UI
bound to localhost and use Windows kiosk restrictions for unattended deployments.

Image retention runs every minute, with a default 24-hour window. It removes only owned capture,
composition, coverage, mask, and final artifacts, excluding active sessions and live QR links.
QR links expire after 15 minutes; link expiry is separate from on-disk image retention.

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
Raw camera acquisition runs separately from subscriber-driven preview processing. YOLO pose
inference is limited to 15 FPS by default and cached results are drawn on intervening previews.
JPEG previews are capped at 1280 pixels on the long edge; saved captures retain camera resolution.

Tune the pose rate and output retention window for a target booth:

```powershell
uv run vpro --kiosk --kiosk-pose-fps 10 --kiosk-output-retention-hours 48
```

The kiosk state endpoint reports stage timings, render and queue wait, queue depth, and rejections.
Outcome metrics are appended to `outputs/kiosk/pipeline_metrics.jsonl` without photo paths or tokens.
`/health` is liveness; `/health/ready` reports camera/renderer readiness and returns 503 while unready.
Re-run `tests/gpu_contention_test.py` on each target hardware configuration; integrated
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
uv run vpro --backend openvino --model-path models/yolo26/yolo26x-pose_openvino_model --device intel:gpu --capture-single-image --camera-index 0 --capture-width 1920 --capture-height 1080 --capture-output outputs/visitor_capture.jpg --capture-rmbg-mask-output outputs/visitor_capture_mask.png --capture-rmbg-foreground-output outputs/visitor_capture_no_bg.png --warmup-frames 12 --capture-delay-seconds 3
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
- live preview with YOLO26 annotations at requested camera resolution (default 1920x1080)
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
