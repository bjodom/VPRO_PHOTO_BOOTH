# Current State

Updated: 2026-09-21
Branch: `experiment/dreamshaper-inpainting`
HEAD: `12319b8 Use local Juggernaut model path`

## Runtime

- OpenVINO `2026.4.x` is the only supported backend.
- DreamShaper INT8 is the production event engine:
  - Model: `OpenVINO/dreamshaper-8-inpainting-int8-ov`
  - Native generation: `512x768`
- Juggernaut INT8 is retained for diagnostics only:
  - Local model: `models/juggernaut-int8`
  - Native generation: `896x1120`
- Engine selection is exposed through `run.ps1`:

```powershell
.\run.ps1 -GenerationEngine DreamShaper -CameraIndex 2 -CaptureRotate 90
```

- The external camera index changes by machine. Confirm the active ID before launch; recent runs used camera `0` and earlier runs used camera `2`.
- The kiosk defaults to random generation seeds. Pass `-JuggernautSeed 1234` for repeatable diagnostics.
- Retry seed behavior uses `5678` after a seeded black/non-finite result.

## Pipeline

1. Camera capture and YOLO framing.
2. RMBG foreground segmentation on CPU by default.
3. Scene-aware composition using destination thumbnail references from `src/vpro/kiosk/static/`.
4. Engine-specific native-size inpainting.
5. Final subject restoration using the composed subject and coverage mask.
6. QR delivery and retention handling.

Synthetic laptop props were removed. Any real prop in the captured foreground is preserved by segmentation and final restoration.

Preset scenes use destination thumbnails as composition references. Custom locations use the neutral fallback canvas because there is no reference image for arbitrary locations.

## Validated Results

- Hardware-free CLI/pipeline gate passes.
- Kiosk app tests pass: 65 checks.
- Performance/composition tests pass: 101-102 checks depending on the current test set.
- Runner tests pass: 32 checks.
- DreamShaper scene-aware tests pass at `512x768`, with safety checker enabled.
- Three cached DreamShaper renders passed at `512x768` in roughly 5 seconds each.
- Earlier cached Juggernaut INT8 Pyramid renders passed at `896x1120`:
  - Approximately 16-19 seconds each.
  - No black-frame failures.
- Earlier real Juggernaut kiosk testing passed two Pyramid guest sessions after native dimensions were wired into `KioskConfig`:
  - Guest 1: 26.66 seconds.
  - Guest 2: 18.97 seconds.
  - Both produced QR delivery successfully.

## Known Limitations

- Landmark fidelity depends on the available thumbnail reference and the inpainting model. Prompt-only generation can produce generic architecture.
- Scene thumbnails are often landscape and require a scene-specific crop anchor to avoid floating subjects. Golden Gate has a tuned foreground crop and placement; other scenes may need tuning.
- DreamShaper is fast enough for the event path and is currently the reliable production choice.
- Juggernaut can look better in successful runs, but recent live tests showed intermittent black/non-finite inpainting failures and slow fallback; do not use it as a guest-facing event engine.
- The diagnostic and kiosk paths have recently been brought closer together, but they still have separate orchestration code.
- The Hugging Face ID for Juggernaut does not resolve correctly in the current offline local cache layout. Use the local path `models/juggernaut-int8` only for diagnostics.
- `docs/session_handoff.md` contains historical notes that are not all current; this file is the current summary.

## Pending Worktree Changes

The current worktree is intentionally dirty with cleanup and simplification changes:

- OpenVINO-only backend cleanup and lockfile update.
- Removal of synthetic prop assets and code.
- Removal of obsolete diagnostic scripts.
- Scene-aware thumbnail composition and prompt updates.
- Native engine-specific generation dimensions.
- Documentation refreshes.
- Local Juggernaut model directory is untracked because model binaries are not committed.

Review and commit these changes before merging the branch.

## Next Steps

1. Run `git diff --check` and review the full diff.
2. Run the hardware-free gate plus kiosk, performance, and runner tests.
3. Run one real DreamShaper kiosk session with the correct camera index.
4. Visually validate the tuned scenes with live DreamShaper output.
5. Centralize engine configuration so model ID, native dimensions, steps, and device settings are defined once.
6. Share the scene-aware guide/restoration helper between diagnostics and production to prevent future drift.
7. Update or archive historical documentation after the current branch is accepted.
8. Commit the branch, then squash-merge into `master` only after the real kiosk checks pass.
