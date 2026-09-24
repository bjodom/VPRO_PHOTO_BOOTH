# Session Handoff: September 21 Post-Reboot Testing

Updated 2026-09-21, later same day as the reboot section below. This section takes precedence over ALL earlier September 21 and September 10 notes, which are historical.

## Current Pause: Save State, Resume Later

- User asked to save state and pause; wait for their next request. Do not automatically start/stop the camera, kiosk, model probes, install packages, upgrade drivers/runtime, or restore anything.
- The kiosk process from this session is still running (`http://127.0.0.1:8000`, camera index 1, delivery on 8765) and was NOT stopped as part of this save. Confirm with the user whether to leave it running or stop it (Ctrl+C in its terminal) before starting new work.
- Uncommitted working tree changes (verify with `git status`): `README.md`, `docs/experience_plan.md`, `docs/session_handoff.md`, `run.ps1`, `src/vpro/cli.py`, `src/vpro/kiosk/service.py`, `src/vpro/vision/juggernaut_runner.py`, `src/vpro/vision/juggernaut_runtime.py`, `src/vpro/vision/rmbg_runtime.py`, `tests/cli_pipeline_test.ps1`, `tests/performance_recommendations_test.py`, plus new untracked `tests/pose_device_test.py` and `tests/rmbg_generation_test.py`. Nothing has been staged, committed, or pushed. User explicitly deferred committing until real-hardware testing was satisfactory; that testing surfaced a new open question, so committing is still pending a decision.

## What Changed and What Was Learned Today

1. Reduced default capture resolution from 2560x1440 to 1920x1080 (camera's native resolution) in `run.ps1` and `src/vpro/cli.py`, to cut per-frame memory/paging. Docs updated to match.
2. Added a targeted VAE-only FP32 precision override (`vae_precision_hint`) threaded through `load_juggernaut_pipeline`, `JuggernautRunner`, `KioskConfig.juggernaut_vae_precision_hint`, and CLI `--juggernaut-vae-precision`, intended to work around the known SDXL fp16 VAE overflow (all-black frames) without the cost of a full-pipeline FP32 override.
3. That override was defaulted ON (`"f32"`) and tested on real hardware: it caused heavy system paging and pushed rendering past the service's hard 170s timeout (`RuntimeError: Rendering exceeded the event time limit`), versus sub-60s at fp16 before. This was a regression introduced by this session's own change, now reverted: `KioskConfig.juggernaut_vae_precision_hint` and `--juggernaut-vae-precision` both default to off (`None` / empty string) again. The override plumbing remains available for future opt-in use but is unproven and should not be re-defaulted on without fresh measurement.
4. After reverting, ran two consecutive real kiosk sessions (same process, same config: YOLO NPU, RMBG CPU, Juggernaut GPU fp16, camera index 1, 1920x1080, rotate 90):
   - Session 1: succeeded. `renders_served=1`, Juggernaut render 28.26s, total pipeline 29.07s, delivery QR generated, no error.
   - Session 2 (immediately after): failed. Render took 78.1s (3 internal retry attempts) then raised the same known error: `RuntimeError: Juggernaut inpaint produced a black frame on all 3 attempts. This is the known SDXL fp16 VAE overflow...`
5. This back-to-back success-then-failure in the identical process/config disproves two hypotheses raised during testing: that moving YOLO off the GPU (NPU contention theory) or that the VAE precision change had fixed generation reliability. Neither did. The user also visually confirmed NPU utilization during pose/framing for the first time this session (previously it was requested in config but never observed active); this may still be a real, separate improvement, but it did not prevent the second failure.
6. Added `tests/pose_device_test.py`: confirms `CameraStream._annotate` forwards its configured `pose_device` to `predict()`, and that `KioskService.start()` constructs `CameraStream` with `pose_device` equal to `config.yolo_device`, for both `intel:npu` and `intel:gpu`. This is hardware-free and only proves the code-level wiring was already correct; it does not explain the NPU-not-observed-until-now behavior, which is a runtime/driver matter outside this code. Registered in `tests/cli_pipeline_test.ps1`.
7. Extended `tests/performance_recommendations_test.py` with an explicit VAE-precision-override contract case (default leaves `unet`/`vae_encoder`/`vae_decoder` untouched; explicit `vae_precision_hint="f32"` only touches the two VAE components). Full hardware-free suite passes (102 checks as of the last run this session).

## Net Conclusion: Generation Reliability Is Still NOT Fixed

- The intermittent SDXL fp16 VAE black-frame overflow remains unresolved and unpredictable on this hardware/driver combination. Today's changes only (a) reduced capture memory footprint, and (b) removed a new 170s-timeout regression that this session's own FP32 VAE experiment introduced. Neither change fixes the underlying overflow.
- Do not claim event readiness. Do not re-enable the FP32 VAE default without a fresh, deliberate benchmark that measures both render time and paging/memory behavior before trusting it again.
- Untried next steps worth considering when resuming: an isolated VAE-only benchmark (encoder-only vs decoder-only vs both at f32, timed and memory-profiled, independent of a full kiosk session); the previously-noted but unauthorized OpenVINO 2026.4 upgrade; or accepting the existing 3-attempt retry-with-reseed as the practical mitigation and tuning the event timeout/expectations around it.

## Historical: September 21 Reboot Pause (superseded by the above)



- User reported another failed generation and requested saving state before reboot. Pause investigation; do not automatically start the camera, kiosk, model probes, install packages, or upgrade drivers/runtime after reboot. Wait for the user's request.
- Latest persisted record in `outputs/kiosk/pipeline_metrics.jsonl`: `2026-09-21T14:49:08.413738`, `success=false`, YOLO `intel:npu`, RMBG `CPU`, render `GPU`, inpaint, 30 steps, strength 0.99. RMBG took 0.727s, YOLO 0.061s, composition 0.095s, total compose 0.947s; Juggernaut took 102.039s (render 102.037s). CPU segmentation/composition completed before generation failed. The exact latest exception has NOT been captured; do not label this latest failure RMBG NaNs, black output, timeout, or OOM without its traceback.
- Earlier September 21 failures used GPU RMBG. Their terminal traceback was `service._produce -> compose_portrait_from_image -> rmbg_runtime.segment -> postprocess_mask`, raising `RuntimeError: RMBG inference returned non-finite values; subject segmentation failed.` That failure occurred before rendering. GPU coexistence is implicated by prior probes, but its underlying cause is not established.
- CPU RMBG is now the production default, chosen by the user for reliability over a small latency saving. It does NOT establish that Juggernaut generation is reliable; the latest record confirms failure still occurs with CPU RMBG.
- Preserve all current outputs, model files, compiled caches, and uncommitted changes across reboot. September 10 claims that no photos/metrics exist are stale: new September 21 captures and metrics exist. No cleanup, commit, push, stash, or upgrade was performed in this session. Current service/process shutdown status has not been verified during this save.

## Current Changes and Verification

- CPU RMBG defaults updated in `run.ps1`, `src/vpro/cli.py`, `src/vpro/kiosk/service.py`, and both constructors/loaders in `src/vpro/vision/rmbg_runtime.py`. Explicit GPU overrides remain supported. YOLO remains NPU and Juggernaut remains GPU in the normal launcher.
- Added `tests/rmbg_generation_test.py`: opt-in saved-photo real RMBG/Juggernaut coexistence regression. Checks segmentation before load, after load, after explicit warmup, and after EVERY repeated render including the final render. Uses fixed input/mask/seed, single render attempt, offline loading before runtime imports, stage JSONL records, fresh output directory, and bounded shutdown. Default RMBG device intentionally remains GPU to reproduce the issue; CPU override provides a control. It does not use the camera, YOLO, or delivery.
- Added hardware-free coverage in `tests/performance_recommendations_test.py` for stage order, corruption/failure/timeout handling, command dispatch, and CPU defaults, using real parser/contracts and autospecced runtime/runner mocks. Added launcher CPU-default/GPU-override checks in `tests/cli_pipeline_test.ps1`. Updated `README.md` with defaults and hardware regression instructions.
- Full hardware-free launcher gate passed 310 checks: performance 99, app 65, session 50, framing 28, runner 32, delivery 36, plus launcher contracts. Focused performance checks passed again after fixing mock-fixture diagnostics. These are earlier session results, not rerun during this save. Some unrelated existing editor diagnostics remain in the performance test.
- The NEW real coexistence regression has NOT been run against hardware. Hardware-free checks do not prove GPU numerical stability. Latest real CPU-default kiosk attempt failed as recorded above; repeated full kiosk acceptance and phone delivery are not established.
- Existing `tests/gpu_contention_test.py` measures YOLO/Juggernaut contention, not RMBG. Launcher test `-Hardware` skips guided generation. Neither replaces the new regression or full kiosk acceptance.
- `docs/session_handoff.md` already had user changes before this session; historical content is preserved below. Do not revert unrelated work. No forced VAE FP32 override should be reintroduced: the user rejected its cost earlier.

## Measured Environment and Upgrade Discussion

- Interpreter: project `.venv\Scripts\python.exe`, Python 3.14. OpenVINO `2026.3.1-22476-759c5a6ab8c-releases/2026/3`; optimum 2.3.0; optimum-intel 2.1.0; diffusers 0.37.1.
- Hardware: Intel Core Ultra X7 358H, Intel Arc B390 iGPU. September 21 driver query: GPU `32.0.101.8724` (April 15, 2026); NPU `32.0.100.5540` (August 19, 2026). This GPU reading differs from historical September 10 notes; no explanation or driver change was attempted.
- Isolated RMBG benchmark used saved capture `outputs/kiosk/capture_1acaf57d1da448958902ec55b224bd2f.jpg`, separate CPU/GPU processes, one first call plus five warmed full `segment` calls, no Juggernaut. Warm median CPU 0.473s vs GPU 0.098s, about 0.38s extra per photo; both produced valid masks. This is NOT a coexistence test.
- Recurring Optimum warning comes from diffusers detecting both `optimum` and `optimum-intel` as owners of the shared namespace. Installed versions satisfy the optimum-intel requirement (`optimum~=2.3.0`); no demonstrated version conflict. Do not uninstall either based on this warning.
- OpenVINO 2026.4 released September 16 and has a Windows Python 3.14 wheel. Official notes explicitly support optimum-intel 2.1.0, but no fix specifically for our RMBG NaNs or Juggernaut black images was identified. Listed Windows NPU driver issue recommends 32.0.100.5540 or newer, which this machine meets. Full dependency resolution was not tested.
- Recommendation was controlled testing in a separate environment with a separate compiled cache, not an urgent in-place upgrade or presumed fix. User has NOT authorized or performed the upgrade. Keep CPU RMBG as default. Release notes: https://docs.openvino.ai/2026/about-openvino/release-notes-openvino.html

## Resume After Reboot

1. Read this section first. Ask for the user's reboot/test result or follow their new request; do not launch anything automatically. Confirm runtime/driver versions and reboot state when a comparison is authorized.
2. Obtain the latest traceback if still available. Metrics record failure and timings, but not its exact cause. If a new test is authorized, retain terminal output and match its timestamp to metrics. Distinguish CPU segmentation success followed by generation failure from prior GPU segmentation NaNs.
3. User's last normal launcher command, with C920-selected index and clockwise rotation, was:

	```powershell
	.\run.ps1 -CameraIndex 1 -CaptureWidth 1920 -CaptureHeight 1080 -CaptureRotate 90
	```

	CPU RMBG now applies without another flag. Camera index 1 was suggested for C920 but OpenCV mapping was not independently verified; Windows also lists ASUS FHD and IR cameras. Browser is normally `http://127.0.0.1:8000`, guest delivery port 8765. Wait for readiness, then complete the guest flow. Ctrl+C stops the service. Do not use `-Sync` casually; it changes the environment.
4. If authorized, run the saved-photo regression documented in README with a currently existing capture and a NEW output directory. Compare CPU and GPU in separate processes with kiosk stopped. Do not assume historical September 10 image paths still exist. Then validate repeated real kiosk sessions and output quality, not just mocked checks.
5. An older detailed inpainting tensor probe remains in local stash `f08e8302accb77d94ce0386c2a6748037adb7934`; do not automatically restore it or overwrite work. Its history and limitations are recorded below. Generation reliability and event readiness remain unresolved.

## Historical September 10 Handoff

Updated 2026-09-10 before closing VS Code. Earlier diagnostic history is retained below.

## Next Session: User-Run Launcher Test

This section takes precedence over all historical notes below.

- User is closing VS Code and plans a clean reboot, then a normal end-user test from PowerShell. Do not start the camera, kiosk, or model probes automatically. Wait for their results or explicit request.
- Stable code checkpoint: `23248da` on `master`, committed and pushed to `origin/master` (`bjodom/VPRO_PHOTO_BOOTH`). The checkpoint passed 262 hardware-free checks plus launcher contracts, JavaScript syntax, and whitespace validation.
- The previous live session succeeded in 35.58 seconds total after removing the forced VAE FP32 override. It used an in-memory wrapper and explicitly set offline environment variables; it was NOT a direct `run.ps1` acceptance test. The upcoming test must use the launcher unchanged, without those wrappers.
- All generated photos, masks, compositions, metrics, and diagnostic logs in `outputs` were deleted at the user's request after the checkpoint. Only `outputs/openvino_cache` remains (approximately 11.7 GiB). Every saved-input/output reference below is now historical, not an available file. A new authorized capture is needed for future compositing experiments.
- The diagnostic probe remains in the local-only stash documented below; it was not deleted by output cleanup or pushed to GitHub.
- The app was stopped and its ports closed after the live session. This handoff update is a local documentation change after the pushed code checkpoint; no additional commit or push is part of this save.

After reboot, the user will run:

```powershell
cd C:\Users\bjodom\ai_projects\vPRO_Photo_Booth
.\run.ps1
```

No environment activation, dependency sync, downloads, or extra model flags are needed for the intended launcher test. If PowerShell blocks script execution, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` in that terminal and retry.

The browser should open at `http://127.0.0.1:8000`. Wait for Start to enable, then complete consent, destination selection, capture, review, and generation. Inspect the image and test QR delivery while the service is running. Stop with Ctrl+C in PowerShell; closing the browser does not stop the application. Normal launching is persistent, not automatic one-session shutdown.

If startup stalls, preserve the terminal error/output and diagnose the actual launcher path before changing settings. The probe's offline import-order fix is in the stash; do not assume it modified production startup. Do not reintroduce FP32 as a workaround.

After launcher validation, the next feature discussion is scene-reference quality and integrated person/scene generation. Current behavior preserves the photographed subject and any real captured prop while generating the background. Phone delivery/full event acceptance remain unverified.

## Current Status After Reboot and Cleanup

This section supersedes the historical pre-reboot instructions below.

- Driver is now 32.0.101.8991; Windows rebooted on 2026-09-10 at 14:11:45.
- Offline loading was corrected in the stage probe before third-party imports. The loading stall was a Hugging Face network metadata request, not demonstrated GPU compilation failure.
- Post-reboot probes passed three renders with CPU RMBG and three with GPU RMBG, using committed precision and compact prompts. All six images were byte-identical, with no non-finite generation tensors. All three segmentation checks in each probe were finite. Checks ran before each render, not after the final render.
- Evidence remains in `outputs/inpaint_stage_probe_post_driver_offline_20260910_01` and `outputs/inpaint_stage_probe_post_driver_gpu_20260910_01`. No camera or kiosk was started.
- At the user's request, the VAE FP32 override was removed. Normal loading uses the original precision configuration without forcing VAE FP32. The updated precision contract and hardware-free performance suite passed 51 checks.
- A subsequent real camera session passed with YOLO on NPU, RMBG on GPU, and Juggernaut on GPU, using normal warmup and offline loading. Machu Picchu generation took 35.02 seconds; total processing through delivery took 35.58 seconds. A final image and QR link were created without a generation error. Phone download access was not verified.
- The one-session launch used a temporary in-memory shutdown wrapper, not a permanent application feature. The process exited after the first session; kiosk port 8880 and delivery port 8765 were verified closed. Saved evidence is in `outputs/kiosk_single_session_20260910`; do not restart the camera automatically.
- This historical checkpoint preserved the photographed person and used a neutral canvas. The current path uses destination thumbnail references, scene-aware placement, and final subject restoration; synthetic prop insertion has been removed.
- The standalone tensor tracer, including its offline fix, is preserved in Git stash `f08e8302accb77d94ce0386c2a6748037adb7934`, named `diagnostics: offline inpainting stage probe`. Only `scripts/probe_inpaint_stages.py` was stashed; other work and output evidence were left intact.
- Restore the probe when needed with `git stash apply f08e8302accb77d94ce0386c2a6748037adb7934`. This retains the stash. Use a fresh output directory for new traces. The stash is local only and is not included in a normal commit or push.
- Tensor tracing scans inputs/outputs and flushes JSON per component call, so it adds unmeasured diagnostic overhead. It was never wired into the kiosk. Production timing records and non-finite/black-frame safety checks remain enabled.
- This is a managed IT machine with monitoring; the intended deployment is clean Windows 11. Driver, reboot, and reduced background load changed together. Full repeated camera/generation/delivery acceptance and visual-quality review are still required on the deployment machine.

## Historical Pre-Reboot Resume Instructions

- Testing is PAUSED at the user's request while they update GPU drivers.
- Wait for confirmation that the update and reboot are complete before loading models or restarting the kiosk.
- Resume from this document and persistent repository memory `generation-diagnostics.md`.
- The real kiosk on port 8880 and the stage probe were stopped. No probe Python process remained at the last check.
- Port 8879 was a synthetic UI preview, not a real camera-backed kiosk; it may disappear on reboot.
- Do not claim event readiness. Generation reliability remains unresolved.
- No commit or new branch was requested or made. Preserve all existing uncommitted work.

## Environment

- Workspace: `C:\Users\bjodom\ai_projects\vPRO_Photo_Booth`, Windows, PowerShell.
- Branch: `master`, tracking `origin/master`; last recorded HEAD: `1149240` (Set optimized launcher device defaults).
- Interpreter: `.venv\Scripts\python.exe`, Python 3.14. Use it directly for existing checks; a plain dependency sync can remove optional generation/kiosk dependencies.
- Installed OpenVINO: 2026.3.1. GPU: Intel Arc B390 integrated graphics; CPU: Intel Core Ultra X7 358H; Intel AI Boost NPU.
- Intended discrete Arc B70 deployment has not been measured.
- Default placement remains YOLO on NPU, RMBG on GPU, Juggernaut on GPU. CPU RMBG was a diagnostic control, not an established complete fix.
- Current generation engines: DreamShaper INT8 at 512x768 or local Juggernaut INT8 at 896x1120; production uses scene references and preserves the captured subject without synthetic prop insertion.
- Cache: `outputs/openvino_cache/juggernaut`. Do not delete it or change precision/device settings as part of an otherwise controlled comparison without recording the change.

## Implemented Work (Uncommitted)

- Event hardening: launcher preflight/offline defaults, CLI argument propagation, startup cleanup, camera/pose scheduling, capture freshness, session/worker exclusion, immutable capture snapshots, cancellation checks, render timeout handling, output retention, delivery retry without rerendering, and metrics.
- Failed generation now reports an error instead of quietly delivering a neutral-background composition. This can expose failures that the earlier fallback concealed.
- Persistent RMBG reuse and inference locking are present; their effect on GPU coexistence has been investigated but not established as the sole regression cause.
- Custom destination selection: normalized/validated location text, UI form, session/API support, generated background prompts, reset behavior, and tests.
- Scene thumbnails are selection assets only. Generation uses a neutral composition and protection mask, not a stock background. Person scale/placement is deterministic; autonomous repositioning/full relighting is not implemented.
- Local destination thumbnails and attribution page added. Custom UI checked at mobile and desktop sizes without observed overflow.
- RMBG now rejects NaN/Inf output instead of silently normalizing it into an empty mask.
- Runner warmup failures now prevent readiness instead of being swallowed.
- Working-tree VAE encoder/decoder FP32 hints remain from an attempted workaround. They are costly and NOT a proven reliability fix. Do not mistake them for the committed precision configuration.
- Documentation includes event readiness, optimization caveats, launcher usage, and custom destination behavior.

## CLIP Warning Fix

- Expanded placement/style wording made the Fuji positive prompt 92 tokens against a 77-token limit. A preset negative prompt also measured 86 tokens.
- `src/vpro/kiosk/scenes.py` now uses compact background style wording and a shorter shared negative list.
- Verified using both actual cached CLIP tokenizers: all eight presets and custom examples Kyoto, Japan / National Museum of Scotland, Edinburgh fit.
- Verified positive lengths: 37-43 tokens; negative lengths: 48-59 tokens; limit: 77.
- This fixes the tested truncation warnings, but truncation has NOT been shown to cause the black frames.
- Remaining gap: the 100-character custom-location limit does not guarantee a token budget, especially for Unicode. Token-aware handling of arbitrary custom locations and corresponding regression coverage are not completed.

## Diagnostic Evidence

Keep these distinct: explicit allocation failure, non-finite segmentation, black generation output, and a model-loading stall are different observations.

1. An earlier full-service run reported an explicit GPU allocation failure of approximately 2 GB. Memory-allocation failure occurred in that run; this does not prove every black frame is OOM.
2. Black frames were accompanied by diffusers invalid-value-to-integer cast warnings. The first numerically failing generation stage has not yet been located. Earlier confident attribution to known VAE overflow was withdrawn; the renderer's error wording may still overstate this attribution.
3. User freed approximately 20 GB of RAM. Failures still occurred. This neither proves nor excludes GPU memory/driver involvement.
4. Original-precision standalone text-to-image succeeded (~39.51s); the working-tree FP32 variant also succeeded (~52.89s). Historical text-to-image benchmarks are not kiosk inpainting acceptance tests.
5. Committed service/runtime/camera/scenes with a saved photo produced a genuine final image (~57.87s). Current service with committed runtime and prompt also passed (~58.34s). The failures are intermittent, not a consistently failing simple code path.
6. Old/new/old prompt comparison: old passed, expanded new passed, later old failed RMBG with NaNs before generation. A prompt-only cause was disproven.
7. Fresh RMBG loading per composition: masking passed all three runs, but generation passed once and then produced two black frames. Persistent RMBG reuse alone is not the explanation.
8. Isolated original-precision inpainting without other models: three direct renders passed (~48.70/46.43/46.63s), byte-identical at seed 1234.
9. Same via current runner, no warmup and no other models: three passed (~54.04/53.23/51.08s), matching the reference. Worker threading alone did not reproduce the failure.
10. Add resident GPU RMBG, no YOLO/camera, serialized operations: all three Juggernaut renders passed (~64.53/58.54/58.62s), but every RMBG output pixel became NaN immediately after the first render and stayed invalid. Before generation it was finite. This isolates an interaction involving resident GPU segmentation and generation, not its underlying driver/runtime cause.
11. Matched CPU RMBG control: all seven segmentation checks stayed finite and identical. Generation passed twice (~59.36/54.98s), then produced a black frame (~46.21s). CPU RMBG avoided observed mask corruption but did not fix generation reliability.

## Stage Probe and Last Stall

- Added `scripts/probe_inpaint_stages.py`: offline checks for both tokenizers; OpenVINO request wrappers for text encoders, VAE encoder, UNet calls, and VAE decoder; logs tensor shape/dtype/min/max/non-finite count without tensor contents; records first non-finite component boundary and output hashes.
- It does not directly trace scheduler internals. Its real component-wrapper path has not yet been validated because loading stalled first.
- `--runtime-ref HEAD` loads the committed runtime in memory before importing the runner, leaving worktree files unchanged and bypassing the unproven FP32 VAE override.
- The last attempt used CPU RMBG, seed 1234, three requested runs, fixed saved composition/mask, and the newly shortened prompts.
- It stalled for over five minutes after the pipeline-loading message. The log contained only a configuration record, not any inference-stage evidence.
- The diagnostic's 180-second startup timeout was hidden by `shutdown(timeout=None)` in `finally`. The run was killed; no surviving probe process was found.
- Fixed the probe to print and log exceptions before `shutdown(wait=True, timeout=10)`. Added a thread-stack dump after 120 seconds via faulthandler.
- A hardware-free mocked startup-timeout test passed: timeout propagates, error is printed/logged, and shutdown receives a 10-second bound. Editor diagnostics reported no errors.
- Bounded thread joining is not a guarantee that native GPU operations are cancellable. If native code still prevents exit, stop the diagnostic process explicitly after retaining the stack/error evidence.
- The updated probe has NOT yet been rerun against models. No conclusion about the cause of the loading stall is available.

## Saved Inputs and Outputs

- Fresh authorized photo: `outputs/event_validation/capture_c41119bd4346453fa96884466492eea1.jpg` (1920x1080). Reuse saved inputs; do not take a new camera photo without authorization.
- Constant generation input: `outputs/regression_per_capture/composed_bde3858ecaed4209918ac866816d7d41.jpg`.
- Constant mask: `outputs/regression_per_capture/mask_bde3858ecaed4209918ac866816d7d41.png`.
- Reference output: `outputs/regression_inpaint_only/final_attempt_1.jpg`.
- Reference SHA256: `8d79886b37e90f0fa13e23c5a2247f2ff237577ee348881fa300e0981ce9d974`.
- CPU mask SHA256: `ae7932209084f1fd66ab5fe4507176f4e3c58731f7e44096ca956403fb451602`.
- Prior comparison directories: `outputs/regression_baseline`, `regression_current`, `regression_per_capture`, `regression_inpaint_only`, `regression_runner_only`, `regression_runner_rmbg`, `regression_runner_rmbg_cpu`.
- Last stalled trace: `outputs/inpaint_stage_probe/stages.jsonl`. Use a new output directory after reboot rather than overwriting evidence.
- Outputs are local/ignored, not committed. Preserve them across reboot.

## Next Authorized Test After Reboot

Record the installed driver version and confirm the reboot first. Rerun the stage probe using the same saved input, CPU RMBG, committed model precision, and current compact prompts:

```powershell
.\.venv\Scripts\python.exe scripts/probe_inpaint_stages.py --runtime-ref HEAD --runs 3 --rmbg-device CPU --output-dir outputs/inpaint_stage_probe_post_driver --image outputs/regression_per_capture/composed_bde3858ecaed4209918ac866816d7d41.jpg --mask outputs/regression_per_capture/mask_bde3858ecaed4209918ac866816d7d41.png
```

- This matches the last stalled probe's settings, but the earlier completed CPU control used longer prompts. Do not describe it as a strict driver-only comparison to those earlier successful/black outputs or expect the historical image hash with changed prompts.
- If loading stalls, use the new thread dump to locate the blocking call before changing unrelated settings.
- If a tensor becomes non-finite, locate its first boundary and distinguish finite-input/non-finite-output from invalid upstream input. Do not assume VAE decoder overflow or OOM.
- The current probe segments the composition before each render; it is not a replacement for the earlier fresh-photo before/after segmentation control. Repeat that control when evaluating GPU RMBG reliability.
- Repeated successful renders plus finite segmentation are necessary before broader live acceptance. They are not by themselves proof of event readiness.
- Later acceptance must include real camera flow, visual inspection/person protection, repeated guests, delivery/phone LAN access, recovery, and timing. Visual quality is not certified.

## Validation Status and Remaining Work

- Prior reported script checks: performance recommendations 49, runner 32, kiosk app 65, kiosk session 50, framing 28, delivery 36. These are historical passing counts, not a newly run full gate after all latest edits.
- Relevant scripts are standalone checks, not assumed pytest tests.
- `tests/juggernaut_test.py` exercises real text-to-image, not production inpainting.
- `tests/cli_pipeline_test.ps1` hardware mode skips guided generation, so it cannot establish generation reliability.
- `tests/kiosk_flow_test.py` is a real capture/generation flow and requires an authorized test subject; do not run it casually.
- Add robust token-budget handling/tests for long custom destinations, validate stage probe against actual component APIs, and correct unsupported error attribution.
- Reassess the unproven FP32 VAE workaround only with evidence; its existing precision-contract test must change if that own workaround is removed.

## Worktree Snapshot

Modified tracked files at save time:

```text
README.md
docs/optimization.md
run.ps1
src/vpro/cli.py
src/vpro/delivery/handoff.py
src/vpro/kiosk/app.py
src/vpro/kiosk/camera.py
src/vpro/kiosk/scenes.py
src/vpro/kiosk/service.py
src/vpro/kiosk/session.py
src/vpro/kiosk/static/app.js
src/vpro/kiosk/static/index.html
src/vpro/kiosk/static/styles.css
src/vpro/vision/juggernaut_runner.py
src/vpro/vision/juggernaut_runtime.py
src/vpro/vision/pipeline.py
src/vpro/vision/rmbg_runtime.py
tests/cli_pipeline_test.ps1
tests/juggernaut_runner_test.py
tests/kiosk_app_test.py
tests/kiosk_flow_test.py
tests/kiosk_session_test.py
tests/performance_recommendations_test.py
```

Untracked additions: `docs/event_readiness.md`, this handoff, `scripts/probe_inpaint_stages.py`, `src/vpro/kiosk/static/credits.html`, and eight destination JPEGs (colosseum, eiffel, fuji, goldengate, machu, pyramids, santorini, tajmahal).

Do not reset, discard, commit, or broadly rewrite this work without the user's request.