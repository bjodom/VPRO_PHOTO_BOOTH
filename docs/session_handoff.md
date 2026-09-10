# Session Handoff: GPU Driver Update

Saved 2026-09-10, before the user's GPU driver update and reboot.

## Current Status After Reboot and Cleanup

This section supersedes the historical pre-reboot instructions below.

- Driver is now 32.0.101.8991; Windows rebooted on 2026-09-10 at 14:11:45.
- Offline loading was corrected in the stage probe before third-party imports. The loading stall was a Hugging Face network metadata request, not demonstrated GPU compilation failure.
- Post-reboot probes passed three renders with CPU RMBG and three with GPU RMBG, using committed precision and compact prompts. All six images were byte-identical, with no non-finite generation tensors. All three segmentation checks in each probe were finite. Checks ran before each render, not after the final render.
- Evidence remains in `outputs/inpaint_stage_probe_post_driver_offline_20260910_01` and `outputs/inpaint_stage_probe_post_driver_gpu_20260910_01`. No camera or kiosk was started.
- At the user's request, the VAE FP32 override was removed. Normal loading uses the original precision configuration without forcing VAE FP32. The updated precision contract and hardware-free performance suite passed 51 checks.
- A subsequent real camera session passed with YOLO on NPU, RMBG on GPU, and Juggernaut on GPU, using normal warmup and offline loading. Machu Picchu generation took 35.02 seconds; total processing through delivery took 35.58 seconds. A final image and QR link were created without a generation error. Phone download access was not verified.
- The one-session launch used a temporary in-memory shutdown wrapper, not a permanent application feature. The process exited after the first session; kiosk port 8880 and delivery port 8765 were verified closed. Saved evidence is in `outputs/kiosk_single_session_20260910`; do not restart the camera automatically.
- This checkpoint preserves the photographed person and pasted prop while Juggernaut generates the background from a neutral canvas. It does not regenerate an integrated portrait. Remaining visual issues include upper-body crop placement, lighting mismatch, and the rectangular prop overlay. Integrated portrait generation is future work, not part of this checkpoint.
- The standalone tensor tracer, including its offline fix, is preserved in Git stash `f08e8302accb77d94ce0386c2a6748037adb7934`, named `diagnostics: offline inpainting stage probe`. Only `scripts/probe_inpaint_stages.py` was stashed; other work and output evidence were left intact.
- Restore the probe when needed with `git stash apply f08e8302accb77d94ce0386c2a6748037adb7934`. This retains the stash. Use a fresh output directory for new traces. The stash is local only and is not included in a normal commit or push.
- Tensor tracing scans inputs/outputs and flushes JSON per component call, so it adds unmeasured diagnostic overhead. It was never wired into the kiosk. Production timing records and non-finite/black-frame safety checks remain enabled.
- This is a managed IT machine with monitoring; the intended deployment is clean Windows 11. Driver, reboot, and reduced background load changed together. Full repeated camera/generation/delivery acceptance and visual-quality review are still required on the deployment machine.

## Resume Instructions

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
- Juggernaut: cached `OpenVINO/Juggernaut-XL-v9-fp16-ov`; production inpainting uses 1080x1350 (aligned internally to 1080x1344), 30 steps, CFG 5, strength 0.99.
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
- Later acceptance must include real camera flow, visual inspection/person protection, repeated guests, delivery/phone LAN access, recovery, and timing. A real photograph previously showed imperfect prop placement/background-person handling; visual quality is not certified.

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