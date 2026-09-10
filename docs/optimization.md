# vPRO Photo Booth — Performance Optimization

Branch: `master` · Last updated: 14 Aug 2026

## Headline

| | At the start | Now | Change |
| --- | ---: | ---: | ---: |
| Pipeline startup | 359.6 s, per process | 17–19 s, once at boot | **20×** |
| Guided render (`identity-lock`) | 22 s, **guide image ignored** | ~5–6 s, guide applied | **4×** + correctness |
| Black frames | delivered to the guest silently | detected, retried, fallback | — |
| Preview during preload | not measured | 29–30 fps, unaffected | — |
| **Per guest, warm booth** | **~400 s** | **~5–6 s** | **~70×** |

Most of that did not come from optimizing anything. It came from finding three defects that were
hiding as performance characteristics.

---

## 1. Current cost model

Measured on the dev laptop: Intel Arc 390 (16 GB), Samsung NVMe SSD, warm `CACHE_DIR`. The target
booth is an Arc B70 (32 GB).

**Startup — paid once per process:**

| Phase | Seconds |
| --- | ---: |
| `optimum.intel` import | ~9.3 |
| `from_pretrained` (metadata) | ~0.5 |
| Component compile (5 components, warm cache) | ~8.0 |
| **Total** | **~17–19** |

Compile detail: `unet` 4.9 s, `text_encoder_2` 2.4 s, `text_encoder` 0.3 s, `vae_decoder` 0.2 s,
`vae_encoder` 0.1 s.

**Render — per guest:**

| Path | Steps | Effective steps | Seconds |
| --- | ---: | ---: | ---: |
| Guided img2img, `identity-lock` (strength 0.16) | 28 | ~4 | **~5–6** |
| Guided img2img, strength 0.95 | 28 | ~27 | ~34 |
| text2img, `quality` | 36 | 36 | ~42 |
| text2img, `fast` | 16 | 16 | ~17–20 |

Derived model: **~7 s fixed per render + ~1.0 s per denoising step.** The fixed portion is VAE
encode/decode, both text encoders, resize and save.

**Live preview:** 29–30 fps (camera-capped); YOLO pose inference 15.9 ms p50, i.e. ~63 fps of
capability and roughly 50% GPU headroom.

---

## 2. Investigation log

Each entry: what we believed, what the evidence showed, what changed as a result.

### 2.1 The 350 s startup — we blamed OpenVINO; it was blocked network calls

**Hypothesis.** Startup was dominated by OpenVINO compiling the SDXL graph, so the fix was better
use of `CACHE_DIR`, static shapes, and a pre-exported local IR. A secondary theory was that the
large blob count in `outputs/openvino_cache/juggernaut` indicated cache-key thrash.

**Evidence.** Instrumenting the load put **337.35 s of a 359.58 s startup inside
`from_pretrained`** — a call that should only read metadata. Each suspect was then eliminated:

| Suspect | Measurement | Verdict |
| --- | --- | --- |
| Cache thrash | Blob count 203 → 203 (**+0**) across runs | Cache works |
| Compilation | All 5 components compile in 12.2 s | 3% of startup |
| Disk I/O | 4.78 GB unet read at **5,776 MB/s** | ~1–2 s total |
| OpenVINO itself | `read_model` 0.19 s, `compile_model` 17.5 s | ~18 s, not 337 s |

That left only optimum/diffusers, so we profiled it.

**Root cause.** `cProfile` attributed **336.5 s to `socket.connect`** across 24 calls, ~14 s per TCP
timeout. Nothing was computing; the process was waiting on unreachable hosts — a corporate VPN was
blackholing connections rather than refusing them.

| Culprit | Seconds | Path |
| --- | ---: | --- |
| Hub repo file listing | 175.5 | `optimum.exporters.tasks.get_model_files` → `huggingface_hub.list_repo_files` |
| Hub library inference | 168.4 | `tasks._infer_library_from_model_name_or_path` → `snapshot_download` → `model_info` |
| OpenVINO telemetry | 84.3 | `openvino_telemetry.backend_ga4.send` → Google Analytics |

Two things worth remembering: **`local_files_only=True` is not honored** — optimum's
`infer_library_from_model` queries the Hub regardless — and **OpenVINO ships analytics that block
startup**, which is a privacy consideration as much as a latency one.

**Fix.** `_configure_offline_defaults` in
[src/vpro/vision/juggernaut_runtime.py](src/vpro/vision/juggernaut_runtime.py) sets
`OPENVINO_TELEMETRY_OPT_OUT=1` always and `HF_HUB_OFFLINE=1` when `local_files_only`, both via
`setdefault` so operators can override. It runs before `optimum` is imported, because
`huggingface_hub` reads that variable at import time.

**Result:** `from_pretrained` 337.35 s → **0.46 s**; total startup 359.58 s → **17.75 s**. Render
times unchanged — this was pure stall.

### 2.2 The guided render — we were tuning a render that ignored the guest

**Hypothesis.** The kiosk's img2img render was simply slow and needed step/scheduler tuning.

**Evidence.** Profiling the "img2img" call produced four independent contradictions:

1. Execution landed in `pipeline_stable_diffusion_xl.py` — the **text2img** class. The img2img
   module never appeared.
2. The UNet ran **28 forwards for `steps=28, strength=0.16`**, where ~4 were expected.
3. `torch.randn` was called 29 times — latents were pure noise, and there was no VAE *encode* at
   all.
4. Decisive: `strength=0.16` and `strength=0.95` produced **byte-identical files**.

**Root cause.** `OVDiffusionPipeline.from_pretrained` returns the text2img class. `render_img2img`
passed `image=` and `strength=`, which diffusers' text2img `__call__` swallows via `**kwargs`
rather than rejecting. Silent failure with plausible-looking output. **The guest's composed
portrait had zero influence on the delivered image, and the `identity-lock` preset was inert.**

**Fix.** `load_juggernaut_pipeline` takes a `task` argument and selects `OVPipelineForImage2Image`
or `OVPipelineForText2Image`. `_require_task` raises on a class/mode mismatch so this cannot recur
silently.

**Result:** correctness restored — and faster, because the render no longer runs every step:
strength 0.16 went **20.2 s → 11.2 s** standalone, later ~5–6 s under the runner (see §2.5).

### 2.3 The black frame — we blamed the seed; it is transient

**Hypothesis.** `run0012` rendered entirely black (`mean=0, max=0`) while runs 1–11 were fine,
matching the known SDXL fp16 VAE overflow on particular seeds.

**Evidence.** Re-running the identical command with the same seed 1236 and preset produced a
**correct** image (`run0014`, mean 158.5). A fixed seed does not reproduce it.

**Root cause.** Still unidentified. Transient — more likely GPU/driver state than seed-specific
arithmetic. That transience is exactly what makes retrying effective.

**Fix.** `_is_degenerate_image` flags all-black or non-finite output; `_render_with_black_guard`
wraps both render paths and retries up to 3 times. Because the fault is transient, **the first
retry reuses the same seed** to preserve reproducibility; only later retries vary it. After
exhausting attempts it raises, so the social pipeline falls back to the deterministic compose.
`render_img2img` also rejects a black *guide* image up front.

This also cost time in §2.2: the img2img probe auto-selected the newest render as its guide, which
happened to be the black `run0012`, making a correct fix look broken.

### 2.4 Retracted: the "~19 s fixed per-render overhead"

An earlier revision claimed ~19 s of fixed per-render cost, derived by comparing a 36-step text2img
run against what was assumed to be a 4-effective-step img2img run. **The premise was wrong** — that
"img2img" run was secretly text2img and executed all 28 steps.

With img2img actually working, the measured model is **~7 s fixed + ~1.0 s/step**. The fixed
portion is real but far smaller than claimed.

### 2.5 The warmup — at the wrong resolution it barely helps

**Hypothesis.** Any cheap render at boot would absorb first-render cost, so 512×512 would do.

**Evidence.** With a 512×512 warmup, the first 1080×1350 guided render took **10.81 s** against
4.97 s for the second. Warming at production resolution dropped the first render to **5.72 s**, in
line with every render after it.

**Consequence beyond the fix:** every standalone measurement taken earlier was inflated by a
~5–6 s first-render penalty, because each fresh process paid it. The steady-state guided render is
**~5–6 s**, not the ~11–12 s measured one-shot.

### 2.6 GPU contention — we expected the load to starve the preview; only the render did

**Hypothesis.** Preloading Juggernaut during camera capture would compete with the live YOLO pose
preview, so `--juggernaut-no-preload` might be necessary.

**Evidence.** Measured on the Arc 390 with the built-in camera
([tests/gpu_contention_test.py](tests/gpu_contention_test.py)), warmup running during preview:

| Window | Preview | p50 | p95 | worst frame |
| --- | --- | ---: | ---: | ---: |
| Before load | 29–30 fps | 16.0 ms | 16.7 ms | 17.5 ms |
| Import + metadata + compile | **29–30 fps, unaffected** | — | — | — |
| **Warmup render** | **8–20 fps** | 17.2 ms | 46.6 ms | **579.7 ms** |
| After ready | 29–30 fps | 16.2 ms | 17.1 ms | 18.6 ms |

**Root cause.** Loading and compiling cost the preview nothing. All the damage came from diffusion
inference — a ~9 s stutter including a 0.58 s freeze, exactly while a guest would be posing. This
is compute contention, not VRAM exhaustion, so more VRAM alone would not fix it.

**Fix.** `submit_warmup()` queues the warmup as a normal request. `_run_social_pipeline` loads with
`warmup=False` so compile still overlaps capture, then calls `submit_warmup()` once the camera is
released, so the warmup overlaps RMBG and compose instead.

**Result:** preview holds **29–30 fps** throughout; p95 rises 17 → 22 ms with a single 52.8 ms
frame, against a 33 ms camera budget. Not perceptible.

### 2.7 NPU offload — works, but does not decouple on an integrated GPU

**Hypothesis.** OpenVINO lets you pick the device, so moving the vision models to the NPU should
free the GPU for diffusion and remove the §2.6 contention entirely.

**Evidence.** This machine exposes three devices:

```
CPU | Intel(R) Core(TM) Ultra X7 358H
GPU | Intel(R) Arc(TM) B390 GPU (iGPU)
NPU | Intel(R) AI Boost
```

YOLO26 pose runs on the NPU without changes, and RMBG validates on it too:

| Model | GPU | NPU |
| --- | ---: | ---: |
| YOLO26 pose, p50 | 15.9 ms (~63 fps) | 26.0 ms (~38 fps) |
| YOLO26 pose, p95 | 16.7 ms | 28.3 ms |
| Preview delivered | 29–30 fps | 29–30 fps |

The NPU is ~1.6× slower per frame but still inside the 33 ms camera budget, so the delivered
preview is identical.

**But it does not remove the contention.** Re-running the §2.6 test with YOLO on the NPU and the
warmup render on the GPU:

| Configuration | During warmup | p95 | worst frame |
| --- | --- | ---: | ---: |
| YOLO on GPU | 8–20 fps | 46.6 ms | 579.7 ms |
| YOLO on NPU | 12–30 fps | 52.2 ms | 522.2 ms |

**Root cause.** The GPU here is *integrated* — OpenVINO reports `Arc B390 (iGPU)`, so its "VRAM" is
shared system memory. A heavy SDXL render saturates memory bandwidth shared by CPU, iGPU and NPU,
so moving the model to a different compute unit does not help; the bottleneck is not the compute
unit. Frame capture, preprocessing and `result.plot()` also remain on the CPU.

**Expectation for the booth.** The target is a **discrete** Arc B70 with dedicated 32 GB VRAM.
There the diffusion workload has its own memory bandwidth, so NPU offload should decouple far more
cleanly — and even GPU-only contention should be milder. **Untested; re-run
`tests/gpu_contention_test.py` there.**

**Result.** `--npu` added, defaulting off. Deferred warmup (§2.6) remains the primary protection
because it works regardless of device topology; NPU offload is complementary, not a replacement.

---

## 3. What was built

| Area | Change |
| --- | --- |
| Startup | `_configure_offline_defaults` — offline/telemetry env defaults before import |
| Correctness | `task` selection for the pipeline class + `_require_task` mismatch guard |
| Correctness | `_is_degenerate_image` / `_render_with_black_guard`; black guide rejection |
| Throughput | `JuggernautRunner` — resident pipeline on a worker thread |
| Kiosk | Load during capture; warmup deferred until the camera is released |
| Kiosk | `--npu` routes YOLO pose to the NPU while RMBG stays on its configured device; Juggernaut stays on the GPU |
| Tooling | Startup/render instrumentation, JSONL metrics, probe modes, brightness scan |

### 3.1 JuggernautRunner

[src/vpro/vision/juggernaut_runner.py](src/vpro/vision/juggernaut_runner.py) holds one compiled
pipeline on a daemon thread and serves requests from a bounded queue. Types are in
[src/vpro/vision/juggernaut_types.py](src/vpro/vision/juggernaut_types.py).

Measured (`probe_load_phases.py --mode runner`):

| Task | Startup (once) | Render 1 | Render 2 | Render 3 |
| --- | ---: | ---: | ---: | ---: |
| text2img, 16 steps | 23.2 s | 18.19 | 17.21 | 17.99 |
| img2img, 28 steps @ 0.16 | 29.2 s | 5.72 | 4.83 | 6.15 |

Design decisions, each covered by
[tests/juggernaut_runner_test.py](tests/juggernaut_runner_test.py) (26 checks, ~1 s, no GPU):

- **Serialized by construction.** One GPU, one compiled pipeline; concurrent calls into the same
  compiled model are not safe to assume. A test asserts renders never overlap and FIFO holds.
- **Submissions before ready are queued, not rejected**, so the kiosk can enqueue during boot.
  `queue_wait_seconds` is reported separately from `render_seconds`.
- **A render failure never kills the worker.** It returns `RenderResult(ok=False, error=...)` and
  the next request proceeds.
- **Load failure is surfaced, not hung.** `wait_until_ready` raises, queued futures resolve with
  the error, further submits are rejected. A silently dead worker would hang the UI forever, which
  is the worst possible outcome.
- **Full queue rejects fast** rather than growing without bound.
- **Cancelled futures are skipped** via `set_running_or_notify_cancel`, so an abandoned request
  does not block the queue. This does *not* abort a running render — that needs a
  `callback_on_step_end` hook.
- **Requests validated at the boundary** (`RenderRequest.validated`): step, dimension and
  prompt-length caps stop a bad UI value wedging the booth.
- **A failed warmup is logged and ignored** rather than preventing service.
- **One runner owns one task.** Supporting both would mean two compiled pipelines; construct two
  runners if needed.

### 3.2 Kiosk wiring

`_run_social_pipeline` starts the runner **before** the camera work, so the load overlaps posing,
countdown, RMBG and compositing. End-to-end with vision stages stubbed (6 s capture, 2 s RMBG, 2 s
compose):

```
Juggernaut pipeline loading in the background during capture.
Juggernaut guided render: 5.79s (waited 18.00s for the pipeline)
capture=6.00 rmbg=2.00 compose=2.00 juggernaut=23.81 total=33.83
```

`--juggernaut-no-preload` defers loading until after capture if it is ever needed.

`--npu` moves YOLO pose off the GPU while leaving RMBG on its configured device. On the measured
development system, RMBG on NPU was ~891 ms p50 versus ~76 ms on GPU, so moving RMBG to the NPU
made the pipeline substantially slower. It only overrides YOLO when `--device` is still at its
default, explicit `--device` / `--rmbg-device` values win, and it fails fast with the available
device list if no NPU is present. Juggernaut is deliberately left on the GPU — SDXL is not a
practical NPU workload.

### 3.3 Deterministic pipeline reuse

The kiosk loads and warms RMBG once during startup rather than compiling it for every guest.
`compose_portrait_from_image` accepts an injected runtime for the kiosk path while retaining
one-shot construction for CLI commands. Scene canvases and prop PNGs are cached by resolved path
and output size. The composition path records `rmbg_seconds`, `yolo_seconds`,
`composition_seconds`, `compose_total_seconds`, and `juggernaut_seconds` in the kiosk status
snapshot so steady-state measurements can distinguish model work from image processing.

Pose inference is capped at 15 FPS by default (`--kiosk-pose-fps`) while camera capture and MJPEG
encoding continue at the camera rate. This reduces GPU contention without making the preview feel
stale. The rate can be set to zero to restore inference on every frame.

Kiosk output files older than 24 hours are removed at startup by default
(`--kiosk-output-retention-hours`). Set the value to zero to disable cleanup. Juggernaut queue
rejections are counted and exposed in the kiosk status response for operational monitoring.

For a steady-state booth benchmark, run `tests/kiosk_flow_test.py --runs 5` against a running kiosk
and compare the per-guest elapsed times with the `pipeline_timings` values from `/api/state`.

Failure path verified: a black guide raises inside the worker, the result returns `ok=False`, and
the deterministic compose is delivered instead.

---

## 4. Measurement tooling

| Tool | Purpose |
| --- | --- |
| `tests/juggernaut_test.py` | text2img bench; logs startup, cache state, per-render timings, JSONL |
| `tests/juggernaut_runner_test.py` | Runner lifecycle/concurrency, stubbed pipeline, no GPU |
| `tests/gpu_contention_test.py` | Preview fps vs. concurrent load; **re-run per target system** |
| `scripts/probe_load_phases.py` | `--mode core \| pipeline \| img2img \| runner` |
| `scripts/check_output_brightness.py` | Bulk scan for black/near-black outputs |

To measure cold-cache startup, point `--openvino-cache-dir` at an empty directory rather than
deleting the existing cache.

---

## 5. Remaining opportunities

Ordered by expected value given the current cost model.

1. **The ~7 s fixed per-render cost.** Now the largest slice of a ~5–6 s guided render, and
   unprofiled. Likely VAE decode at 1080×1350, then the two text encoders.
2. **Cache text-encoder output.** Prompts come from a small fixed scene set plus a constant style
   suffix and constant negative prompt; embeddings could be computed once at warmup.
3. **Check other paths for the §2.1 stall.** RMBG, YOLO26 and Twilio delivery may have the same
   invisible 14 s connect timeouts.
4. **Re-measure contention on the discrete Arc B70.** §2.7 predicts NPU offload decouples properly
   with dedicated VRAM; if so, the warmup could move back into the capture window and shave a few
   more seconds off the first guest.
4. **Decide img2img-guided vs. composite-on-generated-background.** Gates item 6 — the difference
   between hiding ~40 s of latency and not.
5. **Progressive preview.** A step callback decoding a cheap preview so the guest sees the image
   forming instead of a spinner.
6. **Speculative pre-render.** *Not possible as currently built*: the render is guided by the
   composed portrait, so it cannot start before capture. Requires the item-4 decision.
7. **Pre-generated background library.** Pre-render variants for the fixed scene list offline;
   guest-facing latency collapses to compositing time.
8. **Step / scheduler / CFG tuning.** ~1.0 s/step, but at strength 0.16 only ~4 steps run, so this
   is now worth little for the guided path. Higher-order schedulers remain cheap to test.
9. **Static shapes; pre-exported local IR.** Kiosk output is fixed at 1080×1350, batch 1. Mostly a
   throughput argument now that compile is only ~8 s.
10. **Trim import cost.** `optimum.intel` import is ~9.3 s, the single largest startup item.
    `torch` is imported only to build a seeded generator.
11. **Keep the camera open.** A persistent capture thread with a latest-frame buffer removes
    per-run camera init and shutter lag.
12. **Share one process for all models.** YOLO26, RMBG and Juggernaut each pay import/compile; one
    long-lived service also enables overlapping guest N's vision work with guest N-1's render.
13. **Prune the OpenVINO cache.** 203 blobs / 6.67 GB with no size cap or cleanup policy.
14. **Fail-soft delivery.** Decouple render completion from delivery so a guest who leaves still
    gets their image.

---

## 6. Open questions

- **What is inside the ~7 s fixed render cost?** Determines whether item 1 above is worth it.
- **Do RMBG / YOLO26 / Twilio suffer the same blocked-network stalls?** Invisible without profiling.
- **Why does the network blackhole rather than refuse?** A proxy returning `RST` would have made
  §2.1 fail in milliseconds instead of 337 s.
- **What causes the transient black frame?** The guard handles the symptom; the cause is unknown.
  A high retry rate during an event is the signal to investigate.
- **Is arbitrary resolution required?** Static shapes depend on committing to one output size.
- **How many guests per hour must the booth sustain?** Sets the acceptance bar for further work.

---

## 7. Future ideas (deferred)

Recorded so the current design does not accidentally block them.

### 7.1 Out-of-process daemon + IPC

Closes the one gap the in-process runner leaves: `tests/juggernaut_test.py` and the CLI are one
process per invocation, so they still pay full startup each time. Also the only way to let the UI
survive a generation-process crash.

- A `vpro juggernaut-serve` process loads once and listens on a local endpoint; clients fall back
  to a direct in-process load when no server is reachable.
- Transport: length-prefixed JSON over TCP on `127.0.0.1` (no new dependencies), or local HTTP if a
  web kiosk UI arrives. `multiprocessing.connection` is least code but pickle-based — loopback
  only, never across a network.
- Needs: bind to `127.0.0.1`, an auth token, a PID/lock file for single-instance enforcement, and
  validation confining `output_path` / `input_image_path` to configured directories, since
  accepting client-supplied paths is otherwise a path-traversal and arbitrary-write hazard.

Because callers already go through the runner's submit/await interface, this is a wrapper rather
than a rewrite.

### 7.2 Remote render host

One strong GPU host serving several thin kiosks. Attractive, but it invalidates several
assumptions:

1. **Filesystem paths stop working.** Exchange image **bytes** instead — a 1080×1350 JPEG is
   ~0.5–2 MB, negligible on a wired LAN next to a multi-second render. Alternatives: shared
   storage, or two-step HTTP (`POST /render` → job id, `GET /result/{id}`).
2. **Never pickle across a network boundary.** Use HTTP/TLS or length-prefixed JSON over TLS.
3. **Security becomes a real requirement.** TLS with a private CA, per-kiosk revocable tokens,
   isolated VLAN with firewalling. **Reject client-supplied filesystem paths entirely** — the
   server picks its own output location and inputs arrive as bytes. Cap upload size and validate
   decoded dimensions. Guest photos crossing the network also has consent/retention implications
   (see [docs/signage_and_consent_copy.md](docs/signage_and_consent_copy.md)).
4. **The queue becomes multi-tenant.** Round-robin fairness rather than FIFO, admission control
   with estimated wait, queue position in the UI.
5. **Network failure is a new outage mode.** Timeouts, bounded retries, idempotent request ids so a
   retry does not double-render, heartbeats so the kiosk knows before a guest starts, and a decided
   degraded behavior — falling back to a pre-generated background library beats a hung kiosk.

Cheap insurance: keep `RenderRequest` / `RenderResult` flat and primitive so a wire format can be
added without reshaping call sites.

---

## Appendix: two caches, often conflated

| | OpenVINO `CACHE_DIR` (disk) | Compiled pipeline (memory) |
| --- | --- | --- |
| Written | Once, on first compile | Once, per process launch |
| Read | Once **per process launch** | Never re-read; already resident |
| Survives process exit | Yes | **No** |
| Saves | Recompilation of IR → GPU kernels | Everything: imports, disk read, upload, compile |
| Relevant to renders 2..N in one process | **No** | Yes — they cost nothing |

The disk cache does not make later renders faster. Within one process, renders 2..N never consult
it. It only shortens the *startup* of each new process, and even then only removes recompilation —
imports, blob reads and weight upload are paid every launch.

That is the argument for the runner: the only way to stop paying startup is to stop starting up.
