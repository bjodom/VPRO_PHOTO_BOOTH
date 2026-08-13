# vPRO Visitor Photo Experience Plan

## 1) Experience Summary
A visitor walks up, chooses a destination scene (city, beach, mountain, etc.), poses in front of a camera, and receives a stylized image where they appear in that location holding an Intel vPro laptop prop.

Primary goals:
- Fast, delightful experience on-site
- Strong Intel brand association in final output
- No hidden lead-generation behavior
- Safe handling of visitor photos

## 2) End-to-End User Flow
1. Idle screen invites visitor to begin and explains timing (for example: 30-90 seconds).
2. Visitor accepts consent and privacy notice.
3. Visitor chooses location from curated world scenes.
4. Pose guide overlay appears (for example: one hand up for laptop placement).
5. Webcam live preview runs, then one still frame is captured as the render input image.
6. Pipeline generates final composite image from the captured still frame.
7. Watermark/hashtags are applied.
8. Visitor receives image via SMS only.
9. Data is automatically deleted per policy window.

## 3) Technical Architecture (MVP)
### Frontend Kiosk App
- Full-screen kiosk UI (touch + keyboard support)
- Steps: consent -> location -> pose guide -> capture -> generating -> delivery
- Clear progress and retry states

### Capture + Vision Layer
- Webcam capture target: 2560x1440 (2K) when available; fallback to 1920x1080 minimum
- Single-frame extraction from live feed for deterministic rendering input
- Person background removal and alpha matting (RMBG v1.4)
- Pose/keypoint estimation for hand location (YOLO26 keypoint model)

### Framing and Subject Selection Policy
- Single-person framing is preferred, but not required.
- If multiple people are detected, the system selects one primary subject automatically (highest-confidence person with center bias and minimum size threshold).
- Non-primary people are excluded from compositing when possible via mask cleanup.
- If no subject passes minimum confidence/size or required hand-keypoint visibility, show a capture hint and allow retry.

### Output Aspect Policy (Production)
- Portrait-first only: final delivered image is always 1080x1350 (4:5).
- No landscape or square delivery variants are produced in production mode.
- Person detection/extraction runs on the captured frame, then the subject is composited into the portrait canvas.
- For 2K webcam input, inference can run on model-friendly resized tensors (for example 1024x1024) while final compositing/export remains 1080x1350.

### Composition Layer
- Scene template selected from approved library
- Foreground person scaled and blended into scene
- vPro laptop prop composited near hand keypoint with perspective + shadow
- Optional color matching and grain pass for realism

### Delivery Layer
- Packages final branded image for direct MMS delivery
- SMS send only for delivery, no marketing enrollment
- Auto-delete raw and final images after retention window

### Local Processing Policy (Non-Negotiable)
- All image capture, segmentation, pose estimation, compositing, and enhancement run locally on the kiosk laptop.
- No remote image inference or remote image-processing services are used.
- The only third-party integration is Twilio, used strictly for SMS delivery after image generation is complete.

## 4) Recommended Technology Stack
### Frontend (Kiosk UI)
- Framework: React + TypeScript (Vite build)
- UI: Full-screen kiosk web app with route-locked flow and inactivity reset timer
- Camera: Browser media APIs (`getUserMedia`) with live framing guides
- State: lightweight state store (for example, Zustand) for step/session state
- Styling: Tailwind CSS or CSS modules with high-contrast accessibility defaults

### Backend (Orchestration + APIs)
- Framework: FastAPI (Python 3.14)
- Responsibilities: capture session orchestration, pipeline invocation, asset packaging, Twilio SMS delivery, retention jobs
- Validation: Pydantic models for API contracts and consent payloads
- Storage: local disk or object storage for short-lived assets + SQLite/Postgres for session metadata

### Vision + Image Pipeline
- Background removal: OpenVINO-executed RMBG v1.4 model
- Pose/keypoints: YOLO26 keypoint model executed with OpenVINO
- Compositing: OpenCV + Pillow for deterministic placement, color matching, and watermarking
- Prop insertion: hand-keypoint-based laptop placement with angle templates

### RMBG Background Removal Convention
Primary model:
- `briaai/RMBG-1.4` converted to OpenVINO IR and stored in local model assets.

Local OpenVINO IR path convention:
- `models/rmbg/rmbg-1.4/model.xml`
- `models/rmbg/rmbg-1.4/model.bin`
- `models/rmbg/rmbg-1.4/metadata.json`

Startup validation requirements:
- Fail fast if RMBG `model.xml` or `model.bin` is missing.
- Run one warmup inference during app startup.
- Log median foreground-mask generation time over first 5 frames.

Fallback policy:
- If RMBG is unavailable, fail startup in production mode.
- Optional developer-only fallback may use existing portrait-matting model for local experiments.

### YOLO26 Model Selection and Artifact Convention
Recommended starting variants:
- `yolo26n-pose` for lowest latency and fast booth throughput.
- `yolo26s-pose` for higher keypoint stability if latency budget allows.

Selection rule:
- Default to `yolo26n-pose` for production events.
- Promote to `yolo26s-pose` only if p95 latency remains within target on the event laptop.

Local OpenVINO IR path convention:
- `models/yolo26/keypoint/<variant>/model.xml`
- `models/yolo26/keypoint/<variant>/model.bin`
- `models/yolo26/keypoint/<variant>/metadata.json`

Example concrete paths:
- `models/yolo26/keypoint/yolo26n-pose/model.xml`
- `models/yolo26/keypoint/yolo26n-pose/model.bin`

`metadata.json` minimum fields:
- `model_name`
- `model_version`
- `input_shape`
- `keypoint_format`
- `confidence_threshold_default`
- `nms_threshold_default` (if applicable for selected export mode)

Startup validation requirements:
- Fail fast if `model.xml` or `model.bin` is missing.
- Log selected variant and thresholds at boot.
- Run a one-frame warmup inference before opening kiosk capture screen.

### Explicit Person Extraction Stack (Busy Backgrounds)
Primary extraction path (default):
1. YOLO26 keypoint model (OpenVINO IR) identifies the primary visitor and keypoints.
2. RMBG v1.4 (OpenVINO IR) produces foreground alpha matte and isolated subject image.
3. Post-process mask with edge feathering and hole filling, then composite into destination scene.

Why this path:
- Better robustness in cluttered booths than bbox-only segmentation.
- Cleaner hair/edge quality than coarse instance masks.
- Stays fully OpenVINO-first on Intel hardware.

RMBG implementation alignment:
- Follow OpenVINO notebook conversion/inference pattern for RMBG v1.4.
- Keep preprocessing/postprocessing consistent with the converted RMBG model contract.
- Assume batch size 1 for live kiosk pipeline and optimize for latency mode.

### OpenVINO-First Runtime Policy
- OpenVINO is the primary runtime across the pipeline wherever model support exists.
- YOLO26 keypoint, segmentation, and enhancement should run on OpenVINO on the Intel laptop by default.
- PyTorch is allowed only as a secondary path for development, conversion, or unsupported edge cases.
- Release acceptance requires OpenVINO parity on all critical visitor-path features.

### Generative Enhancement Model
- Model: OpenVINO/Juggernaut-XL-v9-fp16-ov
- Status: already converted for OpenVINO runtime; no additional conversion step required
- Role: quality render pass using the captured still frame and compositing outputs (img2img or inpaint enhancement)
- Runtime target: OpenVINO execution path on Intel hardware (first-choice path)
- Guardrail: keep deterministic non-generative fallback path available at all times

Juggernaut artifact map (reference from model repository):
- `model_index.json` (pipeline component mapping)
- `scheduler/` (scheduler config)
- `tokenizer/` and `tokenizer_2/` (text tokenizers)
- `text_encoder/` and `text_encoder_2/` (dual text encoders)
- `unet/` (denoising core)
- `vae_encoder/` and `vae_decoder/` (latent encode/decode)

Implementation note:
- Mirror this folder structure under local model storage so component loading is deterministic.
- Validate component presence at app startup and fail fast with operator-readable errors if any folder is missing.

### Delivery + Messaging
- Provider: Twilio Messaging API (SMS-only flow)
- Payload: direct MMS image payload (no download link)
- Template strategy: Twilio Content Template Builder with approved ContentSid and runtime ContentVariables
- Compliance: explicit delivery consent, separate marketing consent, STOP/HELP handling where required

### Ops, Security, and Monitoring
- Secrets: environment variables + OS secret store
- Logs: structured logs with PII minimization
- Metrics: generation latency, SMS send success, retry counts, session completion rate
- Cleanup: scheduled purge of phone numbers and image assets per retention policy

### Execution Modes
1. Fast mode (default): deterministic compositing only
2. Quality mode (optional): deterministic composite + Juggernaut enhancement

Recommendation:
- Launch MVP with OpenVINO Fast mode for event reliability.
- Enable OpenVINO Quality mode as a controlled toggle after performance validation on the target Intel laptop.

## 5) World Images Strategy
Primary creation path:
- All world-scene images are generated with the Juggernaut model.
- Do not mix in licensed stock, tourism-board photography, or other third-party scene-image sources for production packs.
- Human art direction is still required, but the final approved scene assets should come from the Juggernaut generation workflow.

Requirements:
- Commercial usage of the generated outputs must align with the approved Juggernaut model/license terms and internal legal guidance.
- Consistent resolution, composition framing, and style families across destination sets.
- Metadata tags for quick filtering (city, indoor/outdoor, day/night, region).

Scene pack workflow:
- Generate destination-scene families in batches using locked prompt templates, negative prompts, and seed-tracking so look-and-feel stays consistent.
- Keep a manifest for each approved scene pack (prompt, negative prompt, seed, model version, generation date, owner, approval status).
- Run internal brand/legal review on generated outputs before event use; only publish scenes with explicit internal approval.
- Export final approved scene pack to portrait 1080x1350 templates only.

Generation guardrails:
- Favor stylized travel/postcard realism over photojournalistic depictions of specific copyrighted landmarks when approval is uncertain.
- Avoid generating recognizable people, trademarks, or protected artwork in the background plates.
- Regenerate any output with artifacting, distorted architecture, unreadable signage, or inconsistent lighting before it reaches the approved pack.

Visitor insertion policy (important):
- Use local deterministic compositing for live visitor insertion and laptop prop placement.
- Do not send live visitor images to any remote processing service.

Do not train custom generative models for MVP unless rights and timeline justify it.
Use curated Juggernaut prompt templates first.

## 6) Intel Branding Strategy
- Persistent corner watermark: Intel logo + hashtag
- Suggested tags: #Intel #IntelvPro #OnTheGoWithvPro
- Optional frame style with event branding
- Keep branding readable but not intrusive

## 7) Prop Laptop Placement (vPro Device)
Practical implementation:
- Use transparent PNG render(s) of approved Intel vPro laptop angles
- Detect hand/wrist keypoint and estimate orientation
- Snap to nearest matching prop angle
- Add shadow/contact shadow layer for realism

Future upgrade:
- 3D prop render with dynamic perspective from pose depth estimate

## 8) Privacy, Consent, and Trust (Critical)
Non-negotiables:
- Explicit consent before photo capture
- Explain what is captured, why, and retention period
- Explain delivery channels are not lead generation
- No automatic marketing opt-in

Suggested signage language:
"Photo delivery by SMS only. We do not use your phone number for marketing unless you explicitly opt in."

Data handling baseline:
- Encrypt in transit and at rest
- Minimal metadata collection
- Retention target: delete raw images quickly (for example, within 24 hours), final images within a short event policy window unless visitor requests immediate delete
- Audit log without storing biometric templates

## 9) SMS-Only Delivery Without Lead Gen Feel
Delivery channel:
1. SMS/MMS only with the final image delivered directly in-message

Twilio Rich SMS option:
- Use Twilio Messaging API to send the final image as MMS media.
- Keep message copy short and delivery-only.
- Use branded sender where available (short code, verified toll-free, or approved sender ID based on region).
- Message template should be delivery-only, for example:
	"Here is your Intel vPro photo. Reply STOP to opt out of future SMS from this sender."

If SMS is used:
- Separate delivery consent from marketing consent
- Marketing opt-in default must be off
- Single-purpose message text
- Store phone number only for delivery workflow and purge per retention policy
- Include regional compliance controls (for example STOP/HELP handling where required)

## 10) Performance Targets
MVP latency targets:
- Capture to preview: under 5 seconds
- Capture to final image: under 30-90 seconds depending on mode
- End-to-end visitor session: under 2 minutes

Operational targets:
- 95% session success rate
- Clear fallback path if generation fails (retry capture or alternate template)

Future performance improvements (post-MVP):
- Keep a long-running local render worker that loads Juggernaut once at startup and reuses the in-memory pipeline across kiosk sessions.
- Route kiosk render requests to that local worker instead of spawning a new Python process for each render.
- Batch multiple renders within one process when practical to reduce repeated startup/compile overhead.
- Keep Hugging Face model loading in local-files-only mode by default to avoid network delays during events.
- Enable and tune OpenVINO compile/cache options to reduce first-run warmup latency after reboot.
- Preserve deterministic compositing as a fallback path if the generative worker is unavailable or queue depth exceeds latency targets.

## 11) Deployment Modes
### Edge-first Kiosk (recommended)
- Local inference on event PC with Intel GPU/NPU where possible
- Works even with poor internet
- Sync local logs when operationally needed

Recommendation:
- Run the entire image pipeline locally on the kiosk; use Twilio only for SMS delivery.

## 12) MVP Build Plan (4-6 Weeks)
Week 1:
- Finalize UX wireflow and consent copy
- Build capture screen and template selector

Week 2:
- Integrate segmentation + pose
- Implement deterministic compositing + prop placement

Week 3:
- Add branding overlays and SMS delivery flow
- Add SMS sender with strict consent separation
- Integrate Twilio send path using Content Template Builder with audited delivery template and retention-safe logging

Week 4:
- Hardening, kiosk mode, telemetry, failure recovery
- Pilot test with real users

Weeks 5-6 (optional):
- Add diffusion enhancement mode
- Improve realism and prop perspective

## 13) Risks and Mitigations
Risk: Unrealistic composites
- Mitigation: template curation, lighting normalization, shadow pass

Risk: Slow generation at peak traffic
- Mitigation: deterministic fast path, queue UI, optional high-quality mode toggle

Risk: Privacy trust concerns
- Mitigation: bold signage, transparent policy, short retention, no hidden opt-ins

Risk: Content misuse
- Mitigation: moderation guardrails, staff oversight, terms acceptance

## 14) What to Decide Next
1. MVP visual style: photo-real compositing or stylized postcard look
2. SMS delivery message template and sender strategy (short code, toll-free, or sender ID)
3. Data retention window and legal review owner
4. Approved hashtag and exact branding treatment
5. Initial scene pack (for example, 10-20 destinations)

## 15) Completion Checklist and Acceptance Criteria
Use this section as the release gate for MVP readiness.

### Product and UX Readiness
- Kiosk flow implemented end-to-end: consent -> location -> pose -> capture -> generating -> MMS delivery confirmation.
- Idle reset returns app to attract screen after configured inactivity timeout.
- User-facing copy is approved and consistent with non-lead-gen intent.
- Acceptance target: at least 95% of pilot users complete flow without staff intervention.

### Local-Only Processing Compliance
- All image processing stages (capture, extraction, compositing, enhancement) execute on-device.
- No remote inference endpoints are present in runtime configuration.
- Acceptance target: network disconnect does not block image generation pipeline.

### Model and Pipeline Readiness
- YOLO26 keypoint OpenVINO model is integrated and outputs stable keypoints for laptop placement.
- RMBG v1.4 OpenVINO model is integrated and produces clean alpha for busy backgrounds.
- Juggernaut enhancement path is optional and toggled by quality mode.
- Acceptance target: keypoint placement and mask quality pass internal visual QA on curated test set.

### Performance and Reliability
- Capture-to-preview latency: <= 5 seconds median.
- Capture-to-final-image latency: <= 90 seconds p95 in default mode.
- Session completion success: >= 95% over pilot run.
- Acceptance target: no critical crashes in an 8-hour soak test.

### Twilio MMS Delivery Readiness
- Final image is sent as direct MMS media payload (no download-link dependency).
- Delivery status callbacks are captured and mapped to local session IDs.
- Retry flow is available for transient send failures.
- Acceptance target: >= 95% successful sends on pilot carrier mix.

### Privacy, Retention, and Consent
- Delivery consent is required before SMS/MMS send.
- Marketing consent remains separate and default-off.
- Purge job removes phone numbers and image artifacts per approved policy window.
- Acceptance target: retention policy validated by audit script and spot checks.

### Security and Data Handling
- Secrets are loaded from secure environment configuration.
- Logs avoid raw image data and minimize PII.
- Access to admin or diagnostic screens is restricted.
- Acceptance target: internal security checklist signed off.

### Testing and QA Coverage
- Functional tests: happy path, retry path, invalid phone input, and cancellation flow.
- Edge-case tests: low light, multiple people in frame, partial occlusion, varied skin tones/clothing.
- Offline tests: local generation works without internet; Twilio send is queued or gracefully failed with operator guidance.
- Acceptance target: all critical test cases pass and defects are triaged.

### Operations and Event Runbook
- Startup checklist documented (camera check, model warm-up, sender check).
- Incident playbooks documented (camera failure, MMS failure, low disk space, app restart).
- End-of-day checklist documented (log export, purge verification, health summary).
- Acceptance target: event staff dry run completed successfully.

### Go/No-Go Decision Gate
MVP is release-ready only when all checklist areas above are marked pass by product, engineering, and legal/privacy owners.

## 16) UI Wireframe Artifact
- Primary low-fidelity wireframe document: `docs/ui_wireframe.md`
- Includes: end-to-end screen flow, per-screen layouts, validation rules, and error-state coverage.
