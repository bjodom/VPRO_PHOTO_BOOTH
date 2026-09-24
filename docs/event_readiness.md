# Event Acceptance Checklist

Software checks reduce risk; they do not establish that a booth is flawless. Complete this checklist
on the final camera, display, Intel hardware, power configuration, and event network before opening.

## Before Arrival

1. Install the locked environment with `uv sync --extra gen --extra kiosk`; verify local YOLO/RMBG
  XML and BIN files. Confirm model and brand usage rights for the event.
2. Run `.\tests\cli_pipeline_test.ps1`. This does not open a real camera or load models.
3. Launch `.\run.ps1 -OutputDir outputs/event_validation`. Model loading is offline by default.
   Wait for the Start button to enable. Check `http://127.0.0.1:8000/health/ready` returns 200.
4. Inspect approved sample portraits from the actual pipeline. Destination thumbnails are reference
   photographs, not sample generated outputs; their attribution is in the local Photo credits page.
5. Set Windows power/sleep, camera privacy permissions, display rotation, browser fullscreen, and
   kiosk restrictions. Staff controls are gesture-protected, not authenticated. Do not expose the
   administrative web UI to the guest LAN.

## Camera and Quality

- Check standing and seated guests, different heights, glasses, skin tones, hair, and clothing.
- Check both hands, laptop placement, faces, fingers, hair edges, and complete feet in the final photo.
- Verify portrait rotation (`-CaptureRotate 90` or `270` for a sideways camera) and low-light behavior.
- Run countdown, cancel countdown, retake, and change-scene flows. No previous guest image should remain.
- Disconnect/reconnect the camera and use Retry Camera. Stale frames must not be captured.
- Judge every destination with actual outputs before approving its use. Tune steps/strength only with
  side-by-side output review; smaller diffusion workloads can change quality or the destination.

## Performance and Throughput

With a consenting test subject in position, run:

```powershell
.\.venv\Scripts\python.exe tests/kiosk_flow_test.py --base-url http://127.0.0.1:8000 --runs 10 --p95-budget 60
```

This automatically resets and captures sessions. Do not run it during live guest operation.
It requires an actual render timing, a QR, a reachable handoff page, and a downloaded JPEG.
It reports first-guest time separately from warm p50/p95. Sixty seconds is a provisional acceptance
budget, not a measured promise; set the event's agreed budget explicitly.

Repeat with `-Device intel:gpu` and the default `intel:npu`, keeping RMBG on CPU and DreamShaper on GPU.
Compare tail latency, UI responsiveness, preview FPS, failures, and output quality, not just isolated
model inference. Record the CPU/GPU/NPU names, OpenVINO version, model, dimensions, steps, and strength.
The historical 5-6 second img2img figure does not apply to 30-step, strength-0.99 kiosk inpainting.

## Phones and Privacy

- Configure `-DeliveryAdvertiseHost` to the kiosk's reachable LAN address. Verify Windows firewall
  access to the handoff port (8765); do not disable the firewall globally.
- Test real iPhone and Android scans on the event Wi-Fi, including save/download and copy caption.
  Venue client isolation and cellular-only phones can prevent access to local URLs.
- Verify the URL expires after 15 minutes and a new session cannot see the previous guest's capture.
- Explain that link expiry is not immediate disk deletion. The retention worker checks every minute
  and protects live sessions and QR downloads. Keep the process running for scheduled cleanup.
- Stop the booth before operator-directed bulk deletion; never delete the compiled-model cache as
  part of guest-image cleanup. Treat temporary acceptance photos as personal data too.

## Recovery and Staff

- Hold the Intel vPro header for 1.8 seconds, or focus it and press Ctrl+Enter, to open diagnostics.
  Force Reset asks for confirmation. A reset discards results from an abandoned session.
- Test a renderer failure: the kiosk should show a retryable error, not deliver the gray composition.
- Test a delivery failure: the final image should remain saved and Retry must not regenerate it.
- Test loss of the UI connection: show reconnecting, disable duplicate actions, and cancel countdown.
- Test low disk space in a disposable environment, shutdown/relaunch, and port conflicts before arrival.
- Keep a staff-assisted alternative ready if networking or hardware fails. Do not advertise a fixed
  completion time until the final system has passed its latency and quality gates.

## Review-Only Browser Preview

`.\.venv\Scripts\python.exe tests/kiosk_app_test.py --serve-preview` serves a hardware-free fixture at
http://127.0.0.1:8879. Its gray image is deliberately synthetic. It is only for layout and interaction
checks and cannot pass the real generation acceptance benchmark.