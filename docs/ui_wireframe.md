# vPRO Kiosk UI Wireframe (Low Fidelity)

## Purpose
This wireframe defines the primary kiosk experience screens for the local, OpenVINO-first flow.

Delivery is by **on-screen QR code**: the guest scans it and downloads the image from the kiosk
over the local network. No phone number is collected, which removes the phone-entry screen, the
on-screen keyboard, and most PII handling from the experience. Delivery sits behind the
`DeliveryChannel` interface, so MMS can be added later without changing these screens.

## Screen Flow
```mermaid
flowchart LR
  A[Attract / Idle] --> B[Consent]
  B --> C[Location Select]
  C --> D[Pose Guide + Camera Preview]
  D --> E[Capture Review]
  E --> F[Generating]
  F --> G[Ready: QR Handoff]
  G --> H[Success / Reset]
  H --> A
  D --> D1[Camera Error]
  F --> F1[Generation Error]
  G --> G1[Delivery Error]
  D1 --> D
  F1 --> C
  G1 --> G
```

## Global Layout Rules
- Full-screen kiosk layout, optimized for portrait 1080x1350.
- Touch-first controls with large targets and strong contrast.
- Persistent top-right support button for staff override.
- Session timeout resets to Idle screen.

## 1) Attract / Idle
Goal: invite visitors and explain the value in one glance.

```text
+--------------------------------------------------------------------------------+
| vPRO Travel Photo Experience                                    [Staff Button] |
|--------------------------------------------------------------------------------|
|                                                                                |
|                  Put yourself anywhere. Hold a vPro laptop.                   |
|                                                                                |
|                         [ Start Your Photo Experience ]                        |
|                                                                                |
|                 About 1 minute. Scan a QR code to take it with you.           |
|                                                                                |
+--------------------------------------------------------------------------------+
```

Notes:
- Loop subtle background animation or rotating sample outputs.
- Auto-return here after completion or inactivity.

## 2) Consent
Goal: capture clear consent and trust messaging.

```text
+--------------------------------------------------------------------------------+
| Consent                                                         [Staff Button] |
|--------------------------------------------------------------------------------|
| We capture your photo to generate your personalized image.                     |
| All image processing runs locally on this kiosk.                               |
| We do not ask for your phone number, email, or any personal details.           |
| Download your image by scanning a QR code; the link expires in 15 minutes.     |
|                                                                                |
| [ ] I agree to photo capture and image generation.                             |
|                                                                                |
| [ Cancel ]                                                  [ Accept & Continue ]|
+--------------------------------------------------------------------------------+
```

Validation:
- Continue disabled until checkbox selected.

## 3) Location Select
Goal: choose destination quickly.

```text
+--------------------------------------------------------------------------------+
| Choose Your Destination                                         [Staff Button] |
|--------------------------------------------------------------------------------|
| [ Search ____________________ ]         [ Region: All v ] [ Style: All v ]    |
|                                                                                |
| [ Cairo / Pyramids ] [ Tokyo Night ] [ Swiss Alps ] [ NYC Skyline ]           |
| [ Santorini ]        [ Paris ]       [ Dubai ]      [ Rio ]                   |
| [ Cape Town ]        [ Sydney ]      [ Seoul ]      [ Custom Event Set ]      |
|                                                                                |
| [ Back ]                                                     [ Continue ]       |
+--------------------------------------------------------------------------------+
```

Validation:
- Continue disabled until one location selected.

## 4) Pose Guide + Camera Preview
Goal: guide body position and hand placement for laptop prop.

```text
+--------------------------------------------------------------------------------+
| Strike the Pose                                                [Staff Button]  |
|--------------------------------------------------------------------------------|
| +-----------------------------------+   +------------------------------------+ |
| | Live Camera Feed                  |   | Pose Guide                          | |
| | (person preview with silhouette)  |   | 1. Stand in frame center            | |
| |                                   |   | 2. Raise one hand like a tray       | |
| | [outline + hand target marker]    |   | 3. Hold still on capture            | |
| +-----------------------------------+   +------------------------------------+ |
|                                                                                |
| [ Reposition ]                                     [ Capture ]                 |
+--------------------------------------------------------------------------------+
```

System behavior:
- Show live quality hints: low light, multiple people, out-of-frame.
- Use YOLO26 keypoints to validate hand visibility before enabling Capture.
- If multiple people are present, auto-select one primary subject (center-biased, highest-confidence) and show a non-blocking hint so users can reposition for cleaner results.

## 5) Capture Review
Goal: confirm the shot before processing.

```text
+--------------------------------------------------------------------------------+
| Review Your Photo                                             [Staff Button]   |
|--------------------------------------------------------------------------------|
|                         [ Captured image preview ]                              |
|                                                                                |
|           Looks good? This image will be composited in your scene.            |
|                                                                                |
| [ Retake ]                                                   [ Use This Photo ] |
+--------------------------------------------------------------------------------+
```

## 6) Generating
Goal: reassure while local pipeline runs.

```text
+--------------------------------------------------------------------------------+
| Creating Your Image                                            [Staff Button]  |
|--------------------------------------------------------------------------------|
|                                                                                |
|                      [ Progress Ring / Bar: 0% ... 100% ]                     |
|                                                                                |
|  Running local AI pipeline: pose -> extraction -> composite -> enhancement     |
|                                                                                |
|                         Please keep this screen open.                          |
|                                                                                |
+--------------------------------------------------------------------------------+
```

Notes:
- Guided render is ~5-6s on a warm booth, so this screen is brief; avoid over-designing it.

## 7) Ready: QR Handoff
Goal: get the image onto the guest's phone with no data collection.

```text
+--------------------------------------------------------------------------------+
| Your Photo Is Ready                                           [Staff Button]   |
|--------------------------------------------------------------------------------|
|            [ Final image preview ]            +----------------------+         |
|                                               |                      |         |
|                                               |     [ QR CODE ]      |         |
|            Scan to download to your phone.    |                      |         |
|                                               +----------------------+         |
|            Press and hold the image, then                                      |
|            choose Add to Photos.              Link expires in 15 minutes.      |
|                                                                                |
| [ Show Caption ]                                              [ I'm Done ]     |
+--------------------------------------------------------------------------------+
```

Notes:
- Requires the guest's phone to reach the kiosk. Venue guest WiFi often blocks device-to-device
  traffic (client isolation), and a phone on cellular cannot reach a LAN address at all. Verify
  the venue network before the event; a kiosk-hosted hotspot is the fallback.
- The caption with hashtags is offered with a copy button on the phone page, supporting the social
  post without the kiosk needing to know anything about the guest.
- Do not block the reset on the download: the guest may walk away and still fetch the image until
  the link expires.

## 8) Success / Reset
Goal: close with delight and reset.

```text
+--------------------------------------------------------------------------------+
| All Set!                                                      [Staff Button]   |
|--------------------------------------------------------------------------------|
|                                                                                |
|                   Thanks for trying the Intel vPro experience!                |
|                                                                                |
|                     Returning to home screen in 10 seconds...                 |
|                                                                                |
|                   [ Start Another ]                                            |
|                                                                                |
+--------------------------------------------------------------------------------+
```

## Error States

## E1) Camera Error
- Message: Camera unavailable. Please ask event staff.
- Actions: Retry Camera, Staff Override.

## E2) Generation Error
- Message: We could not generate your image this time.
- Actions: Retry Generation, Retake Photo, Choose Another Scene.

## E3) Delivery Error
- Message: We could not prepare your download link.
- Actions: Retry, Staff Override.
- The image is already saved locally, so staff can hand it over another way.

## E4) Timeout Reset
- Message: Session timed out. Returning to start.
- Action: Auto-return to Idle.

## Staff Controls (Hidden Panel)
- Reinitialize camera
- Reinitialize OpenVINO pipeline
- Delivery self-test (serve a known image and show its QR)
- View recent failed sessions
- Force return to Idle

## Wireframe-to-Build Mapping
- Attract, Consent, Select, Pose, Review, Generating, Ready, Success map 1:1 to frontend routes.
- Error states map to overlay modals with actionable retry buttons.
- All transitions should emit local telemetry events for completion-rate analysis.
