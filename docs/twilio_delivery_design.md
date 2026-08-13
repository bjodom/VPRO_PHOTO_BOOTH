# Twilio Rich SMS Delivery Design

## Goal
Send each visitor a delivery-only SMS/MMS where the final image appears directly in the message.

This channel is for photo delivery only, not lead generation.
All image processing remains local to the kiosk; Twilio is used only for outbound SMS delivery.

## Recommended Flow
1. Visitor selects SMS delivery and enters phone number.
2. App shows delivery consent text and requires explicit confirmation.
3. Backend writes the final branded image to local storage and prepares an MMS-accessible media endpoint.
4. Backend calls Twilio Messaging API using a pre-approved Twilio Content Template (`ContentSid`) and template variables.
5. Delivery status callbacks update message state.
6. Phone number and message metadata are purged per policy.

## Twilio Content Template Builder (Recommended)
Use Twilio Content Template Builder to define one delivery-only template for MMS.

Suggested template behavior:
- Text body: short delivery-only copy (for example, "Here is your Intel vPro photo.")
- Media placeholder: image URL variable bound at send time
- Compliance text: include opt-out guidance if required by your sender profile/region

Send-time fields:
- ContentSid: ID of approved Twilio content template
- ContentVariables: JSON string with placeholders (for example, media URL and optional display text)
- To: visitor phone number
- MessagingServiceSid or From: configured Twilio sender
- StatusCallback: local webhook for delivery events

## Message Strategy
Primary template:
"Here is your Intel vPro photo."

Optional compliance suffix by region/sender policy:
"Reply STOP to opt out."

Guidelines:
- Keep under 320 characters for readability.
- Avoid marketing language.
- Keep message content delivery-only and concise.

## API Contract (Internal)
### POST /api/delivery/sms
Request:
```json
{
  "session_id": "evt_2026_08_10_00123",
  "phone_e164": "+15555550123",
  "asset_id": "img_abc123",
  "asset_path": "outputs/img_abc123.jpg",
  "consent_delivery": true,
  "consent_marketing": false
}
```

Response:
```json
{
  "status": "queued",
  "delivery_id": "sms_789",
  "provider": "twilio"
}
```

### POST /api/delivery/sms/callback
Twilio status callback target. Store only minimal operational fields:
- delivery_id
- twilio_message_sid
- message_status
- timestamp

## Twilio Send Fields
Required:
- To
- ContentSid
- ContentVariables

One sender option is required:
- MessagingServiceSid
- From

Recommended:
- StatusCallback

Optional:
- Body and MediaUrl when using direct send mode instead of template mode

Use MMS-direct by default so the image appears in the message.

## Security and Privacy Controls
- Validate E.164 phone format server-side.
- Rate-limit per kiosk session and IP.
- Protect media endpoint with expiring signed token and short TTL.
- Never store phone numbers longer than policy window.
- Separate delivery consent from marketing consent in data model.

## Operational Notes
- Add retry policy for transient Twilio failures.
- Show fallback in kiosk UI: "SMS failed. Please confirm number and retry with staff assistance."
- Log provider errors without storing message content beyond what is needed for debugging.

Template operations notes:
- Keep template content stable and update only approved variables per send.
- Store template version with each delivery record for auditability.
- Add health check at startup to verify configured ContentSid exists and is active.

## Minimal Python Service Sketch (Template Mode)
```python
from twilio.rest import Client


def send_delivery_mms_template(
  account_sid: str,
  auth_token: str,
  messaging_service_sid: str,
  content_sid: str,
  to: str,
  media_url: str,
) -> str:
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        to=to,
    messaging_service_sid=messaging_service_sid,
    content_sid=content_sid,
    content_variables='{"1": "Here is your Intel vPro photo.", "2": "' + media_url + '"}',
    )
    return message.sid
```

## Minimal Python Service Sketch (Direct MMS Mode)
```python
from twilio.rest import Client


def send_delivery_mms_direct(account_sid: str, auth_token: str, sender: str, to: str, media_url: str) -> str:
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        from_=sender,
        to=to,
        body="Here is your Intel vPro photo.",
        media_url=[media_url],
    )
    return message.sid
```

## Environment Variables
- TWILIO_ACCOUNT_SID
- TWILIO_AUTH_TOKEN
- TWILIO_MESSAGING_FROM (if using direct mode)
- TWILIO_MESSAGING_SERVICE_SID (if using template mode)
- TWILIO_CONTENT_SID (template mode)
- DELIVERY_MEDIA_SIGNING_KEY

## Acceptance Criteria
- MMS can be sent in under 3 seconds median after image finalization.
- Failed sends provide a retry path with no dead-end screens.
- Delivery audit records do not include marketing consent unless explicitly granted.
- Data purge job removes phone numbers within approved retention window.
