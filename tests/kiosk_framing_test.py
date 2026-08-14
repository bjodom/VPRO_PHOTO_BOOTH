"""Framing guidance tests.

    python tests/kiosk_framing_test.py

Pure geometry; no camera needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpro.kiosk.framing import (  # noqa: E402
    STATUS_CROPPED,
    STATUS_NO_LOWER_BODY,
    STATUS_NO_SUBJECT,
    STATUS_OFF_CENTRE,
    STATUS_OK,
    STATUS_TOO_CLOSE,
    STATUS_TOO_FAR,
    evaluate_framing,
    target_box,
)

PASSED: list[str] = []
W, H = 1920, 1080


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed. {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


def centred(height_ratio: float, offset: float = 0.0, top: int | None = None) -> tuple:
    box_h = int(H * height_ratio)
    y1 = (H - box_h) // 2 if top is None else top
    width = int(box_h * 0.42)
    cx = W / 2 + offset * W
    return (int(cx - width / 2), y1, int(cx + width / 2), y1 + box_h)


def test_no_subject() -> None:
    feedback = evaluate_framing(None, W, H)
    check("no subject is not ok", not feedback.ok)
    check("asks guest to step in", feedback.status == STATUS_NO_SUBJECT)


def test_good_framing() -> None:
    feedback = evaluate_framing(centred(0.72), W, H)
    check("ideal framing is ok", feedback.ok, feedback.message)
    check("status ok", feedback.status == STATUS_OK)
    check("reports height ratio", abs(feedback.height_ratio - 0.72) < 0.02)


def test_too_close_and_too_far() -> None:
    close = evaluate_framing(centred(0.95, top=20), W, H)
    check("too close is rejected", not close.ok)
    check("tells guest to step back", "back" in close.message.lower(), close.message)

    far = evaluate_framing(centred(0.30), W, H)
    check("too far is rejected", not far.ok)
    check("status too far", far.status == STATUS_TOO_FAR)
    check("tells guest to step closer", "closer" in far.message.lower(), far.message)


def test_cropped_head_takes_priority() -> None:
    # Touching the top edge means the head is cut off, so "step back" beats any size verdict.
    feedback = evaluate_framing((800, 0, 1100, 900), W, H)
    check("cropped is detected", feedback.status == STATUS_CROPPED, feedback.status)
    check("cropped is not ok", not feedback.ok)


def test_off_centre_direction() -> None:
    left = evaluate_framing(centred(0.72, offset=-0.25), W, H)
    check("off centre rejected", left.status == STATUS_OFF_CENTRE, left.status)
    check("subject left of centre moves right", "right" in left.message.lower(), left.message)

    right = evaluate_framing(centred(0.72, offset=0.25), W, H)
    check("subject right of centre moves left", "left" in right.message.lower(), right.message)


def test_small_offset_is_tolerated() -> None:
    feedback = evaluate_framing(centred(0.72, offset=0.05), W, H)
    check("minor offset still ok", feedback.ok, feedback.message)


def test_size_checked_before_centring() -> None:
    """A guest who is both too far and off centre should be told the more useful thing first."""
    feedback = evaluate_framing(centred(0.30, offset=0.3), W, H)
    check("distance beats centring", feedback.status == STATUS_TOO_FAR, feedback.status)


def test_target_box_is_sane() -> None:
    box = target_box(W, H)
    check("box inside frame", 0 <= box.x1 < box.x2 <= W and 0 <= box.y1 < box.y2 <= H, str(box))
    check("box is portrait", (box.y2 - box.y1) > (box.x2 - box.x1))
    check("box is centred", abs(((box.x1 + box.x2) / 2) - W / 2) < 2)
    check("box sits near the floor", box.y2 > H * 0.9)

    feedback = evaluate_framing(box.as_tuple, W, H)
    check("the guide itself passes", feedback.ok, feedback.message)


def test_degenerate_frame_size() -> None:
    feedback = evaluate_framing((0, 0, 10, 10), 0, 0)
    check("zero frame does not divide by zero", not feedback.ok)


def test_missing_lower_body_is_rejected() -> None:
    """A waist-up shot can pass every box check yet ruin the composite, so keypoints decide."""
    box = centred(0.72)
    check("passes without keypoint info", evaluate_framing(box, W, H, None).ok)
    check("passes when feet are seen", evaluate_framing(box, W, H, True).ok)

    feedback = evaluate_framing(box, W, H, False)
    check("rejected when feet are missing", not feedback.ok)
    check("status names the cause", feedback.status == STATUS_NO_LOWER_BODY, feedback.status)
    check("asks for feet in frame", "feet" in feedback.message.lower(), feedback.message)


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        print(f"\n{test.__name__}")
        test()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
