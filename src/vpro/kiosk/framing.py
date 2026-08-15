"""Framing guidance for the pose screen.

The compositor scales the subject to 78% of the output height and plants the feet near the bottom,
so it needs a roughly full-body capture. A webcam close-up scaled that way produces a giant head,
which is the single biggest quality problem in the delivered image.

Pure geometry so it can be tested without a camera.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Fraction of frame height the subject should occupy. Below this they are too far for detail,
#: above it the head gets cropped and the composite looks like a video call.
MIN_HEIGHT_RATIO = 0.55
MAX_HEIGHT_RATIO = 0.88
IDEAL_HEIGHT_RATIO = 0.72

#: How far the subject centre may sit from the frame centre, as a fraction of width.
MAX_CENTRE_OFFSET = 0.12

#: A box touching the frame edge means the body continues outside it.
EDGE_MARGIN_PX = 4

STATUS_OK = "ok"
STATUS_NO_SUBJECT = "no_subject"
STATUS_TOO_CLOSE = "too_close"
STATUS_TOO_FAR = "too_far"
STATUS_OFF_CENTRE = "off_centre"
STATUS_CROPPED = "cropped"
STATUS_NO_LOWER_BODY = "no_lower_body"


@dataclass(frozen=True)
class TargetBox:
    """Where the subject should land, in pixels."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class FramingFeedback:
    status: str
    message: str
    ok: bool
    height_ratio: float = 0.0
    centre_offset: float = 0.0


def target_box(frame_width: int, frame_height: int) -> TargetBox:
    """Guide rectangle sized for the ideal standing distance."""
    height = IDEAL_HEIGHT_RATIO * frame_height
    # Roughly a standing adult's aspect; only a visual guide, not a constraint.
    width = height * 0.42
    centre_x = frame_width / 2.0
    bottom = frame_height * 0.97
    return TargetBox(
        x1=int(centre_x - width / 2),
        y1=int(bottom - height),
        x2=int(centre_x + width / 2),
        y2=int(bottom),
    )


def evaluate_framing(
    bbox: tuple[int, int, int, int] | None,
    frame_width: int,
    frame_height: int,
    has_lower_body: bool | None = None,
) -> FramingFeedback:
    """Judge a detected subject box against the target and say what to do about it.

    `has_lower_body` comes from ankle keypoints. A bounding box cannot tell a full-body shot with
    feet near the bottom edge from a waist-up shot cut off by the frame, but the compositor needs
    the difference: it scales the subject to 78% of output height and plants the feet.
    """
    if bbox is None or frame_width <= 0 or frame_height <= 0:
        return FramingFeedback(STATUS_NO_SUBJECT, "Step into view", False)

    x1, y1, x2, y2 = bbox
    box_height = max(0, y2 - y1)
    height_ratio = box_height / float(frame_height)
    centre_x = (x1 + x2) / 2.0
    centre_offset = (centre_x - frame_width / 2.0) / float(frame_width)

    # Check cropping before size: a cut-off body reads as "too big" but the fix is to step back.
    touches_top = y1 <= EDGE_MARGIN_PX
    touches_side = x1 <= EDGE_MARGIN_PX or x2 >= frame_width - EDGE_MARGIN_PX
    if touches_top or (touches_side and height_ratio > MAX_HEIGHT_RATIO):
        return FramingFeedback(
            STATUS_CROPPED, "Step back - you're cut off", False, height_ratio, centre_offset
        )

    if has_lower_body is False:
        return FramingFeedback(
            STATUS_NO_LOWER_BODY,
            "Step back - we need your feet in frame",
            False,
            height_ratio,
            centre_offset,
        )

    if height_ratio > MAX_HEIGHT_RATIO:
        return FramingFeedback(
            STATUS_TOO_CLOSE, "Step back", False, height_ratio, centre_offset
        )
    if height_ratio < MIN_HEIGHT_RATIO:
        return FramingFeedback(
            STATUS_TOO_FAR, "Step closer", False, height_ratio, centre_offset
        )
    if abs(centre_offset) > MAX_CENTRE_OFFSET:
        # The camera faces the guest, so their right appears on the image's left, exactly as when
        # facing another person. A subject left of centre is standing to their own right and must
        # move to their own left to centre up.
        direction = "left" if centre_offset < 0 else "right"
        return FramingFeedback(
            STATUS_OFF_CENTRE, f"Move {direction}", False, height_ratio, centre_offset
        )

    return FramingFeedback(STATUS_OK, "Perfect - hold still", True, height_ratio, centre_offset)
