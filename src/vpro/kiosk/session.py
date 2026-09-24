"""Kiosk session state machine.

Pure logic: no camera, no GPU, no HTTP. The web layer renders `state` and forwards guest actions;
long-running work is reported back through `begin_*` / `finish_*` calls. Keeping it free of I/O is
what makes the awkward cases - abandonment, timeout mid-render, retry after failure - testable
without hardware.

Screens follow docs/ui_wireframe.md. Phone entry is intentionally absent: delivery is by QR, so no
personal data is collected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Callable


class State(str, Enum):
    IDLE = "idle"
    CONSENT = "consent"
    SELECT_SCENE = "select_scene"
    POSE = "pose"
    REVIEW = "review"
    GENERATING = "generating"
    READY = "ready"
    DONE = "done"
    CAMERA_ERROR = "camera_error"
    GENERATION_ERROR = "generation_error"
    DELIVERY_ERROR = "delivery_error"


#: Seconds of guest inactivity before a session resets. Screens where the kiosk is working, or
#: where the guest is reading a QR code, get longer or no timeout.
DEFAULT_TIMEOUTS: dict[State, float | None] = {
    State.IDLE: None,
    State.CONSENT: 60.0,
    State.SELECT_SCENE: 90.0,
    State.POSE: 120.0,
    State.REVIEW: 60.0,
    State.GENERATING: 180.0,
    State.READY: 120.0,
    State.DONE: 10.0,
    State.CAMERA_ERROR: 60.0,
    State.GENERATION_ERROR: 60.0,
    State.DELIVERY_ERROR: 60.0,
}

TERMINAL_ERROR_STATES = frozenset(
    {State.CAMERA_ERROR, State.GENERATION_ERROR, State.DELIVERY_ERROR}
)


class InvalidTransition(RuntimeError):
    """Raised when an action does not apply to the current state."""


@dataclass
class SessionData:
    scene: str | None = None
    custom_location: str | None = None
    capture_path: Path | None = None
    composed_path: Path | None = None
    final_path: Path | None = None
    delivery_url: str | None = None
    delivery_qr_svg: str | None = None
    caption: str | None = None
    error: str | None = None
    degraded: bool = False


@dataclass
class KioskSession:
    """Drives the guest flow. One session at a time; `reset()` starts a new guest."""

    timeouts: dict[State, float | None] = field(default_factory=lambda: dict(DEFAULT_TIMEOUTS))
    clock: Callable[[], float] = monotonic
    on_transition: Callable[[State, State], None] | None = None

    state: State = State.IDLE
    data: SessionData = field(default_factory=SessionData)
    session_id: int = 0
    _entered_at: float = field(default_factory=monotonic)

    # -- queries -------------------------------------------------------------------

    @property
    def seconds_in_state(self) -> float:
        return self.clock() - self._entered_at

    @property
    def timeout_seconds(self) -> float | None:
        return self.timeouts.get(self.state)

    @property
    def seconds_remaining(self) -> float | None:
        limit = self.timeout_seconds
        if limit is None:
            return None
        return max(0.0, limit - self.seconds_in_state)

    def is_expired(self) -> bool:
        remaining = self.seconds_remaining
        return remaining is not None and remaining <= 0.0

    # -- guest actions -------------------------------------------------------------

    def start(self) -> State:
        self._require(State.IDLE)
        return self._go(State.CONSENT)

    def accept_consent(self) -> State:
        self._require(State.CONSENT)
        return self._go(State.SELECT_SCENE)

    def choose_scene(self, scene: str, custom_location: str | None = None) -> State:
        self._require(State.SELECT_SCENE)
        if not scene or not scene.strip():
            raise ValueError("scene must be a non-empty string")
        self.data.scene = scene
        self.data.custom_location = custom_location if scene == "custom" else None
        return self._go(State.POSE)

    def capture(self, capture_path: Path) -> State:
        self._require(State.POSE)
        self.data.capture_path = capture_path
        return self._go(State.REVIEW)

    def retake(self) -> State:
        self._require(State.REVIEW, State.CAMERA_ERROR)
        self.data.capture_path = None
        return self._go(State.POSE)

    def accept_capture(self) -> State:
        self._require(State.REVIEW)
        return self._go(State.GENERATING)

    def finish(self) -> State:
        """Guest is done with the QR screen."""
        self._require(State.READY)
        return self._go(State.DONE)

    # -- pipeline callbacks --------------------------------------------------------

    def generation_succeeded(
        self,
        final_path: Path,
        delivery_url: str | None = None,
        delivery_qr_svg: str | None = None,
        caption: str | None = None,
        degraded: bool = False,
    ) -> State:
        self._require(State.GENERATING)
        self.data.final_path = final_path
        self.data.delivery_url = delivery_url
        self.data.delivery_qr_svg = delivery_qr_svg
        self.data.caption = caption
        self.data.error = None
        self.data.degraded = degraded
        return self._go(State.READY)

    def generation_failed(self, error: str) -> State:
        self._require(State.GENERATING)
        self.data.error = error
        return self._go(State.GENERATION_ERROR)

    def delivery_failed(self, error: str) -> State:
        self._require(State.GENERATING, State.READY)
        self.data.error = error
        return self._go(State.DELIVERY_ERROR)

    def camera_failed(self, error: str) -> State:
        self._require(State.POSE, State.SELECT_SCENE, State.REVIEW)
        self.data.error = error
        return self._go(State.CAMERA_ERROR)

    # -- recovery ------------------------------------------------------------------

    def retry(self) -> State:
        """Return to the step that can be attempted again."""
        if self.state == State.GENERATION_ERROR:
            self.data.error = None
            return self._go(State.GENERATING)
        if self.state == State.DELIVERY_ERROR:
            self.data.error = None
            # The image already exists; only the handoff needs redoing.
            return self._go(State.GENERATING)
        if self.state == State.CAMERA_ERROR:
            self.data.error = None
            return self._go(State.POSE)
        raise InvalidTransition(f"retry() is not valid in {self.state.value}")

    def choose_another_scene(self) -> State:
        self._require(State.GENERATION_ERROR, State.REVIEW, State.POSE)
        self.data.scene = None
        self.data.custom_location = None
        return self._go(State.SELECT_SCENE)

    def reset(self, reason: str = "reset") -> State:
        """Abandon the session and return to attract. Safe from any state."""
        self.data = SessionData()
        self.session_id += 1
        return self._go(State.IDLE, reason=reason)

    def tick(self) -> State:
        """Advance time-driven transitions. Call from the UI loop."""
        if self.is_expired():
            return self.reset(reason="timeout")
        return self.state

    # -- internals -----------------------------------------------------------------

    def _require(self, *allowed: State) -> None:
        if self.state not in allowed:
            expected = ", ".join(s.value for s in allowed)
            raise InvalidTransition(
                f"action requires state in ({expected}), but session is {self.state.value}"
            )

    def _go(self, target: State, reason: str | None = None) -> State:
        previous = self.state
        self.state = target
        self._entered_at = self.clock()
        if self.on_transition is not None:
            self.on_transition(previous, target)
        return target
