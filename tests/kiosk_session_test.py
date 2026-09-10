"""Kiosk session state machine tests.

    python tests/kiosk_session_test.py

Pure logic; no camera, GPU or network. Uses a fake clock so timeouts are exact.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from vpro.kiosk import InvalidTransition, KioskSession, State  # noqa: E402

PASSED: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if not condition:
        raise AssertionError(f"{name} failed. {detail}")
    PASSED.append(name)
    print(f"  ok  {name}")


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def new_session(clock: FakeClock | None = None) -> tuple[KioskSession, FakeClock, list[tuple]]:
    clock = clock or FakeClock()
    transitions: list[tuple] = []
    session = KioskSession(
        clock=clock, on_transition=lambda a, b: transitions.append((a.value, b.value))
    )
    return session, clock, transitions


def run_to_generating(session: KioskSession) -> None:
    session.start()
    session.accept_consent()
    session.choose_scene("pyramids")
    session.capture(Path("/tmp/capture.jpg"))
    session.accept_capture()


def test_happy_path() -> None:
    session, _, transitions = new_session()
    check("starts idle", session.state is State.IDLE)

    run_to_generating(session)
    check("reaches generating", session.state is State.GENERATING, session.state.value)
    check("scene recorded", session.data.scene == "pyramids")
    check("capture recorded", session.data.capture_path == Path("/tmp/capture.jpg"))

    session.generation_succeeded(
        Path("/tmp/final.jpg"), delivery_url="http://x/i/tok", delivery_qr_svg="<svg/>"
    )
    check("reaches ready", session.state is State.READY)
    check("delivery url stored", session.data.delivery_url == "http://x/i/tok")

    session.finish()
    check("reaches done", session.state is State.DONE)
    check("transitions recorded", len(transitions) == 7, str(transitions))


def test_invalid_actions_are_rejected() -> None:
    session, _, _ = new_session()
    for label, action in (
        ("consent before start", session.accept_consent),
        ("capture before pose", lambda: session.capture(Path("/tmp/x.jpg"))),
        ("finish before ready", session.finish),
        ("retry when not in error", session.retry),
    ):
        rejected = False
        try:
            action()
        except InvalidTransition:
            rejected = True
        check(f"rejects {label}", rejected)
    check("state unchanged after rejections", session.state is State.IDLE)


def test_empty_scene_rejected() -> None:
    session, _, _ = new_session()
    session.start()
    session.accept_consent()
    rejected = False
    try:
        session.choose_scene("   ")
    except ValueError:
        rejected = True
    check("blank scene rejected", rejected)
    check("still on select screen", session.state is State.SELECT_SCENE)


def test_timeout_resets_session() -> None:
    session, clock, _ = new_session()
    session.start()
    check("consent has a timeout", session.timeout_seconds == 60.0)

    clock.advance(30)
    check("not expired early", not session.is_expired())
    check("remaining counts down", abs((session.seconds_remaining or 0) - 30.0) < 1e-6)

    clock.advance(31)
    check("expired after limit", session.is_expired())
    check("tick resets to idle", session.tick() is State.IDLE)


def test_idle_never_times_out() -> None:
    session, clock, _ = new_session()
    clock.advance(10_000)
    check("idle has no timeout", session.timeout_seconds is None)
    check("idle does not expire", not session.is_expired())
    check("tick keeps idle", session.tick() is State.IDLE)


def test_timeout_during_generation_is_survivable() -> None:
    """A guest may walk away mid-render; the session resets but the callback must not explode."""
    session, clock, _ = new_session()
    run_to_generating(session)
    clock.advance(200)
    check("generation timed out", session.tick() is State.IDLE)

    late = False
    try:
        session.generation_succeeded(Path("/tmp/final.jpg"))
    except InvalidTransition:
        late = True
    check("late success is rejected, not applied", late)
    check("session stays idle", session.state is State.IDLE)


def test_reset_clears_guest_data() -> None:
    session, _, _ = new_session()
    run_to_generating(session)
    session.generation_succeeded(Path("/tmp/final.jpg"), delivery_url="http://x/i/tok")
    first_id = session.session_id

    session.reset()
    check("scene cleared", session.data.scene is None)
    check("capture cleared", session.data.capture_path is None)
    check("final image cleared", session.data.final_path is None)
    check("delivery url cleared", session.data.delivery_url is None)
    check("session id advanced", session.session_id == first_id + 1)


def test_reset_is_safe_from_any_state() -> None:
    for reach in (
        lambda s: None,
        lambda s: s.start(),
        lambda s: run_to_generating(s),
        lambda s: (run_to_generating(s), s.generation_failed("boom")),
    ):
        session, _, _ = new_session()
        reach(session)
        session.reset()
        check(f"reset works from {session.state.value}", session.state is State.IDLE)


def test_generation_error_retry_and_rescene() -> None:
    session, _, _ = new_session()
    run_to_generating(session)
    session.generation_failed("black frame on all attempts")
    check("enters generation error", session.state is State.GENERATION_ERROR)
    check("error recorded", "black frame" in (session.data.error or ""))

    session.retry()
    check("retry returns to generating", session.state is State.GENERATING)
    check("error cleared on retry", session.data.error is None)

    session.generation_failed("again")
    session.choose_another_scene()
    check("can pick another scene", session.state is State.SELECT_SCENE)
    check("scene cleared", session.data.scene is None)


def test_delivery_error_retry_keeps_the_image() -> None:
    session, _, _ = new_session()
    run_to_generating(session)
    session.generation_succeeded(Path("/tmp/final.jpg"))
    session.delivery_failed("handoff server refused")
    check("enters delivery error", session.state is State.DELIVERY_ERROR)

    session.retry()
    check("retry schedules handoff", session.state is State.GENERATING, session.state.value)
    check("image is preserved", session.data.final_path == Path("/tmp/final.jpg"))


def test_delivery_error_before_image_returns_to_generating() -> None:
    session, _, _ = new_session()
    run_to_generating(session)
    session.delivery_failed("no image yet")
    session.retry()
    check("re-renders when no image exists", session.state is State.GENERATING)


def test_camera_error_paths() -> None:
    session, _, _ = new_session()
    session.start()
    session.accept_consent()
    session.choose_scene("fuji")
    session.camera_failed("camera unavailable")
    check("enters camera error", session.state is State.CAMERA_ERROR)

    session.retry()
    check("retry returns to pose", session.state is State.POSE)

    session.camera_failed("again")
    session.retake()
    check("retake from camera error returns to pose", session.state is State.POSE)


def test_retake_clears_capture() -> None:
    session, _, _ = new_session()
    session.start()
    session.accept_consent()
    session.choose_scene("eiffel")
    session.capture(Path("/tmp/first.jpg"))
    session.retake()
    check("returns to pose", session.state is State.POSE)
    check("capture discarded", session.data.capture_path is None)


def main() -> int:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        print(f"\n{test.__name__}")
        test()
    print(f"\n{len(PASSED)} checks passed across {len(tests)} tests.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
