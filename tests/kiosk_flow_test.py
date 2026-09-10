"""End-to-end guest flow against a running kiosk.

Unlike the other kiosk tests this drives the real service: real camera, real compose, real render
and real delivery. Start the kiosk first, then run this against it.

    # terminal 1
    $env:PYTHONPATH="src"; python -m vpro.cli --kiosk --kiosk-port 8010 --delivery-port 8767
    # terminal 2
    python tests/kiosk_flow_test.py --base-url http://127.0.0.1:8010

Add --runs 2 to check that a second guest is faster than the first, which is the whole point of
keeping the pipeline resident.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
import numpy as np
from time import perf_counter, sleep

TERMINAL_STATES = {"ready", "generation_error", "delivery_error", "idle"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--scene", default="fuji")
    parser.add_argument("--runs", type=int, default=5, help="Number of guest sessions to simulate.")
    parser.add_argument("--p95-budget", type=float, default=60.0, help="Warm accept-to-QR p95 budget in seconds.")
    parser.add_argument("--timeout", type=float, default=240.0, help="Seconds to wait per render.")
    return parser


class Kiosk:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def call(self, path: str, payload: dict | None = None, method: str = "GET") -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read())

    def state(self) -> dict:
        return self.call("/api/state")

    def act(self, action: str, payload: dict | None = None) -> dict:
        return self.call(f"/api/action/{action}", payload or {}, "POST")


def run_session(kiosk: Kiosk, scene: str, timeout: float) -> tuple[bool, float, dict]:
    kiosk.act("reset")
    kiosk.act("start")
    kiosk.act("accept_consent")
    state = kiosk.act("choose_scene", {"scene": scene})
    assert state["state"] == "pose", state["state"]

    sleep(1.5)  # let the preview produce a frame to capture
    state = kiosk.act("capture")
    assert state["state"] == "review", state["state"]

    started = perf_counter()
    state = kiosk.act("accept_capture")
    assert state["state"] == "generating", state["state"]

    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        sleep(0.5)
        state = kiosk.state()
        if state["state"] in TERMINAL_STATES:
            break

    elapsed = perf_counter() - started
    generated = "render_seconds" in state.get("status", {}).get("pipeline_timings", {})
    return state["state"] == "ready" and generated, elapsed, state


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.runs < 2:
        raise SystemExit("Use at least two runs to distinguish first-guest and warm latency.")
    kiosk = Kiosk(args.base_url)

    try:
        initial = kiosk.state()
    except urllib.error.URLError as exc:
        raise SystemExit(f"No kiosk at {args.base_url}: {exc}. Start it with --kiosk first.")

    status = initial["status"]
    if not initial.get("generation_enabled", False):
        raise SystemExit("Acceptance requires generation enabled, not a deterministic fixture.")
    print(f"camera open : {status['camera_open']} at {status['camera_fps']} fps")
    print(f"renderer    : {status['renderer_state']} (startup {status['startup_seconds']})")

    failures = 0
    durations = []
    for index in range(1, args.runs + 1):
        ok, elapsed, state = run_session(kiosk, args.scene, args.timeout)
        durations.append(elapsed)
        label = "ok " if ok else "FAIL"
        print(f"\n[{label}] guest {index}: {elapsed:.2f}s -> {state['state']}")
        if ok:
            print(f"       url: {state['delivery_url']}")
            print(f"       qr : {'present' if state['qr_svg'] else 'MISSING'}")
            if not state["qr_svg"]:
                failures += 1
            try:
                with urllib.request.urlopen(state["delivery_url"], timeout=15) as response:
                    if response.status != 200:
                        raise RuntimeError("Handoff page failed")
                image_url = state["delivery_url"].replace("/i/", "/f/", 1)
                with urllib.request.urlopen(image_url, timeout=15) as response:
                    image = response.read()
                if not image.startswith(b"\xff\xd8") or len(image) < 1000:
                    raise RuntimeError("Downloaded file is not a complete JPEG")
            except Exception as exc:
                failures += 1
                print(f"       delivery verification failed: {exc}")
        else:
            print(f"       error: {state['error']}")
            failures += 1
        kiosk.act("reset")

    print(f"\n{args.runs - failures}/{args.runs} sessions completed.")
    warm = durations[1:]
    print(f"First guest: {durations[0]:.2f}s; warm p50={np.percentile(warm, 50):.2f}s p95={np.percentile(warm, 95):.2f}s")
    if np.percentile(warm, 95) > args.p95_budget:
        print(f"FAIL: warm p95 exceeds {args.p95_budget:.1f}s acceptance budget")
        failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
