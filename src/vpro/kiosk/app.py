"""FastAPI app exposing the kiosk session to a local browser.

Binds to localhost by default: the UI is for the kiosk's own screen. The guest-facing image
handoff is a separate server that binds the LAN, so only the finished image is ever exposed to
the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .service import KioskService
from .session import InvalidTransition

STATIC_DIR = Path(__file__).parent / "static"
MJPEG_BOUNDARY = "vproframe"


def create_app(service: KioskService) -> FastAPI:
    app = FastAPI(title="vPRO Photo Booth", docs_url=None, redoc_url=None)
    app.state.service = service

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(service.snapshot(), headers={"Cache-Control": "no-store"})

    @app.post("/api/action/{action}")
    async def act(action: str, request: Request) -> JSONResponse:
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            raise HTTPException(status_code=403, detail="Cross-origin actions are not allowed")
        try:
            payload: dict[str, Any] = await request.json()
        except Exception:
            payload = {}
        try:
            if not isinstance(payload, dict):
                raise ValueError("Action payload must be an object")
            return JSONResponse(await run_in_threadpool(service.act, action, payload))
        except InvalidTransition as exc:
            # The UI and the machine disagree; report it rather than silently ignoring.
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/preview.mjpg")
    def preview() -> StreamingResponse:
        camera = service.camera
        if camera is None:
            raise HTTPException(status_code=503, detail="camera not started")
        return StreamingResponse(
            camera.mjpeg_frames(MJPEG_BOUNDARY),
            media_type=f"multipart/x-mixed-replace; boundary={MJPEG_BOUNDARY}",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/api/final.jpg")
    def final_image() -> FileResponse:
        path = service.session.data.final_path
        if path is None or not Path(path).exists():
            raise HTTPException(status_code=404, detail="no image for this session")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/capture.jpg")
    def capture_image() -> FileResponse:
        path = service.session.data.capture_path
        if path is None or not Path(path).exists():
            raise HTTPException(status_code=404, detail="no capture for this session")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse({"ok": True})

    @app.get("/health/ready")
    def readiness() -> JSONResponse:
        status = service.snapshot()["status"]
        ready = bool(status["camera_open"] and status["camera_fps"] > 0
                     and (not service.config.enable_generation or status["renderer_state"] == "ready"))
        return JSONResponse({"ready": ready, "camera_open": status["camera_open"],
                             "renderer_state": status["renderer_state"]}, status_code=200 if ready else 503)

    return app


def run(service: KioskService, host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False) -> None:
    import uvicorn
    import threading
    import webbrowser

    server = uvicorn.Server(uvicorn.Config(create_app(service), host=host, port=port, log_level="warning"))
    stopped = threading.Event()
    def open_when_ready() -> None:
        while not stopped.wait(0.2):
            if server.started:
                browser_host = "127.0.0.1" if host == "0.0.0.0" else host
                webbrowser.open(f"http://{browser_host}:{port}")
                return
    if open_browser:
        threading.Thread(target=open_when_ready, name="vpro-browser", daemon=True).start()
    try:
        server.run()
    finally:
        stopped.set()
