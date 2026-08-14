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
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/state")
    def state() -> JSONResponse:
        return JSONResponse(service.snapshot())

    @app.post("/api/action/{action}")
    async def act(action: str, request: Request) -> JSONResponse:
        try:
            payload: dict[str, Any] = await request.json()
        except Exception:
            payload = {}
        try:
            return JSONResponse(service.act(action, payload))
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

    return app


def run(service: KioskService, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(service), host=host, port=port, log_level="warning")
