"""Request/result types exchanged with the Juggernaut render worker.

Kept flat and primitive so a wire format can be added later without reshaping call sites
(see docs/optimization.md section 9.1).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

TASK_TEXT2IMG = "text2img"
TASK_IMG2IMG = "img2img"

# Bounds exist so a bad UI value cannot wedge the booth on a multi-minute render.
MAX_STEPS = 150
MAX_DIMENSION = 4096
MIN_DIMENSION = 64
MAX_PROMPT_CHARS = 4000


@dataclass(frozen=True)
class RenderRequest:
    mode: str
    prompt: str
    output_path: Path
    negative_prompt: str | None = None
    steps: int = 24
    guidance_scale: float = 4.5
    width: int = 1080
    height: int = 1350
    seed: int | None = None
    strength: float | None = None
    input_image_path: Path | None = None

    def validated(self) -> RenderRequest:
        """Return a normalized copy, raising ValueError on anything out of bounds."""
        if self.mode not in (TASK_TEXT2IMG, TASK_IMG2IMG):
            raise ValueError(f"mode must be '{TASK_TEXT2IMG}' or '{TASK_IMG2IMG}', got {self.mode!r}")
        if not self.prompt or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if len(self.prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"prompt exceeds {MAX_PROMPT_CHARS} characters")
        if self.negative_prompt is not None and len(self.negative_prompt) > MAX_PROMPT_CHARS:
            raise ValueError(f"negative_prompt exceeds {MAX_PROMPT_CHARS} characters")
        if not 1 <= int(self.steps) <= MAX_STEPS:
            raise ValueError(f"steps must be within 1..{MAX_STEPS}, got {self.steps}")
        for name, value in (("width", self.width), ("height", self.height)):
            if not MIN_DIMENSION <= int(value) <= MAX_DIMENSION:
                raise ValueError(
                    f"{name} must be within {MIN_DIMENSION}..{MAX_DIMENSION}, got {value}"
                )
        if self.mode == TASK_IMG2IMG:
            if self.input_image_path is None:
                raise ValueError("img2img requires input_image_path")
            if self.strength is None:
                raise ValueError("img2img requires strength")
            if not 0.0 <= float(self.strength) <= 1.0:
                raise ValueError(f"strength must be within 0.0..1.0, got {self.strength}")

        return replace(
            self,
            steps=int(self.steps),
            width=int(self.width),
            height=int(self.height),
            guidance_scale=float(self.guidance_scale),
            seed=None if self.seed is None else int(self.seed),
            strength=None if self.strength is None else float(self.strength),
        )


@dataclass(frozen=True)
class RenderResult:
    ok: bool
    output_path: Path | None
    render_seconds: float
    queue_wait_seconds: float
    error: str | None = None
