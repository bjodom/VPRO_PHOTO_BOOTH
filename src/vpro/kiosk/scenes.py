"""Destination catalogue for the kiosk.

Each scene supplies the compositing background and the prompt fragment used to restyle the
composed portrait.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

#: Pre-generated landmark backgrounds, one directory per scene key.
BACKGROUND_ROOT = Path("assets/scenes")

PHOTOREAL_STYLE = (
    "photorealistic travel photograph, shot on Canon EOS R5, 35mm lens, f/8, "
    "natural golden hour light, ultra detailed textures, high dynamic range, "
    "sharp focus, realistic colors, 8k"
)

DEFAULT_NEGATIVE_PROMPT = (
    "illustration, painting, drawing, cartoon, anime, 3d render, cgi, video game, "
    "blurry, lowres, jpeg artifacts, oversaturated, overexposed, distorted perspective, "
    "warped architecture, watermark, text, signature, logo, deformed, extra limbs"
)


@dataclass(frozen=True)
class Scene:
    key: str
    label: str
    description: str

    def prompt(self) -> str:
        return f"a person holding a laptop in front of {self.description}, {PHOTOREAL_STYLE}"

    def background_prompt(self) -> str:
        """Empty-scene prompt: the guest is composited in afterwards, not generated."""
        return f"{self.description}, no people, empty scene, {PHOTOREAL_STYLE}"

    def background_dir(self, root: Path = BACKGROUND_ROOT) -> Path:
        return root / self.key

    def backgrounds(self, root: Path = BACKGROUND_ROOT) -> list[Path]:
        directory = self.background_dir(root)
        if not directory.is_dir():
            return []
        return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".png"})

    def pick_background(self, root: Path = BACKGROUND_ROOT) -> Path | None:
        """Random variant so consecutive guests at the same destination differ."""
        options = self.backgrounds(root)
        return random.choice(options) if options else None


SCENES: tuple[Scene, ...] = (
    Scene("pyramids", "Giza", "the Great Pyramids of Giza rising over the desert, clear sky"),
    Scene("eiffel", "Paris", "the Eiffel Tower seen from the Trocadero, soft morning haze"),
    Scene("colosseum", "Rome", "the Roman Colosseum at sunrise, weathered stone arches"),
    Scene("tajmahal", "Agra", "the Taj Mahal reflected in its water pool, white marble at dawn"),
    Scene("machu", "Machu Picchu", "Machu Picchu terraces with drifting clouds in the valley"),
    Scene("santorini", "Santorini", "a white-washed Santorini cliffside above the Aegean at sunset"),
    Scene("fuji", "Mount Fuji", "Mount Fuji behind Chureito Pagoda, cherry blossoms in bloom"),
    Scene("goldengate", "San Francisco", "the Golden Gate Bridge emerging from low fog"),
)

SCENES_BY_KEY = {scene.key: scene for scene in SCENES}


def get_scene(key: str) -> Scene:
    try:
        return SCENES_BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown scene {key!r}; expected one of {sorted(SCENES_BY_KEY)}") from None
