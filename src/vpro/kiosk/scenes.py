"""Destination catalogue for the kiosk.

Each scene supplies the compositing background and the prompt fragment used to restyle the
composed portrait.
"""

from __future__ import annotations

from dataclasses import dataclass

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

#: The guest is masked out, so any mention of people gets painted into the background.
BACKGROUND_NEGATIVE_PROMPT = (
    f"{DEFAULT_NEGATIVE_PROMPT}, another person, second person, people, crowd, faces, "
    "tourists, mannequin, indoor, table, desk, furniture, railing"
)


@dataclass(frozen=True)
class Scene:
    key: str
    label: str
    description: str

    def prompt(self) -> str:
        return f"a person holding a laptop in front of {self.description}, {PHOTOREAL_STYLE}"

    def background_prompt(self) -> str:
        """Scene without people: the guest is preserved by the mask, so describing a person here
        makes the model paint a second one into the background."""
        return f"{self.description}, no people, empty scene, {PHOTOREAL_STYLE}"


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
