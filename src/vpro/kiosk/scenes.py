"""Destination catalogue for the kiosk.

Scenes supply background-generation prompts, not stock backgrounds for guest portraits.
"""

from __future__ import annotations

from dataclasses import dataclass
import unicodedata

MAX_LOCATION_LENGTH = 100


def normalize_location(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Enter a place name.")
    value = unicodedata.normalize("NFC", value)
    if any(unicodedata.category(character).startswith("C") for character in value):
        raise ValueError("Enter a single-line place name.")
    location = " ".join(value.split())
    if not 2 <= len(location) <= MAX_LOCATION_LENGTH or not any(character.isalpha() for character in location):
        raise ValueError(f"Enter a place name between 2 and {MAX_LOCATION_LENGTH} characters.")
    if any(not (character.isalnum() or character in " .,'\u2019-&()/") for character in location):
        raise ValueError("Use a place name, such as Kyoto, Japan.")
    return location

PHOTOREAL_STYLE = (
    "photorealistic travel photograph, shot on Canon EOS R5, 35mm lens, f/8, "
    "natural golden hour light, ultra detailed textures, high dynamic range, "
    "sharp focus, realistic colors, 8k"
)

DEFAULT_NEGATIVE_PROMPT = (
    "illustration, cartoon, CGI, blurry, low resolution, distorted perspective, "
    "warped architecture, watermark, text, logo, deformed, extra limbs"
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
    subject_scale: float = 0.78
    subject_center_x: float = 0.50
    feet_y: float = 0.92
    scene_crop_center_x: float = 0.50

    def prompt(self) -> str:
        return f"a person in front of {self.description}, {PHOTOREAL_STYLE}"

    def background_prompt(self) -> str:
        """Scene without people: the guest is preserved by the mask, so describing a person here
        makes the model paint a second one into the background."""
        return (
            f"{self.description}, recognizable landmark, photorealistic photo, vertical portrait, "
            "eye-level ground plane, realistic scale, sharp detail"
        )

    def background_negative_prompt(self) -> str:
        if self.key == "custom":
            return f"{DEFAULT_NEGATIVE_PROMPT}, another person, second person, people, crowd, faces, tourists, mannequin"
        return BACKGROUND_NEGATIVE_PROMPT


SCENES: tuple[Scene, ...] = (
    Scene("pyramids", "Giza", "the three Great Pyramids of Giza, limestone blocks, sandy foreground, distant desert plateau, clear blue sky", 0.70, 0.50, 0.94),
    Scene("eiffel", "Paris", "the Eiffel Tower viewed from the Trocadero gardens, broad Parisian avenue, pale limestone buildings, soft morning haze", 0.72, 0.50, 0.94),
    Scene("colosseum", "Rome", "Ancient Rome, recognizable Roman Colosseum, elliptical amphitheater facade, three tiers of travertine arches, Roman columns, broad stone piazza, warm side light", 0.70, 0.62, 0.94, 0.44),
    Scene("tajmahal", "Agra", "the Taj Mahal with white marble dome and four minarets, long reflecting pool, symmetrical Mughal gardens, gentle dawn light", 0.66, 0.35, 0.94, 0.50),
    Scene("machu", "Machu Picchu", "Machu Picchu stone terraces, ancient Inca walls, Huayna Picchu rising behind, layered green Andes, drifting valley clouds", 0.62, 0.32, 0.93, 0.52),
    Scene("santorini", "Santorini", "a Santorini cliffside village, white stucco walls, blue domes, winding stone path, Aegean Sea horizon, warm sunset light", 0.68, 0.36, 0.92, 0.56),
    Scene("fuji", "Mount Fuji", "Mount Fuji behind Chureito Pagoda, red pagoda roof, layered green foothills, cherry blossoms, crisp clear spring morning", 0.66, 0.32, 0.93, 0.50),
    Scene("goldengate", "San Francisco", "the Golden Gate Bridge spanning the bay, red suspension towers, calm water, San Francisco hills, low coastal fog, warm afternoon light", 0.66, 0.30, 0.90, 0.40),
)

SCENES_BY_KEY = {scene.key: scene for scene in SCENES}


def get_scene(key: str, custom_location: str | None = None) -> Scene:
    if key == "custom":
        location = normalize_location(custom_location or "")
        return Scene("custom", location, f"a recognizable view of {location}")
    try:
        return SCENES_BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown scene {key!r}; expected one of {sorted(SCENES_BY_KEY)}") from None
