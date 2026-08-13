from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceConfig:
    """Runtime configuration for model loading and inference."""

    backend: str
    model_path: Path
