from abc import ABC, abstractmethod
from pathlib import Path


class InferenceBackend(ABC):
    """Backend interface for loading a model and running inference."""

    @abstractmethod
    def load_model(self, model_path: Path) -> None:
        """Load model artifacts from disk."""

    @abstractmethod
    def predict(self, input_data: object) -> object:
        """Run inference and return backend-specific output."""
