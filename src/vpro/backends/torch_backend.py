from pathlib import Path

from .base import InferenceBackend


class TorchBackend(InferenceBackend):
    """PyTorch backend scaffold.

    Install dependencies with: `uv sync --extra torch`
    """

    def load_model(self, model_path: Path) -> None:
        # Deferred import keeps base installs lightweight.
        import torch  # noqa: F401

        raise NotImplementedError(
            f"Torch model loading is not implemented yet: {model_path}"
        )

    def predict(self, input_data: object) -> object:
        raise NotImplementedError("Torch inference is not implemented yet")
