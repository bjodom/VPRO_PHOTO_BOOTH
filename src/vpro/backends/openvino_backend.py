from pathlib import Path

from .base import InferenceBackend
from ..vision import resolve_runtime_model


class OpenVINOBackend(InferenceBackend):
    """OpenVINO backend scaffold.
    """

    def __init__(self) -> None:
        self._runtime_model_path: Path | None = None
        self._model = None

    def load_model(self, model_path: Path) -> None:
        # Deferred import ensures runtime-only dependency loading.
        import openvino  # noqa: F401
        from ultralytics import YOLO

        self._runtime_model_path = resolve_runtime_model(str(model_path))
        self._model = YOLO(str(self._runtime_model_path))
        print(f"Loaded OpenVINO model from {self._runtime_model_path}")

    def predict(self, input_data: object) -> object:
        if self._model is None:
            raise RuntimeError("Model is not loaded. Call load_model first.")

        if not isinstance(input_data, dict):
            raise ValueError(
                "input_data must be a dict with keys like 'source', 'device', and optional 'verbose'."
            )

        source = input_data.get("source")
        if source is None:
            raise ValueError("input_data['source'] is required.")

        device = str(input_data.get("device", "intel:gpu"))
        verbose = bool(input_data.get("verbose", False))
        return self._model.predict(source=source, device=device, verbose=verbose)
