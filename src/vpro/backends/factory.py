from .base import InferenceBackend
from .openvino_backend import OpenVINOBackend


def build_backend(name: str) -> InferenceBackend:
    normalized = name.strip().lower()
    if normalized == "openvino":
        return OpenVINOBackend()
    raise ValueError("Unsupported backend. Expected: 'openvino'.")
