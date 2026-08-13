from .base import InferenceBackend
from .openvino_backend import OpenVINOBackend
from .torch_backend import TorchBackend


def build_backend(name: str) -> InferenceBackend:
    normalized = name.strip().lower()
    if normalized == "openvino":
        return OpenVINOBackend()
    if normalized == "torch":
        return TorchBackend()
    raise ValueError(
        "Unsupported backend. Expected one of: 'openvino', 'torch'."
    )
