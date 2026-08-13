from __future__ import annotations

from pathlib import Path


def is_openvino_export_dir(path: Path) -> bool:
    """Return True when a directory looks like a valid Ultralytics OpenVINO export."""
    return (
        path.is_dir()
        and any(path.glob("*.xml"))
        and any(path.glob("*.bin"))
        and (path.joinpath("metadata.yaml").exists() or any(path.glob("*.yaml")))
    )


def resolve_runtime_model(model_arg: str) -> Path:
    """Resolve runtime model path.

    Accepts either:
    - A pre-exported OpenVINO directory with xml/bin/yaml artifacts
    - A .pt checkpoint that is exported once to <stem>_openvino_model
    """
    model_path = Path(model_arg).expanduser().resolve()

    if not model_path.exists():
        raise ValueError(f"Model path does not exist: {model_path}")

    if model_path.is_dir():
        if not is_openvino_export_dir(model_path):
            raise ValueError(
                "OpenVINO model directory is missing expected .xml/.bin/.yaml files: "
                f"{model_path}"
            )
        return model_path

    if model_path.suffix.lower() != ".pt":
        raise ValueError(
            "Model must be a .pt checkpoint or an exported OpenVINO model directory."
        )

    exported_dir = model_path.parent / f"{model_path.stem}_openvino_model"
    if is_openvino_export_dir(exported_dir):
        return exported_dir

    # Deferred import keeps startup light unless export is needed.
    from ultralytics import YOLO

    print(f"Exporting {model_path.name} to OpenVINO format...")
    YOLO(str(model_path)).export(format="openvino")

    if not is_openvino_export_dir(exported_dir):
        raise RuntimeError(
            "Export completed without creating the expected OpenVINO directory with "
            f".xml/.bin/.yaml files: {exported_dir}"
        )

    return exported_dir
