from __future__ import annotations

from pathlib import Path
from typing import Tuple


def export_onnx_model(
    model,
    output_path: Path,
    input_shape: Tuple[int, int, int],
    max_batch: int,
    opset_version: int = 18,
) -> Path:
    """Export one ONNX graph with a dynamic batch axis and validate it."""
    import torch

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError(
            "ONNX export requires the 'onnx' package. Install the project dependencies first."
        ) from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use a non-specialized example batch and allow batch=1 through the declared range.
    dynamic_max_batch = max(2, max_batch)
    example_batch = 2
    example = torch.randn((example_batch, *input_shape), dtype=torch.float32)
    model = model.cpu().eval()

    # The current PyTorch-recommended exporter is torch.onnx.export(..., dynamo=True).
    batch_dim = torch.export.Dim("batch", min=1, max=dynamic_max_batch)

    try:
        program = torch.onnx.export(
            model,
            (example,),
            input_names=["images"],
            output_names=["logits"],
            dynamic_shapes=({0: batch_dim},),
            opset_version=opset_version,
            dynamo=True,
        )
        if program is None or not hasattr(program, "save"):
            raise RuntimeError("Modern ONNX exporter did not return a saveable ONNXProgram")
        program.save(str(output_path))
    except Exception as exc:
        raise RuntimeError(
            "ONNX export failed. Ensure torch, onnx, and onnxscript are installed from "
            "the project requirements. The benchmark intentionally uses the modern "
            "dynamo=True exporter rather than silently falling back to legacy export."
        ) from exc

    model_proto = onnx.load(str(output_path))
    onnx.checker.check_model(model_proto)
    return output_path
