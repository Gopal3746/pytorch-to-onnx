from __future__ import annotations

import os
import platform
import sys
from typing import Dict


def collect_system_info() -> Dict[str, object]:
    import torch

    info: Dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "torch": torch.__version__,
        "torchvision": None,
        "onnx": None,
        "onnxruntime": None,
    }
    try:
        import torchvision

        info["torchvision"] = torchvision.__version__
    except ImportError:
        pass
    try:
        import onnx

        info["onnx"] = onnx.__version__
    except ImportError:
        pass
    try:
        import onnxruntime

        info["onnxruntime"] = onnxruntime.__version__
        info["ort_available_providers"] = onnxruntime.get_available_providers()
    except ImportError:
        pass
    return info
