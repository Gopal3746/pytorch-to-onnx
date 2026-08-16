from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple


def configure_torch_threads(threads: int) -> None:
    import torch

    if threads <= 0:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    # set_num_interop_threads can only be called before parallel work starts.
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _measure(
    fn: Callable[[], object],
    warmup: int,
    iterations: int,
    synchronize: Optional[Callable[[], None]] = None,
) -> Tuple[List[float], object]:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be >= 0 and iterations must be > 0")

    last_output = None
    for _ in range(warmup):
        last_output = fn()
    if synchronize:
        synchronize()

    latencies_ms: List[float] = []
    for _ in range(iterations):
        if synchronize:
            synchronize()
        start_ns = time.perf_counter_ns()
        last_output = fn()
        if synchronize:
            synchronize()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
        latencies_ms.append(elapsed_ms)
    return latencies_ms, last_output


def benchmark_pytorch(model, x, warmup: int, iterations: int, compiled: bool = False):
    import torch

    runner = model
    if compiled:
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is not available in this PyTorch build")
        runner = torch.compile(model)

    sync = torch.cuda.synchronize if x.device.type == "cuda" else None

    def infer():
        with torch.inference_mode():
            return runner(x)

    return _measure(infer, warmup, iterations, sync)


def make_ort_session(onnx_path: Path, threads: int):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "ONNX Runtime backend requires 'onnxruntime'. Install the project dependencies first."
        ) from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(onnx_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def benchmark_onnxruntime(session, x_cpu, warmup: int, iterations: int):
    if x_cpu.device.type != "cpu":
        raise ValueError("The core ONNX Runtime comparison is CPU-only")

    input_name = session.get_inputs()[0].name
    # .numpy() shares the CPU tensor's storage; conversion itself is outside the timed loop.
    feed = {input_name: x_cpu.contiguous().numpy()}

    def infer():
        return session.run(None, feed)[0]

    return _measure(infer, warmup, iterations)
