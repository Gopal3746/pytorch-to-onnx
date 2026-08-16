from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

from .backends import (
    benchmark_onnxruntime,
    benchmark_pytorch,
    configure_torch_threads,
    make_ort_session,
)
from .export import export_onnx_model
from .metrics import summarize_latencies
from .models import get_model_spec, load_model, make_cpu_input
from .reporting import add_speedups, write_reports
from .system_info import collect_system_info


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark native PyTorch vs ONNX Runtime inference for a small vision model."
    )
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnxruntime", "torchcompile", "all"],
        default="all",
        help="Backend to benchmark. 'all' benchmarks PyTorch eager and ONNX Runtime.",
    )
    parser.add_argument("--model", choices=["resnet18"], default="resnet18")
    parser.add_argument(
        "--batch-sizes",
        nargs="+",
        type=positive_int,
        default=[1, 8, 32],
        help="Batch sizes to test (default: 1 8 32).",
    )
    parser.add_argument("--warmup", type=nonnegative_int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument(
        "--threads",
        type=positive_int,
        default=1,
        help="CPU intra-op threads for both PyTorch and ONNX Runtime (default: 1).",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--weights",
        choices=["pretrained", "random"],
        default="pretrained",
        help="Pretrained downloads torchvision weights on first use; random is useful for offline smoke tests.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--opset", type=positive_int, default=18)
    parser.add_argument(
        "--onnx-path",
        type=Path,
        default=Path("artifacts/resnet18_dynamic.onnx"),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("artifacts/results"),
    )
    parser.add_argument(
        "--reuse-onnx",
        action="store_true",
        help="Reuse an existing --onnx-path instead of exporting the currently loaded model.",
    )
    parser.add_argument(
        "--include-compile",
        action="store_true",
        help="When --backend all is used, also benchmark torch.compile.",
    )
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-3)
    return parser


def selected_backends(name: str, include_compile: bool = False) -> List[str]:
    if name == "all":
        backends = ["pytorch", "onnxruntime"]
        if include_compile:
            backends.insert(1, "torchcompile")
        return backends
    return [name]


def _validate(reference, candidate, atol: float, rtol: float) -> Dict[str, object]:
    import torch

    if not isinstance(candidate, torch.Tensor):
        candidate = torch.from_numpy(candidate)
    reference = reference.detach().cpu()
    candidate = candidate.detach().cpu()
    diff = (reference - candidate).abs()
    denom = reference.abs().clamp_min(1e-8)
    top1_ref = reference.argmax(dim=1)
    top1_candidate = candidate.argmax(dim=1)
    return {
        "max_abs_error": float(diff.max().item()),
        "max_rel_error": float((diff / denom).max().item()),
        "top1_agreement": float((top1_ref == top1_candidate).float().mean().item()),
        "allclose": bool(torch.allclose(reference, candidate, atol=atol, rtol=rtol)),
    }


def _row(backend: str, batch_size: int, summary, validation: Dict[str, object]) -> Dict[str, object]:
    return {
        "backend": backend,
        "batch_size": batch_size,
        "mean_ms": summary.mean_ms,
        "p50_ms": summary.p50_ms,
        "p95_ms": summary.p95_ms,
        "std_ms": summary.std_ms,
        "min_ms": summary.min_ms,
        "max_ms": summary.max_ms,
        "throughput_samples_s": summary.throughput_samples_s,
        "speedup_vs_pytorch": None,
        **validation,
    }


def run(args: argparse.Namespace) -> int:
    import torch

    backends = selected_backends(args.backend, args.include_compile)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    if args.device == "cuda" and "onnxruntime" in backends:
        raise RuntimeError(
            "This project's apples-to-apples ONNX Runtime comparison is CPU-only. "
            "Use --device cpu for ONNX Runtime, or benchmark --backend pytorch/torchcompile on CUDA."
        )

    configure_torch_threads(args.threads)
    spec = get_model_spec(args.model)
    model = load_model(args.model, pretrained=(args.weights == "pretrained"))

    if "onnxruntime" in backends:
        if not args.reuse_onnx or not args.onnx_path.exists():
            export_onnx_model(
                model,
                args.onnx_path,
                spec.input_shape,
                max_batch=max(args.batch_sizes),
                opset_version=args.opset,
            )
        ort_session = make_ort_session(args.onnx_path, args.threads)
    else:
        ort_session = None

    device = torch.device(args.device)
    model = model.to(device).eval()
    rows: List[Dict[str, object]] = []

    for batch_size in args.batch_sizes:
        x_cpu = make_cpu_input(batch_size, spec, args.seed)
        x_device = x_cpu.to(device)

        with torch.inference_mode():
            reference = model(x_device).detach().cpu()

        if "pytorch" in backends:
            latencies, output = benchmark_pytorch(
                model, x_device, args.warmup, args.iterations, compiled=False
            )
            summary = summarize_latencies(latencies, batch_size)
            validation = _validate(reference, output, args.atol, args.rtol)
            rows.append(_row("pytorch", batch_size, summary, validation))

        if "torchcompile" in backends:
            latencies, output = benchmark_pytorch(
                model, x_device, args.warmup, args.iterations, compiled=True
            )
            summary = summarize_latencies(latencies, batch_size)
            validation = _validate(reference, output, args.atol, args.rtol)
            rows.append(_row("torchcompile", batch_size, summary, validation))

        if "onnxruntime" in backends:
            assert ort_session is not None
            latencies, output = benchmark_onnxruntime(
                ort_session, x_cpu, args.warmup, args.iterations
            )
            summary = summarize_latencies(latencies, batch_size)
            validation = _validate(reference, output, args.atol, args.rtol)
            rows.append(_row("onnxruntime", batch_size, summary, validation))

    rows = add_speedups(rows)
    metadata = collect_system_info()
    metadata.update(
        {
            "model": args.model,
            "weights": args.weights,
            "device": args.device,
            "threads": args.threads,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "batch_sizes": args.batch_sizes,
            "opset": args.opset,
            "onnx_path": str(args.onnx_path) if "onnxruntime" in backends else None,
            "timing_scope": "steady-state inference; setup/export/warmup excluded",
        }
    )
    paths = write_reports(args.results_dir, metadata, rows)

    print(json.dumps(rows, indent=2))
    print("\nReports:")
    for kind, path in paths.items():
        print(f"  {kind}: {path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    return 2
