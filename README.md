# PyTorch → ONNX → ONNX Runtime Inference Benchmark

A small, reproducible CLI for exporting a torchvision ResNet18 to ONNX and comparing steady-state inference latency between native PyTorch and ONNX Runtime across batch sizes 1, 8, and 32.

The project is deliberately tooling-shaped rather than notebook-shaped: one dynamic-batch ONNX graph, repeatable warmup/measurement loops, backend thread controls, output-parity checks, and machine-readable reports.

## What it demonstrates

- PyTorch model loading and inference (`torch.inference_mode`)
- modern `torch.onnx.export(..., dynamo=True)` export
- dynamic ONNX batch dimension
- ONNX model validation with `onnx.checker`
- ONNX Runtime `InferenceSession` on `CPUExecutionProvider`
- fairer CPU benchmarking via matched intra-op thread counts
- latency distribution statistics and throughput
- numerical parity checks between PyTorch and ONNX Runtime
- optional `torch.compile` backend for a stretch comparison

## Repository layout

```text
.
├── benchmark.py                 # requested top-level CLI entry point
├── src/inference_bench/
│   ├── backends.py              # eager / torch.compile / ORT runners
│   ├── cli.py                   # argument parsing + orchestration
│   ├── export.py                # dynamic-batch ONNX export + checker
│   ├── metrics.py               # P50/P95/throughput statistics
│   ├── models.py                # ResNet18 + deterministic inputs
│   ├── reporting.py             # JSON/CSV/Markdown output
│   └── system_info.py           # runtime/environment metadata
├── tests/
├── artifacts/                   # generated ONNX + result files (gitignored)
└── RESULTS.md                   # canonical-results instructions
```

## Setup

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The default uses pretrained torchvision ResNet18 weights, so the first run may download the weights. If you only want an offline smoke test, use `--weights random`; weight values do not materially change the inference graph or latency, but the default project run remains pretrained.

## Run the comparison

```bash
python benchmark.py --backend all
```

Equivalent explicit run:

```bash
python benchmark.py \
  --backend all \
  --batch-sizes 1 8 32 \
  --warmup 10 \
  --iterations 50 \
  --threads 1
```

Run one backend exactly as a small deployment tool would:

```bash
python benchmark.py --backend onnxruntime
python benchmark.py --backend pytorch
```

By default, an ONNX run exports the currently loaded model before benchmarking so the graph cannot silently go stale. Reuse a previously exported graph only when you intend to:

```bash
python benchmark.py --backend onnxruntime --reuse-onnx
```

## Output

Each run writes three timestamped files to `artifacts/results/`:

```text
benchmark_YYYYMMDDTHHMMSSZ.json
benchmark_YYYYMMDDTHHMMSSZ.csv
benchmark_YYYYMMDDTHHMMSSZ.md
```

The report contains mean/P50/P95 latency, standard deviation, throughput, speedup versus eager PyTorch when both are measured, and output-parity checks.

## Benchmark methodology

For each batch size, input creation happens before timing. Warmup iterations are run and discarded. Only repeated inference calls are timed with `time.perf_counter_ns()`.

On CPU, `--threads` is applied to PyTorch intra-op threads and ONNX Runtime `intra_op_num_threads`; ORT inter-op threads are set to 1. This reduces a common benchmarking confound where one runtime silently uses more CPU parallelism than the other.

The tool intentionally excludes model construction, weight download, ONNX export, ONNX Runtime session creation, and warmup from latency. Those are setup costs, not steady-state inference latency.

## Numerical validation

Before reporting a backend result, the tool compares its logits against PyTorch eager output for the same input and records:

- maximum absolute error
- maximum relative error
- top-1 prediction agreement
- `torch.allclose` using configurable `--atol` and `--rtol`

## Optional `torch.compile` stretch

CPU, including eager + `torch.compile` + ONNX Runtime in one report:

```bash
python benchmark.py --backend all --include-compile --device cpu
```

Or benchmark only the compiled backend:

```bash
python benchmark.py --backend torchcompile --device cpu
```

CUDA, when a compatible PyTorch CUDA build and GPU are available:

```bash
python benchmark.py --backend pytorch --device cuda
python benchmark.py --backend torchcompile --device cuda
```

Compilation happens during warmup and is therefore excluded from steady-state timing. The core ONNX Runtime comparison is intentionally CPU-only so it does not mix host↔device transfer semantics with device-resident PyTorch tensors.

## Useful commands

```bash
make test
make smoke
make benchmark
make compile-benchmark
```

## Notes for interpreting results

A single latency number is not enough. Batch size can change operator efficiency, runtime graph optimizations, cache behavior, and throughput/latency tradeoffs. Report the exact CPU, thread count, package versions, warmup count, and measured iterations alongside the table; the generated Markdown does this automatically.
