# Results

Benchmark values are intentionally not pre-filled because they are hardware-, thread-, runtime-, and build-dependent.

Run:

```bash
python benchmark.py --backend all --batch-sizes 1 8 32 --warmup 10 --iterations 50 --threads 1
```

The CLI writes timestamped JSON, CSV, and Markdown reports under `artifacts/results/`. Copy the Markdown table from your run here if you want one canonical result snapshot in the repository.

## What the benchmark measures

- steady-state inference latency after warmup
- mean, P50, P95, standard deviation, min, and max latency
- throughput in samples/second
- speedup relative to PyTorch eager when eager is included in the same run
- output parity: max absolute/relative error, top-1 agreement, and `allclose`

Model loading, pretrained-weight download, ONNX export, ONNX Runtime session creation, and warmup are excluded from timed inference.
