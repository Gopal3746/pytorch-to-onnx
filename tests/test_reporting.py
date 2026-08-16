import pytest

from inference_bench.reporting import add_speedups, render_markdown


def test_add_speedups_uses_eager_baseline():
    rows = [
        {"backend": "pytorch", "batch_size": 1, "mean_ms": 10.0},
        {"backend": "onnxruntime", "batch_size": 1, "mean_ms": 5.0},
    ]
    out = add_speedups(rows)
    assert out[0]["speedup_vs_pytorch"] == pytest.approx(1.0)
    assert out[1]["speedup_vs_pytorch"] == pytest.approx(2.0)


def test_markdown_contains_table():
    md = render_markdown(
        {"model": "resnet18", "device": "cpu", "threads": 1, "warmup": 1, "iterations": 2},
        [{
            "backend": "pytorch",
            "batch_size": 1,
            "mean_ms": 1.0,
            "p50_ms": 1.0,
            "p95_ms": 1.1,
            "throughput_samples_s": 1000.0,
            "speedup_vs_pytorch": 1.0,
            "max_abs_error": 0.0,
            "top1_agreement": 1.0,
            "allclose": True,
        }],
    )
    assert "| Backend | Batch |" in md
    assert "pytorch" in md
