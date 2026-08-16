import pytest

from inference_bench.metrics import percentile, summarize_latencies


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


def test_summary_computes_throughput():
    summary = summarize_latencies([10.0, 10.0, 10.0], batch_size=8)
    assert summary.mean_ms == pytest.approx(10.0)
    assert summary.throughput_samples_s == pytest.approx(800.0)
