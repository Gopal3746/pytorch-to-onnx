from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import Iterable, Sequence


@dataclass(frozen=True)
class LatencySummary:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    throughput_samples_s: float


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    weight = pos - lo
    return ordered[lo] * (1.0 - weight) + ordered[hi] * weight


def summarize_latencies(latencies_ms: Iterable[float], batch_size: int) -> LatencySummary:
    values = [float(v) for v in latencies_ms]
    if not values:
        raise ValueError("latencies_ms must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    avg = mean(values)
    return LatencySummary(
        mean_ms=avg,
        p50_ms=percentile(values, 0.50),
        p95_ms=percentile(values, 0.95),
        std_ms=pstdev(values),
        min_ms=min(values),
        max_ms=max(values),
        throughput_samples_s=(batch_size * 1000.0 / avg),
    )
