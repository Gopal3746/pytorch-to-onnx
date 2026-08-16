from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List


CSV_FIELDS = [
    "backend",
    "batch_size",
    "mean_ms",
    "p50_ms",
    "p95_ms",
    "std_ms",
    "min_ms",
    "max_ms",
    "throughput_samples_s",
    "speedup_vs_pytorch",
    "max_abs_error",
    "max_rel_error",
    "top1_agreement",
    "allclose",
]


def add_speedups(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baselines = {
        int(row["batch_size"]): float(row["mean_ms"])
        for row in rows
        if row["backend"] == "pytorch"
    }
    for row in rows:
        base = baselines.get(int(row["batch_size"]))
        row["speedup_vs_pytorch"] = (base / float(row["mean_ms"])) if base else None
    return rows


def write_reports(results_dir: Path, metadata: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Path]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = results_dir / f"benchmark_{stamp}.json"
    csv_path = results_dir / f"benchmark_{stamp}.csv"
    md_path = results_dir / f"benchmark_{stamp}.md"

    payload = {"metadata": metadata, "results": rows}
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in CSV_FIELDS})

    md_path.write_text(render_markdown(metadata, rows), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": md_path}


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(metadata: Dict[str, Any], rows: Iterable[Dict[str, Any]]) -> str:
    rows = list(rows)
    lines = [
        "# Inference Benchmark Results",
        "",
        f"- Model: `{metadata.get('model')}`",
        f"- Device: `{metadata.get('device')}`",
        f"- Threads: `{metadata.get('threads')}`",
        f"- Warmup / measured iterations: `{metadata.get('warmup')} / {metadata.get('iterations')}`",
        f"- Python: `{metadata.get('python')}`",
        f"- PyTorch: `{metadata.get('torch')}`",
        f"- ONNX Runtime: `{metadata.get('onnxruntime', 'not loaded')}`",
        "",
        "Steady-state inference only: model loading, ONNX export, session creation, and warmup are excluded from latency.",
        "",
        "| Backend | Batch | Mean ms | P50 ms | P95 ms | Samples/s | Speedup vs eager | Max abs err | Top-1 agreement | Allclose |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        speedup = row.get("speedup_vs_pytorch")
        speedup_text = f"{speedup:.2f}x" if speedup is not None else "—"
        lines.append(
            "| {backend} | {batch} | {mean} | {p50} | {p95} | {throughput} | {speedup} | {max_abs} | {top1} | {allclose} |".format(
                backend=row["backend"],
                batch=row["batch_size"],
                mean=_fmt(row["mean_ms"]),
                p50=_fmt(row["p50_ms"]),
                p95=_fmt(row["p95_ms"]),
                throughput=_fmt(row["throughput_samples_s"], 1),
                speedup=speedup_text,
                max_abs=_fmt(row.get("max_abs_error"), 6),
                top1=_fmt(row.get("top1_agreement"), 4),
                allclose=_fmt(row.get("allclose")),
            )
        )
    lines.append("")
    return "\n".join(lines)
