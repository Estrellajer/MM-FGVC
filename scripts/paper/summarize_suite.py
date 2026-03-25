#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from registry import METHOD_ORDER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a paper suite manifest into markdown.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp, delimiter="\t"))
    if rows and "sequence_index" in rows[0]:
        rows.sort(key=lambda row: int(row["sequence_index"]))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def primary_metric(metrics: dict[str, Any]) -> tuple[str, float]:
    for key in ("accuracy", "pair_accuracy", "g_acc", "raw_acc", "raw_accuracy"):
        if key in metrics:
            return key, float(metrics[key])
    return "metric", 0.0


def fmt_mean_std(values: list[float]) -> str:
    if not values:
        return "-"
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if len(values) > 1 else 0.0
    return f"{mean:.4f}±{std:.4f}"


def fmt_float(value: float | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def task_label(row: dict[str, str]) -> str:
    experiment_id = row["experiment_id"]
    dataset_name = row["dataset_name"]
    return experiment_id if experiment_id != dataset_name else dataset_name


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()

    rows = load_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_path}")

    entries: list[dict[str, Any]] = []
    for row in rows:
        metrics_payload = load_json(Path(row["metrics_path"]))
        timings = metrics_payload.get("timings", {})
        metric_name, metric_value = primary_metric(metrics_payload["metrics"])
        entries.append(
            {
                **row,
                "task_label": task_label(row),
                "metric_name": metric_name,
                "metric_value": metric_value,
                "fit_time_sec": float(timings.get("fit_time_sec", 0.0)),
                "avg_predict_time_sec": float(timings.get("avg_predict_time_sec", 0.0)),
            }
        )

    suite_name = rows[0]["suite_name"]
    timestamp = rows[0]["timestamp"]
    models = list(dict.fromkeys(row["model_name"] for row in rows))
    tasks = list(dict.fromkeys(entry["task_label"] for entry in entries))
    methods = [method_id for method_id in METHOD_ORDER if any(row["method_id"] == method_id for row in rows)]

    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[(entry["model_name"], entry["task_label"], entry["display_name"])].append(entry)

    timing_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        timing_groups[(entry["model_name"], entry["display_name"])].append(entry)

    lines: list[str] = []
    lines.append(f"# {suite_name} Summary")
    lines.append("")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- Timestamp: `{timestamp}`")
    lines.append(f"- Models: `{', '.join(models)}`")
    lines.append(f"- Tasks: `{', '.join(tasks)}`")
    lines.append("")

    for model_name in models:
        lines.append(f"## {model_name}")
        lines.append("")
        present_display_names = []
        for method_id in methods:
            display_name = next((row["display_name"] for row in rows if row["method_id"] == method_id), method_id)
            if any(row["model_name"] == model_name and row["display_name"] == display_name for row in rows):
                present_display_names.append(display_name)

        lines.append("| Task | " + " | ".join(present_display_names) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(present_display_names)) + " |")
        for current_task in tasks:
            row_values = [current_task]
            for display_name in present_display_names:
                bucket = grouped.get((model_name, current_task, display_name), [])
                row_values.append(fmt_mean_std([item["metric_value"] for item in bucket]))
            lines.append("| " + " | ".join(row_values) + " |")

        lines.append("")
        lines.append("| Method | Mean Primary | Mean Fit (s) | Mean Pred (ms/sample) | Runs |")
        lines.append("| --- | --- | --- | --- | --- |")
        for display_name in present_display_names:
            bucket = timing_groups.get((model_name, display_name), [])
            metric_values = [item["metric_value"] for item in bucket]
            fit_values = [item["fit_time_sec"] for item in bucket]
            pred_ms_values = [item["avg_predict_time_sec"] * 1000.0 for item in bucket]
            lines.append(
                "| "
                + " | ".join(
                    [
                        display_name,
                        fmt_mean_std(metric_values),
                        fmt_float(statistics.fmean(fit_values) if fit_values else None),
                        fmt_float(statistics.fmean(pred_ms_values) if pred_ms_values else None),
                        str(len(bucket)),
                    ]
                )
                + " |"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
