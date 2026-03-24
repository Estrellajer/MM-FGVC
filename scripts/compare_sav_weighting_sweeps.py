#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FGVC_EXPERIMENTS = {"pets", "eurosat", "flowers", "cub", "tinyimage"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline SAV sweep against SAV weighted-voting sweep."
    )
    parser.add_argument("--baseline-dir", required=True, help="Directory with baseline SAV metrics files")
    parser.add_argument(
        "--weighted-manifest",
        required=True,
        help="Manifest TSV produced by run_all_task_method_suite.sh for SAV+WVote",
    )
    parser.add_argument("--output", required=True, help="Markdown output path")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def primary_metric(metrics: dict[str, Any]) -> tuple[str, float]:
    for key in ("accuracy", "pair_accuracy", "g_acc", "raw_acc", "raw_accuracy"):
        if key in metrics:
            return key, float(metrics[key])
    return "metric", 0.0


def fmt(value: float) -> str:
    return f"{value:.4f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.4f}"


def infer_experiment_id(metrics_path: Path) -> str:
    name = metrics_path.name
    prefix = "sav_"
    suffix = "_qwen2_vl_diag.metrics.json"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise ValueError(f"Unexpected baseline metrics filename: {name}")
    return name[len(prefix) : -len(suffix)]


def load_weighted_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp, delimiter="\t"))


def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {"count": 0.0, "baseline_mean": 0.0, "weighted_mean": 0.0, "delta_mean": 0.0}
    return {
        "count": float(len(rows)),
        "baseline_mean": sum(row["baseline_value"] for row in rows) / len(rows),
        "weighted_mean": sum(row["weighted_value"] for row in rows) / len(rows),
        "delta_mean": sum(row["delta"] for row in rows) / len(rows),
    }


def main() -> None:
    args = parse_args()
    baseline_dir = Path(args.baseline_dir).resolve()
    weighted_manifest_path = Path(args.weighted_manifest).resolve()
    output_path = Path(args.output).resolve()

    baseline_runs: dict[str, dict[str, Any]] = {}
    for metrics_path in sorted(baseline_dir.glob("*.metrics.json")):
        experiment_id = infer_experiment_id(metrics_path)
        payload = load_json(metrics_path)
        baseline_runs[experiment_id] = {
            "metrics_path": str(metrics_path),
            "metrics": payload["metrics"],
        }

    weighted_rows = load_weighted_manifest(weighted_manifest_path)
    weighted_runs: dict[str, dict[str, Any]] = {}
    for row in weighted_rows:
        if row["method_id"] != "sav_wvote":
            continue
        payload = load_json(Path(row["metrics_path"]))
        weighted_runs[row["experiment_id"]] = {
            "dataset_name": row["dataset_name"],
            "evaluator_name": row["evaluator_name"],
            "run_name": row["run_name"],
            "metrics_path": row["metrics_path"],
            "metrics": payload["metrics"],
        }

    shared_ids = sorted(set(baseline_runs) & set(weighted_runs))
    if not shared_ids:
        raise ValueError("No overlapping experiments found between baseline and weighted sweeps")

    rows: list[dict[str, Any]] = []
    for experiment_id in shared_ids:
        baseline_metrics = baseline_runs[experiment_id]["metrics"]
        weighted_metrics = weighted_runs[experiment_id]["metrics"]
        baseline_name, baseline_value = primary_metric(baseline_metrics)
        weighted_name, weighted_value = primary_metric(weighted_metrics)
        if baseline_name != weighted_name:
            raise ValueError(
                f"Metric mismatch for {experiment_id}: baseline={baseline_name}, weighted={weighted_name}"
            )
        rows.append(
            {
                "experiment_id": experiment_id,
                "dataset_name": weighted_runs[experiment_id]["dataset_name"],
                "evaluator_name": weighted_runs[experiment_id]["evaluator_name"],
                "metric_name": baseline_name,
                "baseline_value": baseline_value,
                "weighted_value": weighted_value,
                "delta": weighted_value - baseline_value,
                "baseline_metrics_path": baseline_runs[experiment_id]["metrics_path"],
                "weighted_metrics_path": weighted_runs[experiment_id]["metrics_path"],
            }
        )

    rows.sort(key=lambda row: row["experiment_id"])

    wins = [row for row in rows if row["delta"] > 1e-9]
    losses = [row for row in rows if row["delta"] < -1e-9]
    ties = [row for row in rows if abs(row["delta"]) <= 1e-9]

    fgvc_rows = [row for row in rows if row["experiment_id"] in FGVC_EXPERIMENTS]
    non_fgvc_rows = [row for row in rows if row["experiment_id"] not in FGVC_EXPERIMENTS]
    bucket_stats = {
        "all": summarize_bucket(rows),
        "fgvc": summarize_bucket(fgvc_rows),
        "non_fgvc": summarize_bucket(non_fgvc_rows),
    }

    largest_gains = sorted(wins, key=lambda row: (-row["delta"], row["experiment_id"]))[:8]
    largest_losses = sorted(losses, key=lambda row: (row["delta"], row["experiment_id"]))[:8]

    per_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        per_dataset[row["dataset_name"]].append(row)

    lines: list[str] = []
    lines.append("# SAV Weighted Voting Comparison")
    lines.append("")
    lines.append(f"- Baseline SAV sweep: `{baseline_dir}`")
    lines.append(f"- SAV+WVote manifest: `{weighted_manifest_path}`")
    lines.append(f"- Overlapping experiments: `{len(rows)}`")
    lines.append(f"- Wins / ties / losses: `{len(wins)} / {len(ties)} / {len(losses)}`")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append("| Bucket | Count | SAV mean | SAV+WVote mean | Delta mean |")
    lines.append("| --- | --- | --- | --- | --- |")
    for bucket_name in ("all", "fgvc", "non_fgvc"):
        stats = bucket_stats[bucket_name]
        lines.append(
            "| "
            + " | ".join(
                [
                    bucket_name,
                    str(int(stats["count"])),
                    fmt(stats["baseline_mean"]),
                    fmt(stats["weighted_mean"]),
                    fmt_delta(stats["delta_mean"]),
                ]
            )
            + " |"
        )

    lines.append("")
    lines.append("## Per Task")
    lines.append("")
    lines.append("| Experiment | Dataset | Metric | SAV | SAV+WVote | Delta |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["experiment_id"],
                    row["dataset_name"],
                    row["metric_name"],
                    fmt(row["baseline_value"]),
                    fmt(row["weighted_value"]),
                    fmt_delta(row["delta"]),
                ]
            )
            + " |"
        )

    if largest_gains:
        lines.append("")
        lines.append("## Largest Gains")
        lines.append("")
        for row in largest_gains:
            lines.append(
                f"- `{row['experiment_id']}` ({row['metric_name']}): "
                f"`{fmt(row['baseline_value'])} -> {fmt(row['weighted_value'])}` "
                f"(`{fmt_delta(row['delta'])}`)"
            )

    if largest_losses:
        lines.append("")
        lines.append("## Largest Losses")
        lines.append("")
        for row in largest_losses:
            lines.append(
                f"- `{row['experiment_id']}` ({row['metric_name']}): "
                f"`{fmt(row['baseline_value'])} -> {fmt(row['weighted_value'])}` "
                f"(`{fmt_delta(row['delta'])}`)"
            )

    lines.append("")
    lines.append("## Dataset Groups")
    lines.append("")
    for dataset_name in sorted(per_dataset):
        group = per_dataset[dataset_name]
        group.sort(key=lambda row: row["experiment_id"])
        mean_delta = sum(row["delta"] for row in group) / len(group)
        wins_count = sum(row["delta"] > 1e-9 for row in group)
        losses_count = sum(row["delta"] < -1e-9 for row in group)
        lines.append(
            f"- `{dataset_name}`: mean delta `{fmt_delta(mean_delta)}`, "
            f"wins `{wins_count}`, losses `{losses_count}`, tasks `{len(group)}`"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
