#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize method-suite results into Markdown.")
    parser.add_argument("--manifest", required=True, help="TSV manifest produced by run_fgvc_method_suite.sh")
    parser.add_argument("--output", required=True, help="Markdown output path")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp, delimiter="\t")
        return [dict(row) for row in reader]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fmt_metric(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def primary_metric(metrics: dict[str, Any]) -> tuple[str, float]:
    if "accuracy" in metrics:
        return "accuracy", float(metrics["accuracy"])
    if "pair_accuracy" in metrics:
        return "pair_accuracy", float(metrics["pair_accuracy"])
    if "g_acc" in metrics:
        return "g_acc", float(metrics["g_acc"])
    if "raw_acc" in metrics:
        return "raw_acc", float(metrics["raw_acc"])
    if "raw_accuracy" in metrics:
        return "raw_accuracy", float(metrics["raw_accuracy"])
    return "metric", 0.0


def top_confusions(metrics: dict[str, Any], limit: int = 3) -> list[tuple[str, str, int]]:
    confusion = metrics.get("confusion_matrix", {})
    labels = confusion.get("labels", [])
    matrix = confusion.get("matrix", [])
    entries: list[tuple[str, str, int]] = []
    for row_idx, actual in enumerate(labels):
        for col_idx, predicted in enumerate(labels):
            if row_idx == col_idx:
                continue
            count = int(matrix[row_idx][col_idx])
            if count > 0:
                entries.append((actual, predicted, count))
    entries.sort(key=lambda item: (-item[2], item[0], item[1]))
    return entries[:limit]


def worst_class_recalls(metrics: dict[str, Any], limit: int = 3) -> list[tuple[str, float, float]]:
    per_class = metrics.get("per_class", {})
    rows: list[tuple[str, float, float]] = []
    for label, stats in per_class.items():
        support = float(stats.get("support", 0.0))
        if support <= 0:
            continue
        rows.append((label, float(stats.get("recall", 0.0)), support))
    rows.sort(key=lambda item: (item[1], -item[2], item[0]))
    return rows[:limit]


def pairwise_complementarity(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complements: list[dict[str, Any]] = []
    for left, right in combinations(run_rows, 2):
        left_preds = left["predictions"]
        right_preds = right["predictions"]
        if len(left_preds) != len(right_preds):
            continue

        total = len(left_preds)
        if total == 0:
            continue
        left_correct = [bool(row["correct"]) for row in left_preds]
        right_correct = [bool(row["correct"]) for row in right_preds]
        oracle_correct = sum(a or b for a, b in zip(left_correct, right_correct))
        one_only = sum((a and not b) or (b and not a) for a, b in zip(left_correct, right_correct))
        disagree = sum(
            left_row["prediction"] != right_row["prediction"]
            for left_row, right_row in zip(left_preds, right_preds)
        )

        complements.append(
            {
                "left": left["display_name"],
                "right": right["display_name"],
                "oracle_accuracy": oracle_correct / total if total else 0.0,
                "one_correct_only": one_only / total if total else 0.0,
                "prediction_disagreement": disagree / total if total else 0.0,
            }
        )

    complements.sort(
        key=lambda item: (
            -item["oracle_accuracy"],
            -item["one_correct_only"],
            -item["prediction_disagreement"],
            item["left"],
            item["right"],
        )
    )
    return complements


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()

    manifest_rows = load_manifest(manifest_path)
    if not manifest_rows:
        raise ValueError(f"Manifest is empty: {manifest_path}")

    runs: list[dict[str, Any]] = []
    for row in manifest_rows:
        metrics_payload = load_json(Path(row["metrics_path"]))
        runs.append(
            {
                "dataset_name": row["dataset_name"],
                "method_id": row["method_id"],
                "display_name": row["display_name"],
                "run_name": row["run_name"],
                "train_subset": row["train_subset"],
                "val_subset": row["val_subset"],
                "metrics_path": row["metrics_path"],
                "predictions_path": row["predictions_path"],
                "metrics": metrics_payload["metrics"],
                "payload": metrics_payload,
                "predictions": load_jsonl(Path(row["predictions_path"])),
            }
        )

    model_name = runs[0]["payload"]["model"]
    datasets = sorted({run["dataset_name"] for run in runs})
    methods = sorted({run["display_name"] for run in runs})
    runs_by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        runs_by_dataset[run["dataset_name"]].append(run)

    lines: list[str] = []
    lines.append("# Method Suite Results")
    lines.append("")
    lines.append(f"- Model: `{model_name}`")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- Datasets: `{', '.join(datasets)}`")
    lines.append(f"- Methods: `{', '.join(methods)}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    header = "| Dataset | " + " | ".join(methods) + " |"
    divider = "| --- | " + " | ".join(["---"] * len(methods)) + " |"
    lines.append(header)
    lines.append(divider)
    for dataset_name in datasets:
        by_name = {run["display_name"]: run for run in runs_by_dataset[dataset_name]}
        row = [dataset_name]
        for display_name in methods:
            run = by_name.get(display_name)
            metric_value = None if run is None else primary_metric(run["metrics"])[1]
            row.append(fmt_metric(metric_value))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    stv_names = ["STV", "STV+QC", "SAV-TV", "SAV-TV+QC"]
    if any(name in methods for name in stv_names):
        lines.append("")
        lines.append("## STV Ablation")
        lines.append("")
        present_stv_names = [name for name in stv_names if name in methods]
        lines.append("| Dataset | " + " | ".join(present_stv_names) + " |")
        lines.append("| --- | " + " | ".join(["---"] * len(present_stv_names)) + " |")
        for dataset_name in datasets:
            by_name = {run["display_name"]: run for run in runs_by_dataset[dataset_name]}
            row = [dataset_name]
            for display_name in present_stv_names:
                run = by_name.get(display_name)
                metric_value = None if run is None else primary_metric(run["metrics"])[1]
                row.append(fmt_metric(metric_value))
            lines.append("| " + " | ".join(row) + " |")

    for dataset_name in datasets:
        lines.append("")
        lines.append(f"## {dataset_name}")
        lines.append("")

        dataset_runs = sorted(
            runs_by_dataset[dataset_name],
            key=lambda run: (-primary_metric(run["metrics"])[1], run["display_name"]),
        )
        best_run = dataset_runs[0]
        best_metric_name, best_metric_value = primary_metric(best_run["metrics"])
        lines.append(
            f"- Best primary metric: `{best_run['display_name']}` = `{best_metric_name}={fmt_metric(best_metric_value)}`"
        )
        lines.append(
            f"- Macro-F1 of best run: `{fmt_metric(float(best_run['metrics'].get('macro_f1', 0.0)))}`"
        )

        confusions = top_confusions(best_run["metrics"])
        if confusions:
            lines.append("- Top confusions of best run:")
            for actual, predicted, count in confusions:
                lines.append(f"  - `{actual}` -> `{predicted}`: `{count}`")

        worst_classes = worst_class_recalls(best_run["metrics"])
        if worst_classes:
            lines.append("- Lowest-recall classes of best run:")
            for label, recall, support in worst_classes:
                lines.append(f"  - `{label}`: recall `{recall:.4f}`, support `{int(support)}`")

        complements = pairwise_complementarity(dataset_runs)
        if complements:
            lines.append("- Most complementary method pairs:")
            for item in complements[:5]:
                lines.append(
                    "  - "
                    f"`{item['left']}` + `{item['right']}`: "
                    f"oracle `{item['oracle_accuracy']:.4f}`, "
                    f"one-correct-only `{item['one_correct_only']:.4f}`, "
                    f"disagreement `{item['prediction_disagreement']:.4f}`"
                )

        lines.append("")
        lines.append("| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for run in dataset_runs:
            metrics = run["metrics"]
            metric_name, metric_value = primary_metric(metrics)
            lines.append(
                "| "
                + " | ".join(
                    [
                        run["display_name"],
                        f"{metric_name}={fmt_metric(metric_value)}",
                        fmt_metric(float(metrics.get("macro_f1", 0.0))),
                        fmt_metric(float(metrics.get("balanced_accuracy", 0.0))),
                        f"`{run['metrics_path']}`",
                        "-" if not Path(run["predictions_path"]).exists() else f"`{run['predictions_path']}`",
                    ]
                )
                + " |"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
