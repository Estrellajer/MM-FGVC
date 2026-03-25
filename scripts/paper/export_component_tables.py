#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export component tables from RSE/RSEv2 diagnostics.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fp:
        rows = list(csv.DictReader(fp, delimiter="\t"))
    if rows and "sequence_index" in rows[0]:
        rows.sort(key=lambda row: int(row["sequence_index"]))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_label(row: dict[str, str]) -> str:
    return row["experiment_id"] if row["experiment_id"] != row["dataset_name"] else row["dataset_name"]


def layer_stage(layer_idx: int, num_layers: int) -> str:
    if num_layers <= 1:
        return "single"
    if layer_idx < num_layers / 3:
        return "early"
    if layer_idx < (2 * num_layers) / 3:
        return "mid"
    return "late"


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    csv_path = Path(args.output_csv).resolve()
    summary_path = Path(args.summary_output).resolve()

    rows = load_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_path}")

    long_rows: list[dict[str, Any]] = []
    selected_levels: Counter[str] = Counter()
    selected_stages: Counter[str] = Counter()
    best_components_by_task: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in rows:
        diagnostics_path = Path(row["diagnostics_path"])
        if not diagnostics_path.exists():
            continue
        diagnostics = load_json(diagnostics_path)
        component_table = diagnostics.get("component_table")
        if not isinstance(component_table, list):
            continue

        num_layers = int(diagnostics.get("num_layers", 0))
        for component in component_table:
            layer_idx = int(component["layer_idx"])
            selected = bool(component.get("selected", False))
            long_rows.append(
                {
                    "model_name": row["model_name"],
                    "task_label": task_label(row),
                    "dataset_name": row["dataset_name"],
                    "method_id": row["method_id"],
                    "display_name": row["display_name"],
                    "level": component.get("level"),
                    "layer_idx": layer_idx,
                    "stage": layer_stage(layer_idx, num_layers),
                    "selected": int(selected),
                    "selection_score": component.get("selection_score"),
                    "fdr": component.get("fdr"),
                    "loo_accuracy": component.get("loo_accuracy"),
                    "cv_accuracy": component.get("cv_accuracy"),
                    "weight": component.get("weight"),
                    "mean_vote_weight": component.get("mean_vote_weight"),
                    "mean_margin": component.get("mean_margin"),
                    "val_accuracy": component.get("val_accuracy"),
                    "shrinkage_alpha": component.get("shrinkage_alpha"),
                }
            )
            if selected:
                selected_levels[str(component.get("level"))] += 1
                selected_stages[layer_stage(layer_idx, num_layers)] += 1

        best_by_val = diagnostics.get("best_component_by_val")
        if isinstance(best_by_val, dict):
            best_components_by_task[(row["model_name"], task_label(row), row["display_name"])] = best_by_val

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        if long_rows:
            writer = csv.DictWriter(fp, fieldnames=list(long_rows[0].keys()))
            writer.writeheader()
            writer.writerows(long_rows)
        else:
            fp.write("")

    lines: list[str] = []
    lines.append("# Component Peak Summary")
    lines.append("")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- Export CSV: `{csv_path}`")
    lines.append("")
    if selected_levels:
        lines.append(
            "- Selected level totals: "
            + ", ".join(f"`{level}`={count}" for level, count in sorted(selected_levels.items()))
        )
    if selected_stages:
        lines.append(
            "- Selected stage totals: "
            + ", ".join(f"`{stage}`={count}" for stage, count in sorted(selected_stages.items()))
        )
    lines.append("")
    lines.append("| Model | Task | Method | Best Level | Best Layer | Best Val Acc |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for (model_name, current_task, display_name), best_by_val in sorted(best_components_by_task.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    model_name,
                    current_task,
                    display_name,
                    str(best_by_val.get("level")),
                    str(best_by_val.get("layer_idx")),
                    f"{float(best_by_val.get('val_accuracy', 0.0)):.4f}",
                ]
            )
            + " |"
        )

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
