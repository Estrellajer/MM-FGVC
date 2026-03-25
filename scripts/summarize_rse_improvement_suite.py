#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


METHOD_ORDER = [
    "Zero-shot",
    "SAV",
    "RSE",
    "RSE-LOO",
    "RSE-Top1",
    "RSE-Greedy",
    "RSE-ZScore",
    "RSE-Fallback",
    "RSE-Route1",
    "RSE-Route2",
    "RSE-Combo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RSE improvement sweeps across multiple manifests.")
    parser.add_argument("--manifest", action="append", required=True, help="Manifest TSV path. Can be passed multiple times.")
    parser.add_argument("--output", required=True, help="Markdown output path")
    return parser.parse_args()


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as fp:
        return list(csv.DictReader(fp, delimiter="\t"))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def primary_metric(metrics: dict[str, Any]) -> tuple[str, float]:
    for key in ("accuracy", "pair_accuracy", "g_acc", "raw_acc", "raw_accuracy"):
        if key in metrics:
            return key, float(metrics[key])
    return "metric", 0.0


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def main() -> None:
    args = parse_args()
    manifests = [Path(path).resolve() for path in args.manifest]
    output_path = Path(args.output).resolve()

    runs: dict[tuple[str, str], dict[str, Any]] = {}
    for manifest_path in manifests:
        for row in load_manifest(manifest_path):
            key = (row["dataset_name"], row["display_name"])
            metrics_payload = load_json(Path(row["metrics_path"]))
            diagnostics_path = Path(row["metrics_path"].replace(".metrics.json", ".diagnostics.json"))
            diagnostics = load_json(diagnostics_path) if diagnostics_path.exists() else None
            runs[key] = {
                "dataset_name": row["dataset_name"],
                "display_name": row["display_name"],
                "metrics_path": row["metrics_path"],
                "metric_name": primary_metric(metrics_payload["metrics"])[0],
                "metric_value": primary_metric(metrics_payload["metrics"])[1],
                "metrics": metrics_payload["metrics"],
                "diagnostics": diagnostics,
                "diagnostics_path": str(diagnostics_path),
            }

    datasets = sorted({key[0] for key in runs})
    method_names = [name for name in METHOD_ORDER if any(key[1] == name for key in runs)]

    lines: list[str] = []
    lines.append("# RSE Improvement Suite")
    lines.append("")
    for manifest_path in manifests:
        lines.append(f"- Manifest: `{manifest_path}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Dataset | " + " | ".join(method_names) + " |")
    lines.append("| --- | " + " | ".join(["---"] * len(method_names)) + " |")
    for dataset_name in datasets:
        row = [dataset_name]
        for method_name in method_names:
            run = runs.get((dataset_name, method_name))
            row.append(fmt(run["metric_value"]) if run else "-")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Aggregate Deltas vs RSE")
    lines.append("")
    lines.append("| Method | Mean Delta vs RSE | Wins | Ties | Losses | Mean Delta vs Zero-shot | Mean Delta vs SAV |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for method_name in method_names:
        if method_name in {"Zero-shot", "SAV", "RSE"}:
            continue
        deltas_vs_rse = []
        deltas_vs_zs = []
        deltas_vs_sav = []
        wins = ties = losses = 0
        for dataset_name in datasets:
            base_rse = runs.get((dataset_name, "RSE"))
            current = runs.get((dataset_name, method_name))
            zero = runs.get((dataset_name, "Zero-shot"))
            sav = runs.get((dataset_name, "SAV"))
            if current is None or base_rse is None:
                continue
            delta = current["metric_value"] - base_rse["metric_value"]
            deltas_vs_rse.append(delta)
            wins += int(delta > 1e-9)
            losses += int(delta < -1e-9)
            ties += int(abs(delta) <= 1e-9)
            if zero is not None:
                deltas_vs_zs.append(current["metric_value"] - zero["metric_value"])
            if sav is not None:
                deltas_vs_sav.append(current["metric_value"] - sav["metric_value"])

        mean_delta_rse = sum(deltas_vs_rse) / len(deltas_vs_rse) if deltas_vs_rse else None
        mean_delta_zs = sum(deltas_vs_zs) / len(deltas_vs_zs) if deltas_vs_zs else None
        mean_delta_sav = sum(deltas_vs_sav) / len(deltas_vs_sav) if deltas_vs_sav else None
        lines.append(
            "| "
            + " | ".join(
                [
                    method_name,
                    fmt_delta(mean_delta_rse),
                    str(wins),
                    str(ties),
                    str(losses),
                    fmt_delta(mean_delta_zs),
                    fmt_delta(mean_delta_sav),
                ]
            )
            + " |"
        )

    for dataset_name in datasets:
        lines.append("")
        lines.append(f"## {dataset_name}")
        lines.append("")
        lines.append("| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        base_rse = runs.get((dataset_name, "RSE"))
        for method_name in method_names:
            run = runs.get((dataset_name, method_name))
            if run is None:
                continue
            diagnostics = run["diagnostics"] or {}
            best_component = diagnostics.get("best_component_by_val") or {}
            selected = diagnostics.get("selected_components") or []
            train_summary = diagnostics.get("train_selection_summary") or {}
            eval_summary = diagnostics.get("eval_summary") or {}
            delta_vs_rse = None
            if base_rse is not None:
                delta_vs_rse = run["metric_value"] - base_rse["metric_value"]
            best_component_text = "-"
            if best_component:
                best_component_text = (
                    f"{best_component.get('level')}@{best_component.get('layer_idx')} "
                    f"({fmt(best_component.get('val_accuracy'))})"
                )
            lines.append(
                "| "
                + " | ".join(
                    [
                        method_name,
                        f"{run['metric_name']}={fmt(run['metric_value'])}",
                        fmt_delta(delta_vs_rse),
                        best_component_text,
                        str(len(selected)) if diagnostics else "-",
                        fmt(train_summary.get("selection_train_accuracy")),
                        str(eval_summary.get("fallback_used")) if diagnostics else "-",
                        "-" if not diagnostics else f"`{run['diagnostics_path']}`",
                    ]
                )
                + " |"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
