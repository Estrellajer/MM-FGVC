#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize RSE diagnostics from a method-suite manifest.")
    parser.add_argument("--manifest", required=True, help="TSV manifest produced by run_all_task_method_suite.sh")
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
    output_path = Path(args.output).resolve()

    rows = load_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_path}")

    runs_by_dataset: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        metrics_payload = load_json(Path(row["metrics_path"]))
        diagnostics_path = Path(row["metrics_path"].replace(".metrics.json", ".diagnostics.json"))
        diagnostics = load_json(diagnostics_path) if diagnostics_path.exists() else None
        runs_by_dataset[row["dataset_name"]][row["display_name"]] = {
            "metrics": metrics_payload["metrics"],
            "metric_name": primary_metric(metrics_payload["metrics"])[0],
            "metric_value": primary_metric(metrics_payload["metrics"])[1],
            "metrics_path": row["metrics_path"],
            "diagnostics_path": str(diagnostics_path),
            "diagnostics": diagnostics,
        }

    datasets = sorted(runs_by_dataset)
    aggregate_level_counts: Counter[str] = Counter()
    aggregate_stage_counts: Counter[str] = Counter()
    rse_vs_zero = []
    rse_vs_sav = []
    best_component_vs_zero = []

    lines: list[str] = []
    lines.append("# RSE Phase-1 Diagnostics")
    lines.append("")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- Datasets: `{', '.join(datasets)}`")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("| Dataset | Zero-shot | SAV | RSE | Best RSE Component | RSE-ZS | RSE-SAV |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    dataset_sections: list[tuple[str, list[str]]] = []
    for dataset_name in datasets:
        runs = runs_by_dataset[dataset_name]
        zero_run = runs.get("Zero-shot")
        sav_run = runs.get("SAV")
        rse_run = runs.get("RSE")
        if rse_run is None or rse_run["diagnostics"] is None:
            continue

        diagnostics = rse_run["diagnostics"]
        component_rows = diagnostics.get("component_table", [])
        component_rows = [row for row in component_rows if row.get("val_accuracy") is not None]
        best_component = max(component_rows, key=lambda row: row["val_accuracy"]) if component_rows else None

        rse_metric = float(rse_run["metric_value"])
        zero_metric = float(zero_run["metric_value"]) if zero_run else None
        sav_metric = float(sav_run["metric_value"]) if sav_run else None
        best_component_metric = float(best_component["val_accuracy"]) if best_component else None

        delta_zero = (rse_metric - zero_metric) if zero_metric is not None else None
        delta_sav = (rse_metric - sav_metric) if sav_metric is not None else None
        best_component_delta_zero = (
            best_component_metric - zero_metric
            if best_component_metric is not None and zero_metric is not None
            else None
        )

        if delta_zero is not None:
            rse_vs_zero.append(delta_zero)
        if delta_sav is not None:
            rse_vs_sav.append(delta_sav)
        if best_component_delta_zero is not None:
            best_component_vs_zero.append(best_component_delta_zero)

        lines.append(
            "| "
            + " | ".join(
                [
                    dataset_name,
                    fmt(zero_metric),
                    fmt(sav_metric),
                    fmt(rse_metric),
                    fmt(best_component_metric),
                    fmt_delta(delta_zero),
                    fmt_delta(delta_sav),
                ]
            )
            + " |"
        )

        section_lines: list[str] = []
        selected = diagnostics.get("selected_components", [])
        num_layers = int(diagnostics.get("num_layers", 0))
        level_counts = Counter(row["level"] for row in selected)
        stage_counts = Counter(layer_stage(int(row["layer_idx"]), num_layers) for row in selected)
        aggregate_level_counts.update(level_counts)
        aggregate_stage_counts.update(stage_counts)

        section_lines.append(f"## {dataset_name}")
        section_lines.append("")
        section_lines.append(
            f"- Metrics: zero-shot `{fmt(zero_metric)}`, SAV `{fmt(sav_metric)}`, RSE `{fmt(rse_metric)}`"
        )
        section_lines.append(
            f"- Delta: RSE-zero-shot `{fmt_delta(delta_zero)}`, RSE-SAV `{fmt_delta(delta_sav)}`"
        )
        if best_component is not None:
            section_lines.append(
                f"- Best standalone component: `{best_component['level']}` layer `{best_component['layer_idx']}` "
                f"with val accuracy `{fmt(best_component_metric)}` and FDR `{fmt(float(best_component['fdr']))}`"
            )
            section_lines.append(
                f"- Best component vs zero-shot: `{fmt_delta(best_component_delta_zero)}`"
            )

        section_lines.append(
            "- Selected component distribution: "
            + ", ".join(f"`{level}`={count}" for level, count in sorted(level_counts.items()))
        )
        section_lines.append(
            "- Selected stage distribution: "
            + ", ".join(f"`{stage}`={count}" for stage, count in sorted(stage_counts.items()))
        )
        section_lines.append("")
        section_lines.append("| Level | Layer | FDR | Val Acc | Weight |")
        section_lines.append("| --- | --- | --- | --- | --- |")
        for row in selected:
            section_lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["level"]),
                        str(row["layer_idx"]),
                        fmt(float(row["fdr"])),
                        fmt(row.get("val_accuracy")),
                        fmt(float(row["weight"])),
                    ]
                )
                + " |"
            )
        section_lines.append("")
        section_lines.append(f"- RSE diagnostics: `{rse_run['diagnostics_path']}`")

        dataset_sections.append((dataset_name, section_lines))

    lines.append("")
    lines.append("## Aggregate Diagnostics")
    lines.append("")
    if rse_vs_zero:
        lines.append(
            f"- RSE better than zero-shot on `{sum(delta > 0 for delta in rse_vs_zero)}/{len(rse_vs_zero)}` tasks"
        )
    if rse_vs_sav:
        lines.append(
            f"- RSE better than SAV on `{sum(delta > 0 for delta in rse_vs_sav)}/{len(rse_vs_sav)}` tasks"
        )
    if best_component_vs_zero:
        lines.append(
            f"- Best standalone RSE component better than zero-shot on "
            f"`{sum(delta > 0 for delta in best_component_vs_zero)}/{len(best_component_vs_zero)}` tasks"
        )
    if aggregate_level_counts:
        lines.append(
            "- Selected level totals: "
            + ", ".join(f"`{level}`={count}" for level, count in sorted(aggregate_level_counts.items()))
        )
    if aggregate_stage_counts:
        lines.append(
            "- Selected stage totals: "
            + ", ".join(f"`{stage}`={count}" for stage, count in sorted(aggregate_stage_counts.items()))
        )

    for _, section_lines in dataset_sections:
        lines.append("")
        lines.extend(section_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
