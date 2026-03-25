#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze representation-generation gap from a paper-suite manifest.")
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


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.4f}"


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:+.4f}"


def task_label(row: dict[str, str]) -> str:
    return row["experiment_id"] if row["experiment_id"] != row["dataset_name"] else row["dataset_name"]


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    output_path = Path(args.output).resolve()
    rows = load_manifest(manifest_path)
    if not rows:
        raise ValueError(f"Empty manifest: {manifest_path}")

    runs: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        metrics_payload = load_json(Path(row["metrics_path"]))
        diagnostics_path = Path(row["diagnostics_path"])
        diagnostics = load_json(diagnostics_path) if diagnostics_path.exists() else {}
        metric_name, metric_value = primary_metric(metrics_payload["metrics"])
        runs[(row["model_name"], task_label(row), row["method_id"])] = {
            "display_name": row["display_name"],
            "metric_name": metric_name,
            "metric_value": metric_value,
            "diagnostics": diagnostics,
        }

    models = list(dict.fromkeys(row["model_name"] for row in rows))
    tasks = list(dict.fromkeys(task_label(row) for row in rows))

    lines: list[str] = []
    lines.append("# Representation-Generation Gap")
    lines.append("")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append("")

    for model_name in models:
        best_component_wins: list[float] = []
        oracle_wins: list[float] = []
        final_wins: list[float] = []

        lines.append(f"## {model_name}")
        lines.append("")
        lines.append("| Task | Zero-shot | SAV | RSEv2 | Best Component | Oracle Upper Bound | Best-vs-ZS | Oracle-vs-ZS | RSEv2-vs-ZS |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

        for current_task in tasks:
            zero_run = runs.get((model_name, current_task, "zero_shot"))
            sav_run = runs.get((model_name, current_task, "sav"))
            rsev2_run = runs.get((model_name, current_task, "rsev2"))
            if zero_run is None or rsev2_run is None:
                continue

            diagnostics = rsev2_run["diagnostics"]
            best_component = ((diagnostics.get("best_component_by_val") or {}).get("val_accuracy"))
            oracle_upper = ((diagnostics.get("oracle_summary") or {}).get("oracle_accuracy"))
            zero_value = float(zero_run["metric_value"])
            rsev2_value = float(rsev2_run["metric_value"])
            sav_value = None if sav_run is None else float(sav_run["metric_value"])

            if best_component is not None:
                best_component_wins.append(float(best_component) - zero_value)
            if oracle_upper is not None:
                oracle_wins.append(float(oracle_upper) - zero_value)
            final_wins.append(rsev2_value - zero_value)

            lines.append(
                "| "
                + " | ".join(
                    [
                        current_task,
                        fmt(zero_value),
                        fmt(sav_value),
                        fmt(rsev2_value),
                        fmt(float(best_component) if best_component is not None else None),
                        fmt(float(oracle_upper) if oracle_upper is not None else None),
                        fmt_delta((float(best_component) - zero_value) if best_component is not None else None),
                        fmt_delta((float(oracle_upper) - zero_value) if oracle_upper is not None else None),
                        fmt_delta(rsev2_value - zero_value),
                    ]
                )
                + " |"
            )

        lines.append("")
        if best_component_wins:
            lines.append(
                f"- Best standalone component beats zero-shot on `{sum(delta > 0 for delta in best_component_wins)}/{len(best_component_wins)}` tasks"
            )
        if oracle_wins:
            lines.append(
                f"- Oracle upper bound beats zero-shot on `{sum(delta > 0 for delta in oracle_wins)}/{len(oracle_wins)}` tasks"
            )
        if final_wins:
            lines.append(
                f"- Final RSEv2 beats zero-shot on `{sum(delta > 0 for delta in final_wins)}/{len(final_wins)}` tasks"
            )
            lines.append(f"- Mean delta (RSEv2 - zero-shot): `{statistics.fmean(final_wins):+.4f}`")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
