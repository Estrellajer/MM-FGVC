#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


WRITE_METHODS = ["stv", "i2cl", "mimic"]
READ_METHOD = "keco"
ALL_METHODS = ["stv", "i2cl", "mimic", READ_METHOD]
LABELS = {"stv": "STV", "i2cl": "I2CL", "mimic": "MimIC", READ_METHOD: "HIRE"}
COLORS = {"stv": "#4c78a8", "i2cl": "#f58518", "mimic": "#54a24b", READ_METHOD: "#1b1f3b"}

plt.rcParams.update(
    {
        "font.size": 15,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.titleweight": "semibold",
        "axes.labelweight": "semibold",
        "font.weight": "semibold",
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a HIRE-vs-write comparison figure on the same analysis subset.")
    parser.add_argument("--source-output-root", required=True, help="Write-failure output root used to define the common subset.")
    parser.add_argument("--timestamp", default=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--suite-name", default="hire_advantage")
    parser.add_argument("--hire-method", default="keco")
    parser.add_argument("--hire-metrics", default=None, help="Optional existing HIRE metrics.json to reuse instead of rerunning.")
    parser.add_argument("--hire-log", default=None, help="Optional existing HIRE log path.")
    parser.add_argument(
        "--runner-python",
        default=str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        help="Python executable used for main.py subprocesses.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_source_diagnostics(source_output_root: Path, method_name: str) -> Path:
    matches = sorted(source_output_root.glob(f"*_{method_name}_*.diagnostics.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source diagnostics file for method={method_name}, found {len(matches)}"
        )
    return matches[0]


def load_write_stats(source_output_root: Path) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for method in WRITE_METHODS:
        diagnostics = load_json(resolve_source_diagnostics(source_output_root, method))
        analysis = diagnostics.get("write_failure_analysis") or {}
        summary = analysis.get("summary") or {}
        rows = list(analysis.get("samples") or [])
        stats[method] = {
            "accuracy": float(summary.get("steered_accuracy") or 0.0),
            "drift": float(summary.get("query_hidden_l2_ratio") or 0.0),
            "norm_ratio": float(summary.get("task_vector_to_hidden_norm_ratio") or 0.0),
        }
        if rows:
            stats[method]["accuracy"] = float(summary.get("steered_accuracy") or 0.0)
    return stats


def run_hire_eval(
    *,
    manifest: dict[str, Any],
    hire_method: str,
    output_root: Path,
    log_root: Path,
    timestamp: str,
    runner_python: str,
) -> tuple[Path, Path]:
    dataset_name = str(manifest["dataset_name"])
    hire_dataset_name = dataset_name.replace("_fgvc", "")
    model_name = str(manifest["model"])
    train_path = Path(manifest["train_path"])
    val_path = Path(manifest["analysis_val_path"])
    seed = int(manifest["seed"])
    run_name = f"hire_advantage_{dataset_name}_{hire_method}_{model_name}_{timestamp}"
    log_path = log_root / f"{run_name}.log"
    cmd = [
        runner_python,
        str(PROJECT_ROOT / "main.py"),
        f"model={model_name}",
        "dataset=general_custom",
        f"method={hire_method}",
        "evaluator=raw",
        f"dataset.name={hire_dataset_name}",
        f"dataset.train_path={train_path}",
        f"dataset.val_path={val_path}",
        f"run.output_dir={output_root}",
        f"run.run_name={run_name}",
        f"run.seed={seed}",
        "run.progress_bar=false",
        "method.params.progress_bar=false",
        "+model.model_args.attn_implementation=eager",
    ]

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(f"# command={' '.join(cmd)}\n\n")
        fp.flush()
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fp, stderr=subprocess.STDOUT, check=True)

    return output_root / f"{run_name}.metrics.json", log_path


def save_figure(figure_path: Path, stats: dict[str, dict[str, float]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.8), dpi=220)
    ax_acc = axes[0, 0]
    ax_drift = axes[0, 1]
    ax_norm = axes[1, 0]
    ax_scatter = axes[1, 1]

    methods = ALL_METHODS
    xs = np.arange(len(methods))
    colors = [COLORS[method] for method in methods]

    acc_values = [stats[method]["accuracy"] for method in methods]
    ax_acc.bar(xs, acc_values, color=colors)
    ax_acc.set_xticks(xs)
    ax_acc.set_xticklabels([LABELS[method] for method in methods])
    ax_acc.set_ylim(0.0, 1.02)
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Accuracy on the same balanced subset")
    ax_acc.grid(alpha=0.25, axis="y")
    ax_acc.tick_params(axis="both", pad=4)
    for label in ax_acc.get_xticklabels() + ax_acc.get_yticklabels():
        label.set_fontweight("semibold")

    drift_values = [stats[method]["drift"] for method in methods]
    ax_drift.bar(xs, drift_values, color=colors)
    ax_drift.set_xticks(xs)
    ax_drift.set_xticklabels([LABELS[method] for method in methods])
    ax_drift.set_ylabel("Median query hidden drift")
    ax_drift.set_title("Internal interference")
    ax_drift.grid(alpha=0.25, axis="y")
    ax_drift.tick_params(axis="both", pad=4)
    for label in ax_drift.get_xticklabels() + ax_drift.get_yticklabels():
        label.set_fontweight("semibold")

    norm_values = [stats[method]["norm_ratio"] for method in methods]
    ax_norm.bar(xs, norm_values, color=colors)
    ax_norm.set_xticks(xs)
    ax_norm.set_xticklabels([LABELS[method] for method in methods])
    ax_norm.set_ylabel("Task vector / hidden norm")
    ax_norm.set_title("Write magnitude")
    ax_norm.grid(alpha=0.25, axis="y")
    ax_norm.tick_params(axis="both", pad=4)
    for label in ax_norm.get_xticklabels() + ax_norm.get_yticklabels():
        label.set_fontweight("semibold")

    max_norm = max(max(norm_values), 1e-6)
    for method in methods:
        x = stats[method]["drift"]
        y = stats[method]["accuracy"]
        size = 90 + 360 * (stats[method]["norm_ratio"] / max_norm)
        ax_scatter.scatter([x], [y], s=size, color=COLORS[method], alpha=0.9, edgecolors="black", linewidths=0.4)
        ax_scatter.text(x + 0.00045, y, LABELS[method], fontsize=14, va="center")
    ax_scatter.set_xlim(-0.0005, max(0.022, max(drift_values) * 1.18))
    ax_scatter.set_ylim(0.55, 1.02)
    ax_scatter.set_xlabel("Median query hidden drift")
    ax_scatter.set_ylabel("Accuracy")
    ax_scatter.set_title("Accuracy vs interference")
    ax_scatter.grid(alpha=0.25)
    ax_scatter.tick_params(axis="both", pad=4)
    for label in ax_scatter.get_xticklabels() + ax_scatter.get_yticklabels():
        label.set_fontweight("semibold")

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_output_root = Path(args.source_output_root).expanduser().resolve()
    source_manifest = load_json(source_output_root / "manifest.json")
    write_stats = load_write_stats(source_output_root)

    paper_root = PROJECT_ROOT / "swap" / "paper"
    output_root = paper_root / "outputs" / f"{args.timestamp}_{args.suite_name}"
    log_root = paper_root / "logs" / f"{args.timestamp}_{args.suite_name}"
    figure_root = paper_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    if args.hire_metrics:
        hire_metrics_path = Path(args.hire_metrics).expanduser().resolve()
        hire_log_path = Path(args.hire_log).expanduser().resolve() if args.hire_log else None
    else:
        hire_metrics_path, hire_log_path = run_hire_eval(
            manifest=source_manifest,
            hire_method=args.hire_method,
            output_root=output_root,
            log_root=log_root,
            timestamp=args.timestamp,
            runner_python=args.runner_python,
        )
    hire_metrics = load_json(hire_metrics_path)
    write_stats[READ_METHOD] = {
        "accuracy": float(((hire_metrics.get("metrics") or {}).get("accuracy")) or 0.0),
        "drift": 0.0,
        "norm_ratio": 0.0,
    }

    figure_path = figure_root / f"{args.suite_name}_{source_manifest['dataset_name']}_{args.timestamp}.png"
    save_figure(figure_path, write_stats)

    manifest = {
        "source_output_root": str(source_output_root),
        "common_subset": str(source_manifest["analysis_val_path"]),
        "hire_method": args.hire_method,
        "hire_metrics": str(hire_metrics_path),
        "hire_log": (str(hire_log_path) if hire_log_path is not None else None),
        "figure_path": str(figure_path),
        "stats": write_stats,
    }
    save_json(output_root / "hire_advantage_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
