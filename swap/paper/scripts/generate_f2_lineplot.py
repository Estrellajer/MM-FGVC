"""Generate a line-plot alternative to the Figure 2 heatmap."""

from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT / "outputs" / "20260325_095501_c1_main_results"
OUTPUT_DIR = ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL = "qwen2_vl"
SEED = "seed42"

TASKS = {
    "flowers_small": {"label": "Flowers-102 (FGVC)"},
    "naturalbench_vqa": {"label": "NaturalBench (VQA)"},
}

LEVELS = ["head", "attn", "mlp", "layer"]
LEVEL_LABELS = {"head": "Head", "attn": "Attn", "mlp": "MLP", "layer": "Layer"}
LEVEL_COLORS = {
    "head": "#1f77b4",
    "attn": "#d62728",
    "mlp": "#2ca02c",
    "layer": "#9467bd",
}
TASK_COLORS = {
    "flowers_small": "#f58518",
    "naturalbench_vqa": "#72b7b2",
}
TASK_SHORT_LABELS = {
    "flowers_small": "Flowers",
    "naturalbench_vqa": "NB-VQA",
}


def find_run_file(task_name: str, method: str, suffix: str) -> Path:
    pattern = str(
        BASE_DIR / f"c1_main_results_{task_name}_{SEED}_{method}_{MODEL}_*.{suffix}.json"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file found for pattern: {pattern}")
    return Path(matches[0])


def load_json(path: Path) -> dict:
    with path.open("r") as f:
        return json.load(f)


def primary_accuracy(metrics_path: Path) -> float:
    metrics = load_json(metrics_path)["metrics"]
    if "accuracy" in metrics:
        return float(metrics["accuracy"])
    if "pair_accuracy" in metrics:
        return float(metrics["pair_accuracy"])
    if "g_acc" in metrics:
        return float(metrics["g_acc"])
    raise KeyError(f"Could not infer primary accuracy from {metrics_path}")


def extract_matrix(diag: dict) -> np.ndarray:
    num_layers = diag["num_layers"]
    matrix = np.full((len(LEVELS), num_layers), np.nan)
    for comp in diag["selected_components"]:
        level = comp["level"]
        layer = comp["layer_idx"]
        val_accuracy = comp.get("val_accuracy")
        if level in LEVELS and val_accuracy is not None:
            matrix[LEVELS.index(level), layer] = val_accuracy
    return matrix


def load_task_data(task_name: str) -> dict:
    diag = load_json(find_run_file(task_name, "rsev2", "diagnostics"))
    zero_shot = primary_accuracy(find_run_file(task_name, "zero_shot", "metrics"))
    matrix = extract_matrix(diag)
    best = diag["best_component_by_val"]
    oracle = float(diag["oracle_summary"]["oracle_accuracy"])
    return {
        "matrix": matrix,
        "num_layers": diag["num_layers"],
        "zero_shot": zero_shot,
        "oracle": oracle,
        "best_level": best["level"],
        "best_layer": best["layer_idx"],
        "best_value": float(best["val_accuracy"]),
    }


def plot_task(
    ax: plt.Axes,
    label: str,
    data: dict,
    *,
    show_legend: bool = False,
    stats_xy: tuple[float, float] = (0.02, 0.04),
    show_xlabel: bool = True,
    show_ylabel: bool = True,
) -> None:
    matrix = data["matrix"]
    x = np.arange(data["num_layers"])

    for row, level in enumerate(LEVELS):
        ax.plot(
            x,
            matrix[row],
            color=LEVEL_COLORS[level],
            linewidth=2.4,
            marker="o",
            markersize=3.2,
            alpha=0.92,
            label=LEVEL_LABELS[level],
        )

    ax.axhline(
        data["zero_shot"],
        color="#555555",
        linewidth=1.8,
        linestyle="--",
    )
    ax.axhline(
        data["oracle"],
        color="#111111",
        linewidth=1.6,
        linestyle=":",
    )

    best_row = LEVELS.index(data["best_level"])
    best_x = data["best_layer"]
    best_y = matrix[best_row, best_x]
    ax.scatter(
        [best_x],
        [best_y],
        s=210,
        marker="*",
        color="gold",
        edgecolors="black",
        linewidths=0.9,
        zorder=10,
    )
    text_x = best_x
    text_y = best_y + 0.055
    if best_y > 0.93:
        text_y = best_y - 0.075
    text_x = min(max(text_x, 0.9), data["num_layers"] - 1.1)
    ax.annotate(
        f"{LEVEL_LABELS[data['best_level']]}-{best_x}\n{best_y:.2f}",
        xy=(best_x, best_y),
        xytext=(text_x, text_y),
        textcoords="data",
        ha="center",
        va="bottom" if text_y >= best_y else "top",
        fontsize=10.0,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "alpha": 0.95,
            "edgecolor": "none",
        },
        arrowprops={
            "arrowstyle": "-",
            "color": "#666666",
            "lw": 0.9,
            "shrinkA": 2,
            "shrinkB": 5,
        },
    )

    ax.set_title(label, fontsize=15.5, fontweight="bold", pad=10)
    ax.set_xlim(-0.5, data["num_layers"] - 0.5)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks(list(range(0, data["num_layers"], 4)))
    ax.set_xlabel(
        "Layer Index" if show_xlabel else "",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.set_ylabel(
        "Val Accuracy" if show_ylabel else "",
        fontsize=12.5,
        fontweight="bold",
    )
    ax.grid(axis="y", linestyle=":", linewidth=0.9, alpha=0.5)
    ax.tick_params(labelsize=11.5)
    ax.text(
        stats_xy[0],
        stats_xy[1],
        f"ZS={data['zero_shot']:.2f}  Oracle={data['oracle']:.2f}",
        transform=ax.transAxes,
        fontsize=10.8,
        fontweight="bold",
        color="#333333",
        bbox={
            "boxstyle": "round,pad=0.24",
            "facecolor": "white",
            "alpha": 0.82,
            "edgecolor": "none",
        },
    )

    if show_legend:
        legend_handles = [
            Line2D([0], [0], color=LEVEL_COLORS[level], lw=2.0, marker="o", markersize=4)
            for level in LEVELS
        ]
        legend_labels = [LEVEL_LABELS[level] for level in LEVELS]
        legend_handles.extend(
            [
                Line2D([0], [0], color="#555555", lw=1.5, linestyle="--"),
                Line2D([0], [0], color="#111111", lw=1.3, linestyle=":"),
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="gold",
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=0.8,
                    linewidth=0,
                ),
            ]
        )
        legend_labels.extend(["Zero-shot", "Oracle", "Peak"])
        ax.legend(
            legend_handles,
            legend_labels,
            ncol=3,
            loc="lower right",
            bbox_to_anchor=(0.99, 0.03),
            fontsize=10.8,
            frameon=True,
            facecolor="white",
            edgecolor="#dddddd",
            framealpha=0.95,
            handlelength=2.4,
            columnspacing=1.0,
            borderpad=0.5,
            labelspacing=0.45,
        )


def main() -> None:
    task_data = {task: load_task_data(task) for task in TASKS}

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.4), sharex=True)
    axes = np.atleast_1d(axes)

    for idx, (ax, (task_name, task_info)) in enumerate(zip(axes, TASKS.items())):
        plot_task(
            ax,
            task_info["label"],
            task_data[task_name],
            show_legend=(task_name == "flowers_small"),
            stats_xy=(0.02, 0.05),
            show_xlabel=(idx == len(TASKS) - 1),
            show_ylabel=True,
        )

    fig.tight_layout(pad=0.8, h_pad=0.8)

    out_png = OUTPUT_DIR / "f2_lineplot.png"
    out_svg = OUTPUT_DIR / "f2_lineplot.svg"
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    fig.savefig(out_svg, bbox_inches="tight", facecolor="white", format="svg")
    plt.close(fig)

    compact_fig, compact_ax = plt.subplots(figsize=(11.2, 4.6))
    for task_name in TASKS:
        data = task_data[task_name]
        layer_best = np.nanmax(data["matrix"], axis=0)
        gains = layer_best - data["zero_shot"]
        x = np.arange(data["num_layers"])

        compact_ax.plot(
            x,
            gains,
            color=TASK_COLORS[task_name],
            linewidth=2.1,
            label=TASK_SHORT_LABELS[task_name],
        )

        peak_x = data["best_layer"]
        peak_y = data["best_value"] - data["zero_shot"]
        compact_ax.scatter(
            [peak_x],
            [peak_y],
            color=TASK_COLORS[task_name],
            s=52,
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
        )
        compact_ax.annotate(
            f"{LEVEL_LABELS[data['best_level']][0]}{peak_x}",
            xy=(peak_x, peak_y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=TASK_COLORS[task_name],
            fontweight="bold",
        )

    compact_ax.axhline(0.0, color="#444444", linestyle="--", linewidth=1.2)
    compact_ax.set_xlim(-0.5, max(data["num_layers"] for data in task_data.values()) - 0.5)
    compact_ax.set_xticks(list(range(0, max(data["num_layers"] for data in task_data.values()), 4)))
    compact_ax.set_xlabel("Layer Index", fontsize=10)
    compact_ax.set_ylabel("Best Component - Zero-shot", fontsize=10)
    compact_ax.set_title(
        "Representation-Generation Gap by Layer",
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    compact_ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.45)
    compact_ax.legend(
        ncol=3,
        loc="upper left",
        frameon=False,
        fontsize=9,
        handlelength=2.5,
    )
    compact_ax.text(
        0.99,
        0.03,
        "Positive values mean some representation component\noutperforms zero-shot generation at that layer.",
        transform=compact_ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    compact_fig.tight_layout()

    compact_png = OUTPUT_DIR / "f2_gainline.png"
    compact_svg = OUTPUT_DIR / "f2_gainline.svg"
    compact_fig.savefig(compact_png, dpi=160, bbox_inches="tight", facecolor="white")
    compact_fig.savefig(compact_svg, bbox_inches="tight", facecolor="white", format="svg")
    plt.close(compact_fig)

    print(f"Saved {out_png}")
    print(f"Saved {out_svg}")
    print(f"Saved {compact_png}")
    print(f"Saved {compact_svg}")


if __name__ == "__main__":
    main()
