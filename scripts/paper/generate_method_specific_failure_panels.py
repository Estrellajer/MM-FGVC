#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper.run_write_failure_cub import build_grids, load_heatmap_payload, overlay_heatmap, resize_grid


METHODS = ["stv", "i2cl", "mimic"]
METHOD_LABELS = {"stv": "STV", "i2cl": "I2CL", "mimic": "MimIC"}
METHOD_COLORS = {"stv": "#4c78a8", "i2cl": "#f58518", "mimic": "#54a24b"}
NEUTRAL = "#d0d0d0"
ANNOTATION_SIZE = 16

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
    parser = argparse.ArgumentParser(description="Generate method-specific failure panels for write-failure analysis.")
    parser.add_argument("--source-output-root", required=True)
    parser.add_argument("--targeted-selection", required=True)
    parser.add_argument("--timestamp", default=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--suite-name", default="write_failure_method_panels")
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


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=np.float32)))


def method_order_labels() -> list[str]:
    return [METHOD_LABELS[method] for method in METHODS]


def load_source_data(source_output_root: Path) -> dict[str, dict[str, Any]]:
    payload: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        diagnostics = load_json(resolve_source_diagnostics(source_output_root, method))
        analysis = diagnostics.get("write_failure_analysis") or {}
        summary = analysis.get("summary") or {}
        rows = list(analysis.get("samples") or [])
        same_prediction_rate = sum(
            1 for row in rows if row.get("normal_prediction") == row.get("steered_prediction")
        ) / max(len(rows), 1)
        payload[method] = {
            "rows": rows,
            "normal_accuracy": float(summary.get("normal_accuracy") or 0.0),
            "steered_accuracy": float(summary.get("steered_accuracy") or 0.0),
            "accuracy_delta": float(summary.get("steered_accuracy") or 0.0) - float(summary.get("normal_accuracy") or 0.0),
            "same_prediction_rate": same_prediction_rate,
            "median_norm_ratio": median(
                [float(row["task_vector_to_hidden_norm_ratio"]) for row in rows if row.get("task_vector_to_hidden_norm_ratio") is not None]
            ),
            "median_query_drift": median(
                [float(row["query_hidden_l2_ratio"]) for row in rows if row.get("query_hidden_l2_ratio") is not None]
            ),
            "median_visual_drop": median(
                [float(row["visual_attention_ratio_drop_percent"]) for row in rows if row.get("visual_attention_ratio_drop_percent") is not None]
            ),
            "median_cosine": median(
                [float(row["representation_cosine_similarity"]) for row in rows if row.get("representation_cosine_similarity") is not None]
            ),
        }
    return payload


def values_for(source_data: dict[str, dict[str, Any]], method: str, key: str) -> np.ndarray:
    rows = source_data[method]["rows"]
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return np.asarray(values, dtype=np.float32)


def load_targeted_replay_info(selection_path: Path) -> tuple[str, dict[str, Any]]:
    selection = load_json(selection_path)
    dataset_label = str(selection["dataset_label"])
    replays = selection["replays"]
    return dataset_label, replays


def build_masks(bundle: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    seq_len = int(bundle["hidden_states"].shape[-2])
    image_mask = torch.zeros(seq_len, dtype=torch.bool)
    for idx in bundle.get("image_token_indices", []) or []:
        if 0 <= int(idx) < seq_len:
            image_mask[int(idx)] = True

    query_mask = torch.zeros(seq_len, dtype=torch.bool)
    for idx in bundle.get("query_token_indices", []) or []:
        if 0 <= int(idx) < seq_len:
            query_mask[int(idx)] = True

    text_mask = (~image_mask) & (~query_mask)
    return image_mask, text_mask, query_mask


def safe_group_ratio(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not mask.any():
        return torch.zeros(values.shape[0], dtype=torch.float32)
    return values[..., mask].mean(dim=tuple(range(1, values[..., mask].ndim)))


def mimic_hidden_drift_curve(targeted_diagnostics: Path) -> np.ndarray:
    diagnostics = load_json(targeted_diagnostics)
    analysis = diagnostics.get("write_failure_analysis") or {}
    curves = []
    for sample in analysis.get("raw_samples") or []:
        raw_paths = sample.get("raw_bundle_paths") or {}
        normal = torch.load(raw_paths["normal"], map_location="cpu")
        steered = torch.load(raw_paths["steered"], map_location="cpu")
        hidden_normal = normal["hidden_states"].to(dtype=torch.float32)
        hidden_steered = steered["hidden_states"].to(dtype=torch.float32)
        image_mask, text_mask, query_mask = build_masks(normal)
        delta = (hidden_steered - hidden_normal).norm(p=2, dim=-1)
        base = hidden_normal.norm(p=2, dim=-1).clamp_min(1e-8)
        ratio = delta / base
        curve = torch.stack(
            [
                safe_group_ratio(ratio, image_mask),
                safe_group_ratio(ratio, text_mask),
                safe_group_ratio(ratio, query_mask),
            ],
            dim=0,
        )
        curves.append(curve.cpu().numpy())
    return np.stack(curves, axis=0).mean(axis=0)


def resample_curve(values: np.ndarray, size: int = 128) -> np.ndarray:
    if values.size == size:
        return values.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, size)
    return np.interp(x_new, x_old, values).astype(np.float32)


def mimic_attention_concentration(targeted_diagnostics: Path, curve_size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    diagnostics = load_json(targeted_diagnostics)
    analysis = diagnostics.get("write_failure_analysis") or {}
    normal_curves = []
    steered_curves = []
    for sample in analysis.get("raw_samples") or []:
        raw_bundle_paths = sample.get("raw_bundle_paths") or {}
        meta_path = raw_bundle_paths.get("meta")
        if not meta_path:
            continue
        _meta, _bundle, metrics = load_heatmap_payload(Path(meta_path))
        attention = metrics["attention"]
        normal = np.asarray(attention["image_attention_profile_normal"], dtype=np.float32)
        steered = np.asarray(attention["image_attention_profile_steered"], dtype=np.float32)
        normal = normal / max(float(normal.sum()), 1e-8)
        steered = steered / max(float(steered.sum()), 1e-8)
        normal = np.sort(normal)[::-1]
        steered = np.sort(steered)[::-1]
        normal_curves.append(resample_curve(normal, size=curve_size))
        steered_curves.append(resample_curve(steered, size=curve_size))

    if not normal_curves:
        xs = np.linspace(0.0, 1.0, curve_size)
        zeros = np.zeros(curve_size, dtype=np.float32)
        return xs, zeros, zeros

    xs = np.linspace(0.0, 1.0, curve_size)
    return xs, np.mean(normal_curves, axis=0), np.mean(steered_curves, axis=0)


def strongest_sample(targeted_diagnostics: Path) -> dict[str, Any] | None:
    diagnostics = load_json(targeted_diagnostics)
    analysis = diagnostics.get("write_failure_analysis") or {}
    raw_samples = list(analysis.get("raw_samples") or [])
    if not raw_samples:
        return None

    def score(row: dict[str, Any]) -> float:
        drop = abs(float(row.get("visual_attention_ratio_drop_percent") or 0.0))
        entropy_delta = abs(
            float(row.get("normalized_attention_entropy_steered") or 0.0)
            - float(row.get("normalized_attention_entropy_normal") or 0.0)
        )
        cosine = float(row.get("representation_cosine_similarity") or 1.0)
        l2 = float(row.get("query_hidden_l2_ratio") or 0.0)
        return drop + 25.0 * entropy_delta + 100.0 * (1.0 - cosine) + 10.0 * l2

    return max(raw_samples, key=score)


def build_delta_overlay(sample: dict[str, Any]) -> tuple[np.ndarray, str]:
    meta_path = Path(sample["raw_bundle_paths"]["meta"])
    meta, _bundle, metrics = load_heatmap_payload(meta_path)
    image = Image.open(meta["image"]).convert("RGB")
    _normal_grid, _steered_grid, delta_grid = build_grids(meta=meta, metrics=metrics)
    overlay = overlay_heatmap(
        image,
        resize_grid(delta_grid, image.size),
        cmap_name="coolwarm",
        alpha=0.62,
        signed=True,
    )
    title = (
        f"Target: {sample.get('label', '')}\n"
        f"drop = {float(sample.get('visual_attention_ratio_drop_percent') or 0.0):.2f}%"
    )
    return overlay, title


def plot_bar(ax, values: list[float], *, highlight: str, title: str, ylabel: str, log_scale: bool = False) -> None:
    xs = np.arange(len(METHODS))
    colors = [METHOD_COLORS[method] if method == highlight else NEUTRAL for method in METHODS]
    ax.bar(xs, values, color=colors)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(method_order_labels())
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, axis="y")
    ax.tick_params(axis="both", pad=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("semibold")


def plot_accuracy_delta(ax, source_data: dict[str, dict[str, Any]], *, highlight: str) -> None:
    values = [source_data[method]["accuracy_delta"] for method in METHODS]
    xs = np.arange(len(METHODS))
    colors = [METHOD_COLORS[method] if method == highlight else NEUTRAL for method in METHODS]
    ax.bar(xs, values, color=colors)
    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(xs)
    ax.set_xticklabels(method_order_labels())
    ax.set_title("Accuracy change after write")
    ax.set_ylabel("steered acc - normal acc")
    ax.grid(alpha=0.25, axis="y")
    ax.tick_params(axis="both", pad=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("semibold")


def plot_box_strip(
    ax,
    source_data: dict[str, dict[str, Any]],
    *,
    metric_key: str,
    title: str,
    ylabel: str,
    highlight: str,
    log_scale: bool = False,
) -> None:
    series = [values_for(source_data, method, metric_key) for method in METHODS]
    box = ax.boxplot(series, positions=np.arange(len(METHODS)), widths=0.55, patch_artist=True, showfliers=False)
    for idx, patch in enumerate(box["boxes"]):
        method = METHODS[idx]
        patch.set_facecolor(METHOD_COLORS[method] if method == highlight else NEUTRAL)
        patch.set_edgecolor("#555555")
    for element in ["whiskers", "caps", "medians"]:
        for artist in box[element]:
            artist.set_color("#555555")
            artist.set_linewidth(1.1)

    rng = np.random.default_rng(0)
    for idx, method in enumerate(METHODS):
        values = series[idx]
        if values.size == 0:
            continue
        jitter = rng.normal(loc=0.0, scale=0.04, size=values.size)
        ax.scatter(
            np.full(values.size, idx, dtype=np.float32) + jitter,
            values,
            s=18,
            alpha=0.40,
            color=METHOD_COLORS[method] if method == highlight else "#7f7f7f",
            edgecolors="none",
        )

    if log_scale:
        ax.set_yscale("log")
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels(method_order_labels())
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, axis="y")
    ax.tick_params(axis="both", pad=4)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("semibold")


def plot_scatter_norm_vs_drift(ax, source_data: dict[str, dict[str, Any]], *, highlight: str, title: str) -> None:
    for method in METHODS:
        xs = values_for(source_data, method, "task_vector_to_hidden_norm_ratio")
        ys = values_for(source_data, method, "query_hidden_l2_ratio")
        alpha = 0.75 if method == highlight else 0.30
        size = 22 if method == highlight else 16
        ax.scatter(xs, ys, s=size, alpha=alpha, color=METHOD_COLORS[method], label=METHOD_LABELS[method])
        if xs.size and ys.size:
            med_x = float(np.median(xs))
            med_y = float(np.median(ys))
            ax.scatter([med_x], [med_y], marker="X", s=110, color=METHOD_COLORS[method], edgecolors="black", linewidths=0.4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("task vector / hidden norm")
    ax.set_ylabel("query hidden drift")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.tick_params(axis="both", pad=4)
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("semibold")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("semibold")


def save_stv_panel(figure_path: Path, *, dataset_label: str, source_data: dict[str, dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(14.5, 8.8), dpi=220)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_bar(
        ax1,
        [source_data[method]["median_norm_ratio"] for method in METHODS],
        highlight="stv",
        title="Median task vector / hidden norm",
        ylabel="ratio",
        log_scale=True,
    )
    plot_bar(
        ax2,
        [source_data[method]["median_query_drift"] for method in METHODS],
        highlight="stv",
        title="Median query hidden drift",
        ylabel="L2 ratio",
        log_scale=True,
    )
    plot_bar(
        ax3,
        [source_data[method]["same_prediction_rate"] for method in METHODS],
        highlight="stv",
        title="Same prediction rate",
        ylabel="fraction",
    )
    ax3.set_ylim(0.0, 1.0)
    plot_scatter_norm_vs_drift(
        ax4,
        source_data,
        highlight="stv",
        title="Per-sample write magnitude map",
    )

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_i2cl_panel(figure_path: Path, *, dataset_label: str, source_data: dict[str, dict[str, Any]]) -> None:
    fig = plt.figure(figsize=(14.5, 8.8), dpi=220)
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_box_strip(
        ax1,
        source_data,
        metric_key="task_vector_to_hidden_norm_ratio",
        title="Task vector / hidden norm distribution",
        ylabel="ratio",
        highlight="i2cl",
        log_scale=True,
    )
    plot_box_strip(
        ax2,
        source_data,
        metric_key="query_hidden_l2_ratio",
        title="Query hidden drift distribution",
        ylabel="L2 ratio",
        highlight="i2cl",
        log_scale=True,
    )
    plot_accuracy_delta(ax3, source_data, highlight="i2cl")
    plot_scatter_norm_vs_drift(
        ax4,
        source_data,
        highlight="i2cl",
        title="High-norm write also causes larger drift",
    )

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_mimic_panel(
    figure_path: Path,
    *,
    dataset_label: str,
    source_data: dict[str, dict[str, Any]],
    targeted_diagnostics: Path,
) -> None:
    fig = plt.figure(figsize=(14.5, 9.2), dpi=220)
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    plot_bar(
        ax1,
        [source_data[method]["median_query_drift"] for method in METHODS],
        highlight="mimic",
        title="Median query hidden drift",
        ylabel="L2 ratio",
        log_scale=True,
    )

    heatmap = mimic_hidden_drift_curve(targeted_diagnostics)
    vmax = max(float(np.percentile(heatmap, 98.0)), 1e-6)
    image = ax2.imshow(heatmap, aspect="auto", cmap="magma", vmin=0.0, vmax=vmax)
    ax2.set_title("Layer-wise hidden drift")
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels(["Image", "Text", "Query"])
    ax2.set_xticks(np.linspace(0, heatmap.shape[1] - 1, 5, dtype=int))
    colorbar = fig.colorbar(image, ax=ax2, fraction=0.046, pad=0.04)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontweight("semibold")
    for label in colorbar.ax.get_yticklabels():
        label.set_fontweight("semibold")

    xs, normal_curve, steered_curve = mimic_attention_concentration(targeted_diagnostics)
    uniform = np.full_like(xs, 1.0 / max(xs.size, 1), dtype=np.float32)
    ax3.plot(xs, normal_curve, linewidth=2.0, color="#1f77b4", label="normal")
    ax3.plot(xs, steered_curve, linewidth=2.0, color="#d62728", label="steered")
    ax3.plot(xs, uniform, linewidth=1.5, linestyle="--", color="#7f7f7f", label="uniform")
    ax3.set_title("Patch-normalized attention concentration")
    ax3.set_xlabel("Sorted image patches")
    ax3.set_ylabel("attention mass")
    ax3.grid(alpha=0.25)
    ax3.legend(frameon=False, fontsize=13, loc="upper right")
    legend = ax3.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("semibold")
    for label in ax3.get_xticklabels() + ax3.get_yticklabels():
        label.set_fontweight("semibold")

    sample = strongest_sample(targeted_diagnostics)
    if sample is not None:
        overlay, overlay_title = build_delta_overlay(sample)
        ax4.imshow(overlay)
        ax4.text(
            0.5,
            1.04,
            overlay_title,
            transform=ax4.transAxes,
            ha="center",
            va="bottom",
            fontsize=ANNOTATION_SIZE,
            fontweight="semibold",
        )
        ax4.text(
            0.5,
            -0.16,
            "delta overlay is contrast-rescaled for visibility",
            transform=ax4.transAxes,
            fontsize=ANNOTATION_SIZE,
            fontweight="semibold",
            ha="center",
            va="top",
        )
    ax4.axis("off")

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_output_root = Path(args.source_output_root).expanduser().resolve()
    targeted_selection_path = Path(args.targeted_selection).expanduser().resolve()
    dataset_label, targeted_replays = load_targeted_replay_info(targeted_selection_path)
    source_data = load_source_data(source_output_root)

    figure_root = PROJECT_ROOT / "swap" / "paper" / "figures"
    output_root = PROJECT_ROOT / "swap" / "paper" / "outputs" / f"{args.timestamp}_{args.suite_name}"
    output_root.mkdir(parents=True, exist_ok=True)

    panels = {
        "stv": figure_root / f"{args.suite_name}_stv_{args.timestamp}.png",
        "i2cl": figure_root / f"{args.suite_name}_i2cl_{args.timestamp}.png",
        "mimic": figure_root / f"{args.suite_name}_mimic_{args.timestamp}.png",
    }

    save_stv_panel(
        panels["stv"],
        dataset_label=dataset_label,
        source_data=source_data,
    )
    save_i2cl_panel(
        panels["i2cl"],
        dataset_label=dataset_label,
        source_data=source_data,
    )
    save_mimic_panel(
        panels["mimic"],
        dataset_label=dataset_label,
        source_data=source_data,
        targeted_diagnostics=Path(targeted_replays["mimic"]["replay_diagnostics"]),
    )

    manifest = {
        "source_output_root": str(source_output_root),
        "targeted_selection": str(targeted_selection_path),
        "dataset_label": dataset_label,
        "panels": {name: str(path) for name, path in panels.items()},
        "panel_order": METHODS,
    }
    save_json(output_root / "method_specific_panels_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
