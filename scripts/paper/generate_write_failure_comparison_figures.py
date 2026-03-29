#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from swap.paper.scripts.analyze_write_failure import compute_attention_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate comparison figures for write-failure analysis.")
    parser.add_argument("--source-output-root", required=True, help="Original full write-failure output root.")
    parser.add_argument("--targeted-selection", required=True, help="Targeted replay selection manifest.")
    parser.add_argument("--timestamp", default=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--methods", default="stv,i2cl,mimic")
    parser.add_argument(
        "--suite-name",
        default="write_failure_comparison",
        help="Prefix used for generated figure filenames.",
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


def resolve_targeted_diagnostics(targeted_output_root: Path, method_name: str) -> Path:
    matches = sorted(targeted_output_root.glob(f"*_{method_name}_*.diagnostics.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one targeted diagnostics file for method={method_name}, found {len(matches)}"
        )
    return matches[0]


def load_raw_samples(diagnostics_path: Path) -> list[dict[str, Any]]:
    diagnostics = load_json(diagnostics_path)
    analysis = diagnostics.get("write_failure_analysis") or {}
    return list(analysis.get("raw_samples") or [])


def load_raw_pair(raw_sample: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    raw_paths = raw_sample.get("raw_bundle_paths") or {}
    meta = load_json(Path(raw_paths["meta"]))
    normal = torch.load(raw_paths["normal"], map_location="cpu")
    steered = torch.load(raw_paths["steered"], map_location="cpu")
    return meta, normal, steered


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


def compute_layerwise_hidden_drift(raw_sample: dict[str, Any]) -> np.ndarray:
    _meta, normal, steered = load_raw_pair(raw_sample)
    hidden_normal = normal["hidden_states"].to(dtype=torch.float32)
    hidden_steered = steered["hidden_states"].to(dtype=torch.float32)
    image_mask, text_mask, query_mask = build_masks(normal)

    delta = (hidden_steered - hidden_normal).norm(p=2, dim=-1)
    base = hidden_normal.norm(p=2, dim=-1).clamp_min(1e-8)
    ratio = delta / base

    image_curve = safe_group_ratio(ratio, image_mask)
    text_curve = safe_group_ratio(ratio, text_mask)
    query_curve = safe_group_ratio(ratio, query_mask)
    return torch.stack([image_curve, text_curve, query_curve], dim=0).cpu().numpy()


def mean_curves(curves: list[np.ndarray]) -> np.ndarray:
    return np.stack(curves, axis=0).mean(axis=0)


def save_hidden_drift_figure(
    targeted_diagnostics: dict[str, Path],
    figure_path: Path,
    *,
    dataset_label: str,
) -> None:
    methods = list(targeted_diagnostics.keys())
    per_method = {}
    vmax = 0.0
    for method in methods:
        curves = [compute_layerwise_hidden_drift(sample) for sample in load_raw_samples(targeted_diagnostics[method])]
        mean_curve = mean_curves(curves)
        per_method[method] = mean_curve
        vmax = max(vmax, float(mean_curve.max()))

    fig, axes = plt.subplots(1, len(methods), figsize=(5.6 * len(methods), 3.9), dpi=180)
    if len(methods) == 1:
        axes = [axes]
    row_labels = ["Image", "Text", "Query"]
    for ax, method in zip(axes, methods):
        curve = per_method[method]
        im = ax.imshow(curve, aspect="auto", cmap="magma", vmin=0.0, vmax=max(vmax, 1e-6))
        ax.set_title(method.upper())
        ax.set_xlabel("Layer")
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels)
        ax.set_xticks(np.linspace(0, curve.shape[1] - 1, 5, dtype=int))
        for row in range(curve.shape[0]):
            peak = int(np.argmax(curve[row]))
            ax.text(
                peak,
                row,
                f"{curve[row, peak]:.3f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                bbox={"facecolor": "black", "alpha": 0.25, "pad": 1.0, "edgecolor": "none"},
            )

    fig.colorbar(im, ax=axes, fraction=0.024, pad=0.02, label="Relative hidden drift (L2 / base norm)")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def cumulative_concentration_curve(profile: np.ndarray, points: int = 200) -> np.ndarray:
    sorted_profile = np.sort(np.asarray(profile, dtype=np.float32))[::-1]
    sorted_profile = np.maximum(sorted_profile, 0.0)
    total = float(sorted_profile.sum())
    if total <= 0.0:
        xs = np.linspace(0.0, 1.0, points)
        return np.clip(xs, 0.0, 1.0)
    cumulative = np.cumsum(sorted_profile) / total
    src_x = np.linspace(1.0 / len(sorted_profile), 1.0, len(sorted_profile))
    dst_x = np.linspace(0.0, 1.0, points)
    return np.interp(dst_x, np.concatenate([[0.0], src_x]), np.concatenate([[0.0], cumulative]))


def sample_attention_profiles(raw_sample: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    _meta, normal, steered = load_raw_pair(raw_sample)
    attentions_normal = normal["attentions"].to(dtype=torch.float32)
    attentions_steered = steered["attentions"].to(dtype=torch.float32)
    image_indices = list(normal.get("image_token_indices") or [])
    image_mask = torch.zeros(int(attentions_normal.shape[-1]), dtype=torch.bool)
    for idx in image_indices:
        if 0 <= int(idx) < int(attentions_normal.shape[-1]):
            image_mask[int(idx)] = True
    metrics = compute_attention_metrics(
        attentions_normal,
        attentions_steered,
        image_mask=image_mask,
        query_indices=list(range(int(attentions_normal.shape[-2]))),
    )
    return (
        np.asarray(metrics["image_attention_profile_normal"], dtype=np.float32),
        np.asarray(metrics["image_attention_profile_steered"], dtype=np.float32),
    )


def save_attention_concentration_figure(
    targeted_diagnostics: dict[str, Path],
    figure_path: Path,
    *,
    dataset_label: str,
) -> None:
    methods = list(targeted_diagnostics.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(5.6 * len(methods), 3.8), dpi=180)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        normal_curves = []
        steered_curves = []
        for raw_sample in load_raw_samples(targeted_diagnostics[method]):
            profile_normal, profile_steered = sample_attention_profiles(raw_sample)
            normal_curves.append(cumulative_concentration_curve(profile_normal))
            steered_curves.append(cumulative_concentration_curve(profile_steered))

        normal_arr = np.stack(normal_curves, axis=0)
        steered_arr = np.stack(steered_curves, axis=0)
        xs = np.linspace(0.0, 1.0, normal_arr.shape[1])
        uniform = xs

        ax.plot(xs, normal_arr.mean(axis=0), color="#1f77b4", linewidth=2.2, label="normal")
        ax.fill_between(
            xs,
            normal_arr.mean(axis=0) - normal_arr.std(axis=0),
            normal_arr.mean(axis=0) + normal_arr.std(axis=0),
            color="#1f77b4",
            alpha=0.18,
        )
        ax.plot(xs, steered_arr.mean(axis=0), color="#d62728", linewidth=2.2, label="steered")
        ax.fill_between(
            xs,
            steered_arr.mean(axis=0) - steered_arr.std(axis=0),
            steered_arr.mean(axis=0) + steered_arr.std(axis=0),
            color="#d62728",
            alpha=0.18,
        )
        ax.plot(xs, uniform, color="#7f7f7f", linewidth=1.5, linestyle="--", label="uniform")
        ax.set_title(method.upper())
        ax.set_xlabel("Fraction of most-attended image patches")
        ax.set_ylabel("Cumulative attention mass")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.02)
        ax.grid(alpha=0.25)
        if method == methods[-1]:
            ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_norm_mismatch_figure(
    source_diagnostics: dict[str, Path],
    figure_path: Path,
    *,
    dataset_label: str,
) -> None:
    methods = list(source_diagnostics.keys())
    values = []
    for method in methods:
        diagnostics = load_json(source_diagnostics[method])
        rows = ((diagnostics.get("write_failure_analysis") or {}).get("samples") or [])
        method_values = [
            float(row["task_vector_to_hidden_norm_ratio"])
            for row in rows
            if row.get("task_vector_to_hidden_norm_ratio") is not None
        ]
        values.append(method_values)

    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=180)
    violin = ax.violinplot(values, showmeans=False, showmedians=True, widths=0.8)
    colors = ["#4c78a8", "#f58518", "#54a24b"]
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.35)
    violin["cmedians"].set_color("black")
    violin["cmedians"].set_linewidth(1.6)

    rng = np.random.default_rng(42)
    for idx, (method, method_values, color) in enumerate(zip(methods, values, colors), start=1):
        xs = np.full(len(method_values), idx, dtype=np.float32)
        xs += rng.uniform(-0.08, 0.08, size=len(method_values))
        ax.scatter(xs, method_values, s=14, alpha=0.45, color=color, edgecolors="none")
        ax.text(idx, max(method_values) * 1.15, f"median={np.median(method_values):.3g}", ha="center", fontsize=8)

    ax.set_xticks(range(1, len(methods) + 1))
    ax.set_xticklabels([method.upper() for method in methods])
    ax.set_yscale("log")
    ax.set_ylabel("Task vector / hidden norm ratio")
    ax.set_title("Norm mismatch distribution")
    ax.grid(alpha=0.25, axis="y")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    source_output_root = Path(args.source_output_root).expanduser().resolve()
    targeted_selection_path = Path(args.targeted_selection).expanduser().resolve()
    targeted_selection = load_json(targeted_selection_path)
    targeted_output_root = targeted_selection_path.parent

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    dataset_label = str(targeted_selection["dataset_label"])
    dataset_name = str(targeted_selection["dataset_name"])

    source_diagnostics = {method: resolve_source_diagnostics(source_output_root, method) for method in methods}
    targeted_diagnostics = {method: resolve_targeted_diagnostics(targeted_output_root, method) for method in methods}

    figure_root = PROJECT_ROOT / "swap" / "paper" / "figures"
    output_root = PROJECT_ROOT / "swap" / "paper" / "outputs" / f"{args.timestamp}_{args.suite_name}"
    output_root.mkdir(parents=True, exist_ok=True)

    paths = {
        "hidden_drift": figure_root / f"{args.suite_name}_{dataset_name}_hidden_drift_{args.timestamp}.png",
        "norm_mismatch": figure_root / f"{args.suite_name}_{dataset_name}_norm_mismatch_{args.timestamp}.png",
        "attention_concentration": figure_root
        / f"{args.suite_name}_{dataset_name}_attention_concentration_{args.timestamp}.png",
    }

    save_hidden_drift_figure(targeted_diagnostics, paths["hidden_drift"], dataset_label=dataset_label)
    save_norm_mismatch_figure(source_diagnostics, paths["norm_mismatch"], dataset_label=dataset_label)
    save_attention_concentration_figure(
        targeted_diagnostics,
        paths["attention_concentration"],
        dataset_label=dataset_label,
    )

    manifest = {
        "source_output_root": str(source_output_root),
        "targeted_selection": str(targeted_selection_path),
        "dataset_name": dataset_name,
        "dataset_label": dataset_label,
        "methods": methods,
        "figures": {name: str(path) for name, path in paths.items()},
    }
    save_json(output_root / "comparison_figures_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
