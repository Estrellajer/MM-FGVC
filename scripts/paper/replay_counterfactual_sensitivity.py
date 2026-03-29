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
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper.run_write_failure_cub import load_json, method_overrides, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run counterfactual image sensitivity replays for write-failure methods.")
    parser.add_argument("--targeted-selection", required=True, help="Targeted replay selection manifest.")
    parser.add_argument("--timestamp", default=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--methods", default="stv,i2cl,mimic")
    parser.add_argument("--suite-name", default="write_failure_counterfactual")
    parser.add_argument(
        "--runner-python",
        default=str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        help="Python executable used for main.py subprocesses.",
    )
    parser.add_argument("--shuffle-grid", type=int, default=4)
    parser.add_argument("--occlusion-frac", type=float, default=0.42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def center_occlude(image: Image.Image, frac: float) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    occ_w = max(1, int(width * frac))
    occ_h = max(1, int(height * frac))
    left = (width - occ_w) // 2
    top = (height - occ_h) // 2
    occluded = image.copy()
    draw = ImageDraw.Draw(occluded)
    draw.rectangle([left, top, left + occ_w, top + occ_h], fill=(0, 0, 0))
    return occluded


def patch_shuffle(image: Image.Image, grid: int, seed: int) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    xs = np.linspace(0, width, grid + 1, dtype=int)
    ys = np.linspace(0, height, grid + 1, dtype=int)
    tiles = []
    boxes = []
    for row in range(grid):
        for col in range(grid):
            box = (xs[col], ys[row], xs[col + 1], ys[row + 1])
            boxes.append(box)
            tiles.append(image.crop(box))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(tiles))
    shuffled = Image.new("RGB", image.size)
    for dst_box, tile_idx in zip(boxes, order):
        tile = tiles[int(tile_idx)]
        dst_w = max(1, int(dst_box[2] - dst_box[0]))
        dst_h = max(1, int(dst_box[3] - dst_box[1]))
        if tile.size != (dst_w, dst_h):
            tile = tile.resize((dst_w, dst_h), Image.Resampling.BILINEAR)
        shuffled.paste(tile, dst_box)
    return shuffled


def choose_wrong_class_donor(
    sample: dict[str, Any],
    *,
    donor_pool: list[dict[str, Any]],
    offset: int,
) -> dict[str, Any]:
    label = str(sample["label"])
    candidates = [item for item in donor_pool if str(item["label"]) != label]
    if not candidates:
        raise ValueError(f"No wrong-class donor available for label={label}")
    return candidates[offset % len(candidates)]


def build_variant_subsets(
    *,
    method_name: str,
    base_subset: list[dict[str, Any]],
    donor_pool: list[dict[str, Any]],
    output_root: Path,
    shuffle_grid: int,
    occlusion_frac: float,
) -> dict[str, Path]:
    subset_root = output_root / "subsets" / method_name
    image_root = output_root / "images" / method_name
    subset_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    variants: dict[str, list[dict[str, Any]]] = {
        "occluded": [],
        "shuffled": [],
        "wrong_class": [],
    }

    for idx, sample in enumerate(base_subset):
        original_image = Image.open(sample["image"]).convert("RGB")
        qid = sample.get("question_id", idx)

        occluded_image = center_occlude(original_image, occlusion_frac)
        occluded_path = image_root / f"{qid}_occluded.png"
        occluded_image.save(occluded_path)

        shuffled_image = patch_shuffle(original_image, grid=shuffle_grid, seed=int(qid))
        shuffled_path = image_root / f"{qid}_shuffled.png"
        shuffled_image.save(shuffled_path)

        wrong_donor = choose_wrong_class_donor(sample, donor_pool=donor_pool, offset=idx + 1)
        wrong_path = str(wrong_donor["image"])

        for variant_name, image_path in (
            ("occluded", str(occluded_path)),
            ("shuffled", str(shuffled_path)),
            ("wrong_class", wrong_path),
        ):
            item = dict(sample)
            item["image"] = image_path
            if "images" in item and item["images"]:
                item["images"] = [image_path]
            variants[variant_name].append(item)

    paths: dict[str, Path] = {}
    for variant_name, items in variants.items():
        subset_path = subset_root / f"{variant_name}.json"
        save_json(subset_path, items)
        paths[variant_name] = subset_path
    return paths


def run_variant(
    *,
    method_name: str,
    dataset_name: str,
    model_name: str,
    train_path: Path,
    val_path: Path,
    run_name: str,
    output_root: Path,
    log_root: Path,
    seed: int,
    runner_python: str,
) -> dict[str, Path]:
    log_path = log_root / f"{run_name}.log"
    cmd = [
        runner_python,
        str(PROJECT_ROOT / "main.py"),
        f"model={model_name}",
        "dataset=general_custom",
        f"method={method_name}",
        "evaluator=raw",
        f"dataset.name={dataset_name}",
        f"dataset.train_path={train_path}",
        f"dataset.val_path={val_path}",
        f"run.output_dir={output_root}",
        f"run.run_name={run_name}",
        f"run.seed={seed}",
        "run.progress_bar=false",
        "method.params.progress_bar=false",
        "+model.model_args.attn_implementation=eager",
    ]
    cmd.extend(method_overrides(method_name))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(f"# command={' '.join(cmd)}\n\n")
        fp.flush()
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fp, stderr=subprocess.STDOUT, check=True)

    return {
        "metrics": output_root / f"{run_name}.metrics.json",
        "predictions": output_root / f"{run_name}.predictions.jsonl",
        "log": log_path,
    }


def compute_variant_summary(original_predictions_path: Path, variant_predictions_path: Path, metrics_path: Path) -> dict[str, Any]:
    original_rows = {str(row["question_id"]): row for row in load_jsonl(original_predictions_path)}
    variant_rows = {str(row["question_id"]): row for row in load_jsonl(variant_predictions_path)}
    common_ids = sorted(set(original_rows) & set(variant_rows))
    flips = 0
    label_kept = 0
    for qid in common_ids:
        if str(original_rows[qid]["prediction"]) != str(variant_rows[qid]["prediction"]):
            flips += 1
        if str(variant_rows[qid]["prediction"]) == str(original_rows[qid]["label"]):
            label_kept += 1
    metrics_payload = load_json(metrics_path)
    return {
        "accuracy": float(((metrics_payload.get("metrics") or {}).get("accuracy")) or 0.0),
        "prediction_flip_rate": flips / max(len(common_ids), 1),
        "label_retention_rate": label_kept / max(len(common_ids), 1),
        "num_samples": len(common_ids),
    }


def save_counterfactual_figure(results: dict[str, dict[str, dict[str, float]]], figure_path: Path, *, dataset_label: str) -> None:
    methods = list(results.keys())
    variants = ["original", "occluded", "shuffled", "wrong_class"]
    colors = {
        "original": "#4c78a8",
        "occluded": "#f58518",
        "shuffled": "#54a24b",
        "wrong_class": "#e45756",
    }
    labels = {
        "original": "Original",
        "occluded": "Center occlusion",
        "shuffled": "Patch shuffled",
        "wrong_class": "Wrong-class image",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.2), dpi=180)
    x = np.arange(len(methods), dtype=np.float32)
    width = 0.18

    for idx, variant in enumerate(variants):
        offset = (idx - 1.5) * width
        acc = [results[method][variant]["accuracy"] for method in methods]
        flips = [results[method][variant]["prediction_flip_rate"] for method in methods]
        axes[0].bar(x + offset, acc, width=width, color=colors[variant], label=labels[variant])
        axes[1].bar(x + offset, flips, width=width, color=colors[variant], label=labels[variant])

    for ax, title, ylabel in (
        (axes[0], "Exact-match accuracy", "Accuracy"),
        (axes[1], "Prediction flip rate vs original", "Flip rate"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels([method.upper() for method in methods])
        ax.set_ylim(0.0, 1.02)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y")

    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    selection_path = Path(args.targeted_selection).expanduser().resolve()
    selection = load_json(selection_path)
    source_output_root = Path(selection["source_output_root"])
    source_manifest = load_json(source_output_root / "manifest.json")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    dataset_name = str(selection["dataset_name"])
    dataset_label = str(selection["dataset_label"])
    model_name = str(selection["model"])
    train_path = Path(source_manifest["train_path"])
    source_analysis_val_path = Path(source_manifest["analysis_val_path"])
    donor_pool = load_json(source_analysis_val_path)
    seed = int(source_manifest["seed"])

    paper_root = PROJECT_ROOT / "swap" / "paper"
    output_root = paper_root / "outputs" / f"{args.timestamp}_{args.suite_name}"
    log_root = paper_root / "logs" / f"{args.timestamp}_{args.suite_name}"
    figure_root = paper_root / "figures"
    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    manifest: dict[str, Any] = {
        "targeted_selection": str(selection_path),
        "timestamp": args.timestamp,
        "dataset_name": dataset_name,
        "dataset_label": dataset_label,
        "methods": methods,
        "variants": ["original", "occluded", "shuffled", "wrong_class"],
        "replays": {},
    }

    for method_name in methods:
        replay_info = selection["replays"][method_name]
        base_subset = load_json(Path(replay_info["subset_path"]))
        original_predictions = Path(replay_info["replay_predictions"])
        original_metrics = Path(replay_info["replay_metrics"])

        variant_root = output_root / "counterfactual_assets"
        subset_paths = build_variant_subsets(
            method_name=method_name,
            base_subset=base_subset,
            donor_pool=donor_pool,
            output_root=variant_root,
            shuffle_grid=args.shuffle_grid,
            occlusion_frac=args.occlusion_frac,
        )

        method_summary: dict[str, dict[str, float]] = {
            "original": compute_variant_summary(original_predictions, original_predictions, original_metrics)
        }
        method_manifest: dict[str, Any] = {
            "original": {
                "predictions": str(original_predictions),
                "metrics": str(original_metrics),
                "summary": method_summary["original"],
            }
        }

        for variant_name, subset_path in subset_paths.items():
            run_name = f"{args.suite_name}_{dataset_name}_{method_name}_{variant_name}_{args.timestamp}"
            saved = run_variant(
                method_name=method_name,
                dataset_name=dataset_name,
                model_name=model_name,
                train_path=train_path,
                val_path=subset_path,
                run_name=run_name,
                output_root=output_root,
                log_root=log_root,
                seed=seed,
                runner_python=args.runner_python,
            )
            method_summary[variant_name] = compute_variant_summary(
                original_predictions,
                saved["predictions"],
                saved["metrics"],
            )
            method_manifest[variant_name] = {
                "subset_path": str(subset_path),
                "predictions": str(saved["predictions"]),
                "metrics": str(saved["metrics"]),
                "log": str(saved["log"]),
                "summary": method_summary[variant_name],
            }

        summary[method_name] = method_summary
        manifest["replays"][method_name] = method_manifest

    figure_path = figure_root / f"{args.suite_name}_{dataset_name}_{args.timestamp}.png"
    save_counterfactual_figure(summary, figure_path, dataset_label=dataset_label)
    manifest["figure_path"] = str(figure_path)
    manifest["summary"] = summary
    manifest_path = output_root / "counterfactual_sensitivity_manifest.json"
    save_json(manifest_path, manifest)
    print(json.dumps({"manifest": str(manifest_path), "figure": str(figure_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
