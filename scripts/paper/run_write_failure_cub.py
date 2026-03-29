#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from swap.paper.scripts.analyze_write_failure import compute_attention_metrics


DEFAULT_TRAIN_PATH = (
    PROJECT_ROOT / "swap" / "paper" / "subsets" / "20260327_020551_c3_efficiency" / "seed_42" / "cub_fgvc" / "train_subset.json"
)
DEFAULT_VAL_PATH = (
    PROJECT_ROOT / "swap" / "paper" / "subsets" / "20260327_020551_c3_efficiency" / "seed_42" / "cub_fgvc" / "val_subset.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run write-failure analysis for STV / I2CL / MimIC.")
    parser.add_argument("--timestamp", default=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--model", default="qwen2_vl")
    parser.add_argument("--dataset-name", default="cub_fgvc")
    parser.add_argument("--dataset-label", default="CUB")
    parser.add_argument("--methods", default="stv,i2cl,mimic")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-per-label", type=int, default=1)
    parser.add_argument("--query-last-k", type=int, default=3)
    parser.add_argument("--heatmap-samples", type=int, default=3)
    parser.add_argument(
        "--answer-source",
        default="label",
        choices=["label", "normal_prediction", "steered_prediction"],
        help="Which answer text to teacher-force during write-failure analysis.",
    )
    parser.add_argument("--train-path", default=str(DEFAULT_TRAIN_PATH))
    parser.add_argument("--val-path", default=str(DEFAULT_VAL_PATH))
    parser.add_argument(
        "--suite-name",
        default="write_failure_cub",
    )
    parser.add_argument(
        "--runner-python",
        default=str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        help="Python executable used for main.py experiment subprocesses.",
    )
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_balanced_val_subset(
    *,
    src_path: Path,
    dst_path: Path,
    count_per_label: int,
    seed: int,
) -> list[dict[str, Any]]:
    data = load_json(src_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in data:
        grouped[str(item["label"])].append(item)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        selected.extend(items[: min(count_per_label, len(items))])

    rng.shuffle(selected)
    save_json(dst_path, selected)
    return selected


def method_overrides(method_name: str) -> list[str]:
    overrides: dict[str, list[str]] = {
        "stv": [],
        "i2cl": [
            "method.params.max_steps=48",
        ],
        "mimic": [
            "method.params.max_steps=32",
        ],
    }
    return overrides.get(method_name, [])


def run_method(
    *,
    method_name: str,
    model_name: str,
    dataset_name: str,
    dataset_tag: str,
    train_path: Path,
    val_path: Path,
    output_root: Path,
    log_root: Path,
    timestamp: str,
    seed: int,
    query_last_k: int,
    heatmap_samples: int,
    answer_source: str,
    runner_python: str,
) -> dict[str, Path]:
    run_name = f"write_failure_{dataset_tag}_{method_name}_{model_name}_{timestamp}"
    log_path = log_root / f"{run_name}.log"
    write_failure_dir = output_root / "write_failure" / method_name

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
        f"method.params.write_failure_dump_dir={write_failure_dir}",
        f"method.params.write_failure_max_samples={load_json(val_path).__len__()}",
        f"method.params.write_failure_heatmap_samples={heatmap_samples}",
        f"method.params.write_failure_query_last_k={query_last_k}",
        f"method.params.write_failure_answer_source={answer_source}",
    ]
    cmd.extend(method_overrides(method_name))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        fp.write(f"# command={' '.join(cmd)}\n\n")
        fp.flush()
        subprocess.run(cmd, cwd=PROJECT_ROOT, stdout=fp, stderr=subprocess.STDOUT, check=True)

    return {
        "run_name": Path(run_name),
        "metrics": output_root / f"{run_name}.metrics.json",
        "predictions": output_root / f"{run_name}.predictions.jsonl",
        "diagnostics": output_root / f"{run_name}.diagnostics.json",
        "log": log_path,
    }


def build_summary_rows(results: dict[str, dict[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name, paths in results.items():
        metrics_payload = load_json(paths["metrics"])
        diagnostics_payload = load_json(paths["diagnostics"])
        analysis = diagnostics_payload.get("write_failure_analysis") or {}
        summary = analysis.get("summary") or {}
        rows.append(
            {
                "method": method_name,
                "run_name": str(paths["run_name"]),
                "raw_accuracy": ((metrics_payload.get("metrics") or {}).get("accuracy")),
                "analysis_samples": int(analysis.get("num_analyzed_samples", 0)),
                "normal_accuracy": summary.get("normal_accuracy"),
                "steered_accuracy": summary.get("steered_accuracy"),
                "visual_attention_ratio_normal": summary.get("visual_attention_ratio_normal"),
                "visual_attention_ratio_steered": summary.get("visual_attention_ratio_steered"),
                "visual_attention_ratio_drop_percent": summary.get("visual_attention_ratio_drop_percent"),
                "normalized_attention_entropy_normal": summary.get("normalized_attention_entropy_normal"),
                "normalized_attention_entropy_steered": summary.get("normalized_attention_entropy_steered"),
                "representation_cosine_similarity": summary.get("representation_cosine_similarity"),
                "query_hidden_l2_ratio": summary.get("query_hidden_l2_ratio"),
                "task_vector_to_hidden_norm_ratio": summary.get("task_vector_to_hidden_norm_ratio"),
                "diagnostics_path": str(paths["diagnostics"]),
                "log_path": str(paths["log"]),
            }
        )
    return rows


def save_summary_table(
    rows: list[dict[str, Any]],
    csv_path: Path,
    md_path: Path,
    *,
    dataset_label: str,
    analysis_val_size: int,
    answer_source: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    lines = [
        f"# {dataset_label} Write-Failure Summary",
        "",
        f"- Analysis samples: {analysis_val_size}",
        f"- Answer source: {answer_source}",
        "",
        "| Method | Raw Acc | Normal Acc | Steered Acc | Visual Ratio N | Visual Ratio S | Drop % | Entropy N | Entropy S | Cosine | L2 Ratio | Task/Hidden |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {method} | {raw_accuracy:.4f} | {normal_accuracy:.4f} | {steered_accuracy:.4f} | "
            "{visual_attention_ratio_normal:.4f} | {visual_attention_ratio_steered:.4f} | "
            "{visual_attention_ratio_drop_percent:.2f} | {normalized_attention_entropy_normal:.4f} | "
            "{normalized_attention_entropy_steered:.4f} | {representation_cosine_similarity:.4f} | "
            "{query_hidden_l2_ratio:.4f} | {task_vector_to_hidden_norm_ratio:.4f} |".format(
                **{k: (0.0 if row.get(k) is None else row.get(k)) for k in row}
            )
        )
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_heatmap_payload(meta_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    meta = load_json(meta_path)
    normal_bundle = torch.load(meta["normal_bundle"], map_location="cpu")
    steered_bundle = torch.load(meta["steered_bundle"], map_location="cpu")
    image_indices = list(normal_bundle.get("image_token_indices") or meta.get("image_token_indices") or [])
    attentions_normal = normal_bundle["attentions"].to(dtype=torch.float32)
    attentions_steered = steered_bundle["attentions"].to(dtype=torch.float32)
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
    return meta, normal_bundle, {"attention": metrics}


def infer_grid(num_tokens: int, image_size: tuple[int, int]) -> tuple[int, int]:
    width, height = image_size
    target_ratio = float(width) / max(float(height), 1.0)
    best = (1, num_tokens)
    best_score = float("inf")
    for h in range(1, int(num_tokens**0.5) + 1):
        if num_tokens % h != 0:
            continue
        w = num_tokens // h
        for hh, ww in ((h, w), (w, h)):
            ratio = float(ww) / max(float(hh), 1.0)
            score = abs(ratio - target_ratio)
            if score < best_score:
                best = (hh, ww)
                best_score = score
    return best


def normalize_grid(grid: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(grid, 5.0))
    hi = float(np.percentile(grid, 95.0))
    if hi <= lo:
        lo = float(grid.min())
        hi = float(grid.max())
    if hi <= lo:
        return np.zeros_like(grid, dtype=np.float32)
    return np.clip((grid - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def resize_grid(grid: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    width, height = image_size
    tensor = torch.tensor(grid, dtype=torch.float32)[None, None, :, :]
    up = F.interpolate(tensor, size=(height, width), mode="bilinear", align_corners=False)
    return up[0, 0].cpu().numpy()


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    *,
    cmap_name: str,
    alpha: float,
    signed: bool = False,
) -> np.ndarray:
    image_np = np.asarray(image).astype(np.float32) / 255.0
    if signed:
        scale = float(np.percentile(np.abs(heatmap), 95.0))
        if scale <= 0.0:
            scale = float(np.max(np.abs(heatmap)))
        if scale <= 0.0:
            scaled = np.zeros_like(heatmap, dtype=np.float32)
        else:
            scaled = np.clip(heatmap / scale, -1.0, 1.0)
            scaled = ((scaled + 1.0) / 2.0).astype(np.float32)
    else:
        scaled = normalize_grid(heatmap)

    cmap = plt.get_cmap(cmap_name)
    color = cmap(scaled)[..., :3].astype(np.float32)
    overlay = (1.0 - alpha) * image_np + alpha * color
    return np.clip(overlay, 0.0, 1.0)


def select_best_raw_sample(raw_samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not raw_samples:
        return None

    def score(row: dict[str, Any]) -> float:
        drop = abs(float(row.get("visual_attention_ratio_drop_percent") or 0.0))
        entropy_n = float(row.get("normalized_attention_entropy_normal") or 0.0)
        entropy_s = float(row.get("normalized_attention_entropy_steered") or 0.0)
        entropy_delta = abs(entropy_s - entropy_n)
        return drop + 25.0 * entropy_delta

    return max(raw_samples, key=score)


def build_grids(
    *,
    meta: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    image = Image.open(meta["image"]).convert("RGB")
    image_size = image.size
    normal = np.asarray(metrics["attention"]["image_attention_profile_normal"], dtype=np.float32)
    steered = np.asarray(metrics["attention"]["image_attention_profile_steered"], dtype=np.float32)
    delta = steered - normal
    height, width = infer_grid(int(normal.size), image_size)
    return normal.reshape(height, width), steered.reshape(height, width), delta.reshape(height, width)


def save_overlay_gallery(
    rows: list[tuple[str, dict[str, Any], dict[str, Any], dict[str, Any]]],
    figure_path: Path,
    *,
    title: str | None = None,
) -> None:
    if not rows:
        return

    fig, axes = plt.subplots(len(rows), 5, figsize=(19, 4.6 * len(rows)), dpi=170)
    if len(rows) == 1:
        axes = [axes]

    for row_axes, (row_title, meta, _bundle, metrics) in zip(axes, rows):
        image = Image.open(meta["image"]).convert("RGB")
        normal_grid, steered_grid, delta_grid = build_grids(meta=meta, metrics=metrics)
        normal_overlay = overlay_heatmap(
            image,
            resize_grid(normal_grid, image.size),
            cmap_name="magma",
            alpha=0.48,
        )
        steered_overlay = overlay_heatmap(
            image,
            resize_grid(steered_grid, image.size),
            cmap_name="magma",
            alpha=0.48,
        )
        delta_overlay = overlay_heatmap(
            image,
            resize_grid(delta_grid, image.size),
            cmap_name="coolwarm",
            alpha=0.60,
            signed=True,
        )
        attn = metrics["attention"]

        row_axes[0].imshow(image)
        row_axes[0].set_title(row_title)
        row_axes[0].axis("off")

        row_axes[1].imshow(normal_overlay)
        row_axes[1].set_title(
            f"Normal overlay\nvisual={attn['visual_attention_ratio_normal']:.3f}, H={attn['normalized_attention_entropy_normal']:.3f}"
        )
        row_axes[1].axis("off")

        row_axes[2].imshow(steered_overlay)
        row_axes[2].set_title(
            f"Steered overlay\nvisual={attn['visual_attention_ratio_steered']:.3f}, H={attn['normalized_attention_entropy_steered']:.3f}"
        )
        row_axes[2].axis("off")

        row_axes[3].imshow(delta_overlay)
        row_axes[3].set_title(
            f"Delta overlay\nratio drop={attn['visual_attention_ratio_drop_percent']:.2f}%"
        )
        row_axes[3].axis("off")

        normal_sorted = np.sort(np.asarray(attn["image_attention_profile_normal"], dtype=np.float32))[::-1]
        steered_sorted = np.sort(np.asarray(attn["image_attention_profile_steered"], dtype=np.float32))[::-1]
        xs = np.linspace(0.0, 1.0, normal_sorted.size, endpoint=False)
        uniform = np.full_like(xs, 1.0 / max(normal_sorted.size, 1), dtype=np.float32)
        row_axes[4].plot(xs, normal_sorted, label="normal", linewidth=2.0, color="#1f77b4")
        row_axes[4].plot(xs, steered_sorted, label="steered", linewidth=2.0, color="#d62728")
        row_axes[4].plot(xs, uniform, label="uniform", linewidth=1.5, linestyle="--", color="#7f7f7f")
        row_axes[4].set_title("Token concentration")
        row_axes[4].set_xlabel("Sorted image patches")
        row_axes[4].set_ylabel("Attention weight")
        row_axes[4].grid(alpha=0.25)
        row_axes[4].legend(frameon=False, fontsize=8, loc="upper right")

        caption = (
            f"label={meta.get('label', '')}\n"
            f"note: delta overlay is contrast-rescaled for visibility"
        )
        row_axes[0].text(
            0.0,
            -0.10,
            caption,
            transform=row_axes[0].transAxes,
            fontsize=8,
            va="top",
        )

    if title:
        fig.suptitle(title, fontsize=15)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    else:
        fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_heatmap_figure(results: dict[str, dict[str, Path]], figure_path: Path, *, dataset_label: str) -> None:
    rows = []
    for method_name, paths in results.items():
        diagnostics = load_json(paths["diagnostics"])
        raw_samples = (diagnostics.get("write_failure_analysis") or {}).get("raw_samples") or []
        best = select_best_raw_sample(raw_samples)
        if not best:
            continue
        meta_path = Path(best["raw_bundle_paths"]["meta"])
        rows.append((method_name.upper(), *load_heatmap_payload(meta_path)))

    save_overlay_gallery(rows, figure_path, title=None)


def save_diagnostics_gallery(
    diagnostics_path: Path,
    figure_path: Path,
    *,
    dataset_label: str,
    title: str | None = None,
    max_rows: int | None = None,
) -> None:
    diagnostics = load_json(diagnostics_path)
    analysis = diagnostics.get("write_failure_analysis") or {}
    method_name = str(analysis.get("method") or diagnostics_path.stem).upper()
    raw_samples = list(analysis.get("raw_samples") or [])
    if max_rows is not None:
        raw_samples = raw_samples[: max_rows]

    rows = []
    for idx, sample in enumerate(raw_samples, start=1):
        raw_bundle_paths = sample.get("raw_bundle_paths") or {}
        meta = raw_bundle_paths.get("meta")
        if not meta:
            continue
        row_title = f"{method_name} target #{idx}"
        rows.append((row_title, *load_heatmap_payload(Path(meta))))

    save_overlay_gallery(rows, figure_path, title=title)


def main() -> None:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    dataset_tag = args.dataset_name.replace("_fgvc", "").replace("_", "-")

    paper_root = PROJECT_ROOT / "swap" / "paper"
    output_root = paper_root / "outputs" / f"{args.timestamp}_{args.suite_name}"
    log_root = paper_root / "logs" / f"{args.timestamp}_{args.suite_name}"
    record_root = paper_root / "records"
    figure_root = paper_root / "figures"
    subset_root = paper_root / "subsets" / f"{args.timestamp}_{args.suite_name}"

    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    subset_root.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train_path)
    val_path = Path(args.val_path)
    analysis_val_path = subset_root / args.dataset_name / "val_balanced.json"
    analysis_val = build_balanced_val_subset(
        src_path=val_path,
        dst_path=analysis_val_path,
        count_per_label=args.val_per_label,
        seed=args.seed,
    )

    manifest_rows = {
        "suite_name": args.suite_name,
        "timestamp": args.timestamp,
        "model": args.model,
        "dataset_name": args.dataset_name,
        "dataset_label": args.dataset_label,
        "seed": args.seed,
        "train_path": str(train_path),
        "source_val_path": str(val_path),
        "analysis_val_path": str(analysis_val_path),
        "analysis_val_size": len(analysis_val),
        "methods": methods,
        "answer_source": args.answer_source,
    }
    save_json(output_root / "manifest.json", manifest_rows)

    results: dict[str, dict[str, Path]] = {}
    for method_name in methods:
        results[method_name] = run_method(
            method_name=method_name,
            model_name=args.model,
            dataset_name=args.dataset_name,
            dataset_tag=dataset_tag,
            train_path=train_path,
            val_path=analysis_val_path,
            output_root=output_root,
            log_root=log_root,
            timestamp=args.timestamp,
            seed=args.seed,
            query_last_k=args.query_last_k,
            heatmap_samples=args.heatmap_samples,
            answer_source=args.answer_source,
            runner_python=args.runner_python,
        )

    summary_rows = build_summary_rows(results)
    summary_json_path = output_root / "write_failure_summary.json"
    summary_csv_path = record_root / f"{args.suite_name}_{args.timestamp}.csv"
    summary_md_path = record_root / f"{args.suite_name}_{args.timestamp}.md"
    save_json(summary_json_path, summary_rows)
    save_summary_table(
        summary_rows,
        summary_csv_path,
        summary_md_path,
        dataset_label=args.dataset_label,
        analysis_val_size=len(analysis_val),
        answer_source=args.answer_source,
    )

    heatmap_path = figure_root / f"{args.suite_name}_{args.timestamp}.png"
    save_heatmap_figure(results, heatmap_path, dataset_label=args.dataset_label)

    print(json.dumps(
        {
            "summary_json": str(summary_json_path),
            "summary_csv": str(summary_csv_path),
            "summary_md": str(summary_md_path),
            "heatmap_figure": str(heatmap_path),
            "output_root": str(output_root),
            "analysis_val_size": len(analysis_val),
            "answer_source": args.answer_source,
            "dataset_name": args.dataset_name,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
