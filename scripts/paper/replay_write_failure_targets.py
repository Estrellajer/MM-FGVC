#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.paper.run_write_failure_cub import (
    load_json,
    run_method,
    save_diagnostics_gallery,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay the strongest write-failure samples from an existing diagnostics run."
    )
    parser.add_argument("--source-output-root", required=True, help="Original write-failure output root.")
    parser.add_argument("--timestamp", default=datetime.now(UTC).strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--methods", default="stv,i2cl,mimic")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--score-mode",
        default="combined",
        choices=["combined", "drop_abs", "entropy_abs", "hidden_shift"],
        help="How to rank candidate samples from diagnostics.",
    )
    parser.add_argument("--suite-name", default="write_failure_targeted_replay")
    parser.add_argument("--query-last-k", type=int, default=None)
    parser.add_argument(
        "--answer-source",
        default=None,
        choices=["label", "normal_prediction", "steered_prediction", None],
        help="Override answer source used for the replay. Defaults to the source run setting.",
    )
    parser.add_argument(
        "--runner-python",
        default=str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        help="Python executable used for main.py subprocesses.",
    )
    return parser.parse_args()


def dataset_tag(dataset_name: str) -> str:
    return str(dataset_name).replace("_fgvc", "").replace("_", "-")


def score_sample(row: dict[str, Any], score_mode: str) -> float:
    drop = float(row.get("visual_attention_ratio_drop_percent") or 0.0)
    entropy_n = float(row.get("normalized_attention_entropy_normal") or 0.0)
    entropy_s = float(row.get("normalized_attention_entropy_steered") or 0.0)
    entropy_delta = entropy_s - entropy_n
    l2 = float(row.get("query_hidden_l2_ratio") or 0.0)
    cosine = float(row.get("representation_cosine_similarity") or 1.0)

    if score_mode == "drop_abs":
        return abs(drop)
    if score_mode == "entropy_abs":
        return abs(entropy_delta)
    if score_mode == "hidden_shift":
        return l2 + (1.0 - cosine)

    return abs(drop) + 100.0 * abs(entropy_delta) + 50.0 * (1.0 - cosine) + 10.0 * l2


def find_diagnostics_path(source_output_root: Path, method_name: str) -> Path:
    matches = sorted(source_output_root.glob(f"*_{method_name}_*.diagnostics.json"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one diagnostics file for method={method_name}, found {len(matches)} in {source_output_root}"
        )
    return matches[0]


def index_subset_items(subset_path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in load_json(subset_path):
        key = str(item.get("question_id"))
        indexed[key] = item
    return indexed


def select_top_samples(
    diagnostics_path: Path,
    *,
    subset_index: dict[str, dict[str, Any]],
    top_k: int,
    score_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    diagnostics = load_json(diagnostics_path)
    rows = ((diagnostics.get("write_failure_analysis") or {}).get("samples") or [])
    ranked: list[dict[str, Any]] = []
    seen_question_ids: set[str] = set()

    for row in rows:
        question_id = str(row.get("question_id"))
        if question_id in seen_question_ids:
            continue
        if question_id not in subset_index:
            continue
        ranked.append(
            {
                "question_id": question_id,
                "label": row.get("label"),
                "score": score_sample(row, score_mode),
                "visual_attention_ratio_drop_percent": row.get("visual_attention_ratio_drop_percent"),
                "normalized_attention_entropy_normal": row.get("normalized_attention_entropy_normal"),
                "normalized_attention_entropy_steered": row.get("normalized_attention_entropy_steered"),
                "representation_cosine_similarity": row.get("representation_cosine_similarity"),
                "query_hidden_l2_ratio": row.get("query_hidden_l2_ratio"),
                "task_vector_to_hidden_norm_ratio": row.get("task_vector_to_hidden_norm_ratio"),
            }
        )
        seen_question_ids.add(question_id)

    ranked.sort(key=lambda item: float(item["score"]), reverse=True)
    selected = ranked[:top_k]
    subset = [subset_index[item["question_id"]] for item in selected]
    return selected, subset


def infer_query_last_k(diagnostics_path: Path, fallback: int) -> int:
    diagnostics = load_json(diagnostics_path)
    analysis = diagnostics.get("write_failure_analysis") or {}
    value = analysis.get("query_last_k")
    if value is None:
        return fallback
    return int(value)


def main() -> None:
    args = parse_args()
    source_output_root = Path(args.source_output_root).expanduser().resolve()
    manifest = load_json(source_output_root / "manifest.json")
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]

    dataset_name = str(manifest["dataset_name"])
    dataset_label = str(manifest["dataset_label"])
    model_name = str(manifest["model"])
    seed = int(manifest["seed"])
    train_path = Path(manifest["train_path"])
    analysis_val_path = Path(manifest["analysis_val_path"])
    answer_source = str(args.answer_source or manifest.get("answer_source") or "label")
    subset_index = index_subset_items(analysis_val_path)

    paper_root = PROJECT_ROOT / "swap" / "paper"
    output_root = paper_root / "outputs" / f"{args.timestamp}_{args.suite_name}"
    log_root = paper_root / "logs" / f"{args.timestamp}_{args.suite_name}"
    subset_root = paper_root / "subsets" / f"{args.timestamp}_{args.suite_name}"
    figure_root = paper_root / "figures"

    output_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    subset_root.mkdir(parents=True, exist_ok=True)

    selection_manifest: dict[str, Any] = {
        "source_output_root": str(source_output_root),
        "timestamp": args.timestamp,
        "dataset_name": dataset_name,
        "dataset_label": dataset_label,
        "model": model_name,
        "score_mode": args.score_mode,
        "top_k": args.top_k,
        "methods": methods,
        "answer_source": answer_source,
        "replays": {},
    }

    replay_outputs: dict[str, dict[str, str]] = {}

    for method_name in methods:
        source_diagnostics = find_diagnostics_path(source_output_root, method_name)
        selected_rows, subset = select_top_samples(
            source_diagnostics,
            subset_index=subset_index,
            top_k=args.top_k,
            score_mode=args.score_mode,
        )
        if not subset:
            continue

        method_subset_path = subset_root / dataset_name / f"{method_name}_top{len(subset)}.json"
        save_json(method_subset_path, subset)

        replay_result = run_method(
            method_name=method_name,
            model_name=model_name,
            dataset_name=dataset_name,
            dataset_tag=dataset_tag(dataset_name),
            train_path=train_path,
            val_path=method_subset_path,
            output_root=output_root,
            log_root=log_root,
            timestamp=args.timestamp,
            seed=seed,
            query_last_k=(
                int(args.query_last_k)
                if args.query_last_k is not None
                else infer_query_last_k(source_diagnostics, fallback=3)
            ),
            heatmap_samples=len(subset),
            answer_source=answer_source,
            runner_python=args.runner_python,
        )

        figure_path = figure_root / f"{args.suite_name}_{dataset_tag(dataset_name)}_{method_name}_{args.timestamp}.png"
        save_diagnostics_gallery(
            replay_result["diagnostics"],
            figure_path,
            dataset_label=dataset_label,
            title=None,
            max_rows=len(subset),
        )

        selection_manifest["replays"][method_name] = {
            "source_diagnostics": str(source_diagnostics),
            "selected_samples": selected_rows,
            "subset_path": str(method_subset_path),
            "replay_diagnostics": str(replay_result["diagnostics"]),
            "replay_metrics": str(replay_result["metrics"]),
            "replay_predictions": str(replay_result["predictions"]),
            "log_path": str(replay_result["log"]),
            "figure_path": str(figure_path),
        }
        replay_outputs[method_name] = {
            "diagnostics": str(replay_result["diagnostics"]),
            "figure": str(figure_path),
            "subset": str(method_subset_path),
        }

    selection_path = output_root / "targeted_replay_selection.json"
    save_json(selection_path, selection_manifest)

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "selection_manifest": str(selection_path),
                "replays": replay_outputs,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
