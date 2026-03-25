#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from src.data import build_prompt, load_train_val

from registry import DEFAULT_MODELS, METHOD_GROUPS, METHOD_SPECS, TASK_GROUPS, TASK_SPECS, MethodSpec, TaskSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paper-oriented experiment suites with a shared clean runner.")
    parser.add_argument("--suite-name", required=True)
    parser.add_argument("--task-groups", default="")
    parser.add_argument("--tasks", default="")
    parser.add_argument("--method-groups", default="")
    parser.add_argument("--methods", default="")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--timestamp", required=True)
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    parser.add_argument(
        "--gpu-workers",
        default=os.environ.get("PAPER_GPU_WORKERS", ""),
        help=(
            "Semicolon-separated CUDA_VISIBLE_DEVICES groups. "
            "Examples: '0;1;2;3', '0,1;2,3', 'auto', or 'all'. "
            "Leave empty to keep the original sequential behavior."
        ),
    )
    return parser.parse_args()


def parse_csv_arg(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def expand_items(
    *,
    explicit_items: list[str],
    group_items: list[str],
    registry: dict[str, object],
    groups: dict[str, tuple[str, ...]],
) -> list[str]:
    resolved: "OrderedDict[str, None]" = OrderedDict()
    for group_name in group_items:
        if group_name not in groups:
            supported = ", ".join(sorted(groups))
            raise ValueError(f"Unknown group '{group_name}'. Supported groups: {supported}")
        for item in groups[group_name]:
            resolved[item] = None
    for item in explicit_items:
        if item not in registry:
            supported = ", ".join(sorted(registry))
            raise ValueError(f"Unknown item '{item}'. Supported items: {supported}")
        resolved[item] = None
    return list(resolved)


def method_supports_task(method_spec: MethodSpec, task_spec: TaskSpec) -> bool:
    return not method_spec.supported_datasets or task_spec.dataset_name in method_spec.supported_datasets


def subset_extension(path_str: str) -> str:
    return ".jsonl" if path_str.endswith(".jsonl") else ".json"


def visible_cuda_devices() -> list[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        return [item.strip() for item in raw.split(",") if item.strip()]
    if torch.cuda.is_available():
        return [str(index) for index in range(torch.cuda.device_count())]
    return []


def resolve_gpu_workers(raw: str) -> list[list[str]] | None:
    raw = raw.strip()
    if not raw:
        return None

    visible_devices = visible_cuda_devices()
    lowered = raw.lower()

    if lowered in {"cpu", "none", "off"}:
        return [[]]
    if lowered == "auto":
        if not visible_devices:
            raise ValueError("gpu-workers=auto requested, but no CUDA devices are visible")
        return [[device] for device in visible_devices]
    if lowered == "all":
        if not visible_devices:
            raise ValueError("gpu-workers=all requested, but no CUDA devices are visible")
        return [visible_devices]

    workers: list[list[str]] = []
    for chunk in raw.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        chunk_lower = chunk.lower()
        if chunk_lower in {"cpu", "none", "off"}:
            workers.append([])
            continue
        if chunk_lower == "all":
            if not visible_devices:
                raise ValueError("gpu-workers contains 'all', but no CUDA devices are visible")
            workers.append(list(visible_devices))
            continue

        devices = [item.strip() for item in chunk.split(",") if item.strip()]
        if not devices:
            raise ValueError(f"Invalid gpu worker chunk: '{chunk}'")
        if visible_devices:
            unknown = [device for device in devices if device not in visible_devices]
            if unknown:
                supported = ", ".join(visible_devices)
                raise ValueError(
                    f"Unknown CUDA devices in gpu-workers: {', '.join(unknown)}. "
                    f"Visible devices: {supported}"
                )
        workers.append(devices)

    if not workers:
        raise ValueError("gpu-workers did not resolve to any runnable worker")
    return workers


def format_cuda_visible_devices(devices: list[str] | None) -> str:
    if devices is None:
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        return inherited.strip() if inherited and inherited.strip() else "inherit"
    if not devices:
        return "cpu"
    return ",".join(devices)


def build_subprocess_env(devices: list[str] | None) -> dict[str, str] | None:
    if devices is None:
        return None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    return env


def run_subprocess(
    cmd: list[str],
    log_path: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    log_prefix_lines: list[str] | None = None,
) -> None:
    if log_path is None:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True, env=env)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as fp:
        if log_prefix_lines:
            for line in log_prefix_lines:
                fp.write(f"{line}\n")
            fp.write("\n")
            fp.flush()
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            stdout=fp,
            stderr=subprocess.STDOUT,
            check=True,
            env=env,
        )


def build_subset_file(
    *,
    task_spec: TaskSpec,
    src_path: str,
    dst_path: Path,
    mode: str,
    count: int,
    group_size: int,
    seed: int,
    shuffle: bool,
    exclude_path: Path | None,
    restrict_labels_from: Path | None,
    meta_path: Path,
) -> None:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "build_author_subset.py"),
        "--dataset-name",
        task_spec.dataset_name,
        "--src",
        src_path,
        "--dst",
        str(dst_path),
        "--mode",
        mode,
        "--count",
        str(count),
        "--group-size",
        str(group_size),
        "--meta-path",
        str(meta_path),
        "--seed",
        str(seed),
    ]
    if shuffle:
        cmd.append("--shuffle")
    if exclude_path is not None:
        cmd.extend(["--exclude-path", str(exclude_path)])
    if restrict_labels_from is not None:
        cmd.extend(["--restrict-labels-from", str(restrict_labels_from)])
    run_subprocess(cmd)


def validate_subset(task_spec: TaskSpec, train_path: Path, val_path: Path) -> None:
    train_data, val_data = load_train_val(task_spec.dataset_name, str(train_path), str(val_path))
    if not train_data:
        raise ValueError(f"{task_spec.experiment_id}: train subset is empty")
    if not val_data:
        raise ValueError(f"{task_spec.experiment_id}: val subset is empty")

    for split_name, samples in (("train", train_data), ("val", val_data)):
        for idx, sample in enumerate(samples):
            prompt = build_prompt(task_spec.dataset_name, sample)
            if not str(prompt).strip():
                raise ValueError(f"{task_spec.experiment_id}: {split_name}[{idx}] produced an empty prompt")
            for image_path in sample.get("images") or [sample.get("image")]:
                if not Path(str(image_path)).exists():
                    raise FileNotFoundError(f"{task_spec.experiment_id}: missing image {image_path}")

    if task_spec.evaluator_name == "pair" and len(val_data) % 2 != 0:
        raise ValueError(f"{task_spec.experiment_id}: pair evaluator requires an even-sized validation subset")
    if task_spec.evaluator_name == "naturalbench_group" and len(val_data) % 4 != 0:
        raise ValueError(f"{task_spec.experiment_id}: naturalbench_group evaluator requires val size % 4 == 0")


def build_task_subsets(
    *,
    task_spec: TaskSpec,
    seed: int,
    subset_root: Path,
) -> tuple[Path, Path]:
    experiment_root = subset_root / f"seed_{seed}" / task_spec.experiment_id
    experiment_root.mkdir(parents=True, exist_ok=True)

    ext = subset_extension(task_spec.train_src)
    train_subset = experiment_root / f"train_subset{ext}"
    val_subset = experiment_root / f"val_subset{subset_extension(task_spec.val_src)}"

    label_subset: Path | None = None
    if task_spec.label_seed_count > 0:
        label_subset = experiment_root / f"label_seed{ext}"
        build_subset_file(
            task_spec=task_spec,
            src_path=task_spec.train_src,
            dst_path=label_subset,
            mode="distinct_labels",
            count=task_spec.label_seed_count,
            group_size=0,
            seed=seed,
            shuffle=True,
            exclude_path=None,
            restrict_labels_from=None,
            meta_path=experiment_root / "label_seed.meta.json",
        )

    build_subset_file(
        task_spec=task_spec,
        src_path=task_spec.train_src,
        dst_path=train_subset,
        mode=task_spec.train_mode,
        count=task_spec.train_count,
        group_size=task_spec.train_group_size,
        seed=seed,
        shuffle=task_spec.shuffle_train,
        exclude_path=None,
        restrict_labels_from=label_subset,
        meta_path=experiment_root / "train_subset.meta.json",
    )
    build_subset_file(
        task_spec=task_spec,
        src_path=task_spec.val_src,
        dst_path=val_subset,
        mode=task_spec.val_mode,
        count=task_spec.val_count,
        group_size=task_spec.val_group_size,
        seed=seed,
        shuffle=task_spec.shuffle_val,
        exclude_path=train_subset,
        restrict_labels_from=train_subset,
        meta_path=experiment_root / "val_subset.meta.json",
    )

    validate_subset(task_spec, train_subset, val_subset)
    return train_subset, val_subset


def run_method(
    *,
    suite_name: str,
    timestamp: str,
    output_root: Path,
    log_root: Path,
    task_spec: TaskSpec,
    seed: int,
    model_name: str,
    method_spec: MethodSpec,
    train_subset: Path,
    val_subset: Path,
    save_predictions: bool,
    worker_id: int | None,
    gpu_devices: list[str] | None,
) -> tuple[str, Path, Path, Path, Path, str, str]:
    run_name = f"{suite_name}_{task_spec.experiment_id}_seed{seed}_{method_spec.method_id}_{model_name}_{timestamp}"
    log_path = log_root / f"{run_name}.log"
    worker_label = f"worker_{worker_id}" if worker_id is not None else "main"
    cuda_visible_devices = format_cuda_visible_devices(gpu_devices)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "main.py"),
        f"model={model_name}",
        "dataset=general_custom",
        f"method={method_spec.method_name}",
        f"evaluator={task_spec.evaluator_name}",
        f"dataset.name={task_spec.dataset_name}",
        f"dataset.train_path={train_subset}",
        f"dataset.val_path={val_subset}",
        f"run.output_dir={output_root}",
        f"run.run_name={run_name}",
        "run.progress_bar=false",
        f"run.save_predictions={'true' if save_predictions else 'false'}",
        f"run.seed={seed}",
    ]
    if method_spec.method_name != "zero_shot":
        cmd.append("method.params.progress_bar=false")
    cmd.extend(method_spec.overrides)
    print(
        f"[run_suite][{worker_label}][cuda={cuda_visible_devices}] start {run_name}",
        flush=True,
    )
    run_subprocess(
        cmd,
        log_path=log_path,
        env=build_subprocess_env(gpu_devices),
        log_prefix_lines=[
            f"# worker={worker_label}",
            f"# CUDA_VISIBLE_DEVICES={cuda_visible_devices}",
            f"# command={shlex.join(cmd)}",
        ],
    )
    metrics_path = output_root / f"{run_name}.metrics.json"
    predictions_path = output_root / f"{run_name}.predictions.jsonl"
    diagnostics_path = output_root / f"{run_name}.diagnostics.json"
    print(
        f"[run_suite][{worker_label}][cuda={cuda_visible_devices}] done {run_name}",
        flush=True,
    )
    return (
        run_name,
        metrics_path,
        predictions_path,
        diagnostics_path,
        log_path,
        worker_label,
        cuda_visible_devices,
    )


def write_manifest_header(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(
            [
                "suite_name",
                "timestamp",
                "seed",
                "model_name",
                "experiment_id",
                "dataset_name",
                "evaluator_name",
                "method_id",
                "display_name",
                "sequence_index",
                "worker_id",
                "cuda_visible_devices",
                "run_name",
                "train_subset",
                "val_subset",
                "metrics_path",
                "predictions_path",
                "diagnostics_path",
                "log_path",
            ]
        )


def append_manifest_row(path: Path, row: Iterable[object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as fp:
        writer = csv.writer(fp, delimiter="\t")
        writer.writerow(list(row))


def build_run_specs(
    *,
    task_ids: list[str],
    method_ids: list[str],
    models: list[str],
    seeds: list[int],
    subset_root: Path,
) -> list[dict[str, object]]:
    run_specs: list[dict[str, object]] = []
    sequence_index = 0
    for seed in seeds:
        built_subsets: dict[str, tuple[Path, Path]] = {}
        for task_id in task_ids:
            task_spec = TASK_SPECS[task_id]
            built_subsets[task_id] = build_task_subsets(
                task_spec=task_spec,
                seed=seed,
                subset_root=subset_root,
            )

        for model_name in models:
            for task_id in task_ids:
                task_spec = TASK_SPECS[task_id]
                train_subset, val_subset = built_subsets[task_id]
                for method_id in method_ids:
                    method_spec = METHOD_SPECS[method_id]
                    if not method_supports_task(method_spec, task_spec):
                        continue
                    run_specs.append(
                        {
                            "sequence_index": sequence_index,
                            "seed": seed,
                            "model_name": model_name,
                            "task_spec": task_spec,
                            "method_spec": method_spec,
                            "train_subset": train_subset,
                            "val_subset": val_subset,
                        }
                    )
                    sequence_index += 1
    return run_specs


def execute_run_spec(
    *,
    run_spec: dict[str, object],
    suite_name: str,
    timestamp: str,
    output_root: Path,
    log_root: Path,
    save_predictions: bool,
    worker_id: int | None,
    gpu_devices: list[str] | None,
) -> list[object]:
    task_spec = run_spec["task_spec"]
    method_spec = run_spec["method_spec"]
    if not isinstance(task_spec, TaskSpec):
        raise TypeError("run_spec.task_spec must be a TaskSpec")
    if not isinstance(method_spec, MethodSpec):
        raise TypeError("run_spec.method_spec must be a MethodSpec")

    seed = int(run_spec["seed"])
    model_name = str(run_spec["model_name"])
    train_subset = Path(run_spec["train_subset"])
    val_subset = Path(run_spec["val_subset"])

    (
        run_name,
        metrics_path,
        predictions_path,
        diagnostics_path,
        log_path,
        worker_label,
        cuda_visible_devices,
    ) = run_method(
        suite_name=suite_name,
        timestamp=timestamp,
        output_root=output_root,
        log_root=log_root,
        task_spec=task_spec,
        seed=seed,
        model_name=model_name,
        method_spec=method_spec,
        train_subset=train_subset,
        val_subset=val_subset,
        save_predictions=save_predictions,
        worker_id=worker_id,
        gpu_devices=gpu_devices,
    )

    return [
        suite_name,
        timestamp,
        seed,
        model_name,
        task_spec.experiment_id,
        task_spec.dataset_name,
        task_spec.evaluator_name,
        method_spec.method_id,
        method_spec.display_name,
        int(run_spec["sequence_index"]),
        worker_label,
        cuda_visible_devices,
        run_name,
        train_subset,
        val_subset,
        metrics_path,
        predictions_path,
        diagnostics_path,
        log_path,
    ]


def main() -> None:
    args = parse_args()
    task_ids = expand_items(
        explicit_items=parse_csv_arg(args.tasks),
        group_items=parse_csv_arg(args.task_groups),
        registry=TASK_SPECS,
        groups=TASK_GROUPS,
    )
    method_ids = expand_items(
        explicit_items=parse_csv_arg(args.methods),
        group_items=parse_csv_arg(args.method_groups),
        registry=METHOD_SPECS,
        groups=METHOD_GROUPS,
    )
    models = parse_csv_arg(args.models)
    seeds = [int(seed) for seed in parse_csv_arg(args.seeds)]
    if not task_ids:
        raise ValueError("run_suite.py requires at least one task via --task-groups or --tasks")
    if not method_ids:
        raise ValueError("run_suite.py requires at least one method via --method-groups or --methods")
    if not models:
        raise ValueError("run_suite.py requires at least one model via --models")
    if not seeds:
        raise ValueError("run_suite.py requires at least one seed via --seeds")
    gpu_workers = resolve_gpu_workers(args.gpu_workers)

    suite_key = f"{args.timestamp}_{args.suite_name}"
    paper_root = PROJECT_ROOT / "swap" / "paper"
    subset_root = paper_root / "subsets" / suite_key
    log_root = paper_root / "logs" / suite_key
    output_root = paper_root / "outputs" / suite_key
    record_root = paper_root / "records"
    summary_path = record_root / f"{args.suite_name}_{args.timestamp}.md"
    manifest_path = output_root / "manifest.tsv"
    config_path = output_root / "suite_config.json"

    subset_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    record_root.mkdir(parents=True, exist_ok=True)
    write_manifest_header(manifest_path)

    config_payload = {
        "suite_name": args.suite_name,
        "timestamp": args.timestamp,
        "task_ids": task_ids,
        "method_ids": method_ids,
        "models": models,
        "seeds": seeds,
        "save_predictions": bool(args.save_predictions),
        "gpu_workers": None if gpu_workers is None else [format_cuda_visible_devices(worker) for worker in gpu_workers],
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    run_specs = build_run_specs(
        task_ids=task_ids,
        method_ids=method_ids,
        models=models,
        seeds=seeds,
        subset_root=subset_root,
    )

    if gpu_workers is None or len(gpu_workers) <= 1:
        single_worker_devices = None if gpu_workers is None else gpu_workers[0]
        for run_spec in run_specs:
            try:
                row = execute_run_spec(
                    run_spec=run_spec,
                    suite_name=args.suite_name,
                    timestamp=args.timestamp,
                    output_root=output_root,
                    log_root=log_root,
                    save_predictions=bool(args.save_predictions),
                    worker_id=1 if gpu_workers else None,
                    gpu_devices=single_worker_devices,
                )
            except subprocess.CalledProcessError:
                if args.stop_on_error:
                    raise
                continue
            append_manifest_row(manifest_path, row)
    else:
        cursor_lock = threading.Lock()
        manifest_lock = threading.Lock()
        failure_lock = threading.Lock()
        stop_event = threading.Event()
        failures: list[subprocess.CalledProcessError] = []
        next_index = 0

        def worker_loop(worker_idx: int, devices: list[str]) -> None:
            nonlocal next_index
            while True:
                with cursor_lock:
                    if next_index >= len(run_specs):
                        return
                    if stop_event.is_set() and args.stop_on_error:
                        return
                    run_spec = run_specs[next_index]
                    next_index += 1

                try:
                    row = execute_run_spec(
                        run_spec=run_spec,
                        suite_name=args.suite_name,
                        timestamp=args.timestamp,
                        output_root=output_root,
                        log_root=log_root,
                        save_predictions=bool(args.save_predictions),
                        worker_id=worker_idx,
                        gpu_devices=devices,
                    )
                except subprocess.CalledProcessError as exc:
                    failed_spec = str(run_spec.get("sequence_index", "?"))
                    print(
                        f"[run_suite][worker_{worker_idx}][cuda={format_cuda_visible_devices(devices)}] "
                        f"failed spec={failed_spec}: {exc}",
                        flush=True,
                    )
                    with failure_lock:
                        failures.append(exc)
                    if args.stop_on_error:
                        stop_event.set()
                        return
                    continue

                with manifest_lock:
                    append_manifest_row(manifest_path, row)

        threads: list[threading.Thread] = []
        for worker_idx, devices in enumerate(gpu_workers, start=1):
            thread = threading.Thread(
                target=worker_loop,
                args=(worker_idx, devices),
                name=f"paper_gpu_worker_{worker_idx}",
            )
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        if failures and args.stop_on_error:
            raise failures[0]

    if not args.skip_summary:
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "summarize_suite.py"),
                "--manifest",
                str(manifest_path),
                "--output",
                str(summary_path),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )

    print(f"manifest={manifest_path}")
    if not args.skip_summary:
        print(f"summary={summary_path}")
    print(f"config={config_path}")


if __name__ == "__main__":
    main()
