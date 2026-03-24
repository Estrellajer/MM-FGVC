#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset


DISALLOWED_EVAL_LABELS = {"HIDDEN", "unknown"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build author-style support/eval subsets.")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--mode", required=True, choices=["author_ref", "per_label", "first", "distinct_labels", "grouped"])
    parser.add_argument("--count", type=int, default=0, help="Per-label count for per_label mode, total count otherwise.")
    parser.add_argument("--group-size", type=int, default=0)
    parser.add_argument("--exclude-path", type=str, default="")
    parser.add_argument("--restrict-labels-from", type=str, default="")
    parser.add_argument("--meta-path", type=str, default="")
    parser.add_argument("--allow-hidden-labels", action="store_true")
    return parser.parse_args()


def _dataset_key(dataset_name: str) -> str:
    return str(dataset_name).strip().lower().replace("-", "_")


def _dataset_family(dataset_name: str) -> str:
    key = _dataset_key(dataset_name)
    if key.startswith("blink_"):
        return "blink"
    if key in {"naturalbench_ret", "naturalbench_vqa"}:
        return "naturalbench"
    return key


def _canonical_source_globs(dataset_name: str) -> list[str]:
    family = _dataset_family(dataset_name)
    root = PROJECT_ROOT / "dataset" / "converted_from_data"
    mapping = {
        "blink": [str(root / "blink" / "*.json")],
        "cub": [str(root / "cub" / "*.json")],
        "eurosat": [str(root / "eurosat" / "*.json")],
        "flowers": [str(root / "flowers" / "*.json")],
        "mhalubench": [str(root / "mhalubench" / "*.json")],
        "naturalbench": [str(root / "naturalbench" / "*.jsonl")],
        "pets": [str(root / "pets" / "*.json")],
        "sugarcrepe": [str(root / "sugarcrepe" / "*.jsonl")],
        "tinyimage": [str(root / "tinyimage" / "*.json")],
        "vizwiz": [str(root / "vizwiz" / "*.jsonl")],
        "vlguard": [str(root / "vlguard" / "*.json")],
    }
    return mapping.get(family, [])


@lru_cache(maxsize=None)
def _canonical_lookup(dataset_name: str) -> tuple[dict[str, str], dict[str, str]]:
    image_by_basename: dict[str, str] = {}
    label_by_basename: dict[str, str] = {}

    for pattern in _canonical_source_globs(dataset_name):
        for path in sorted(PROJECT_ROOT.glob(str(Path(pattern).relative_to(PROJECT_ROOT)))):
            try:
                rows = load_dataset(dataset_name, str(path))
            except Exception:
                continue
            for row in rows:
                images = row.get("images") or [row.get("image")]
                for image_path in images:
                    if not image_path:
                        continue
                    image_path = str(image_path).strip()
                    if not image_path:
                        continue
                    basename = Path(image_path).name
                    image_by_basename.setdefault(basename, image_path)
                    label_by_basename.setdefault(basename, str(row.get("label", "")).strip())

    return image_by_basename, label_by_basename


def _try_blink_ref_shift(basename: str) -> str | None:
    match = re.fullmatch(r"(.+_)(\d+)(\.[^.]+)", basename)
    if not match:
        return None
    prefix, numeric, suffix = match.groups()
    return f"{prefix}{int(numeric) + 1}{suffix}"


def _should_canonicalize_labels(dataset_name: str) -> bool:
    return _dataset_family(dataset_name) in {"pets", "eurosat"}


def _repair_images_and_labels(
    rows: list[dict[str, Any]],
    dataset_name: str,
    src_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    needs_canonical_lookup = "ref-data" in src_path.parts or not src_path.is_relative_to(PROJECT_ROOT / "dataset" / "converted_from_data")
    image_by_basename, label_by_basename = ({}, {})
    if needs_canonical_lookup:
        image_by_basename, label_by_basename = _canonical_lookup(dataset_name)

    stats = {
        "canonicalized_paths": 0,
        "canonicalized_labels": 0,
        "skipped_missing_images": 0,
    }

    repaired: list[dict[str, Any]] = []
    family = _dataset_family(dataset_name)

    for row in rows:
        images = list(row.get("images") or [row.get("image")])
        fixed_images: list[str] = []
        row_missing = False

        for image_path in images:
            if image_path is None:
                row_missing = True
                break

            candidate = Path(str(image_path))
            if candidate.exists():
                fixed_images.append(str(candidate.resolve()))
                continue

            basename = candidate.name
            resolved_path = ""

            relative_candidate = (src_path.parent / str(image_path)).resolve()
            if relative_candidate.exists():
                resolved_path = str(relative_candidate)
            elif basename in image_by_basename:
                resolved_path = image_by_basename[basename]
            elif family == "blink":
                shifted = _try_blink_ref_shift(basename)
                if shifted and shifted in image_by_basename:
                    resolved_path = image_by_basename[shifted]

            if not resolved_path:
                row_missing = True
                break

            fixed_images.append(resolved_path)
            stats["canonicalized_paths"] += 1

        if row_missing or not fixed_images:
            stats["skipped_missing_images"] += 1
            continue

        fixed_row = dict(row)
        fixed_row["image"] = fixed_images[0]
        fixed_row["images"] = fixed_images

        if _should_canonicalize_labels(dataset_name):
            basename = Path(fixed_row["image"]).name
            canonical_label = label_by_basename.get(basename)
            if canonical_label and canonical_label != fixed_row["label"]:
                fixed_row["label"] = canonical_label
                stats["canonicalized_labels"] += 1

        repaired.append(fixed_row)

    return repaired, stats


def _row_signature(row: dict[str, Any]) -> str:
    payload = {
        "images": [Path(p).name for p in (row.get("images") or [row.get("image")])],
        "question": str(row.get("question", "")),
        "label": str(row.get("label", "")),
        "question_id": row.get("question_id"),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _load_exclude_signatures(exclude_path: str, dataset_name: str) -> set[str]:
    if not exclude_path:
        return set()
    path = Path(exclude_path)
    if not path.exists():
        return set()
    rows = load_dataset(dataset_name, str(path))
    rows, _ = _repair_images_and_labels(rows, dataset_name, path)
    return {_row_signature(row) for row in rows}


def _load_allowed_labels(restrict_labels_from: str, dataset_name: str) -> set[str]:
    if not restrict_labels_from:
        return set()
    path = Path(restrict_labels_from)
    if not path.exists():
        return set()
    rows = load_dataset(dataset_name, str(path))
    rows, _ = _repair_images_and_labels(rows, dataset_name, path)
    return {str(row["label"]) for row in rows}


def _write_subset(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as fp:
            for row in rows:
                fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        return
    if path.suffix == ".json":
        path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    raise ValueError(f"Unsupported subset extension: {path.suffix}")


def _take_per_label(rows: list[dict[str, Any]], count: int, excluded: set[str], allow_hidden_labels: bool) -> tuple[list[dict[str, Any]], int, int]:
    selected: list[dict[str, Any]] = []
    per_label: Counter[str] = Counter()
    excluded_overlap = 0
    dropped_labels = 0

    for row in rows:
        sig = _row_signature(row)
        if sig in excluded:
            excluded_overlap += 1
            continue
        label = str(row["label"])
        if not allow_hidden_labels and label in DISALLOWED_EVAL_LABELS:
            dropped_labels += 1
            continue
        if per_label[label] >= count:
            continue
        selected.append(row)
        per_label[label] += 1

    return selected, excluded_overlap, dropped_labels


def _take_first(rows: list[dict[str, Any]], count: int, excluded: set[str], allow_hidden_labels: bool, distinct_labels: bool) -> tuple[list[dict[str, Any]], int, int]:
    selected: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    excluded_overlap = 0
    dropped_labels = 0

    for row in rows:
        if count > 0 and len(selected) >= count:
            break
        sig = _row_signature(row)
        if sig in excluded:
            excluded_overlap += 1
            continue
        label = str(row["label"])
        if not allow_hidden_labels and label in DISALLOWED_EVAL_LABELS:
            dropped_labels += 1
            continue
        if distinct_labels and label in seen_labels:
            continue
        selected.append(row)
        seen_labels.add(label)

    return selected, excluded_overlap, dropped_labels


def _take_grouped(rows: list[dict[str, Any]], count: int, group_size: int, excluded: set[str], allow_hidden_labels: bool) -> tuple[list[dict[str, Any]], int, int]:
    if group_size <= 0:
        raise ValueError("grouped mode requires --group-size > 0")

    selected: list[dict[str, Any]] = []
    excluded_overlap = 0
    dropped_labels = 0

    for start in range(0, len(rows), group_size):
        group = rows[start : start + group_size]
        if len(group) < group_size:
            break
        if count > 0 and len(selected) + group_size > count:
            break

        group_sigs = [_row_signature(row) for row in group]
        if any(sig in excluded for sig in group_sigs):
            excluded_overlap += sum(1 for sig in group_sigs if sig in excluded)
            continue

        labels = [str(row["label"]) for row in group]
        if not allow_hidden_labels and any(label in DISALLOWED_EVAL_LABELS for label in labels):
            dropped_labels += sum(1 for label in labels if label in DISALLOWED_EVAL_LABELS)
            continue

        selected.extend(group)

    return selected, excluded_overlap, dropped_labels


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name
    src_path = (PROJECT_ROOT / args.src).resolve() if not Path(args.src).is_absolute() else Path(args.src).resolve()
    dst_path = (PROJECT_ROOT / args.dst).resolve() if not Path(args.dst).is_absolute() else Path(args.dst).resolve()
    meta_path = (
        (PROJECT_ROOT / args.meta_path).resolve()
        if args.meta_path and not Path(args.meta_path).is_absolute()
        else Path(args.meta_path).resolve() if args.meta_path else None
    )

    rows = load_dataset(dataset_name, str(src_path))
    rows, repair_stats = _repair_images_and_labels(rows, dataset_name, src_path)
    excluded = _load_exclude_signatures(args.exclude_path, dataset_name)
    allowed_labels = _load_allowed_labels(args.restrict_labels_from, dataset_name)

    if allowed_labels:
        rows = [row for row in rows if str(row["label"]) in allowed_labels]

    if args.mode == "author_ref":
        selected, excluded_overlap, dropped_labels = _take_first(
            rows=rows,
            count=0,
            excluded=excluded,
            allow_hidden_labels=args.allow_hidden_labels,
            distinct_labels=False,
        )
    elif args.mode == "per_label":
        selected, excluded_overlap, dropped_labels = _take_per_label(
            rows=rows,
            count=args.count,
            excluded=excluded,
            allow_hidden_labels=args.allow_hidden_labels,
        )
    elif args.mode == "first":
        selected, excluded_overlap, dropped_labels = _take_first(
            rows=rows,
            count=args.count,
            excluded=excluded,
            allow_hidden_labels=args.allow_hidden_labels,
            distinct_labels=False,
        )
    elif args.mode == "distinct_labels":
        selected, excluded_overlap, dropped_labels = _take_first(
            rows=rows,
            count=args.count,
            excluded=excluded,
            allow_hidden_labels=args.allow_hidden_labels,
            distinct_labels=True,
        )
    elif args.mode == "grouped":
        selected, excluded_overlap, dropped_labels = _take_grouped(
            rows=rows,
            count=args.count,
            group_size=args.group_size,
            excluded=excluded,
            allow_hidden_labels=args.allow_hidden_labels,
        )
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    _write_subset(dst_path, selected)

    label_counts = Counter(str(row["label"]) for row in selected)
    metadata = {
        "count": len(selected),
        "mode": args.mode,
        "requested_count": args.count,
        "group_size": args.group_size,
        "src": str(src_path),
        "source_kind": "author_ref" if "ref-data" in src_path.parts else "converted",
        "unique_labels": len(label_counts),
        "per_label_min": min(label_counts.values()) if label_counts else 0,
        "per_label_max": max(label_counts.values()) if label_counts else 0,
        "canonicalized_paths": repair_stats["canonicalized_paths"],
        "canonicalized_labels": repair_stats["canonicalized_labels"],
        "skipped_missing_images": repair_stats["skipped_missing_images"],
        "dropped_disallowed_labels": dropped_labels,
        "excluded_overlap": excluded_overlap,
        "restricted_label_space": len(allowed_labels),
    }

    if meta_path is not None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(len(selected))


if __name__ == "__main__":
    main()
