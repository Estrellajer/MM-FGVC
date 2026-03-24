#!/usr/bin/env python3
"""
Fix missing images for MHaluBench by copying them from source datasets
into a unified MHaluBench image directory.

Usage example (from project root):

    uv run python scripts/fix_mhalubench_missing_images.py \
        --mahalu-ann Data/MHaluBench/MHaluBench_train.json \
        --extra-ann Data/MHaluBench/MHaluBench_val-v0.1.json \
        --extra-ann Data/MHaluBench/MHaluBench_val-v0.2.json \
        --nlg-ann Data/MHaluBench/test_for_nlpcc.json \
        --mhalu-root Data/MHaluBench \
        --coco2014-root train2014 \
        --coco2017-root test2017 \
        --textvqa-root train_val_images

The script:
1. 解析提供的标注文件，收集其中出现的所有圖像文件名。
2. 在指定的 COCO2014 / COCO2017 / TextVQA 根目录下查找这些文件。
3. 对于尚未出现在 MHaluBench 目录中的图片，将其复制到
   `{mhalu_root}/data/image-to-text/{file_name}` 下（若目录不存在则自动创建）。

这样可以保证：
- MHaluBench 转换代码只需基于文件名在 MHaluBench 目录下查找图像；
- 缺失图片可以从原始数据集中补齐，并与已有 MHaluBench 图片放在一起。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix missing MHaluBench images by copying from COCO/TextVQA.")
    parser.add_argument(
        "--mahalu-ann",
        type=str,
        default="Data/MHaluBench/MHaluBench_train.json",
        help="Path to the primary MHaluBench annotation JSON.",
    )
    parser.add_argument(
        "--extra-ann",
        type=str,
        action="append",
        default=[],
        help="Additional annotation JSON/JSONL files to scan for image_path references. Can be passed multiple times.",
    )
    parser.add_argument(
        "--nlg-ann",
        type=str,
        default="",
        help="Optional MHaluBench extra annotation JSON/JSONL (e.g., Data/MHaluBench/test_for_nlpcc.json).",
    )
    parser.add_argument(
        "--mhalu-root",
        type=str,
        default="Data/MHaluBench",
        help="Root directory of MHaluBench (images will be placed under {mhalu_root}/data/image-to-text).",
    )
    parser.add_argument(
        "--coco2014-root",
        type=str,
        default=None,
        help="Root directory of COCO2014 images (e.g., path that directly contains COCO_train2014_*.jpg).",
    )
    parser.add_argument(
        "--coco2017-root",
        type=str,
        default=None,
        help="Root directory of COCO2017/val2017 (e.g., extracted test2017/ or val2017/).",
    )
    parser.add_argument(
        "--textvqa-root",
        type=str,
        default=None,
        help="Root directory of TextVQA train images (e.g., train_val_images/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report planned copy operations without actually copying files.",
    )
    return parser.parse_args()


def load_json_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        return []
    # 支持 JSON list 或 JSONL
    if text.startswith("["):
        return json.loads(text)
    records: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def collect_filenames_from_ann(records: Iterable[Mapping], key: str) -> Set[str]:
    names: Set[str] = set()
    for r in records:
        v = r.get(key)
        if not v:
            continue
        name = Path(str(v)).name
        if name:
            names.add(name)
    return names


def index_source_root(root: Path) -> Mapping[str, Path]:
    """Index all image files under a root directory by basename."""
    index: dict[str, Path] = {}
    if not root.exists():
        return index
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        index[p.name] = p
    return index


def build_source_index(roots: Iterable[Path]) -> Mapping[str, Path]:
    merged: dict[str, Path] = {}
    for root in roots:
        if not root:
            continue
        idx = index_source_root(root)
        for name, p in idx.items():
            # 如果多个根目录中出现同名文件，优先保留第一个，避免意外覆盖
            merged.setdefault(name, p)
    return merged


def main() -> None:
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]

    mahalu_ann_path = (project_root / args.mahalu_ann).resolve()
    extra_ann_paths = [(project_root / path).resolve() for path in args.extra_ann]
    nlg_ann_path = (project_root / args.nlg_ann).resolve() if args.nlg_ann else None
    mhalu_root = (project_root / args.mhalu_root).resolve()
    target_root = mhalu_root / "data" / "image-to-text"

    coco2014_root = Path(args.coco2014_root).resolve() if args.coco2014_root else None
    coco2017_root = Path(args.coco2017_root).resolve() if args.coco2017_root else None
    textvqa_root = Path(args.textvqa_root).resolve() if args.textvqa_root else None

    print(f"[info] mahalu_ann: {mahalu_ann_path}")
    if extra_ann_paths:
        for extra_path in extra_ann_paths:
            print(f"[info] extra_ann: {extra_path if extra_path.exists() else '(missing, skip)'}")
    else:
        print("[info] extra_ann: (unset)")
    if nlg_ann_path is None:
        print("[info] nlg_ann: (unset)")
    else:
        print(f"[info] nlg_ann: {nlg_ann_path if nlg_ann_path.exists() else '(missing, skip)'}")
    print(f"[info] mhalu_root: {mhalu_root}")
    print(f"[info] target_root: {target_root}")

    source_roots = [r for r in [coco2014_root, coco2017_root, textvqa_root] if r is not None]
    if not source_roots:
        print("[error] No source roots provided. Please set at least one of --coco2014-root/--coco2017-root/--textvqa-root.")
        raise SystemExit(1)

    print("[info] indexing source roots...")
    source_index = build_source_index(source_roots)
    print(f"[info] indexed {len(source_index)} image files from source roots.")

    # 收集需要的文件名
    mahalu_records = load_json_records(mahalu_ann_path)
    extra_ann_records = []
    for extra_ann_path in extra_ann_paths:
        if extra_ann_path.exists():
            extra_ann_records.extend(load_json_records(extra_ann_path))
    nlg_records = load_json_records(nlg_ann_path) if (nlg_ann_path and nlg_ann_path.exists()) else []

    needed_from_mahalu = collect_filenames_from_ann(mahalu_records, "image_path")
    needed_from_extra = collect_filenames_from_ann(extra_ann_records, "image_path")
    needed_from_nlg = collect_filenames_from_ann(nlg_records, "image_path")
    needed = needed_from_mahalu | needed_from_extra | needed_from_nlg

    print(f"[info] total distinct filenames referenced: {len(needed)}")

    # 检查 MHaluBench 目录下已有的图片
    existing: Set[str] = set()
    if mhalu_root.exists():
        for p in mhalu_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                existing.add(p.name)
    print(f"[info] images already under MHaluBench: {len(existing)} (by basename)")

    target_root.mkdir(parents=True, exist_ok=True)

    missing: list[str] = []
    copied: list[tuple[Path, Path]] = []

    for name in sorted(needed):
        if name in existing:
            continue
        src = source_index.get(name)
        if not src:
            missing.append(name)
            continue
        dst = target_root / name
        copied.append((src, dst))
        if args.dry_run:
            continue
        if not dst.parent.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        existing.add(name)

    print(f"[result] planned copies: {len(copied)}")
    if args.dry_run:
        for src, dst in copied[:20]:
            print(f"[dry-run] copy {src} -> {dst}")
        if len(copied) > 20:
            print(f"... {len(copied) - 20} more")
    else:
        for src, dst in copied[:20]:
            print(f"[copied] {src} -> {dst}")
        if len(copied) > 20:
            print(f"... {len(copied) - 20} more")

    if missing:
        print(f"[warning] {len(missing)} files were referenced but not found in source roots.")
        for name in missing[:50]:
            print(f"  - {name}")
        if len(missing) > 50:
            print(f"... {len(missing) - 50} more")


if __name__ == "__main__":
    main()
