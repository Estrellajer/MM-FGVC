#!/usr/bin/env python3
"""Check dataset availability: annotations, path remapping, and actual file existence."""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from paths import (
    DATASET_ROOT,
    DATA_ROOT,
    remap_path,
    NATURAL_BENCH_DIR,
    VL_GUARD_DIR,
    COCO_VAL2017_DIR,
    FLOWERS102_DIR,
    PETS_DIR,
    CUB200_DIR,
    EUROSAT_DIR,
    TINY_IMAGENET_DIR,
)


def check_paths_exist(paths, sample_size=5):
    """Return (ok_count, total_checked, missing_sample)."""
    ok, total, missing = 0, 0, []
    for p in paths[:sample_size]:
        total += 1
        if os.path.exists(p):
            ok += 1
        else:
            missing.append(p)
    return ok, total, missing


def main():
    issues = []
    summary = []

    # 1. Core datasets (ref-data / converted annotations)
    for name, p, key in [
        ("naturalbench_ret(ref)", PROJECT_ROOT / "ref-data" / "naturalbench_ret_train.jsonl", "image"),
        ("vlguard(conv)", PROJECT_ROOT / "dataset" / "converted_from_data" / "vlguard" / "train.json", "image"),
        ("mhalu(ref)", PROJECT_ROOT / "ref-data" / "mahalu_train.json", "image_path"),
    ]:
        if not p.exists():
            issues.append(f"{name}: annotation not found: {p}")
            continue
        if p.name.endswith(".jsonl"):
            items = [json.loads(line) for line in open(p, encoding="utf-8")][:10]
        else:
            items = json.load(open(p, encoding="utf-8"))[:10]
        paths = [remap_path(x[key]) for x in items]
        ok, n, missing = check_paths_exist(paths, 5)
        if ok < n:
            issues.append(f"{name}: remapped image paths not found (sample {n-ok}/{n})")
        summary.append(f"  {name}: {len(items)} samples checked, paths exist: {ok}/{n}")

    # 2. Blink (no path remap for images - uses BLINK_DIR)
    p_ref = PROJECT_ROOT / "ref-data" / "blink_art_style_train.json"
    if p_ref.exists():
        _ = json.load(open(p_ref, encoding="utf-8"))[:3]
        blink_images = DATA_ROOT / "blink"
        has_task_dirs = any((blink_images / x).exists() for x in ["Art_Style", "Jigsaw", "Relative_Depth", "Visual_Similarity"])
        summary.append(f"  blink(ref): images in {blink_images}, has_task_dirs: {has_task_dirs}")

    # 3. Image classification (direct dirs)
    dirs = [
        ("flowers102", FLOWERS102_DIR),
        ("pets", PETS_DIR),
        ("cub200", CUB200_DIR),
        ("eurosat", EUROSAT_DIR),
        ("tiny_imagenet", TINY_IMAGENET_DIR),
    ]
    for name, d in dirs:
        if d.exists():
            n_files = sum(1 for _ in d.rglob("*") if _.is_file())
            summary.append(f"  {name}: {d}, {n_files} files")
        else:
            issues.append(f"{name}: {d} not found")

    # 4. COCO
    if COCO_VAL2017_DIR.exists():
        n = sum(1 for _ in COCO_VAL2017_DIR.rglob("*.jpg") if _.is_file())
        summary.append(f"  coco/val2017: {n} jpg files")
    else:
        issues.append(f"coco/val2017: {COCO_VAL2017_DIR} not found")

    # 5. Converted outputs quick summary
    conv_root = PROJECT_ROOT / "dataset" / "converted_from_data"
    if conv_root.exists():
        for d in sorted([p for p in conv_root.iterdir() if p.is_dir()]):
            summary.append(f"  [converted] {d.name}/")

    # 6. Potential missing: SugarCrepe, CameraBench, VizWiz, EuroSAT/Pets (from dataset/)
    sugarcrepe = DATASET_ROOT / "sugarcrepe_testset"
    camerabench = DATASET_ROOT / "camerabench_trainset"
    if sugarcrepe.exists():
        summary.append(f"  sugarcrepe: annotations in dataset/, needs COCO val2017")
    if camerabench.exists():
        summary.append(f"  camerabench: annotations in dataset/, needs video files (not in data/)")

    print("\n=== Dataset check ===\n")
    for s in summary:
        print(s)
    if issues:
        print("\nIssues:")
        for i in issues:
            print(f"  - {i}")
        print("\nNote: naturalbench/vlguard from HF use Arrow format (images in .arrow).")
        print("dataset/ annotations expect loose images. Use HF load_dataset or extract images from Arrow.")
    else:
        print("\nNo critical issues.")
    return 0 if not issues else 1


if __name__ == "__main__":
    sys.exit(main())
