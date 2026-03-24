"""Path configuration for FGVC datasets. Override via environment variables."""

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("FGVC_PROJECT_ROOT", Path(__file__).resolve().parent))

# Default to `Data/` (uppercase) if it exists, otherwise `data/`.
_default_data_root = PROJECT_ROOT / "data"
if (PROJECT_ROOT / "Data").exists() and not _default_data_root.exists():
    _default_data_root = PROJECT_ROOT / "Data"

DATA_ROOT = Path(os.environ.get("FGVC_DATA_ROOT", _default_data_root))
DATASET_ROOT = Path(os.environ.get("FGVC_DATASET_ROOT", PROJECT_ROOT / "dataset"))

def _pick_subdir(root: Path, *candidates: str) -> Path:
    """
    Pick the first existing subdir under `root` from candidates.
    Fallback to the first candidate path (even if missing).
    """
    for name in candidates:
        p = root / name
        if p.exists():
            return p
    return root / candidates[0]


# Per-dataset directories under DATA_ROOT (support both `data/` and `Data/` layouts)
NATURAL_BENCH_DIR = _pick_subdir(DATA_ROOT, "natural_bench", "NaturalBench")
BLINK_DIR = _pick_subdir(DATA_ROOT, "blink", "BLINK")
BLINK_IMAGES_DIR = _pick_subdir(BLINK_DIR, "images")  # BLINK HF export uses images/
M_HALU_DIR = _pick_subdir(DATA_ROOT, "m_halu_bench", "MHaluBench")
VL_GUARD_DIR = _pick_subdir(DATA_ROOT, "vl_guard", "VLGuard")
FLOWERS102_DIR = _pick_subdir(DATA_ROOT, "flowers102", "Flowers")
PETS_DIR = _pick_subdir(DATA_ROOT, "pets", "Pets")
CUB200_DIR = _pick_subdir(DATA_ROOT, "cub200", "Cub")
EUROSAT_DIR = _pick_subdir(DATA_ROOT, "eurosat", "Eurosat")
TINY_IMAGENET_DIR = _pick_subdir(DATA_ROOT, "tiny_imagenet", "TinyImage")
COCO_VAL2017_DIR = _pick_subdir(DATA_ROOT, "coco", "Coco2017") / "val2017"

# Path remapping: old prefix -> new root (Path)
# Used to rewrite absolute paths in annotations to local extracted paths
PATH_REMAP = {
    "/home/chancharikm/vqascore_data/naturalbench": NATURAL_BENCH_DIR,
    "/home/zhaobin/Qwen-VL/data/vlguard": VL_GUARD_DIR,
    "/home/zhaobin/Qwen-VL/data/blink/images": BLINK_DIR / "images",
    # M-HaluBench: we store patched images by basename under MHaluBench/data/image-to-text/
    "/home/zhaobin/Qwen-VL/data/hallucination/images/data/image-to-text": M_HALU_DIR,
    "/datasets/coco2014_2024-02-22_2010": M_HALU_DIR,  # COCO2014 train
    "/home/zhaobin/Emu/dataset/textvqa/train_images": M_HALU_DIR,  # TextVQA
    "/TextVQA/train_images": M_HALU_DIR,  # converted MHaluBench snapshots
    "/datasets/coco2017_2024-01-05_0047/val2017": COCO_VAL2017_DIR,
}


def remap_path(old_path: str, remap: dict = None) -> str:
    """
    Remap an absolute path from annotation to local extracted path.
    Returns the new path if a prefix matches; otherwise returns old_path unchanged.
    """
    if remap is None:
        remap = PATH_REMAP
    for prefix, new_root in remap.items():
        if old_path.startswith(prefix):
            # Special-case M-HaluBench sources: keep everything in one place by basename.
            if Path(new_root) == M_HALU_DIR and prefix in {
                "/home/zhaobin/Qwen-VL/data/hallucination/images/data/image-to-text",
                "/datasets/coco2014_2024-02-22_2010",
                "/home/zhaobin/Emu/dataset/textvqa/train_images",
                "/TextVQA/train_images",
            }:
                return str((M_HALU_DIR / "data" / "image-to-text" / Path(old_path).name).resolve())

            rel = old_path[len(prefix) :].lstrip("/")
            return str(Path(new_root) / rel)
    return old_path
