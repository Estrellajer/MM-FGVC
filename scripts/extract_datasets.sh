#!/bin/bash
# Extract dataset zip/tar/tgz files from data/ to data/ with canonical directory names.
# Run from project root: bash scripts/extract_datasets.sh
# Expects archives in data/ (or pass DATA_ROOT). Extracts directly under data/.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="${FGVC_DATA_ROOT:-${PROJECT_ROOT}/data}"
TMP="${DATA_ROOT}/.tmp_extract"

mkdir -p "$DATA_ROOT"
cd "$DATA_ROOT"

extract_zip() {
    local zip_path="$1"
    local target_dir="$2"
    if [[ ! -f "$zip_path" ]]; then
        echo "Skip: $zip_path not found"
        return 0
    fi
    echo "Extracting $(basename "$zip_path") -> $target_dir"
    rm -rf "$TMP"
    mkdir -p "$(dirname "$target_dir")"
    unzip -o -q "$zip_path" -d "$TMP" 2>/dev/null || true
    if [[ -d "$TMP" ]]; then
        first_dir=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -1)
        if [[ -n "$first_dir" ]]; then
            rm -rf "$target_dir"
            mv "$first_dir" "$target_dir"
        else
            mkdir -p "$target_dir"
            mv "$TMP"/* "$target_dir/" 2>/dev/null || true
        fi
        rm -rf "$TMP"
    fi
}

extract_tar() {
    local tar_path="$1"
    local target_dir="$2"
    if [[ ! -f "$tar_path" ]]; then
        echo "Skip: $tar_path not found"
        return 0
    fi
    echo "Extracting $(basename "$tar_path") -> $target_dir"
    rm -rf "$TMP"
    mkdir -p "$TMP"
    mkdir -p "$(dirname "$target_dir")"
    tar -xzf "$tar_path" -C "$TMP"
    first_dir=$(find "$TMP" -mindepth 1 -maxdepth 1 -type d | head -1)
    if [[ -n "$first_dir" ]]; then
        rm -rf "$target_dir"
        mv "$first_dir" "$target_dir"
    else
        mkdir -p "$target_dir"
        mv "$TMP"/* "$target_dir/" 2>/dev/null || true
    fi
    rm -rf "$TMP"
}

# ----- HF / existing -----
extract_zip "${DATA_ROOT}/BaiqiL___natural_bench.zip" "natural_bench"
extract_zip "${DATA_ROOT}/BLINK-Benchmark___blink.zip" "blink"
extract_zip "${DATA_ROOT}/openkg___m_halu_bench.zip" "m_halu_bench"
extract_zip "${DATA_ROOT}/ys-zong___vl_guard.zip" "vl_guard"

# ----- External: Oxford Flowers 102 -----
extract_tar "${DATA_ROOT}/102flowers.tgz" "flowers102"

# ----- External: Oxford-IIIT Pet -----
extract_tar "${DATA_ROOT}/annotations.tar.gz" "pets/annotations"
extract_tar "${DATA_ROOT}/images.tar.gz" "pets/images"

# ----- External: CUB-200, EuroSAT, Tiny-ImageNet -----
extract_zip "${DATA_ROOT}/cub200.zip" "cub200"
extract_zip "${DATA_ROOT}/EuroSAT_MS.zip" "eurosat"
extract_zip "${DATA_ROOT}/tiny-imagenet-200.zip" "tiny_imagenet"

# ----- External: COCO -----
extract_zip "${DATA_ROOT}/val2017.zip" "coco/val2017"
extract_zip "${DATA_ROOT}/val.zip" "coco/val"

echo "Done. Data structure:"
ls -la "$DATA_ROOT"
