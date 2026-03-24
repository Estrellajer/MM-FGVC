#!/usr/bin/env bash
set -euo pipefail

# One-click dataset rebuild / verification.
#
# Runs:
# 1) MHaluBench missing image fix (copy from COCO/TextVQA into Data/MHaluBench)
# 2) Convert datasets under Data/ into dataset/converted_from_data/*
# 3) Verify/check dataset schemas and path remapping
#
# You can configure the source image roots via env vars:
#   COCO2014_ROOT  (e.g. /root/autodl-tmp/train2014)
#   COCO2017_ROOT  (optional; only needed if you also want to fix images referenced by extra MHalu files)
#   TEXTVQA_ROOT   (e.g. /root/autodl-tmp/train_val_images)
#
# Optional:
#   MHALU_ANN        (default: Data/MHaluBench/MHaluBench_train.json)
#   MHALU_EXTRA_ANNS (default: official MHaluBench val files)
#   MHALU_NLG_ANN  (e.g. Data/MHaluBench/test_for_nlpcc.json)
#
# And you can limit conversion to specific datasets:
#   DATASETS="mhalubench vlguard blink naturalbench"  (default: all)
#
# Example:
#   COCO2014_ROOT=/root/autodl-tmp/train2014 \
#   TEXTVQA_ROOT=/root/autodl-tmp/train_val_images \
#   DATASETS="mhalubench vlguard blink naturalbench" \
#   bash scripts/rebuild_datasets_one_click.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

COCO2014_ROOT="${COCO2014_ROOT:-}"
COCO2017_ROOT="${COCO2017_ROOT:-}"
TEXTVQA_ROOT="${TEXTVQA_ROOT:-}"
MHALU_ANN="${MHALU_ANN:-Data/MHaluBench/MHaluBench_train.json}"
MHALU_EXTRA_ANNS="${MHALU_EXTRA_ANNS:-Data/MHaluBench/MHaluBench_val-v0.1.json Data/MHaluBench/MHaluBench_val-v0.2.json}"
MHALU_NLG_ANN="${MHALU_NLG_ANN:-}"
DATASETS="${DATASETS:-all}"

echo "[one-click] project_root: $PROJECT_ROOT"
echo "[one-click] COCO2014_ROOT: ${COCO2014_ROOT:-'(unset)'}"
echo "[one-click] COCO2017_ROOT: ${COCO2017_ROOT:-'(unset)'}"
echo "[one-click] TEXTVQA_ROOT: ${TEXTVQA_ROOT:-'(unset)'}"
echo "[one-click] MHALU_ANN: ${MHALU_ANN:-'(unset)'}"
echo "[one-click] MHALU_EXTRA_ANNS: ${MHALU_EXTRA_ANNS:-'(unset)'}"
echo "[one-click] MHALU_NLG_ANN: ${MHALU_NLG_ANN:-'(unset)'}"
echo "[one-click] DATASETS: $DATASETS"

echo
echo "[one-click] Step 1/3: fix MHaluBench missing images"

fix_args=(uv run python scripts/fix_mhalubench_missing_images.py)
if [[ -n "$MHALU_ANN" ]]; then
  fix_args+=("--mahalu-ann" "$MHALU_ANN")
fi
if [[ -n "$MHALU_EXTRA_ANNS" ]]; then
  # shellcheck disable=SC2206
  extra_ann_list=($MHALU_EXTRA_ANNS)
  for extra_ann in "${extra_ann_list[@]}"; do
    if [[ -n "$extra_ann" ]]; then
      fix_args+=("--extra-ann" "$extra_ann")
    fi
  done
fi
if [[ -n "$MHALU_NLG_ANN" ]]; then
  fix_args+=("--nlg-ann" "$MHALU_NLG_ANN")
fi
if [[ -n "$COCO2014_ROOT" ]]; then
  fix_args+=("--coco2014-root" "$COCO2014_ROOT")
fi
if [[ -n "$COCO2017_ROOT" ]]; then
  fix_args+=("--coco2017-root" "$COCO2017_ROOT")
fi
if [[ -n "$TEXTVQA_ROOT" ]]; then
  fix_args+=("--textvqa-root" "$TEXTVQA_ROOT")
fi

if [[ -z "$COCO2014_ROOT" && -z "$COCO2017_ROOT" && -z "$TEXTVQA_ROOT" ]]; then
  echo "[one-click][warn] No source roots provided. Skipping MHaluBench fix."
  echo "[one-click][warn] Set COCO2014_ROOT and TEXTVQA_ROOT to enable this step."
else
  "${fix_args[@]}"
fi

echo
echo "[one-click] Step 2/3: convert datasets under Data/"

convert_args=(uv run python scripts/convert_data_to_json.py --data-root Data)
if [[ "$DATASETS" == "all" ]]; then
  convert_args+=(--datasets all)
else
  # shellcheck disable=SC2206
  ds_list=($DATASETS)
  convert_args+=(--datasets "${ds_list[@]}")
fi

"${convert_args[@]}"

echo
echo "[one-click] Step 3/3: verify + check"
uv run python scripts/verify_datasets.py
uv run python scripts/check_datasets.py || true

echo
echo "[one-click] DONE"
