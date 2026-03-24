#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FGVC

MODEL="${MODEL:-qwen2_vl}"
NUM_HEADS="${NUM_HEADS:-20}"

TASKS=(
  art_style
  counting
  forensic_detection
  functional_correspondence
  iq_test
  jigsaw
  multi-view_reasoning
  object_localization
  relative_depth
  relative_reflectance
  semantic_correspondence
  spatial_relation
  visual_correspondence
  visual_similarity
)

for TASK in "${TASKS[@]}"; do
  echo "[blink] running task: ${TASK}"
  uv run python main.py \
    model="$MODEL" \
    dataset=general_custom \
    method=sav \
    evaluator=raw \
    dataset.name="blink_${TASK}" \
    dataset.train_path="dataset/converted_from_data/blink/${TASK}_val.json" \
    dataset.val_path="dataset/converted_from_data/blink/${TASK}_test.json" \
    method.params.num_heads="$NUM_HEADS" \
    run.run_name="sav_blink_${TASK}_${MODEL}_full"
done
