#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FGVC

MODEL="${MODEL:-qwen2_vl}"
NUM_HEADS="${NUM_HEADS:-20}"

for VAL_NAME in val_v01 val_v02; do
  echo "[mhalubench] running validation split: ${VAL_NAME}"
  uv run python main.py \
    model="$MODEL" \
    dataset=general_custom \
    method=sav \
    evaluator=raw \
    dataset.name=mhalubench \
    dataset.train_path=dataset/converted_from_data/mhalubench/train.json \
    dataset.val_path="dataset/converted_from_data/mhalubench/${VAL_NAME}.json" \
    method.params.num_heads="$NUM_HEADS" \
    run.run_name="sav_mhalubench_${VAL_NAME}_${MODEL}_full"
done
