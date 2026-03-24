#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FGVC

MODEL="${MODEL:-qwen2_vl}"
NUM_HEADS="${NUM_HEADS:-20}"

uv run python main.py \
  model="$MODEL" \
  dataset=sugarcrepe_sample \
  method=sav \
  evaluator=auto \
  dataset.train_path=dataset/converted_from_data/sugarcrepe/sugarcrepe_all.jsonl \
  dataset.val_path=dataset/converted_from_data/sugarcrepe/sugarcrepe_all.jsonl \
  method.params.num_heads="$NUM_HEADS" \
  run.run_name="sav_sugarcrepe_${MODEL}_full"
