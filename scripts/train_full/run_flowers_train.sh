#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FGVC

MODEL="${MODEL:-qwen2_vl}"
NUM_HEADS="${NUM_HEADS:-20}"

uv run python main.py \
  model="$MODEL" \
  dataset=general_custom \
  method=sav \
  evaluator=raw \
  dataset.name=flowers \
  dataset.train_path=dataset/converted_from_data/flowers/train.json \
  dataset.val_path=dataset/converted_from_data/flowers/test.json \
  method.params.num_heads="$NUM_HEADS" \
  run.run_name="sav_flowers_${MODEL}_full"
