#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FGVC

MODEL="${MODEL:-qwen2_vl}"
NUM_HEADS="${NUM_HEADS:-20}"

uv run python main.py \
  model="$MODEL" \
  dataset=naturalbench_vqa \
  method=sav \
  evaluator=raw \
  method.params.num_heads="$NUM_HEADS" \
  run.run_name="sav_naturalbench_vqa_${MODEL}_full"

