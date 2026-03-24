#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FGVC

MODEL="${MODEL:-qwen2_vl}"
NUM_HEADS="${NUM_HEADS:-20}"

uv run python main.py \
  model="$MODEL" \
  dataset=naturalbench_ret \
  method=sav \
  evaluator=auto \
  method.params.num_heads="$NUM_HEADS" \
  run.run_name="sav_naturalbench_ret_${MODEL}_full"

