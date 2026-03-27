#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FGVC

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[rerun] start c3_efficiency 20260327_020551"
TIMESTAMP=20260327_020551 bash scripts/paper/run_c3_efficiency.sh

echo "[rerun] start c4_cross_model 20260327_020651"
TIMESTAMP=20260327_020651 bash scripts/paper/run_c4_cross_model.sh
