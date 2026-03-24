#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/FGVC

SCRIPTS=(
  scripts/train_full/run_naturalbench_train.sh
  scripts/train_full/run_naturalbench_ret_train.sh
  scripts/train_full/run_pets_train.sh
  scripts/train_full/run_eurosat_train.sh
  scripts/train_full/run_sugarcrepe_train.sh
  scripts/train_full/run_cub_train.sh
  scripts/train_full/run_flowers_train.sh
  scripts/train_full/run_tinyimage_train.sh
  scripts/train_full/run_vizwiz_train.sh
  scripts/train_full/run_vlguard_train.sh
  scripts/train_full/run_mhalubench_train.sh
  scripts/train_full/run_blink_train.sh
)

for script in "${SCRIPTS[@]}"; do
  echo "=============================="
  echo "Running: ${script}"
  bash "${script}"
done

echo "All non-COCO full-data training scripts finished."
