#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="$PROJECT_ROOT/swap/paper/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
LOG_FILE="$LOG_DIR/${TIMESTAMP}_c2_c4_overnight.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "[INFO] starting overnight paper batch"
echo "[INFO] log file: $LOG_FILE"

bash scripts/paper/run_c2_ablations.sh "$@"
bash scripts/paper/run_c3_efficiency.sh "$@"
bash scripts/paper/run_c4_cross_model.sh "$@"

echo "[INFO] overnight paper batch completed"