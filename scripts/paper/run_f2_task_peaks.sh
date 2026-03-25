#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/paper/common.sh"

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
SUITE_NAME="f2_task_peaks"
MODELS="${MODELS:-qwen2_vl,qwen3_vl,idefics3}"
SEEDS="${SEEDS:-42}"
TASK_GROUPS="${TASK_GROUPS:-all_tasks_current}"
METHOD_GROUPS="${METHOD_GROUPS:-task_peaks}"
PRIMARY_SUMMARY_PATH="$PROJECT_ROOT/swap/paper/records/${SUITE_NAME}_${TIMESTAMP}.md"
CSV_PATH="$PROJECT_ROOT/swap/paper/records/${SUITE_NAME}_${TIMESTAMP}_components.csv"
PEAK_SUMMARY_PATH="$PROJECT_ROOT/swap/paper/records/${SUITE_NAME}_${TIMESTAMP}_peaks.md"
EXTRA_NOTIFY_PATHS="${CSV_PATH};${PEAK_SUMMARY_PATH}"

trap 'paper_notify_on_exit "$?" "$SUITE_NAME" "$TIMESTAMP" "$MODELS" "$SEEDS" "$TASK_GROUPS" "$METHOD_GROUPS" "$PRIMARY_SUMMARY_PATH" "$EXTRA_NOTIFY_PATHS"' EXIT

uv run python scripts/paper/run_suite.py \
  --suite-name "$SUITE_NAME" \
  --timestamp "$TIMESTAMP" \
  --task-groups "$TASK_GROUPS" \
  --method-groups "$METHOD_GROUPS" \
  --models "$MODELS" \
  --seeds "$SEEDS" \
  "$@"

MANIFEST_PATH="$PROJECT_ROOT/swap/paper/outputs/${TIMESTAMP}_${SUITE_NAME}/manifest.tsv"

uv run python scripts/paper/export_component_tables.py \
  --manifest "$MANIFEST_PATH" \
  --output-csv "$CSV_PATH" \
  --summary-output "$PEAK_SUMMARY_PATH"

printf 'component_csv=%s\npeak_summary=%s\n' "$CSV_PATH" "$PEAK_SUMMARY_PATH"
