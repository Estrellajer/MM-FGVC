#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/paper/common.sh"

TIMESTAMP="${TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
SUITE_NAME="c3_efficiency"
MODELS="${MODELS:-qwen2_vl,qwen3_vl,idefics3}"
SEEDS="${SEEDS:-42}"
TASK_GROUPS="${TASK_GROUPS:-efficiency_core6}"
METHOD_GROUPS="${METHOD_GROUPS:-efficiency}"
SUMMARY_PATH="$PROJECT_ROOT/swap/paper/records/${SUITE_NAME}_${TIMESTAMP}.md"
EXTRA_NOTIFY_PATHS=""

trap 'paper_notify_on_exit "$?" "$SUITE_NAME" "$TIMESTAMP" "$MODELS" "$SEEDS" "$TASK_GROUPS" "$METHOD_GROUPS" "$SUMMARY_PATH" "$EXTRA_NOTIFY_PATHS"' EXIT

uv run python scripts/paper/run_suite.py \
  --suite-name "$SUITE_NAME" \
  --timestamp "$TIMESTAMP" \
  --task-groups "$TASK_GROUPS" \
  --method-groups "$METHOD_GROUPS" \
  --models "$MODELS" \
  --seeds "$SEEDS" \
  "$@"
