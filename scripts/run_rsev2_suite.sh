#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen2_vl}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
DATASET_FILTER="${DATASET_FILTER:-}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
BASE_MANIFEST="${BASE_MANIFEST:-$PROJECT_ROOT/swap/outputs/20260325_040658_rse_expanded_suite/manifest.tsv}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

SWAP_ROOT="$PROJECT_ROOT/swap"
LOG_ROOT="$SWAP_ROOT/logs/${TIMESTAMP}_rsev2_suite"
OUTPUT_ROOT="$SWAP_ROOT/outputs/${TIMESTAMP}_rsev2_suite"
RECORD_ROOT="$SWAP_ROOT/records"
MANIFEST_PATH="$OUTPUT_ROOT/manifest.tsv"
SUMMARY_PATH="$RECORD_ROOT/rsev2_suite_${TIMESTAMP}.md"
TMP_EXPERIMENTS="$OUTPUT_ROOT/experiments.tsv"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$RECORD_ROOT"

if [[ ! -f "$BASE_MANIFEST" ]]; then
  echo "Missing base manifest: $BASE_MANIFEST" >&2
  exit 1
fi

python - <<'PY' "$BASE_MANIFEST" "$TMP_EXPERIMENTS"
import csv
import sys
from pathlib import Path

base_manifest = Path(sys.argv[1])
output_path = Path(sys.argv[2])
seen = set()
rows = []
with base_manifest.open("r", encoding="utf-8") as fp:
    reader = csv.DictReader(fp, delimiter="\t")
    for row in reader:
        key = (
            row["experiment_id"],
            row["dataset_name"],
            row["evaluator_name"],
            row["train_subset"],
            row["val_subset"],
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)

with output_path.open("w", encoding="utf-8", newline="") as fp:
    writer = csv.writer(fp, delimiter="\t", lineterminator="\n")
    writer.writerow(["experiment_id", "dataset_name", "evaluator_name", "train_subset", "val_subset"])
    writer.writerows(rows)
PY

run_method() {
  local experiment_id="$1"
  local dataset_name="$2"
  local evaluator_name="$3"
  local train_path="$4"
  local val_path="$5"
  local log_path="$6"
  local run_name="${experiment_id}_rsev2_${MODEL}_${TIMESTAMP}"
  local -a cmd=(
    uv run python main.py
    "model=${MODEL}"
    "dataset=general_custom"
    "method=rsev2"
    "evaluator=${evaluator_name}"
    "dataset.name=${dataset_name}"
    "dataset.train_path=${train_path}"
    "dataset.val_path=${val_path}"
    "run.output_dir=${OUTPUT_ROOT}"
    "run.run_name=${run_name}"
    "run.progress_bar=false"
    "run.save_predictions=${SAVE_PREDICTIONS}"
    "method.params.progress_bar=false"
  )

  if ! "${cmd[@]}" > "$log_path" 2>&1; then
    echo "FAILED: ${experiment_id} / RSEv2" >&2
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    return 1
  fi

  local metrics_path="${OUTPUT_ROOT}/${run_name}.metrics.json"
  local predictions_path="${OUTPUT_ROOT}/${run_name}.predictions.jsonl"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_id" \
    "$dataset_name" \
    "$evaluator_name" \
    "rsev2" \
    "RSEv2" \
    "$run_name" \
    "$train_path" \
    "$val_path" \
    "$metrics_path" \
    "$predictions_path" >> "$MANIFEST_PATH"
}

printf 'experiment_id\tdataset_name\tevaluator_name\tmethod_id\tdisplay_name\trun_name\ttrain_subset\tval_subset\tmetrics_path\tpredictions_path\n' > "$MANIFEST_PATH"

while IFS=$'\t' read -r experiment_id dataset_name evaluator_name train_subset val_subset; do
  if [[ "$experiment_id" == "experiment_id" ]]; then
    continue
  fi
  if [[ -n "$DATASET_FILTER" && ! "$experiment_id" =~ $DATASET_FILTER && ! "$dataset_name" =~ $DATASET_FILTER ]]; then
    continue
  fi
  log_path="$LOG_ROOT/${experiment_id}_rsev2.log"
  run_method "$experiment_id" "$dataset_name" "$evaluator_name" "$train_subset" "$val_subset" "$log_path"
done < "$TMP_EXPERIMENTS"

uv run python scripts/summarize_rse_improvement_suite.py \
  --manifest "$BASE_MANIFEST" \
  --manifest "$MANIFEST_PATH" \
  --output "$SUMMARY_PATH"

echo "RSEv2 suite complete"
echo "Manifest: $MANIFEST_PATH"
echo "Summary: $SUMMARY_PATH"
