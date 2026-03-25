#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen2_vl}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
METHOD_FILTER="${METHOD_FILTER:-}"
DATASET_FILTER="${DATASET_FILTER:-}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
BASE_MANIFEST="${BASE_MANIFEST:-$PROJECT_ROOT/swap/outputs/20260325_040658_rse_expanded_suite/manifest.tsv}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

OUTPUT_ROOT="$PROJECT_ROOT/swap/outputs/${TIMESTAMP}_rse_adaptive_combo_suite"
LOG_ROOT="$PROJECT_ROOT/swap/logs/${TIMESTAMP}_rse_adaptive_combo_suite"
RECORD_ROOT="$PROJECT_ROOT/swap/records"
MANIFEST_PATH="$OUTPUT_ROOT/manifest.tsv"
SUMMARY_PATH="$RECORD_ROOT/rse_adaptive_combo_suite_${TIMESTAMP}.md"

mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT" "$RECORD_ROOT"

matches_method_filter() {
  local method_id="$1"
  local display_name="$2"
  if [[ -z "$METHOD_FILTER" ]]; then
    return 0
  fi
  [[ "$method_id" =~ $METHOD_FILTER || "$display_name" =~ $METHOD_FILTER ]]
}

matches_dataset_filter() {
  local dataset_name="$1"
  if [[ -z "$DATASET_FILTER" ]]; then
    return 0
  fi
  [[ "$dataset_name" =~ $DATASET_FILTER ]]
}

run_method() {
  local dataset_name="$1"
  local evaluator_name="$2"
  local train_path="$3"
  local val_path="$4"
  local method_id="$5"
  local display_name="$6"
  local method_name="$7"
  local extra_args_spec="$8"
  local log_path="$9"

  local run_name="${dataset_name}_${method_id}_${MODEL}_${TIMESTAMP}"
  local -a cmd=(
    uv run python main.py
    "model=${MODEL}"
    "dataset=general_custom"
    "method=${method_name}"
    "evaluator=${evaluator_name}"
    "dataset.name=${dataset_name}"
    "dataset.train_path=${train_path}"
    "dataset.val_path=${val_path}"
    "run.output_dir=${OUTPUT_ROOT}"
    "run.run_name=${run_name}"
    "run.progress_bar=false"
    "run.save_predictions=${SAVE_PREDICTIONS}"
  )

  if [[ "$method_name" != "zero_shot" ]]; then
    cmd+=("method.params.progress_bar=false")
  fi

  if [[ -n "$extra_args_spec" ]]; then
    IFS=';' read -r -a extra_args <<< "$extra_args_spec"
    for arg in "${extra_args[@]}"; do
      if [[ -n "$arg" ]]; then
        cmd+=("$arg")
      fi
    done
  fi

  if ! "${cmd[@]}" > "$log_path" 2>&1; then
    echo "FAILED: ${dataset_name} / ${display_name}" >&2
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    return 1
  fi

  local metrics_path="${OUTPUT_ROOT}/${run_name}.metrics.json"
  local predictions_path="${OUTPUT_ROOT}/${run_name}.predictions.jsonl"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$dataset_name" \
    "$dataset_name" \
    "$evaluator_name" \
    "$method_id" \
    "$display_name" \
    "$run_name" \
    "$train_path" \
    "$val_path" \
    "$metrics_path" \
    "$predictions_path" >> "$MANIFEST_PATH"
}

if [[ ! -f "$BASE_MANIFEST" ]]; then
  echo "Base manifest not found: $BASE_MANIFEST" >&2
  exit 1
fi

printf 'experiment_id\tdataset_name\tevaluator_name\tmethod_id\tdisplay_name\trun_name\ttrain_subset\tval_subset\tmetrics_path\tpredictions_path\n' > "$MANIFEST_PATH"

METHOD_SPECS=(
  "rse_adaptive_only|RSE-AdaptiveOnly|rse|method.params.selection_metric=loo_accuracy;method.params.routing_mode=adaptive;method.params.adaptive_margin_quantile=0.6;method.params.score_normalization=none"
  "rse_adaptive_pca|RSE-Adaptive+PCA|rse|method.params.selection_metric=loo_accuracy;method.params.routing_mode=adaptive;method.params.adaptive_margin_quantile=0.6;method.params.score_normalization=none;method.params.feature_reduction=pca;method.params.pca_dim=64"
)

while IFS=$'\t' read -r dataset_name evaluator_name train_subset val_subset; do
  if ! matches_dataset_filter "$dataset_name"; then
    continue
  fi

  for method_entry in "${METHOD_SPECS[@]}"; do
    IFS='|' read -r method_id display_name method_name extra_args_spec <<< "$method_entry"
    if ! matches_method_filter "$method_id" "$display_name"; then
      continue
    fi
    log_path="$LOG_ROOT/${dataset_name}_${method_id}.log"
    run_method "$dataset_name" "$evaluator_name" "$train_subset" "$val_subset" "$method_id" "$display_name" "$method_name" "$extra_args_spec" "$log_path"
  done
done < <(
  python - "$BASE_MANIFEST" <<'PY'
import csv
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
rows = list(csv.DictReader(manifest_path.open("r", encoding="utf-8"), delimiter="\t"))
seen = {}
for row in rows:
    dataset_name = row["dataset_name"]
    if dataset_name in seen:
        continue
    seen[dataset_name] = row

for dataset_name in sorted(seen):
    row = seen[dataset_name]
    print(
        "\t".join(
            [
                dataset_name,
                row["evaluator_name"],
                row["train_subset"],
                row["val_subset"],
            ]
        )
    )
PY
)

uv run python scripts/summarize_rse_improvement_suite.py \
  --manifest "$BASE_MANIFEST" \
  --manifest "$MANIFEST_PATH" \
  --output "$SUMMARY_PATH"

printf 'manifest=%s\nsummary=%s\n' "$MANIFEST_PATH" "$SUMMARY_PATH"
