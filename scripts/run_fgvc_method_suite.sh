#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen2_vl}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
DATASET_FILTER="${DATASET_FILTER:-}"
METHOD_FILTER="${METHOD_FILTER:-}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

SWAP_ROOT="$PROJECT_ROOT/swap"
SUBSET_ROOT="$SWAP_ROOT/subsets/${TIMESTAMP}_fgvc_method_suite"
LOG_ROOT="$SWAP_ROOT/logs/${TIMESTAMP}_fgvc_method_suite"
OUTPUT_ROOT="$SWAP_ROOT/outputs/${TIMESTAMP}_fgvc_method_suite"
RECORD_ROOT="$SWAP_ROOT/records"
MANIFEST_PATH="$OUTPUT_ROOT/manifest.tsv"
SUMMARY_PATH="$RECORD_ROOT/fgvc_method_suite_${TIMESTAMP}.md"

mkdir -p "$SUBSET_ROOT" "$LOG_ROOT" "$OUTPUT_ROOT" "$RECORD_ROOT"

matches_filter() {
  local dataset_name="$1"
  if [[ -z "$DATASET_FILTER" ]]; then
    return 0
  fi
  [[ "$dataset_name" =~ $DATASET_FILTER ]]
}

matches_method_filter() {
  local method_id="$1"
  local display_name="$2"
  if [[ -z "$METHOD_FILTER" ]]; then
    return 0
  fi
  [[ "$method_id" =~ $METHOD_FILTER || "$display_name" =~ $METHOD_FILTER ]]
}

build_subset_file() {
  local dataset_name="$1"
  local src_path="$2"
  local dst_path="$3"
  local mode="$4"
  local count="$5"
  local exclude_path="$6"
  local restrict_labels_from="$7"
  local meta_path="$8"

  local -a cmd=(
    uv run python scripts/build_author_subset.py
    --dataset-name "$dataset_name"
    --src "$src_path"
    --dst "$dst_path"
    --mode "$mode"
    --count "$count"
    --group-size 0
    --meta-path "$meta_path"
  )

  if [[ -n "$exclude_path" ]]; then
    cmd+=(--exclude-path "$exclude_path")
  fi
  if [[ -n "$restrict_labels_from" ]]; then
    cmd+=(--restrict-labels-from "$restrict_labels_from")
  fi

  "${cmd[@]}"
}

validate_subset() {
  local dataset_name="$1"
  local train_path="$2"
  local val_path="$3"
  uv run python - "$dataset_name" "$train_path" "$val_path" <<'PY'
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_prompt, load_train_val

dataset_name = sys.argv[1]
train_path = sys.argv[2]
val_path = sys.argv[3]

train_data, val_data = load_train_val(dataset_name, train_path, val_path)

if not train_data:
    raise ValueError("Train subset is empty")
if not val_data:
    raise ValueError("Validation subset is empty")

for split_name, samples in [("train", train_data), ("val", val_data)]:
    for idx, sample in enumerate(samples):
        prompt = build_prompt(dataset_name, sample)
        if not str(prompt).strip():
            raise ValueError(f"{split_name}[{idx}] produced an empty prompt")
        for image_path in sample.get("images") or [sample.get("image")]:
            if not Path(image_path).exists():
                raise FileNotFoundError(f"{split_name}[{idx}] missing image: {image_path}")

if {row["question_id"] for row in train_data} & {row["question_id"] for row in val_data}:
    print("warning: train/val question_id overlap detected")

print(f"train={len(train_data)} val={len(val_data)} labels={len({row['label'] for row in train_data})}")
PY
}

run_method() {
  local dataset_name="$1"
  local train_path="$2"
  local val_path="$3"
  local method_id="$4"
  local display_name="$5"
  local method_name="$6"
  local extra_args_spec="$7"
  local log_path="$8"

  local run_name="${dataset_name}_${method_id}_${MODEL}_${TIMESTAMP}"
  local -a cmd=(
    uv run python main.py
    "model=${MODEL}"
    "dataset=general_custom"
    "method=${method_name}"
    "evaluator=raw"
    "dataset.name=${dataset_name}"
    "dataset.train_path=${train_path}"
    "dataset.val_path=${val_path}"
    "run.output_dir=${OUTPUT_ROOT}"
    "run.run_name=${run_name}"
    "run.progress_bar=false"
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
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$dataset_name" \
    "$method_id" \
    "$display_name" \
    "$run_name" \
    "$train_path" \
    "$val_path" \
    "$metrics_path" \
    "$predictions_path" \
    "$log_path" >> "$MANIFEST_PATH"
}

printf 'dataset_name\tmethod_id\tdisplay_name\trun_name\ttrain_subset\tval_subset\tmetrics_path\tpredictions_path\tlog_path\n' > "$MANIFEST_PATH"

DATASET_SPECS=(
  "eurosat|dataset/converted_from_data/eurosat/train.json|dataset/converted_from_data/eurosat/test.json|10|10|10"
  "pets|dataset/converted_from_data/pets/train.json|dataset/converted_from_data/pets/test.json|20|5|5"
  "cub|dataset/converted_from_data/cub/train.json|dataset/converted_from_data/cub/test.json|20|5|5"
)

METHOD_SPECS=(
  "zero_shot|Zero-shot|zero_shot|"
  "keco|KeCO|keco|"
  "sav|SAV|sav|"
  "mimic|MimIC|mimic|"
  "i2cl|I2CL|i2cl|"
  "stv|STV|stv|method.params.head_selection_mode=sensitivity;method.params.cluster_selection_mode=rl"
  "stv_qc|STV+QC|stv|method.params.head_selection_mode=sensitivity;method.params.cluster_selection_mode=query_adaptive"
  "sav_tv|SAV-TV|stv|method.params.head_selection_mode=sav_accuracy;method.params.cluster_selection_mode=rl"
  "sav_tv_qc|SAV-TV+QC|stv|method.params.head_selection_mode=sav_accuracy;method.params.cluster_selection_mode=query_adaptive"
)

for dataset_entry in "${DATASET_SPECS[@]}"; do
  IFS='|' read -r dataset_name train_src val_src class_count train_per_class val_per_class <<< "$dataset_entry"
  if ! matches_filter "$dataset_name"; then
    continue
  fi

  dataset_dir="$SUBSET_ROOT/$dataset_name"
  mkdir -p "$dataset_dir"
  suffix="${train_src##*.}"

  label_subset="$dataset_dir/label_seed.${suffix}"
  train_subset="$dataset_dir/train_subset.${suffix}"
  val_subset="$dataset_dir/val_subset.${suffix}"

  build_subset_file "$dataset_name" "$train_src" "$label_subset" "distinct_labels" "$class_count" "" "" "$dataset_dir/label_seed.meta.json"
  build_subset_file "$dataset_name" "$train_src" "$train_subset" "per_label" "$train_per_class" "" "$label_subset" "$dataset_dir/train_subset.meta.json"
  build_subset_file "$dataset_name" "$val_src" "$val_subset" "per_label" "$val_per_class" "$train_subset" "$train_subset" "$dataset_dir/val_subset.meta.json"
  validate_subset "$dataset_name" "$train_subset" "$val_subset"

  for method_entry in "${METHOD_SPECS[@]}"; do
    IFS='|' read -r method_id display_name method_name extra_args_spec <<< "$method_entry"
    if ! matches_method_filter "$method_id" "$display_name"; then
      continue
    fi
    log_path="$LOG_ROOT/${dataset_name}_${method_id}.log"
    run_method "$dataset_name" "$train_subset" "$val_subset" "$method_id" "$display_name" "$method_name" "$extra_args_spec" "$log_path"
  done
done

uv run python scripts/summarize_fgvc_method_suite.py \
  --manifest "$MANIFEST_PATH" \
  --output "$SUMMARY_PATH"

printf 'manifest=%s\nsummary=%s\n' "$MANIFEST_PATH" "$SUMMARY_PATH"
