#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen2_vl}"
STOP_ON_ERROR="${STOP_ON_ERROR:-1}"
DATASET_FILTER="${DATASET_FILTER:-}"
METHOD_FILTER="${METHOD_FILTER:-}"
SAVE_PREDICTIONS="${SAVE_PREDICTIONS:-0}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

SWAP_ROOT="$PROJECT_ROOT/swap"
SUBSET_ROOT="$SWAP_ROOT/subsets/${TIMESTAMP}_all_task_method_suite"
LOG_ROOT="$SWAP_ROOT/logs/${TIMESTAMP}_all_task_method_suite"
OUTPUT_ROOT="$SWAP_ROOT/outputs/${TIMESTAMP}_all_task_method_suite"
RECORD_ROOT="$SWAP_ROOT/records"
MANIFEST_PATH="$OUTPUT_ROOT/manifest.tsv"
SUMMARY_PATH="$RECORD_ROOT/all_task_method_suite_${TIMESTAMP}.md"

mkdir -p "$SUBSET_ROOT" "$LOG_ROOT" "$OUTPUT_ROOT" "$RECORD_ROOT"

matches_filter() {
  local experiment_id="$1"
  local dataset_name="$2"
  if [[ -z "$DATASET_FILTER" ]]; then
    return 0
  fi
  [[ "$experiment_id" =~ $DATASET_FILTER || "$dataset_name" =~ $DATASET_FILTER ]]
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
  local group_size="$6"
  local exclude_path="$7"
  local restrict_labels_from="$8"
  local meta_path="$9"

  local -a cmd=(
    uv run python scripts/build_author_subset.py
    --dataset-name "$dataset_name"
    --src "$src_path"
    --dst "$dst_path"
    --mode "$mode"
    --count "$count"
    --group-size "$group_size"
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
  local evaluator_name="$4"
  uv run python - "$dataset_name" "$train_path" "$val_path" "$evaluator_name" <<'PY'
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_prompt, load_train_val

dataset_name = sys.argv[1]
train_path = sys.argv[2]
val_path = sys.argv[3]
evaluator_name = sys.argv[4]

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

if evaluator_name == "pair" and len(val_data) % 2 != 0:
    raise ValueError("Pair evaluator requires an even-sized validation subset")
if evaluator_name == "naturalbench_group" and len(val_data) % 4 != 0:
    raise ValueError("NaturalBench group evaluator requires validation subset size % 4 == 0")

print(f"train={len(train_data)} val={len(val_data)} labels={len({row['label'] for row in train_data})}")
PY
}

run_method() {
  local experiment_id="$1"
  local dataset_name="$2"
  local evaluator_name="$3"
  local train_path="$4"
  local val_path="$5"
  local method_id="$6"
  local display_name="$7"
  local method_name="$8"
  local extra_args_spec="$9"
  local log_path="${10}"

  local run_name="${experiment_id}_${method_id}_${MODEL}_${TIMESTAMP}"
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
    echo "FAILED: ${experiment_id} / ${display_name}" >&2
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
    "$method_id" \
    "$display_name" \
    "$run_name" \
    "$train_path" \
    "$val_path" \
    "$metrics_path" \
    "$predictions_path" >> "$MANIFEST_PATH"
}

printf 'experiment_id\tdataset_name\tevaluator_name\tmethod_id\tdisplay_name\trun_name\ttrain_subset\tval_subset\tmetrics_path\tpredictions_path\n' > "$MANIFEST_PATH"

EXPERIMENTS=(
  "naturalbench_ret|naturalbench_ret|naturalbench_group|ref-data/naturalbench_ret_train.jsonl|ref-data/naturalbench_ret_test.jsonl|author_ref|0|grouped|40|4"
  "naturalbench_vqa|naturalbench_vqa|raw|dataset/converted_from_data/naturalbench/train.jsonl|dataset/converted_from_data/naturalbench/train.jsonl|per_label|20|grouped|40|4"
  "sugarcrepe|sugarcrepe|pair|dataset/converted_from_data/sugarcrepe/sugarcrepe_all.jsonl|dataset/converted_from_data/sugarcrepe/sugarcrepe_all.jsonl|per_label|20|grouped|40|2"
  "pets|pets|raw|ref-data/pets_train.json|dataset/converted_from_data/pets/test.json|per_label|5|per_label|2|0"
  "eurosat|eurosat|raw|ref-data/eurosat_train.json|dataset/converted_from_data/eurosat/test.json|per_label|10|per_label|5|0"
  "flowers|flowers|raw|dataset/converted_from_data/flowers/train.json|dataset/converted_from_data/flowers/test.json|per_label|2|per_label|1|0"
  "cub|cub|raw|dataset/converted_from_data/cub/train.json|dataset/converted_from_data/cub/test.json|per_label|2|per_label|1|0"
  "tinyimage|tinyimage|raw|dataset/converted_from_data/tinyimage/train.json|dataset/converted_from_data/tinyimage/val.json|per_label|2|per_label|1|0"
  "vizwiz|vizwiz|raw|ref-data/vizwiz_sav_train.jsonl|dataset/converted_from_data/vizwiz/val.jsonl|author_ref|0|per_label|20|0"
  "vlguard|vlguard|raw|ref-data/vlguard_train.json|dataset/converted_from_data/vlguard/test.json|author_ref|0|per_label|20|0"
  "mhalubench_val_v01|mhalubench|raw|ref-data/mahalu_train.json|dataset/converted_from_data/mhalubench/val_v01.json|author_ref|0|per_label|20|0"
  "mhalubench_val_v02|mhalubench|raw|ref-data/mahalu_train.json|dataset/converted_from_data/mhalubench/val_v02.json|author_ref|0|per_label|20|0"
  "blink_art_style|blink_art_style|raw|ref-data/blink_art_style_train.json|dataset/converted_from_data/blink/art_style_val.json|author_ref|0|per_label|10|0"
  "blink_counting|blink_counting|raw|dataset/converted_from_data/blink/counting_val.json|dataset/converted_from_data/blink/counting_val.json|per_label|20|per_label|5|0"
  "blink_forensic_detection|blink_forensic_detection|raw|dataset/converted_from_data/blink/forensic_detection_val.json|dataset/converted_from_data/blink/forensic_detection_val.json|per_label|20|per_label|5|0"
  "blink_functional_correspondence|blink_functional_correspondence|raw|dataset/converted_from_data/blink/functional_correspondence_val.json|dataset/converted_from_data/blink/functional_correspondence_val.json|per_label|20|per_label|5|0"
  "blink_iq_test|blink_iq_test|raw|dataset/converted_from_data/blink/iq_test_val.json|dataset/converted_from_data/blink/iq_test_val.json|per_label|20|per_label|5|0"
  "blink_jigsaw|blink_jigsaw|raw|dataset/converted_from_data/blink/jigsaw_val.json|dataset/converted_from_data/blink/jigsaw_val.json|per_label|20|per_label|10|0"
  "blink_multi-view_reasoning|blink_multi-view_reasoning|raw|dataset/converted_from_data/blink/multi-view_reasoning_val.json|dataset/converted_from_data/blink/multi-view_reasoning_val.json|per_label|20|per_label|10|0"
  "blink_object_localization|blink_object_localization|raw|dataset/converted_from_data/blink/object_localization_val.json|dataset/converted_from_data/blink/object_localization_val.json|per_label|20|per_label|10|0"
  "blink_relative_depth|blink_relative_depth|raw|dataset/converted_from_data/blink/relative_depth_val.json|dataset/converted_from_data/blink/relative_depth_val.json|per_label|20|per_label|10|0"
  "blink_relative_reflectance|blink_relative_reflectance|raw|dataset/converted_from_data/blink/relative_reflectance_val.json|dataset/converted_from_data/blink/relative_reflectance_val.json|per_label|20|per_label|5|0"
  "blink_semantic_correspondence|blink_semantic_correspondence|raw|dataset/converted_from_data/blink/semantic_correspondence_val.json|dataset/converted_from_data/blink/semantic_correspondence_val.json|per_label|20|per_label|5|0"
  "blink_spatial_relation|blink_spatial_relation|raw|dataset/converted_from_data/blink/spatial_relation_val.json|dataset/converted_from_data/blink/spatial_relation_val.json|per_label|20|per_label|10|0"
  "blink_visual_correspondence|blink_visual_correspondence|raw|dataset/converted_from_data/blink/visual_correspondence_val.json|dataset/converted_from_data/blink/visual_correspondence_val.json|per_label|20|per_label|5|0"
  "blink_visual_similarity|blink_visual_similarity|raw|dataset/converted_from_data/blink/visual_similarity_val.json|dataset/converted_from_data/blink/visual_similarity_val.json|per_label|20|per_label|10|0"
)

METHOD_SPECS=(
  "zero_shot|Zero-shot|zero_shot|"
  "sav|SAV|sav|"
  "sav_wvote|SAV+WVote|sav|method.params.vote_weighting=head_accuracy"
  "mimic|MimIC|mimic|"
  "i2cl|I2CL|i2cl|"
  "stv|STV|stv|method.params.head_selection_mode=sensitivity;method.params.cluster_selection_mode=rl"
)

for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r experiment_id dataset_name evaluator_name train_src val_src train_mode train_count val_mode val_count val_group_size <<< "$entry"

  if ! matches_filter "$experiment_id" "$dataset_name"; then
    continue
  fi

  experiment_root="$SUBSET_ROOT/$experiment_id"
  mkdir -p "$experiment_root"

  if [[ "$train_src" == *.jsonl ]]; then
    train_subset="$experiment_root/train_subset.jsonl"
  else
    train_subset="$experiment_root/train_subset.json"
  fi

  if [[ "$val_src" == *.jsonl ]]; then
    val_subset="$experiment_root/val_subset.jsonl"
  else
    val_subset="$experiment_root/val_subset.json"
  fi

  build_subset_file "$dataset_name" "$train_src" "$train_subset" "$train_mode" "$train_count" "0" "" "" "$experiment_root/train_subset.meta.json"
  build_subset_file "$dataset_name" "$val_src" "$val_subset" "$val_mode" "$val_count" "$val_group_size" "$train_subset" "$train_subset" "$experiment_root/val_subset.meta.json"
  validate_subset "$dataset_name" "$train_subset" "$val_subset" "$evaluator_name"

  for method_entry in "${METHOD_SPECS[@]}"; do
    IFS='|' read -r method_id display_name method_name extra_args_spec <<< "$method_entry"
    if ! matches_method_filter "$method_id" "$display_name"; then
      continue
    fi
    log_path="$LOG_ROOT/${experiment_id}_${method_id}.log"
    run_method "$experiment_id" "$dataset_name" "$evaluator_name" "$train_subset" "$val_subset" "$method_id" "$display_name" "$method_name" "$extra_args_spec" "$log_path"
  done
done

uv run python scripts/summarize_fgvc_method_suite.py \
  --manifest "$MANIFEST_PATH" \
  --output "$SUMMARY_PATH"

printf 'manifest=%s\nsummary=%s\n' "$MANIFEST_PATH" "$SUMMARY_PATH"
