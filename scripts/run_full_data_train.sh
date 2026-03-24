#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

RUN_MODE="${RUN_MODE:-subset_smoke}"
MODEL="${MODEL:-qwen2_vl}"
METHODS="${METHODS:-zero_shot}"
NUM_HEADS="${NUM_HEADS:-20}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16}"
SUPPORT_PER_LABEL="${SUPPORT_PER_LABEL:-20}"
DO_VALIDATE="${DO_VALIDATE:-1}"
DO_RUN="${DO_RUN:-1}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
DATASET_FILTER="${DATASET_FILTER:-}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

SWAP_ROOT="$PROJECT_ROOT/swap"
SUBSET_ROOT="$SWAP_ROOT/subsets/$TIMESTAMP"
LOG_ROOT="$SWAP_ROOT/logs/$TIMESTAMP"
OUTPUT_ROOT="$SWAP_ROOT/outputs/$TIMESTAMP"
RECORD_ROOT="$SWAP_ROOT/records"
RECORD_PATH="$RECORD_ROOT/dataset_subset_smoke_${TIMESTAMP}.md"

if [[ "$RUN_MODE" == "full_matrix" ]]; then
  echo "[run_full_data_train] RUN_MODE=full_matrix -> dispatching full-data matrix"
  bash scripts/train_full/run_all_non_coco_train.sh
  exit $?
fi

mkdir -p "$SUBSET_ROOT" "$LOG_ROOT" "$OUTPUT_ROOT" "$RECORD_ROOT"
read -r -a METHOD_LIST <<< "$METHODS"

sanitize_cell() {
  local value="$1"
  value="${value//$'\n'/ }"
  value="${value//|/\\|}"
  printf '%s' "$value" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//'
}

write_record_header() {
  cat > "$RECORD_PATH" <<EOF
# Author-Style Dataset Subset Smoke Test

- Timestamp (UTC): \`$TIMESTAMP\`
- Run mode: \`$RUN_MODE\`
- Model: \`$MODEL\`
- Methods: \`$METHODS\`
- Support rule: \`ref-data\` when available, otherwise first \`$SUPPORT_PER_LABEL\` valid examples per label
- Eval rule: official labeled eval split when available; same-source tasks exclude support rows
- Validate datasets: \`$DO_VALIDATE\`
- Execute \`main.py\`: \`$DO_RUN\`
- Dataset filter: \`${DATASET_FILTER:-all}\`

## Output Layout

- Subsets: \`swap/subsets/$TIMESTAMP\`
- Logs: \`swap/logs/$TIMESTAMP\`
- Metrics: \`swap/outputs/$TIMESTAMP\`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
EOF
}

append_record_row() {
  local experiment="$1"
  local dataset_name="$2"
  local evaluator="$3"
  local train_count="$4"
  local val_count="$5"
  local validate_status="$6"
  local run_status="$7"
  local notes="$8"

  printf '| %s | %s | %s | %s | %s | %s | %s | %s |\n' \
    "$(sanitize_cell "$experiment")" \
    "$(sanitize_cell "$dataset_name")" \
    "$(sanitize_cell "$evaluator")" \
    "$(sanitize_cell "$train_count")" \
    "$(sanitize_cell "$val_count")" \
    "$(sanitize_cell "$validate_status")" \
    "$(sanitize_cell "$run_status")" \
    "$(sanitize_cell "$notes")" \
    >> "$RECORD_PATH"
}

matches_filter() {
  local experiment_id="$1"
  local dataset_name="$2"
  if [[ -z "$DATASET_FILTER" ]]; then
    return 0
  fi
  [[ "$experiment_id" =~ $DATASET_FILTER || "$dataset_name" =~ $DATASET_FILTER ]]
}

build_subset_file() {
  local dataset_name="$1"
  local src_path="$2"
  local dst_path="$3"
  local mode="$4"
  local count="$5"
  local group_size="$6"
  local exclude_path="$7"
  local meta_path="$8"

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

  "${cmd[@]}"
}

meta_note() {
  local meta_path="$1"
  python - "$meta_path" <<'PY'
import json
import sys
from pathlib import Path

meta = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
bits = [
    f"mode={meta['mode']}",
    f"source={meta['source_kind']}",
    f"selected={meta['count']}",
    f"labels={meta['unique_labels']}",
]

per_label_min = int(meta.get("per_label_min", 0))
per_label_max = int(meta.get("per_label_max", 0))
if per_label_min > 0:
    if per_label_min == per_label_max:
        bits.append(f"per_label={per_label_min}")
    else:
        bits.append(f"per_label={per_label_min}-{per_label_max}")

for key, name in [
    ("canonicalized_paths", "path_fix"),
    ("canonicalized_labels", "label_fix"),
    ("excluded_overlap", "exclude"),
    ("dropped_disallowed_labels", "drop_label"),
    ("skipped_missing_images", "skip_missing"),
]:
    value = int(meta.get(key, 0))
    if value > 0:
        bits.append(f"{name}={value}")

print(" ".join(bits))
PY
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

multi_image_count = 0
image_count = 0

for split_name, samples in [("train", train_data), ("val", val_data)]:
    for idx, sample in enumerate(samples):
        question = str(sample.get("question", "")).strip()
        label = str(sample.get("label", "")).strip()
        if not question:
            raise ValueError(f"{split_name}[{idx}] is missing normalized question text")
        if not label:
            raise ValueError(f"{split_name}[{idx}] is missing normalized label")

        prompt = build_prompt(dataset_name, sample)
        if not str(prompt).strip():
            raise ValueError(f"{split_name}[{idx}] produced an empty prompt")

        images = sample.get("images") or [sample.get("image")]
        if len(images) > 1:
            multi_image_count += 1
        for image_path in images:
            if image_path is None:
                raise ValueError(f"{split_name}[{idx}] contains a null image reference")
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"{split_name}[{idx}] missing image: {path}")
            image_count += 1

if evaluator_name == "pair" and len(val_data) % 2 != 0:
    raise ValueError("Pair evaluator requires an even-sized validation subset")
if evaluator_name == "naturalbench_group" and len(val_data) % 4 != 0:
    raise ValueError("NaturalBench group evaluator requires validation subset size % 4 == 0")

print(
    f"train={len(train_data)} val={len(val_data)} "
    f"image_paths={image_count} multi_image_samples={multi_image_count}"
)
PY
}

run_main_subset() {
  local method_name="$1"
  local dataset_name="$2"
  local evaluator_name="$3"
  local train_path="$4"
  local val_path="$5"
  local run_name="$6"
  local log_path="$7"

  local -a cmd=(
    uv run python main.py
    model="$MODEL"
    dataset=general_custom
    method="$method_name"
    evaluator="$evaluator_name"
    dataset.name="$dataset_name"
    dataset.train_path="$train_path"
    dataset.val_path="$val_path"
    run.output_dir="$OUTPUT_ROOT"
    run.run_name="$run_name"
    run.save_predictions=false
    run.progress_bar=false
  )

  if [[ "$method_name" == "sav" ]]; then
    cmd+=(
      method.params.num_heads="$NUM_HEADS"
      method.params.progress_bar=false
    )
  else
    cmd+=(
      method.params.max_new_tokens="$MAX_NEW_TOKENS"
      method.params.do_sample=false
      method.params.temperature=0.0
    )
  fi

  "${cmd[@]}" > "$log_path" 2>&1
}

write_record_header

EXPERIMENTS=(
  "naturalbench_ret|naturalbench_ret|naturalbench_group|ref-data/naturalbench_ret_train.jsonl|ref-data/naturalbench_ret_test.jsonl|author_ref|0|grouped|4|4"
  "naturalbench_vqa|naturalbench_vqa|raw|dataset/converted_from_data/naturalbench/train.jsonl|dataset/converted_from_data/naturalbench/train.jsonl|per_label|$SUPPORT_PER_LABEL|grouped|4|4"
  "sugarcrepe|sugarcrepe|pair|dataset/converted_from_data/sugarcrepe/sugarcrepe_all.jsonl|dataset/converted_from_data/sugarcrepe/sugarcrepe_all.jsonl|per_label|$SUPPORT_PER_LABEL|grouped|4|2"
  "pets|pets|raw|ref-data/pets_train.json|dataset/converted_from_data/pets/test.json|author_ref|0|distinct_labels|4|0"
  "eurosat|eurosat|raw|ref-data/eurosat_train.json|dataset/converted_from_data/eurosat/test.json|author_ref|0|distinct_labels|4|0"
  "flowers|flowers|raw|dataset/converted_from_data/flowers/train.json|dataset/converted_from_data/flowers/test.json|per_label|$SUPPORT_PER_LABEL|distinct_labels|4|0"
  "cub|cub|raw|dataset/converted_from_data/cub/train.json|dataset/converted_from_data/cub/test.json|per_label|$SUPPORT_PER_LABEL|distinct_labels|4|0"
  "tinyimage|tinyimage|raw|dataset/converted_from_data/tinyimage/train.json|dataset/converted_from_data/tinyimage/val.json|per_label|$SUPPORT_PER_LABEL|distinct_labels|4|0"
  "vizwiz|vizwiz|raw|ref-data/vizwiz_sav_train.jsonl|dataset/converted_from_data/vizwiz/val.jsonl|author_ref|0|per_label|2|0"
  "vlguard|vlguard|raw|ref-data/vlguard_train.json|dataset/converted_from_data/vlguard/test.json|author_ref|0|per_label|2|0"
  "mhalubench_val_v01|mhalubench|raw|ref-data/mahalu_train.json|dataset/converted_from_data/mhalubench/val_v01.json|author_ref|0|per_label|2|0"
  "mhalubench_val_v02|mhalubench|raw|ref-data/mahalu_train.json|dataset/converted_from_data/mhalubench/val_v02.json|author_ref|0|per_label|2|0"
)

BLINK_TASKS=(
  art_style
  counting
  forensic_detection
  functional_correspondence
  iq_test
  jigsaw
  multi-view_reasoning
  object_localization
  relative_depth
  relative_reflectance
  semantic_correspondence
  spatial_relation
  visual_correspondence
  visual_similarity
)

for task in "${BLINK_TASKS[@]}"; do
  train_src="dataset/converted_from_data/blink/${task}_val.json"
  train_mode="per_label"
  train_count="$SUPPORT_PER_LABEL"
  if [[ "$task" == "art_style" ]]; then
    train_src="ref-data/blink_art_style_train.json"
    train_mode="author_ref"
    train_count="0"
  fi
  EXPERIMENTS+=(
    "blink_${task}|blink_${task}|raw|${train_src}|dataset/converted_from_data/blink/${task}_val.json|${train_mode}|${train_count}|distinct_labels|4|0"
  )
done

selected=0
failures=0

for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r experiment_id dataset_name evaluator_name train_src val_src train_mode train_count_requested val_mode val_count_requested val_group_size <<< "$entry"

  if ! matches_filter "$experiment_id" "$dataset_name"; then
    continue
  fi

  selected=$((selected + 1))

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

  echo "=================================================="
  echo "[subset_smoke] Preparing ${experiment_id}"

  train_meta="$experiment_root/train_subset.meta.json"
  val_meta="$experiment_root/val_subset.meta.json"

  if ! train_count="$(build_subset_file "$dataset_name" "$train_src" "$train_subset" "$train_mode" "$train_count_requested" "0" "" "$train_meta" 2>&1)"; then
    failures=$((failures + 1))
    append_record_row "$experiment_id" "$dataset_name" "$evaluator_name" "ERR" "ERR" "FAIL" "SKIP" "subset build failed: $train_count"
    echo "[subset_smoke][FAIL] train subset build failed for ${experiment_id}"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi
  train_note="$(meta_note "$train_meta")"

  if ! val_count="$(build_subset_file "$dataset_name" "$val_src" "$val_subset" "$val_mode" "$val_count_requested" "$val_group_size" "$train_subset" "$val_meta" 2>&1)"; then
    failures=$((failures + 1))
    append_record_row "$experiment_id" "$dataset_name" "$evaluator_name" "$train_count" "ERR" "FAIL" "SKIP" "subset build failed: $val_count"
    echo "[subset_smoke][FAIL] val subset build failed for ${experiment_id}"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi
  val_note="$(meta_note "$val_meta")"

  validate_status="SKIP"
  validate_note="validation disabled"
  validate_log="$LOG_ROOT/${experiment_id}.validate.log"

  if [[ "$DO_VALIDATE" == "1" ]]; then
    if validate_output="$(validate_subset "$dataset_name" "$train_subset" "$val_subset" "$evaluator_name" 2>&1)"; then
      validate_status="PASS"
      validate_note="$validate_output"
      printf '%s\n' "$validate_output" > "$validate_log"
      echo "[subset_smoke][PASS] validation: ${validate_output}"
    else
      validate_status="FAIL"
      validate_note="$validate_output"
      printf '%s\n' "$validate_output" > "$validate_log"
      failures=$((failures + 1))
      append_record_row "$experiment_id" "$dataset_name" "$evaluator_name" "$train_count" "$val_count" "$validate_status" "SKIP" "train=[$train_note] val=[$val_note] validate_log=$(basename "$validate_log") $validate_note"
      echo "[subset_smoke][FAIL] validation failed for ${experiment_id}"
      if [[ "$STOP_ON_ERROR" == "1" ]]; then
        exit 1
      fi
      continue
    fi
  fi

  if [[ "$DO_RUN" != "1" ]]; then
    for method_name in "${METHOD_LIST[@]}"; do
      append_record_row "${experiment_id}:${method_name}" "$dataset_name" "$evaluator_name" "$train_count" "$val_count" "$validate_status" "SKIP" "train=[$train_note] val=[$val_note] validate=[$validate_note]"
    done
    continue
  fi

  for method_name in "${METHOD_LIST[@]}"; do
    run_name="${method_name}_${experiment_id}_${MODEL}_subset"
    run_log="$LOG_ROOT/${run_name}.log"
    echo "[subset_smoke] Running ${run_name}"

    if run_main_subset "$method_name" "$dataset_name" "$evaluator_name" "$train_subset" "$val_subset" "$run_name" "$run_log"; then
      append_record_row "${experiment_id}:${method_name}" "$dataset_name" "$evaluator_name" "$train_count" "$val_count" "$validate_status" "PASS" "train=[$train_note] val=[$val_note] validate=[$validate_note] log=$(basename "$run_log")"
      echo "[subset_smoke][PASS] ${run_name}"
    else
      failures=$((failures + 1))
      append_record_row "${experiment_id}:${method_name}" "$dataset_name" "$evaluator_name" "$train_count" "$val_count" "$validate_status" "FAIL" "train=[$train_note] val=[$val_note] validate=[$validate_note] log=$(basename "$run_log")"
      echo "[subset_smoke][FAIL] ${run_name}"
      if [[ "$STOP_ON_ERROR" == "1" ]]; then
        exit 1
      fi
    fi
  done
done

if [[ "$selected" -eq 0 ]]; then
  echo "[subset_smoke] No experiments matched DATASET_FILTER='${DATASET_FILTER}'"
  exit 1
fi

{
  echo
  echo "## Summary"
  echo
  echo "- Selected experiments: \`$selected\`"
  echo "- Failures: \`$failures\`"
  echo "- Record: \`swap/records/$(basename "$RECORD_PATH")\`"
} >> "$RECORD_PATH"

echo "=================================================="
echo "[subset_smoke] Selected experiments: ${selected}"
echo "[subset_smoke] Failures: ${failures}"
echo "[subset_smoke] Markdown record: ${RECORD_PATH}"

if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
