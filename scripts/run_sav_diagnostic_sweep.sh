#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen2_vl}"
NUM_HEADS="${NUM_HEADS:-20}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
DATASET_FILTER="${DATASET_FILTER:-}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

SWAP_ROOT="$PROJECT_ROOT/swap"
SUBSET_ROOT="$SWAP_ROOT/subsets/${TIMESTAMP}_sav_diag"
LOG_ROOT="$SWAP_ROOT/logs/${TIMESTAMP}_sav_diag"
OUTPUT_ROOT="$SWAP_ROOT/outputs/${TIMESTAMP}_sav_diag"
RECORD_ROOT="$SWAP_ROOT/records"
RECORD_PATH="$RECORD_ROOT/sav_diagnostic_sweep_${TIMESTAMP}.md"

mkdir -p "$SUBSET_ROOT" "$LOG_ROOT" "$OUTPUT_ROOT" "$RECORD_ROOT"

sanitize_cell() {
  local value="$1"
  value="${value//$'\n'/ }"
  value="${value//|/\\|}"
  printf '%s' "$value" | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//'
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
    ("restricted_label_space", "restrict_labels"),
    ("excluded_overlap", "exclude"),
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
  local dataset_name="$1"
  local evaluator_name="$2"
  local train_path="$3"
  local val_path="$4"
  local run_name="$5"
  local log_path="$6"

  uv run python main.py \
    model="$MODEL" \
    dataset=general_custom \
    method=sav \
    evaluator="$evaluator_name" \
    dataset.name="$dataset_name" \
    dataset.train_path="$train_path" \
    dataset.val_path="$val_path" \
    run.output_dir="$OUTPUT_ROOT" \
    run.run_name="$run_name" \
    run.save_predictions=false \
    run.progress_bar=false \
    method.params.num_heads="$NUM_HEADS" \
    method.params.progress_bar=false \
    > "$log_path" 2>&1
}

metrics_note() {
  local metrics_path="$1"
  python - "$metrics_path" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["metrics"]
if "accuracy" in metrics:
    print(f"accuracy={metrics['accuracy']:.4f}")
elif "pair_accuracy" in metrics:
    print(f"pair_accuracy={metrics['pair_accuracy']:.4f} raw_accuracy={metrics['raw_accuracy']:.4f}")
elif "g_acc" in metrics:
    print(f"g_acc={metrics['g_acc']:.4f} raw_acc={metrics['raw_acc']:.4f} q_acc={metrics['q_acc']:.4f} i_acc={metrics['i_acc']:.4f}")
else:
    print(json.dumps(metrics, ensure_ascii=False))
PY
}

cat > "$RECORD_PATH" <<EOF
# SAV Diagnostic Sweep

- Timestamp (UTC): \`$TIMESTAMP\`
- Model: \`$MODEL\`
- Method: \`sav\`
- Goal: larger-scale, author-style subset diagnostics across all supported tasks
- Principle:
  - use author \`ref-data\` support when available
  - otherwise use balanced per-label support from converted official data
  - use labeled eval splits
  - keep train/eval disjoint for same-source tasks
  - cap very high-class datasets to tractable balanced subsets for diagnostics

## Output Layout

- Subsets: \`swap/subsets/${TIMESTAMP}_sav_diag\`
- Logs: \`swap/logs/${TIMESTAMP}_sav_diag\`
- Metrics: \`swap/outputs/${TIMESTAMP}_sav_diag\`

## Result Table

| Experiment | Dataset | Evaluator | Train | Val | Metric | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
EOF

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

selected=0
failures=0

for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r experiment_id dataset_name evaluator_name train_src val_src train_mode train_count val_mode val_count val_group_size <<< "$entry"

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

  train_meta="$experiment_root/train_subset.meta.json"
  val_meta="$experiment_root/val_subset.meta.json"
  run_name="sav_${experiment_id}_${MODEL}_diag"
  run_log="$LOG_ROOT/${run_name}.log"

  echo "=================================================="
  echo "[sav_diag] Preparing ${experiment_id}"

  if ! train_n="$(build_subset_file "$dataset_name" "$train_src" "$train_subset" "$train_mode" "$train_count" "0" "" "" "$train_meta" 2>&1)"; then
    failures=$((failures + 1))
    printf '| %s | %s | %s | ERR | ERR | - | FAIL | %s |\n' \
      "$(sanitize_cell "$experiment_id")" \
      "$(sanitize_cell "$dataset_name")" \
      "$(sanitize_cell "$evaluator_name")" \
      "$(sanitize_cell "train subset build failed: $train_n")" >> "$RECORD_PATH"
    echo "[sav_diag][FAIL] train subset build failed for ${experiment_id}"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  if ! val_n="$(build_subset_file "$dataset_name" "$val_src" "$val_subset" "$val_mode" "$val_count" "$val_group_size" "$train_subset" "$train_subset" "$val_meta" 2>&1)"; then
    failures=$((failures + 1))
    printf '| %s | %s | %s | %s | ERR | - | FAIL | %s |\n' \
      "$(sanitize_cell "$experiment_id")" \
      "$(sanitize_cell "$dataset_name")" \
      "$(sanitize_cell "$evaluator_name")" \
      "$(sanitize_cell "$train_n")" \
      "$(sanitize_cell "val subset build failed: $val_n")" >> "$RECORD_PATH"
    echo "[sav_diag][FAIL] val subset build failed for ${experiment_id}"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  train_note="$(meta_note "$train_meta")"
  val_note="$(meta_note "$val_meta")"

  if ! validate_output="$(validate_subset "$dataset_name" "$train_subset" "$val_subset" "$evaluator_name" 2>&1)"; then
    failures=$((failures + 1))
    printf '| %s | %s | %s | %s | %s | - | FAIL | %s |\n' \
      "$(sanitize_cell "$experiment_id")" \
      "$(sanitize_cell "$dataset_name")" \
      "$(sanitize_cell "$evaluator_name")" \
      "$(sanitize_cell "$train_n")" \
      "$(sanitize_cell "$val_n")" \
      "$(sanitize_cell "train=[$train_note] val=[$val_note] validate=$validate_output")" >> "$RECORD_PATH"
    echo "[sav_diag][FAIL] validation failed for ${experiment_id}"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  echo "[sav_diag][PASS] validation: ${validate_output}"
  echo "[sav_diag] Running ${run_name}"

  if ! run_main_subset "$dataset_name" "$evaluator_name" "$train_subset" "$val_subset" "$run_name" "$run_log"; then
    failures=$((failures + 1))
    printf '| %s | %s | %s | %s | %s | - | FAIL | %s |\n' \
      "$(sanitize_cell "$experiment_id")" \
      "$(sanitize_cell "$dataset_name")" \
      "$(sanitize_cell "$evaluator_name")" \
      "$(sanitize_cell "$train_n")" \
      "$(sanitize_cell "$val_n")" \
      "$(sanitize_cell "train=[$train_note] val=[$val_note] validate=[$validate_output] log=$(basename "$run_log")")" >> "$RECORD_PATH"
    echo "[sav_diag][FAIL] ${run_name}"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  metrics_path="$OUTPUT_ROOT/${run_name}.metrics.json"
  metric_text="$(metrics_note "$metrics_path")"

  printf '| %s | %s | %s | %s | %s | %s | PASS | %s |\n' \
    "$(sanitize_cell "$experiment_id")" \
    "$(sanitize_cell "$dataset_name")" \
    "$(sanitize_cell "$evaluator_name")" \
    "$(sanitize_cell "$train_n")" \
    "$(sanitize_cell "$val_n")" \
    "$(sanitize_cell "$metric_text")" \
    "$(sanitize_cell "train=[$train_note] val=[$val_note] validate=[$validate_output] log=$(basename "$run_log")")" >> "$RECORD_PATH"
  echo "[sav_diag][PASS] ${run_name} ${metric_text}"
done

if [[ "$selected" -eq 0 ]]; then
  echo "[sav_diag] No experiments matched DATASET_FILTER='${DATASET_FILTER}'"
  exit 1
fi

python - "$OUTPUT_ROOT" "$RECORD_PATH" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
record_path = Path(sys.argv[2])

rows = []
for path in sorted(output_root.glob("*.metrics.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = path.stem.replace(".metrics", "")
    metrics = payload["metrics"]
    if "accuracy" in metrics:
        score = float(metrics["accuracy"])
        score_label = "accuracy"
    elif "pair_accuracy" in metrics:
        score = float(metrics["pair_accuracy"])
        score_label = "pair_accuracy"
    else:
        score = float(metrics.get("g_acc", metrics.get("raw_acc", 0.0)))
        score_label = "g_acc" if "g_acc" in metrics else "raw_acc"
    rows.append((name, payload["dataset"], score_label, score))

rows_sorted = sorted(rows, key=lambda x: x[3])
with record_path.open("a", encoding="utf-8") as fp:
    fp.write("\n## Summary\n\n")
    fp.write(f"- Completed runs: `{len(rows)}`\n")
    weak = [row for row in rows_sorted if row[3] < 0.5]
    strong = [row for row in rows_sorted if row[3] >= 0.8]
    fp.write(f"- Weak tasks (< 0.5): `{len(weak)}`\n")
    fp.write(f"- Strong tasks (>= 0.8): `{len(strong)}`\n")
    if rows_sorted:
        fp.write("\n### Lowest Scores\n\n")
        for name, dataset, score_label, score in rows_sorted[:8]:
            fp.write(f"- `{name}` ({dataset}): `{score_label}={score:.4f}`\n")
        fp.write("\n### Highest Scores\n\n")
        for name, dataset, score_label, score in rows_sorted[-8:][::-1]:
            fp.write(f"- `{name}` ({dataset}): `{score_label}={score:.4f}`\n")
PY

{
  echo
  echo "[sav_diag] Selected experiments: ${selected}"
  echo "[sav_diag] Failures: ${failures}"
  echo "[sav_diag] Markdown record: ${RECORD_PATH}"
}

if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
