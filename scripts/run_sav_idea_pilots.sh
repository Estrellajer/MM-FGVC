#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen2_vl}"
NUM_HEADS="${NUM_HEADS:-20}"
SELECTION_STRATEGY="${SELECTION_STRATEGY:-topk}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
DATASET_FILTER="${DATASET_FILTER:-}"
IDEA_FILTER="${IDEA_FILTER:-}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

SWAP_ROOT="$PROJECT_ROOT/swap"
SUBSET_ROOT="$SWAP_ROOT/subsets/${TIMESTAMP}_sav_idea_pilots"
LOG_ROOT="$SWAP_ROOT/logs/${TIMESTAMP}_sav_idea_pilots"
OUTPUT_ROOT="$SWAP_ROOT/outputs/${TIMESTAMP}_sav_idea_pilots"
RECORD_ROOT="$SWAP_ROOT/records"
RECORD_PATH="$RECORD_ROOT/sav_idea_pilots_${TIMESTAMP}.md"
MANIFEST_PATH="$OUTPUT_ROOT/manifest.tsv"

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

matches_idea_filter() {
  local idea_name="$1"
  if [[ -z "$IDEA_FILTER" ]]; then
    return 0
  fi
  [[ "$idea_name" =~ $IDEA_FILTER ]]
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
  local prototype_mode="$7"
  local class_bank_size="$8"

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
    method.params.selection_strategy="$SELECTION_STRATEGY" \
    method.params.selection_seed=42 \
    method.params.prototype_mode="$prototype_mode" \
    method.params.class_bank_size="$class_bank_size" \
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
# SAV Easy Idea Pilots

- Timestamp (UTC): \`$TIMESTAMP\`
- Model: \`$MODEL\`
- Method family: \`sav\`
- Goal: quick pilots for easy-to-test ideas from \`testidea.md\`
- Fixed head selector: \`$SELECTION_STRATEGY\`
- Fixed \`num_heads\`: \`$NUM_HEADS\`
- Idea mapping:
  - \`baseline_mean\`: current SAV class-mean prototype
  - \`support_nn\`: query-conditioned nearest-support retrieval (QC-style lightweight proxy)
  - \`class_bank4\`: class-conditional prototype bank with up to 4 diverse prototypes per class (CC-style lightweight proxy)
- Principle:
  - reuse author-style subsets
  - keep the "where" part fixed to current SAV \`topk\`
  - only test lightweight "what" / retrieval-bank changes

## Output Layout

- Subsets: \`swap/subsets/${TIMESTAMP}_sav_idea_pilots\`
- Logs: \`swap/logs/${TIMESTAMP}_sav_idea_pilots\`
- Metrics: \`swap/outputs/${TIMESTAMP}_sav_idea_pilots\`

## Result Table

| Experiment | Dataset | Idea | Train | Val | Metric | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
EOF

printf 'experiment_id\tdataset_name\tevaluator_name\tidea_name\tprototype_mode\tclass_bank_size\tmetrics_path\tlog_path\n' > "$MANIFEST_PATH"

EXPERIMENTS=(
  "naturalbench_vqa|naturalbench_vqa|raw|dataset/converted_from_data/naturalbench/train.jsonl|dataset/converted_from_data/naturalbench/train.jsonl|per_label|20|grouped|80|4"
  "eurosat|eurosat|raw|ref-data/eurosat_train.json|dataset/converted_from_data/eurosat/test.json|per_label|10|per_label|10|0"
  "tinyimage|tinyimage|raw|dataset/converted_from_data/tinyimage/train.json|dataset/converted_from_data/tinyimage/val.json|per_label|2|per_label|1|0"
  "blink_art_style|blink_art_style|raw|ref-data/blink_art_style_train.json|dataset/converted_from_data/blink/art_style_val.json|author_ref|0|per_label|20|0"
  "blink_semantic_correspondence|blink_semantic_correspondence|raw|dataset/converted_from_data/blink/semantic_correspondence_val.json|dataset/converted_from_data/blink/semantic_correspondence_val.json|per_label|20|per_label|10|0"
)

IDEA_SPECS=(
  "baseline_mean|mean|1"
  "support_nn|support_nn|1"
  "class_bank4|class_bank|4"
)

selected_experiments=0
selected_runs=0
failures=0

for entry in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r experiment_id dataset_name evaluator_name train_src val_src train_mode train_count val_mode val_count val_group_size <<< "$entry"

  if ! matches_filter "$experiment_id" "$dataset_name"; then
    continue
  fi

  selected_experiments=$((selected_experiments + 1))
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

  echo "=================================================="
  echo "[sav_ideas] Preparing ${experiment_id}"

  if ! train_n="$(build_subset_file "$dataset_name" "$train_src" "$train_subset" "$train_mode" "$train_count" "0" "" "" "$train_meta" 2>&1)"; then
    failures=$((failures + 1))
    printf '| %s | %s | setup | ERR | ERR | - | FAIL | %s |\n' \
      "$(sanitize_cell "$experiment_id")" \
      "$(sanitize_cell "$dataset_name")" \
      "$(sanitize_cell "train subset build failed: $train_n")" >> "$RECORD_PATH"
    echo "[sav_ideas][FAIL] train subset build failed for ${experiment_id}"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  if ! val_n="$(build_subset_file "$dataset_name" "$val_src" "$val_subset" "$val_mode" "$val_count" "$val_group_size" "$train_subset" "$train_subset" "$val_meta" 2>&1)"; then
    failures=$((failures + 1))
    printf '| %s | %s | setup | %s | ERR | - | FAIL | %s |\n' \
      "$(sanitize_cell "$experiment_id")" \
      "$(sanitize_cell "$dataset_name")" \
      "$(sanitize_cell "$train_n")" \
      "$(sanitize_cell "val subset build failed: $val_n")" >> "$RECORD_PATH"
    echo "[sav_ideas][FAIL] val subset build failed for ${experiment_id}"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  train_note="$(meta_note "$train_meta")"
  val_note="$(meta_note "$val_meta")"

  if ! validate_output="$(validate_subset "$dataset_name" "$train_subset" "$val_subset" "$evaluator_name" 2>&1)"; then
    failures=$((failures + 1))
    printf '| %s | %s | setup | %s | %s | - | FAIL | %s |\n' \
      "$(sanitize_cell "$experiment_id")" \
      "$(sanitize_cell "$dataset_name")" \
      "$(sanitize_cell "$train_n")" \
      "$(sanitize_cell "$val_n")" \
      "$(sanitize_cell "train=[$train_note] val=[$val_note] validate=$validate_output")" >> "$RECORD_PATH"
    echo "[sav_ideas][FAIL] validation failed for ${experiment_id}"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  echo "[sav_ideas][PASS] validation: ${validate_output}"

  for idea_entry in "${IDEA_SPECS[@]}"; do
    IFS='|' read -r idea_name prototype_mode class_bank_size <<< "$idea_entry"

    if ! matches_idea_filter "$idea_name"; then
      continue
    fi

    selected_runs=$((selected_runs + 1))
    run_name="sav_${experiment_id}_${idea_name}_${MODEL}_pilot"
    run_log="$LOG_ROOT/${run_name}.log"

    echo "[sav_ideas] Running ${run_name}"

    if ! run_main_subset "$dataset_name" "$evaluator_name" "$train_subset" "$val_subset" "$run_name" "$run_log" "$prototype_mode" "$class_bank_size"; then
      failures=$((failures + 1))
      printf '| %s | %s | %s | %s | %s | - | FAIL | %s |\n' \
        "$(sanitize_cell "$experiment_id")" \
        "$(sanitize_cell "$dataset_name")" \
        "$(sanitize_cell "$idea_name")" \
        "$(sanitize_cell "$train_n")" \
        "$(sanitize_cell "$val_n")" \
        "$(sanitize_cell "train=[$train_note] val=[$val_note] validate=[$validate_output] log=$(basename "$run_log")")" >> "$RECORD_PATH"
      echo "[sav_ideas][FAIL] ${run_name}"
      if [[ "$STOP_ON_ERROR" == "1" ]]; then
        exit 1
      fi
      continue
    fi

    metrics_path="$OUTPUT_ROOT/${run_name}.metrics.json"
    metric_text="$(metrics_note "$metrics_path")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$experiment_id" \
      "$dataset_name" \
      "$evaluator_name" \
      "$idea_name" \
      "$prototype_mode" \
      "$class_bank_size" \
      "$metrics_path" \
      "$run_log" >> "$MANIFEST_PATH"

    printf '| %s | %s | %s | %s | %s | %s | PASS | %s |\n' \
      "$(sanitize_cell "$experiment_id")" \
      "$(sanitize_cell "$dataset_name")" \
      "$(sanitize_cell "$idea_name")" \
      "$(sanitize_cell "$train_n")" \
      "$(sanitize_cell "$val_n")" \
      "$(sanitize_cell "$metric_text")" \
      "$(sanitize_cell "train=[$train_note] val=[$val_note] validate=[$validate_output] prototype_mode=${prototype_mode} bank=${class_bank_size} log=$(basename "$run_log")")" >> "$RECORD_PATH"

    echo "[sav_ideas][PASS] ${run_name} ${metric_text}"
  done
done

if [[ "$selected_experiments" -eq 0 ]]; then
  echo "[sav_ideas] No experiments matched DATASET_FILTER='${DATASET_FILTER}'"
  exit 1
fi

python - "$MANIFEST_PATH" "$RECORD_PATH" <<'PY'
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path


def primary_metric(metrics: dict) -> tuple[str, float]:
    if "accuracy" in metrics:
        return "accuracy", float(metrics["accuracy"])
    if "pair_accuracy" in metrics:
        return "pair_accuracy", float(metrics["pair_accuracy"])
    if "g_acc" in metrics:
        return "g_acc", float(metrics["g_acc"])
    if "raw_acc" in metrics:
        return "raw_acc", float(metrics["raw_acc"])
    key = sorted(metrics)[0]
    return key, float(metrics[key])


manifest_path = Path(sys.argv[1])
record_path = Path(sys.argv[2])

rows = []
with manifest_path.open("r", encoding="utf-8") as fp:
    reader = csv.DictReader(fp, delimiter="\t")
    for row in reader:
        payload = json.loads(Path(row["metrics_path"]).read_text(encoding="utf-8"))
        metric_name, score = primary_metric(payload["metrics"])
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "dataset_name": row["dataset_name"],
                "idea_name": row["idea_name"],
                "metric_name": metric_name,
                "score": score,
            }
        )

by_experiment = defaultdict(list)
for row in rows:
    by_experiment[row["experiment_id"]].append(row)

mode_scores = defaultdict(list)
task_rows = []
improvements = []
regressions = []

for experiment_id, experiment_rows in sorted(by_experiment.items()):
    score_by_idea = {row["idea_name"]: row for row in experiment_rows}
    baseline = score_by_idea.get("baseline_mean")
    support_nn = score_by_idea.get("support_nn")
    class_bank = score_by_idea.get("class_bank4")

    for idea_name, row in score_by_idea.items():
        mode_scores[idea_name].append(row["score"])

    if baseline is None:
        continue

    best_row = max(experiment_rows, key=lambda row: row["score"])
    task_rows.append(
        {
            "experiment_id": experiment_id,
            "dataset_name": baseline["dataset_name"],
            "metric_name": baseline["metric_name"],
            "baseline": baseline["score"],
            "support_nn": support_nn["score"] if support_nn else math.nan,
            "class_bank4": class_bank["score"] if class_bank else math.nan,
            "best_name": best_row["idea_name"],
            "best_score": best_row["score"],
        }
    )

    for candidate_name, candidate_row in [("support_nn", support_nn), ("class_bank4", class_bank)]:
        if candidate_row is None:
            continue
        delta = candidate_row["score"] - baseline["score"]
        if delta > 0:
            improvements.append((delta, experiment_id, candidate_name, candidate_row["score"], baseline["score"]))
        elif delta < 0:
            regressions.append((delta, experiment_id, candidate_name, candidate_row["score"], baseline["score"]))

with record_path.open("a", encoding="utf-8") as fp:
    fp.write("\n## Aggregate Summary\n\n")
    fp.write(f"- Completed runs: `{len(rows)}`\n")
    for idea_name in ["baseline_mean", "support_nn", "class_bank4"]:
        scores = mode_scores.get(idea_name, [])
        if scores:
            fp.write(f"- Mean primary score `{idea_name}`: `{sum(scores) / len(scores):.4f}` (n={len(scores)})\n")

    fp.write("\n## Per-Task Comparison\n\n")
    fp.write("| Experiment | Dataset | Metric | baseline_mean | support_nn | class_bank4 | Best |\n")
    fp.write("| --- | --- | --- | --- | --- | --- | --- |\n")
    for row in task_rows:
        fp.write(
            f"| {row['experiment_id']} | {row['dataset_name']} | {row['metric_name']} | "
            f"{row['baseline']:.4f} | {row['support_nn']:.4f} | {row['class_bank4']:.4f} | "
            f"{row['best_name']} ({row['best_score']:.4f}) |\n"
        )

    fp.write("\n## Key Deltas\n\n")
    if improvements:
        fp.write("- Positive moves over baseline:\n")
        for delta, experiment_id, idea_name, score, baseline_score in sorted(improvements, reverse=True):
            fp.write(
                f"  - `{experiment_id}` with `{idea_name}`: `{score:.4f}` vs baseline `{baseline_score:.4f}` "
                f"(delta `+{delta:.4f}`)\n"
            )
    if regressions:
        fp.write("- Negative moves over baseline:\n")
        for delta, experiment_id, idea_name, score, baseline_score in sorted(regressions):
            fp.write(
                f"  - `{experiment_id}` with `{idea_name}`: `{score:.4f}` vs baseline `{baseline_score:.4f}` "
                f"(delta `{delta:.4f}`)\n"
            )
    if not improvements and not regressions:
        fp.write("- No measurable difference detected.\n")
PY

{
  echo
  echo "[sav_ideas] Selected experiments: ${selected_experiments}"
  echo "[sav_ideas] Selected runs: ${selected_runs}"
  echo "[sav_ideas] Failures: ${failures}"
  echo "[sav_ideas] Markdown record: ${RECORD_PATH}"
}

if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
