#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

MODEL="${MODEL:-qwen2_vl}"
NUM_HEADS="${NUM_HEADS:-20}"
STOP_ON_ERROR="${STOP_ON_ERROR:-0}"
DATASET_FILTER="${DATASET_FILTER:-}"
STRATEGY_FILTER="${STRATEGY_FILTER:-}"
RANDOM_SEEDS="${RANDOM_SEEDS:-11 22 33}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

SWAP_ROOT="$PROJECT_ROOT/swap"
SUBSET_ROOT="$SWAP_ROOT/subsets/${TIMESTAMP}_sav_head_ablation"
LOG_ROOT="$SWAP_ROOT/logs/${TIMESTAMP}_sav_head_ablation"
OUTPUT_ROOT="$SWAP_ROOT/outputs/${TIMESTAMP}_sav_head_ablation"
RECORD_ROOT="$SWAP_ROOT/records"
RECORD_PATH="$RECORD_ROOT/sav_head_ablation_${TIMESTAMP}.md"
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

matches_strategy_filter() {
  local strategy_name="$1"
  if [[ -z "$STRATEGY_FILTER" ]]; then
    return 0
  fi
  [[ "$strategy_name" =~ $STRATEGY_FILTER ]]
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
  local selection_strategy="$7"
  local selection_seed="$8"

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
    method.params.selection_strategy="$selection_strategy" \
    method.params.selection_seed="$selection_seed" \
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
# SAV Head Selection Ablation

- Timestamp (UTC): \`$TIMESTAMP\`
- Model: \`$MODEL\`
- Method: \`sav\`
- Goal: test whether explicit head selection is necessary, and whether random or non-adaptive choices can match \`topk\`
- Fixed \`num_heads\`: \`$NUM_HEADS\`
- Strategies:
  - \`topk\`
  - \`firstk\`
  - \`all\`
  - \`random\` with seeds: \`$RANDOM_SEEDS\`
- Principle:
  - reuse author-style subset construction
  - use larger-than-smoke, compute-capped validation subsets
  - compare strategies on the same train/eval subsets

## Output Layout

- Subsets: \`swap/subsets/${TIMESTAMP}_sav_head_ablation\`
- Logs: \`swap/logs/${TIMESTAMP}_sav_head_ablation\`
- Metrics: \`swap/outputs/${TIMESTAMP}_sav_head_ablation\`

## Result Table

| Experiment | Dataset | Strategy | Train | Val | Metric | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
EOF

printf 'experiment_id\tdataset_name\tevaluator_name\tstrategy_name\tselection_strategy\tselection_seed\tmetrics_path\tlog_path\n' > "$MANIFEST_PATH"

EXPERIMENTS=(
  "vlguard|vlguard|raw|ref-data/vlguard_train.json|dataset/converted_from_data/vlguard/test.json|author_ref|0|per_label|40|0"
  "mhalubench_val_v01|mhalubench|raw|ref-data/mahalu_train.json|dataset/converted_from_data/mhalubench/val_v01.json|author_ref|0|per_label|40|0"
  "naturalbench_vqa|naturalbench_vqa|raw|dataset/converted_from_data/naturalbench/train.jsonl|dataset/converted_from_data/naturalbench/train.jsonl|per_label|20|grouped|80|4"
  "sugarcrepe|sugarcrepe|pair|dataset/converted_from_data/sugarcrepe/sugarcrepe_all.jsonl|dataset/converted_from_data/sugarcrepe/sugarcrepe_all.jsonl|per_label|20|grouped|80|2"
  "eurosat|eurosat|raw|ref-data/eurosat_train.json|dataset/converted_from_data/eurosat/test.json|per_label|10|per_label|10|0"
  "tinyimage|tinyimage|raw|dataset/converted_from_data/tinyimage/train.json|dataset/converted_from_data/tinyimage/val.json|per_label|2|per_label|1|0"
  "blink_art_style|blink_art_style|raw|ref-data/blink_art_style_train.json|dataset/converted_from_data/blink/art_style_val.json|author_ref|0|per_label|20|0"
  "blink_semantic_correspondence|blink_semantic_correspondence|raw|dataset/converted_from_data/blink/semantic_correspondence_val.json|dataset/converted_from_data/blink/semantic_correspondence_val.json|per_label|20|per_label|10|0"
)

STRATEGY_SPECS=(
  "topk|topk|42"
  "firstk|firstk|42"
  "all|all|42"
)

for seed in $RANDOM_SEEDS; do
  STRATEGY_SPECS+=("random_s${seed}|random|${seed}")
done

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
  echo "[sav_ablation] Preparing ${experiment_id}"

  if ! train_n="$(build_subset_file "$dataset_name" "$train_src" "$train_subset" "$train_mode" "$train_count" "0" "" "" "$train_meta" 2>&1)"; then
    failures=$((failures + 1))
    printf '| %s | %s | setup | ERR | ERR | - | FAIL | %s |\n' \
      "$(sanitize_cell "$experiment_id")" \
      "$(sanitize_cell "$dataset_name")" \
      "$(sanitize_cell "train subset build failed: $train_n")" >> "$RECORD_PATH"
    echo "[sav_ablation][FAIL] train subset build failed for ${experiment_id}"
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
    echo "[sav_ablation][FAIL] val subset build failed for ${experiment_id}"
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
    echo "[sav_ablation][FAIL] validation failed for ${experiment_id}"
    if [[ "$STOP_ON_ERROR" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  echo "[sav_ablation][PASS] validation: ${validate_output}"

  for strategy_entry in "${STRATEGY_SPECS[@]}"; do
    IFS='|' read -r strategy_name selection_strategy selection_seed <<< "$strategy_entry"

    if ! matches_strategy_filter "$strategy_name"; then
      continue
    fi

    selected_runs=$((selected_runs + 1))
    run_name="sav_${experiment_id}_${strategy_name}_${MODEL}_ablation"
    run_log="$LOG_ROOT/${run_name}.log"

    echo "[sav_ablation] Running ${run_name}"

    if ! run_main_subset "$dataset_name" "$evaluator_name" "$train_subset" "$val_subset" "$run_name" "$run_log" "$selection_strategy" "$selection_seed"; then
      failures=$((failures + 1))
      printf '| %s | %s | %s | %s | %s | - | FAIL | %s |\n' \
        "$(sanitize_cell "$experiment_id")" \
        "$(sanitize_cell "$dataset_name")" \
        "$(sanitize_cell "$strategy_name")" \
        "$(sanitize_cell "$train_n")" \
        "$(sanitize_cell "$val_n")" \
        "$(sanitize_cell "train=[$train_note] val=[$val_note] validate=[$validate_output] log=$(basename "$run_log")")" >> "$RECORD_PATH"
      echo "[sav_ablation][FAIL] ${run_name}"
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
      "$strategy_name" \
      "$selection_strategy" \
      "$selection_seed" \
      "$metrics_path" \
      "$run_log" >> "$MANIFEST_PATH"

    printf '| %s | %s | %s | %s | %s | %s | PASS | %s |\n' \
      "$(sanitize_cell "$experiment_id")" \
      "$(sanitize_cell "$dataset_name")" \
      "$(sanitize_cell "$strategy_name")" \
      "$(sanitize_cell "$train_n")" \
      "$(sanitize_cell "$val_n")" \
      "$(sanitize_cell "$metric_text")" \
      "$(sanitize_cell "train=[$train_note] val=[$val_note] validate=[$validate_output] strategy=${selection_strategy} seed=${selection_seed} log=$(basename "$run_log")")" >> "$RECORD_PATH"

    echo "[sav_ablation][PASS] ${run_name} ${metric_text}"
  done
done

if [[ "$selected_experiments" -eq 0 ]]; then
  echo "[sav_ablation] No experiments matched DATASET_FILTER='${DATASET_FILTER}'"
  exit 1
fi

python - "$MANIFEST_PATH" "$RECORD_PATH" <<'PY'
import csv
import json
import math
import statistics
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
        metrics_payload = json.loads(Path(row["metrics_path"]).read_text(encoding="utf-8"))
        metric_name, score = primary_metric(metrics_payload["metrics"])
        rows.append(
            {
                "experiment_id": row["experiment_id"],
                "dataset_name": row["dataset_name"],
                "strategy_name": row["strategy_name"],
                "selection_strategy": row["selection_strategy"],
                "selection_seed": int(row["selection_seed"]),
                "metric_name": metric_name,
                "score": score,
            }
        )

by_experiment: dict[str, list[dict]] = defaultdict(list)
for row in rows:
    by_experiment[row["experiment_id"]].append(row)

strategy_averages: dict[str, list[float]] = defaultdict(list)
task_rows: list[dict] = []
selection_helpful = []
selection_not_critical = []
all_better = []

for experiment_id, experiment_rows in sorted(by_experiment.items()):
    metrics_by_strategy = {row["strategy_name"]: row for row in experiment_rows}
    topk = metrics_by_strategy.get("topk")
    firstk = metrics_by_strategy.get("firstk")
    all_heads = metrics_by_strategy.get("all")
    random_rows = [row for row in experiment_rows if row["selection_strategy"] == "random"]
    random_scores = [row["score"] for row in random_rows]

    if topk:
        strategy_averages["topk"].append(topk["score"])
    if firstk:
        strategy_averages["firstk"].append(firstk["score"])
    if all_heads:
        strategy_averages["all"].append(all_heads["score"])
    if random_scores:
        strategy_averages["random_mean"].append(sum(random_scores) / len(random_scores))

    metric_name = topk["metric_name"] if topk else experiment_rows[0]["metric_name"]
    random_mean = sum(random_scores) / len(random_scores) if random_scores else math.nan
    random_std = statistics.pstdev(random_scores) if len(random_scores) > 1 else 0.0
    best_random = max(random_scores) if random_scores else math.nan

    notes = []
    if topk and random_scores:
        gap = topk["score"] - random_mean
        if gap >= 0.05:
            notes.append("topk clearly beats random")
            selection_helpful.append((gap, experiment_id, topk["score"], random_mean))
        elif abs(gap) <= 0.02:
            notes.append("random is close to topk")
            selection_not_critical.append((abs(gap), experiment_id, topk["score"], random_mean))
    if topk and all_heads:
        all_gap = all_heads["score"] - topk["score"]
        if all_gap >= 0.05:
            notes.append("all heads beats topk")
            all_better.append((all_gap, experiment_id, all_heads["score"], topk["score"]))
        elif topk["score"] - all_heads["score"] >= 0.05:
            notes.append("selection helps over all-head voting")
    if topk and firstk and topk["score"] - firstk["score"] >= 0.05:
        notes.append("adaptive selection beats fixed first-k")

    task_rows.append(
        {
            "experiment_id": experiment_id,
            "dataset_name": experiment_rows[0]["dataset_name"],
            "metric_name": metric_name,
            "topk": topk["score"] if topk else math.nan,
            "firstk": firstk["score"] if firstk else math.nan,
            "all": all_heads["score"] if all_heads else math.nan,
            "random_mean": random_mean,
            "random_std": random_std,
            "best_random": best_random,
            "notes": "; ".join(notes) if notes else "mixed",
        }
    )

with record_path.open("a", encoding="utf-8") as fp:
    fp.write("\n## Aggregate Summary\n\n")
    fp.write(f"- Completed runs: `{len(rows)}`\n")
    for strategy_name in ["topk", "firstk", "all", "random_mean"]:
        scores = strategy_averages.get(strategy_name, [])
        if scores:
            fp.write(
                f"- Mean primary score `{strategy_name}`: `{sum(scores) / len(scores):.4f}` "
                f"(n={len(scores)})\n"
            )

    fp.write("\n## Per-Task Comparison\n\n")
    fp.write("| Experiment | Dataset | Metric | topk | firstk | all | random mean | random std | best random | Notes |\n")
    fp.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for row in sorted(task_rows, key=lambda item: item["topk"] - item["random_mean"] if not math.isnan(item["random_mean"]) else -999, reverse=True):
        fp.write(
            f"| {row['experiment_id']} | {row['dataset_name']} | {row['metric_name']} | "
            f"{row['topk']:.4f} | {row['firstk']:.4f} | {row['all']:.4f} | "
            f"{row['random_mean']:.4f} | {row['random_std']:.4f} | {row['best_random']:.4f} | {row['notes']} |\n"
        )

    fp.write("\n## Automated Takeaways\n\n")
    if selection_helpful:
        fp.write("- Tasks where adaptive `topk` head selection clearly helps over random mean:\n")
        for gap, experiment_id, topk_score, random_mean in sorted(selection_helpful, reverse=True):
            fp.write(
                f"  - `{experiment_id}`: `topk={topk_score:.4f}` vs `random_mean={random_mean:.4f}` "
                f"(gap `{gap:.4f}`)\n"
            )
    if selection_not_critical:
        fp.write("- Tasks where random is already close to `topk`:\n")
        for gap, experiment_id, topk_score, random_mean in sorted(selection_not_critical):
            fp.write(
                f"  - `{experiment_id}`: `topk={topk_score:.4f}` vs `random_mean={random_mean:.4f}` "
                f"(gap `{gap:.4f}`)\n"
            )
    if all_better:
        fp.write("- Tasks where using all heads beats `topk`:\n")
        for gap, experiment_id, all_score, topk_score in sorted(all_better, reverse=True):
            fp.write(
                f"  - `{experiment_id}`: `all={all_score:.4f}` vs `topk={topk_score:.4f}` "
                f"(gap `{gap:.4f}`)\n"
            )

    if not selection_helpful and not selection_not_critical and not all_better:
        fp.write("- No strong pattern detected in the completed runs.\n")
PY

{
  echo
  echo "[sav_ablation] Selected experiments: ${selected_experiments}"
  echo "[sav_ablation] Selected runs: ${selected_runs}"
  echo "[sav_ablation] Failures: ${failures}"
  echo "[sav_ablation] Markdown record: ${RECORD_PATH}"
}

if [[ "$failures" -ne 0 ]]; then
  exit 1
fi
