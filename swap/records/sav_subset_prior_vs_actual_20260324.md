# SAV Subset Prior vs Actual

- Date (UTC): `2026-03-24`
- Main smoke run record: `swap/records/dataset_subset_smoke_20260324_020033.md`
- Main smoke outputs: `swap/outputs/20260324_020033`
- Follow-up labeled outputs: `swap/outputs/20260324_020033_followup3`
- Model: `qwen2_vl`
- Method: `sav`
- Default subset sizes:
  - most tasks: train `4`, val `4`
  - `pets/eurosat/flowers/cub/tinyimage`: train `3`, val `3`

## How I Judged Before Running

- `High`: expected `>= 0.75` on tiny labeled subsets
- `Medium`: expected around `0.50 - 0.75`
- `Low`: expected `< 0.50`

The prior is based on the current `SAV` implementation: it builds class prototypes from selected attention heads, so it should favor closed-set visual recognition and simpler binary discrimination, and be less stable on reasoning-heavy or multi-image comparison tasks.

## Experiment Table

| Task | Task Type | Prior | Main Subset Result | Follow-up Result | Judgment |
| --- | --- | --- | --- | --- | --- |
| `pets` | fine-grained classification | High | `1.00` | `-` | matched |
| `eurosat` | remote sensing classification | High | `1.00` | `-` | matched |
| `flowers` | fine-grained classification | High | `1.00` | `-` | matched |
| `cub` | fine-grained classification | High | `1.00` | `-` | matched |
| `tinyimage` | 200-way classification | Medium | `0.00` | `0.00` | mismatch, but subset is degenerate |
| `vizwiz` | answerable vs unanswerable | Medium | `0.75` | `-` | matched |
| `vlguard` | harmful vs unharmful | Medium | `0.50` | `-` | matched |
| `mhalubench_val_v01` | hallucination detection | High | `1.00` | `-` | matched |
| `mhalubench_val_v02` | hallucination detection | High | `1.00` | `-` | matched |
| `sugarcrepe` | pairwise yes/no | High | `1.00` pair / `1.00` raw | `-` | matched |
| `naturalbench_ret` | grouped yes/no retrieval | Medium | `0.50` raw / `0.00` group | `-` | partly matched |
| `naturalbench_vqa` | VQA-style classification | Medium | `1.00` | `-` | better than expected |
| `blink_art_style` | multi-image style match | Low | `0.00` with hidden test labels | `0.00` | matched |
| `blink_counting` | counting MCQ | Medium | `0.00` with hidden test labels | `0.25` | weaker than expected |
| `blink_forensic_detection` | forensic reasoning | Low | `0.00` with hidden test labels | `0.50` | better than expected |
| `blink_functional_correspondence` | correspondence reasoning | Low | `0.00` with hidden test labels | `0.50` | better than expected |
| `blink_iq_test` | puzzle reasoning | Low | `0.00` with hidden test labels | `0.50` | better than expected |
| `blink_jigsaw` | layout matching | Medium | `0.00` with hidden test labels | `0.75` | matched |
| `blink_multi-view_reasoning` | multi-view reasoning | Low | `0.00` with hidden test labels | `0.00` | matched |
| `blink_object_localization` | localization | Medium | `0.00` with hidden test labels | `0.50` | matched |
| `blink_relative_depth` | relative depth | Medium | `0.00` with hidden test labels | `0.50` | matched |
| `blink_relative_reflectance` | reflectance reasoning | Medium | `0.00` with hidden test labels | `0.25` | weaker than expected |
| `blink_semantic_correspondence` | semantic correspondence | Low | `0.00` with hidden test labels | `0.50` | better than expected |
| `blink_spatial_relation` | spatial relation | Medium | `0.00` with hidden test labels | `1.00` | better than expected |
| `blink_visual_correspondence` | visual correspondence | Low | `0.00` with hidden test labels | `0.25` | matched |
| `blink_visual_similarity` | visual similarity | Medium | `0.00` with hidden test labels | `0.50` | matched |

## Important Caveats

- The main smoke script currently evaluates `BLINK` on `*_test.json`, whose labels are `HIDDEN`. So the `0.00` scores in the main run are not meaningful as method-quality signals.
- The main smoke script currently evaluates `tinyimage` on `test.json`, whose labels are `unknown`. So its `0.00` is also not a meaningful benchmark result.
- The follow-up run fixes that by using labeled `BLINK *_val.json` and labeled `tinyimage/val.json`.
- `tinyimage` still scored `0.00` in the follow-up because the naive 4-shot train subset came from the start of `train.json`, where all four training examples were the same class (`Egyptian cat`), while the validation subset contained different classes. This is a subset-construction issue, not strong evidence that `SAV` categorically fails on TinyImage.
- `naturalbench_ret` uses a group metric. With only one 4-sample group, `raw_acc = 0.50` can still become `g_acc = 0.00`.

## What the Runs Say

- On standard closed-set visual classification, `SAV` looks strong on tiny subsets: `pets`, `eurosat`, `flowers`, and `cub` all reached `1.00`.
- On binary vision-language classification, `SAV` is also promising: `mhalubench` was `1.00`, `sugarcrepe` was `1.00`, `vizwiz` was `0.75`, and `vlguard` was `0.50`.
- On `BLINK`, the corrected labeled follow-up shows a mixed picture rather than total failure. It is weak on `art_style`, `multi-view_reasoning`, and some harder option-matching tasks, but workable on `jigsaw`, `spatial_relation`, `forensic_detection`, `functional_correspondence`, `semantic_correspondence`, and several perceptual tasks.
- The broad prior was mostly right: `SAV` is best when the task can be reduced to stable visual prototypes, and less reliable when the label depends on harder relational reasoning or when the subset construction is poor.

## Aggregate Readout

- Main smoke run:
  - classic FGVC family average: `0.80`
  - binary vision-language family average: `0.85`
  - NaturalBench family average: `0.75`
  - BLINK family average: `0.00`, but this is not interpretable because the evaluation labels are hidden
- Labeled follow-up:
  - BLINK family average: `0.43`
  - best BLINK tasks in this run: `spatial_relation = 1.00`, `jigsaw = 0.75`
  - weakest BLINK tasks in this run: `art_style = 0.00`, `multi-view_reasoning = 0.00`

## Next Fix I Would Recommend

- Make `scripts/run_full_data_train.sh` use labeled validation splits for `BLINK` and `tinyimage`, and make subset selection stratified by label instead of taking the first `N` rows. That would turn this from a smoke pipeline into a much more trustworthy quick benchmark for `SAV`.
