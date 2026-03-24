# SAV All-Task Weighted Voting Report

## Goal

Evaluate the "route A" idea from the latest analysis on **all supported tasks**, not only FGVC image classification:

- baseline: `SAV`
- new variant: `SAV+WVote`
- model: `qwen2_vl`

This report focuses on whether **head-accuracy weighted voting** improves the current SAV pipeline when we expand evaluation to the full multi-task suite.

## Task Coverage

The sweep covers `26` experiments across classification and non-classification settings:

- FGVC / image classification: `pets`, `eurosat`, `flowers`, `cub`, `tinyimage`
- retrieval / VQA / pair tasks: `naturalbench_ret`, `naturalbench_vqa`, `sugarcrepe`
- robustness / hallucination / safety: `vizwiz`, `vlguard`, `mhalubench_val_v01`, `mhalubench_val_v02`
- multi-image BLINK tasks: `blink_art_style`, `blink_counting`, `blink_forensic_detection`, `blink_functional_correspondence`, `blink_iq_test`, `blink_jigsaw`, `blink_multi-view_reasoning`, `blink_object_localization`, `blink_relative_depth`, `blink_relative_reflectance`, `blink_semantic_correspondence`, `blink_spatial_relation`, `blink_visual_correspondence`, `blink_visual_similarity`

## Experimental Artifacts

- Baseline SAV sweep: `swap/records/sav_diagnostic_sweep_20260324_025703.md`
- New SAV+WVote sweep: `swap/records/all_task_method_suite_20260324_133026.md`
- Direct comparison: `swap/records/sav_weighting_comparison_20260324_133026.md`

## Main Result

`SAV+WVote` is a complete no-op on the current all-task suite.

| Bucket | Count | SAV mean | SAV+WVote mean | Delta mean |
| --- | --- | --- | --- | --- |
| all | 26 | 0.5890 | 0.5890 | +0.0000 |
| fgvc | 5 | 0.6851 | 0.6851 | +0.0000 |
| non_fgvc | 21 | 0.5661 | 0.5661 | +0.0000 |

- Wins / ties / losses: `0 / 26 / 0`
- No task changed metric value
- No gain on FGVC tasks
- No gain on non-classification tasks either

## Representative Examples

| Experiment | Metric | SAV | SAV+WVote | Delta |
| --- | --- | --- | --- | --- |
| `naturalbench_ret` | `g_acc` | 0.8000 | 0.8000 | +0.0000 |
| `naturalbench_vqa` | `accuracy` | 0.8000 | 0.8000 | +0.0000 |
| `sugarcrepe` | `pair_accuracy` | 0.5000 | 0.5000 | +0.0000 |
| `eurosat` | `accuracy` | 0.6400 | 0.6400 | +0.0000 |
| `cub` | `accuracy` | 0.7100 | 0.7100 | +0.0000 |
| `pets` | `accuracy` | 1.0000 | 1.0000 | +0.0000 |
| `blink_art_style` | `accuracy` | 0.9000 | 0.9000 | +0.0000 |
| `blink_semantic_correspondence` | `accuracy` | 0.1000 | 0.1000 | +0.0000 |

## Interpretation

The latest hypothesis was:

- keep SAV's current representation pipeline
- replace uniform head voting with train-accuracy weighted voting

Empirically, this does **not** buy us anything on the current implementation. The strongest conclusion from this run is:

1. `SAV` remains the right family to prioritize.
2. `head-accuracy weighted voting` is not a meaningful improvement direction for the current SAV formulation.
3. This negative result is not limited to FGVC classification; it also holds on retrieval, VQA, safety, hallucination, and BLINK multi-image tasks.

## Publishing Implication

If we follow the latest strategy analysis, the next SAV-focused iterations should move away from weighted voting and prioritize ideas with a higher chance of changing predictions:

1. `SAV + constrained decoding`
2. task-adaptive retrieval / `support_nn` style routing on multi-image tasks
3. adding visual-side features instead of only changing the final voting rule

For cross-method comparison among `Zero-shot / SAV / MimIC / I2CL / STV / SAV-TV / QC-TV`, the current best reference is still [FGVC_METHOD_SUITE_RESULTS.md](/root/autodl-tmp/FGVC/docs/FGVC_METHOD_SUITE_RESULTS.md).
