# SAV Head Selection Ablation

- Timestamp (UTC): `20260324_034917`
- Model: `qwen2_vl`
- Method: `sav`
- Goal: test whether explicit head selection is necessary, and whether random or non-adaptive choices can match `topk`
- Fixed `num_heads`: `20`
- Strategies:
  - `topk`
  - `firstk`
  - `all`
  - `random` with seeds: `11 22 33`
- Principle:
  - reuse author-style subset construction
  - use larger-than-smoke, compute-capped validation subsets
  - compare strategies on the same train/eval subsets

## Output Layout

- Subsets: `swap/subsets/20260324_034917_sav_head_ablation`
- Logs: `swap/logs/20260324_034917_sav_head_ablation`
- Metrics: `swap/outputs/20260324_034917_sav_head_ablation`

## Result Table

| Experiment | Dataset | Strategy | Train | Val | Metric | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vlguard | vlguard | topk | 40 | 80 | accuracy=0.9750 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=topk seed=42 log=sav_vlguard_topk_qwen2_vl_ablation.log |
| vlguard | vlguard | firstk | 40 | 80 | accuracy=0.7375 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=firstk seed=42 log=sav_vlguard_firstk_qwen2_vl_ablation.log |
| vlguard | vlguard | all | 40 | 80 | accuracy=0.9000 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=all seed=42 log=sav_vlguard_all_qwen2_vl_ablation.log |
| vlguard | vlguard | random_s11 | 40 | 80 | accuracy=0.6375 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=random seed=11 log=sav_vlguard_random_s11_qwen2_vl_ablation.log |

## Aggregate Summary

- Completed runs: `4`
- Mean primary score `topk`: `0.9750` (n=1)
- Mean primary score `firstk`: `0.7375` (n=1)
- Mean primary score `all`: `0.9000` (n=1)
- Mean primary score `random_mean`: `0.6375` (n=1)

## Per-Task Comparison

| Experiment | Dataset | Metric | topk | firstk | all | random mean | random std | best random | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| vlguard | vlguard | accuracy | 0.9750 | 0.7375 | 0.9000 | 0.6375 | 0.0000 | 0.6375 | topk clearly beats random; selection helps over all-head voting; adaptive selection beats fixed first-k |

## Automated Takeaways

- Tasks where adaptive `topk` head selection clearly helps over random mean:
  - `vlguard`: `topk=0.9750` vs `random_mean=0.6375` (gap `0.3375`)
