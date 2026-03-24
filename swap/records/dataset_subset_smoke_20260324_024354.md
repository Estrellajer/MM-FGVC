# Author-Style Dataset Subset Smoke Test

- Timestamp (UTC): `20260324_024354`
- Run mode: `subset_smoke`
- Model: `qwen2_vl`
- Methods: `sav`
- Support rule: `ref-data` when available, otherwise first `20` valid examples per label
- Eval rule: official labeled eval split when available; same-source tasks exclude support rows
- Validate datasets: `1`
- Execute `main.py`: `0`
- Dataset filter: `^vizwiz$`

## Output Layout

- Subsets: `swap/subsets/20260324_024354`
- Logs: `swap/logs/20260324_024354`
- Metrics: `swap/outputs/20260324_024354`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vizwiz:sav | vizwiz | raw | 40 | 4 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=4 labels=2 per_label=2] validate=[train=40 val=4 image_paths=44 multi_image_samples=0] |

## Summary

- Selected experiments: `1`
- Failures: `0`
- Record: `swap/records/dataset_subset_smoke_20260324_024354.md`
