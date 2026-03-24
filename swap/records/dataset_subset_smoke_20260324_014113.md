# Dataset Subset Smoke Test

- Timestamp (UTC): `20260324_014113`
- Run mode: `subset_smoke`
- Model: `qwen2_vl`
- Methods: `zero_shot`
- Validate datasets: `1`
- Execute `main.py`: `0`
- Dataset filter: `^(mhalubench|vizwiz)$`

## Output Layout

- Subsets: `swap/subsets/20260324_014113`
- Logs: `swap/logs/20260324_014113`
- Metrics: `swap/outputs/20260324_014113`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vizwiz:zero_shot | vizwiz | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| mhalubench_val_v01:zero_shot | mhalubench | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| mhalubench_val_v02:zero_shot | mhalubench | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |

## Summary

- Selected experiments: `3`
- Failures: `0`
- Record: `swap/records/dataset_subset_smoke_20260324_014113.md`
