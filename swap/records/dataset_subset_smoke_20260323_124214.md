# Dataset Subset Smoke Test

- Timestamp (UTC): `20260323_124214`
- Run mode: `subset_smoke`
- Model: `qwen2_vl`
- Methods: `zero_shot`
- Validate datasets: `1`
- Execute `main.py`: `0`
- Dataset filter: `mhalubench`

## Output Layout

- Subsets: `swap/subsets/20260323_124214`
- Logs: `swap/logs/20260323_124214`
- Metrics: `swap/outputs/20260323_124214`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mhalubench_val_v01:zero_shot | mhalubench | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| mhalubench_val_v02:zero_shot | mhalubench | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |

## Summary

- Selected experiments: `2`
- Failures: `0`
- Record: `swap/records/dataset_subset_smoke_20260323_124214.md`
