# Dataset Subset Smoke Test

- Timestamp (UTC): `20260323_124319`
- Run mode: `subset_smoke`
- Model: `qwen2_vl`
- Methods: `sav`
- Validate datasets: `1`
- Execute `main.py`: `1`
- Dataset filter: `^pets$`

## Output Layout

- Subsets: `swap/subsets/20260323_124319`
- Logs: `swap/logs/20260323_124319`
- Metrics: `swap/outputs/20260323_124319`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pets:sav | pets | raw | 3 | 3 | PASS | PASS | validate=[train=3 val=3 image_paths=6 multi_image_samples=0] log=sav_pets_qwen2_vl_subset.log |

## Summary

- Selected experiments: `1`
- Failures: `0`
- Record: `swap/records/dataset_subset_smoke_20260323_124319.md`
