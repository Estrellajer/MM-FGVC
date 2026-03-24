# Dataset Subset Smoke Test

- Timestamp (UTC): `20260323_124242`
- Run mode: `subset_smoke`
- Model: `qwen2_vl`
- Methods: `zero_shot`
- Validate datasets: `1`
- Execute `main.py`: `1`
- Dataset filter: `pets|blink_art_style`

## Output Layout

- Subsets: `swap/subsets/20260323_124242`
- Logs: `swap/logs/20260323_124242`
- Metrics: `swap/outputs/20260323_124242`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pets:zero_shot | pets | raw | 3 | 3 | PASS | PASS | validate=[train=3 val=3 image_paths=6 multi_image_samples=0] log=zero_shot_pets_qwen2_vl_subset.log |
| blink_art_style:zero_shot | blink_art_style | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=24 multi_image_samples=8] log=zero_shot_blink_art_style_qwen2_vl_subset.log |

## Summary

- Selected experiments: `2`
- Failures: `0`
- Record: `swap/records/dataset_subset_smoke_20260323_124242.md`
