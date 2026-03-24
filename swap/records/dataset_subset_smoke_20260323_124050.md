# Dataset Subset Smoke Test

- Timestamp (UTC): `20260323_124050`
- Run mode: `subset_smoke`
- Model: `qwen2_vl`
- Methods: `zero_shot`
- Validate datasets: `1`
- Execute `main.py`: `0`
- Dataset filter: `all`

## Output Layout

- Subsets: `swap/subsets/20260323_124050`
- Logs: `swap/logs/20260323_124050`
- Metrics: `swap/outputs/20260323_124050`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| naturalbench_ret:zero_shot | naturalbench_ret | naturalbench_group | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| naturalbench_vqa:zero_shot | naturalbench_vqa | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| sugarcrepe:zero_shot | sugarcrepe | pair | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| pets:zero_shot | pets | raw | 3 | 3 | PASS | SKIP | train=3 val=3 image_paths=6 multi_image_samples=0 |
| eurosat:zero_shot | eurosat | raw | 3 | 3 | PASS | SKIP | train=3 val=3 image_paths=6 multi_image_samples=0 |
| flowers:zero_shot | flowers | raw | 3 | 3 | PASS | SKIP | train=3 val=3 image_paths=6 multi_image_samples=0 |
| cub:zero_shot | cub | raw | 3 | 3 | PASS | SKIP | train=3 val=3 image_paths=6 multi_image_samples=0 |
| tinyimage:zero_shot | tinyimage | raw | 3 | 3 | PASS | SKIP | train=3 val=3 image_paths=6 multi_image_samples=0 |
| vizwiz:zero_shot | vizwiz | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| vlguard:zero_shot | vlguard | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| mhalubench_val_v01 | mhalubench | raw | 4 | 4 | FAIL | SKIP | validate_log=mhalubench_val_v01.validate.log Traceback (most recent call last): File "<stdin>", line 45, in <module> FileNotFoundError: train[0] missing image: /root/autodl-tmp/FGVC/Data/MHaluBench/data/image-to-text/b4e93b744f962240.jpg |
| mhalubench_val_v02 | mhalubench | raw | 4 | 4 | FAIL | SKIP | validate_log=mhalubench_val_v02.validate.log Traceback (most recent call last): File "<stdin>", line 45, in <module> FileNotFoundError: train[0] missing image: /root/autodl-tmp/FGVC/Data/MHaluBench/data/image-to-text/b4e93b744f962240.jpg |
| blink_art_style:zero_shot | blink_art_style | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=24 multi_image_samples=8 |
| blink_counting:zero_shot | blink_counting | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| blink_forensic_detection:zero_shot | blink_forensic_detection | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=32 multi_image_samples=8 |
| blink_functional_correspondence:zero_shot | blink_functional_correspondence | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=16 multi_image_samples=8 |
| blink_iq_test:zero_shot | blink_iq_test | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| blink_jigsaw:zero_shot | blink_jigsaw | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=24 multi_image_samples=8 |
| blink_multi-view_reasoning:zero_shot | blink_multi-view_reasoning | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=16 multi_image_samples=8 |
| blink_object_localization:zero_shot | blink_object_localization | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| blink_relative_depth:zero_shot | blink_relative_depth | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| blink_relative_reflectance:zero_shot | blink_relative_reflectance | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| blink_semantic_correspondence:zero_shot | blink_semantic_correspondence | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=16 multi_image_samples=8 |
| blink_spatial_relation:zero_shot | blink_spatial_relation | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=8 multi_image_samples=0 |
| blink_visual_correspondence:zero_shot | blink_visual_correspondence | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=16 multi_image_samples=8 |
| blink_visual_similarity:zero_shot | blink_visual_similarity | raw | 4 | 4 | PASS | SKIP | train=4 val=4 image_paths=24 multi_image_samples=8 |

## Summary

- Selected experiments: `26`
- Failures: `2`
- Record: `swap/records/dataset_subset_smoke_20260323_124050.md`
