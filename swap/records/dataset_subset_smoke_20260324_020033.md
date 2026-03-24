# Dataset Subset Smoke Test

- Timestamp (UTC): `20260324_020033`
- Run mode: `subset_smoke`
- Model: `qwen2_vl`
- Methods: `sav`
- Validate datasets: `1`
- Execute `main.py`: `1`
- Dataset filter: `all`

## Output Layout

- Subsets: `swap/subsets/20260324_020033`
- Logs: `swap/logs/20260324_020033`
- Metrics: `swap/outputs/20260324_020033`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| naturalbench_ret:sav | naturalbench_ret | naturalbench_group | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_naturalbench_ret_qwen2_vl_subset.log |
| naturalbench_vqa:sav | naturalbench_vqa | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_naturalbench_vqa_qwen2_vl_subset.log |
| sugarcrepe:sav | sugarcrepe | pair | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_sugarcrepe_qwen2_vl_subset.log |
| pets:sav | pets | raw | 3 | 3 | PASS | PASS | validate=[train=3 val=3 image_paths=6 multi_image_samples=0] log=sav_pets_qwen2_vl_subset.log |
| eurosat:sav | eurosat | raw | 3 | 3 | PASS | PASS | validate=[train=3 val=3 image_paths=6 multi_image_samples=0] log=sav_eurosat_qwen2_vl_subset.log |
| flowers:sav | flowers | raw | 3 | 3 | PASS | PASS | validate=[train=3 val=3 image_paths=6 multi_image_samples=0] log=sav_flowers_qwen2_vl_subset.log |
| cub:sav | cub | raw | 3 | 3 | PASS | PASS | validate=[train=3 val=3 image_paths=6 multi_image_samples=0] log=sav_cub_qwen2_vl_subset.log |
| tinyimage:sav | tinyimage | raw | 3 | 3 | PASS | PASS | validate=[train=3 val=3 image_paths=6 multi_image_samples=0] log=sav_tinyimage_qwen2_vl_subset.log |
| vizwiz:sav | vizwiz | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_vizwiz_qwen2_vl_subset.log |
| vlguard:sav | vlguard | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_vlguard_qwen2_vl_subset.log |
| mhalubench_val_v01:sav | mhalubench | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_mhalubench_val_v01_qwen2_vl_subset.log |
| mhalubench_val_v02:sav | mhalubench | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_mhalubench_val_v02_qwen2_vl_subset.log |
| blink_art_style:sav | blink_art_style | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=24 multi_image_samples=8] log=sav_blink_art_style_qwen2_vl_subset.log |
| blink_counting:sav | blink_counting | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_blink_counting_qwen2_vl_subset.log |
| blink_forensic_detection:sav | blink_forensic_detection | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=32 multi_image_samples=8] log=sav_blink_forensic_detection_qwen2_vl_subset.log |
| blink_functional_correspondence:sav | blink_functional_correspondence | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=16 multi_image_samples=8] log=sav_blink_functional_correspondence_qwen2_vl_subset.log |
| blink_iq_test:sav | blink_iq_test | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_blink_iq_test_qwen2_vl_subset.log |
| blink_jigsaw:sav | blink_jigsaw | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=24 multi_image_samples=8] log=sav_blink_jigsaw_qwen2_vl_subset.log |
| blink_multi-view_reasoning:sav | blink_multi-view_reasoning | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=16 multi_image_samples=8] log=sav_blink_multi-view_reasoning_qwen2_vl_subset.log |
| blink_object_localization:sav | blink_object_localization | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_blink_object_localization_qwen2_vl_subset.log |
| blink_relative_depth:sav | blink_relative_depth | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_blink_relative_depth_qwen2_vl_subset.log |
| blink_relative_reflectance:sav | blink_relative_reflectance | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_blink_relative_reflectance_qwen2_vl_subset.log |
| blink_semantic_correspondence:sav | blink_semantic_correspondence | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=16 multi_image_samples=8] log=sav_blink_semantic_correspondence_qwen2_vl_subset.log |
| blink_spatial_relation:sav | blink_spatial_relation | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=8 multi_image_samples=0] log=sav_blink_spatial_relation_qwen2_vl_subset.log |
| blink_visual_correspondence:sav | blink_visual_correspondence | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=16 multi_image_samples=8] log=sav_blink_visual_correspondence_qwen2_vl_subset.log |
| blink_visual_similarity:sav | blink_visual_similarity | raw | 4 | 4 | PASS | PASS | validate=[train=4 val=4 image_paths=24 multi_image_samples=8] log=sav_blink_visual_similarity_qwen2_vl_subset.log |

## Summary

- Selected experiments: `26`
- Failures: `0`
- Record: `swap/records/dataset_subset_smoke_20260324_020033.md`
