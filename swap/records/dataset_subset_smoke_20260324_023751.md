# Author-Style Dataset Subset Smoke Test

- Timestamp (UTC): `20260324_023751`
- Run mode: `subset_smoke`
- Model: `qwen2_vl`
- Methods: `zero_shot`
- Support rule: `ref-data` when available, otherwise first `20` valid examples per label
- Eval rule: official labeled eval split when available; same-source tasks exclude support rows
- Validate datasets: `1`
- Execute `main.py`: `0`
- Dataset filter: `all`

## Output Layout

- Subsets: `swap/subsets/20260324_023751`
- Logs: `swap/logs/20260324_023751`
- Metrics: `swap/outputs/20260324_023751`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| naturalbench_ret:zero_shot | naturalbench_ret | naturalbench_group | 40 | 4 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=40 labels=4 per_label=8-12 label_fix=32] val=[mode=grouped source=author_ref selected=4 labels=2 per_label=2 label_fix=2598 exclude=1] validate=[train=40 val=4 image_paths=44 multi_image_samples=0] |
| naturalbench_vqa:zero_shot | naturalbench_vqa | raw | 80 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=4 labels=2 per_label=2 exclude=30] validate=[train=80 val=4 image_paths=84 multi_image_samples=0] |
| sugarcrepe:zero_shot | sugarcrepe | pair | 40 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=grouped source=converted selected=4 labels=2 per_label=2 exclude=20] validate=[train=40 val=4 image_paths=44 multi_image_samples=0] |
| pets:zero_shot | pets | raw | 700 | 4 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=700 labels=35 per_label=20 path_fix=700 label_fix=340] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1] validate=[train=700 val=4 image_paths=704 multi_image_samples=0] |
| eurosat:zero_shot | eurosat | raw | 200 | 4 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=200 labels=10 per_label=20 path_fix=200 label_fix=160] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1] validate=[train=200 val=4 image_paths=204 multi_image_samples=0] |
| flowers:zero_shot | flowers | raw | 2040 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=2040 labels=102 per_label=20] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1] validate=[train=2040 val=4 image_paths=2044 multi_image_samples=0] |
| cub:zero_shot | cub | raw | 4000 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=4000 labels=200 per_label=20] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1] validate=[train=4000 val=4 image_paths=4004 multi_image_samples=0] |
| tinyimage:zero_shot | tinyimage | raw | 4000 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=4000 labels=200 per_label=20] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1] validate=[train=4000 val=4 image_paths=4004 multi_image_samples=0] |
| vizwiz:zero_shot | vizwiz | raw | 40 | 4 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=14-26 path_fix=40 label_fix=6] val=[mode=per_label source=converted selected=4 labels=2 per_label=2] validate=[train=40 val=4 image_paths=44 multi_image_samples=0] |
| vlguard:zero_shot | vlguard | raw | 40 | 4 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=9-31 path_fix=40 label_fix=11] val=[mode=per_label source=converted selected=4 labels=2 per_label=2] validate=[train=40 val=4 image_paths=44 multi_image_samples=0] |
| mhalubench_val_v01:zero_shot | mhalubench | raw | 40 | 4 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=4 labels=2 per_label=2] validate=[train=40 val=4 image_paths=44 multi_image_samples=0] |
| mhalubench_val_v02:zero_shot | mhalubench | raw | 40 | 4 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=4 labels=2 per_label=2] validate=[train=40 val=4 image_paths=44 multi_image_samples=0] |
| blink_art_style:zero_shot | blink_art_style | raw | 40 | 2 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=distinct_labels source=converted selected=2 labels=2 per_label=1] validate=[train=40 val=2 image_paths=86 multi_image_samples=42] |
| blink_counting:zero_shot | blink_counting | raw | 80 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1 exclude=80] validate=[train=80 val=4 image_paths=84 multi_image_samples=0] |
| blink_forensic_detection:zero_shot | blink_forensic_detection | raw | 80 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1 exclude=80] validate=[train=80 val=4 image_paths=336 multi_image_samples=84] |
| blink_functional_correspondence:zero_shot | blink_functional_correspondence | raw | 80 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1 exclude=80] validate=[train=80 val=4 image_paths=168 multi_image_samples=84] |
| blink_iq_test:zero_shot | blink_iq_test | raw | 80 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1 exclude=80] validate=[train=80 val=4 image_paths=84 multi_image_samples=0] |
| blink_jigsaw:zero_shot | blink_jigsaw | raw | 40 | 2 | PASS | SKIP | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=distinct_labels source=converted selected=2 labels=2 per_label=1 exclude=40] validate=[train=40 val=2 image_paths=126 multi_image_samples=42] |
| blink_multi-view_reasoning:zero_shot | blink_multi-view_reasoning | raw | 40 | 2 | PASS | SKIP | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=distinct_labels source=converted selected=2 labels=2 per_label=1 exclude=40] validate=[train=40 val=2 image_paths=84 multi_image_samples=42] |
| blink_object_localization:zero_shot | blink_object_localization | raw | 40 | 2 | PASS | SKIP | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=distinct_labels source=converted selected=2 labels=2 per_label=1 exclude=40] validate=[train=40 val=2 image_paths=42 multi_image_samples=0] |
| blink_relative_depth:zero_shot | blink_relative_depth | raw | 40 | 2 | PASS | SKIP | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=distinct_labels source=converted selected=2 labels=2 per_label=1 exclude=40] validate=[train=40 val=2 image_paths=42 multi_image_samples=0] |
| blink_relative_reflectance:zero_shot | blink_relative_reflectance | raw | 60 | 3 | PASS | SKIP | train=[mode=per_label source=converted selected=60 labels=3 per_label=20] val=[mode=distinct_labels source=converted selected=3 labels=3 per_label=1 exclude=60] validate=[train=60 val=3 image_paths=63 multi_image_samples=0] |
| blink_semantic_correspondence:zero_shot | blink_semantic_correspondence | raw | 80 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1 exclude=80] validate=[train=80 val=4 image_paths=168 multi_image_samples=84] |
| blink_spatial_relation:zero_shot | blink_spatial_relation | raw | 40 | 2 | PASS | SKIP | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=distinct_labels source=converted selected=2 labels=2 per_label=1 exclude=40] validate=[train=40 val=2 image_paths=42 multi_image_samples=0] |
| blink_visual_correspondence:zero_shot | blink_visual_correspondence | raw | 80 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1 exclude=80] validate=[train=80 val=4 image_paths=168 multi_image_samples=84] |
| blink_visual_similarity:zero_shot | blink_visual_similarity | raw | 40 | 2 | PASS | SKIP | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=distinct_labels source=converted selected=2 labels=2 per_label=1 exclude=40] validate=[train=40 val=2 image_paths=126 multi_image_samples=42] |

## Summary

- Selected experiments: `26`
- Failures: `0`
- Record: `swap/records/dataset_subset_smoke_20260324_023751.md`
