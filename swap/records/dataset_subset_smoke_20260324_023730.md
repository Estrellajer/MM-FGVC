# Author-Style Dataset Subset Smoke Test

- Timestamp (UTC): `20260324_023730`
- Run mode: `subset_smoke`
- Model: `qwen2_vl`
- Methods: `zero_shot`
- Support rule: `ref-data` when available, otherwise first `20` valid examples per label
- Eval rule: official labeled eval split when available; same-source tasks exclude support rows
- Validate datasets: `1`
- Execute `main.py`: `0`
- Dataset filter: `^(pets|vizwiz|naturalbench_vqa|sugarcrepe|blink_art_style|tinyimage)$`

## Output Layout

- Subsets: `swap/subsets/20260324_023730`
- Logs: `swap/logs/20260324_023730`
- Metrics: `swap/outputs/20260324_023730`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| naturalbench_vqa:zero_shot | naturalbench_vqa | raw | 80 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=4 labels=2 per_label=2 exclude=30] validate=[train=80 val=4 image_paths=84 multi_image_samples=0] |
| sugarcrepe:zero_shot | sugarcrepe | pair | 40 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=grouped source=converted selected=4 labels=2 per_label=2 exclude=20] validate=[train=40 val=4 image_paths=44 multi_image_samples=0] |
| pets:zero_shot | pets | raw | 700 | 4 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=700 labels=35 per_label=20 path_fix=700 label_fix=340] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1] validate=[train=700 val=4 image_paths=704 multi_image_samples=0] |
| tinyimage:zero_shot | tinyimage | raw | 4000 | 4 | PASS | SKIP | train=[mode=per_label source=converted selected=4000 labels=200 per_label=20] val=[mode=distinct_labels source=converted selected=4 labels=4 per_label=1] validate=[train=4000 val=4 image_paths=4004 multi_image_samples=0] |
| vizwiz:zero_shot | vizwiz | raw | 40 | 4 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=14-26 path_fix=40 label_fix=6] val=[mode=per_label source=converted selected=4 labels=2 per_label=2] validate=[train=40 val=4 image_paths=44 multi_image_samples=0] |
| blink_art_style:zero_shot | blink_art_style | raw | 40 | 2 | PASS | SKIP | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=distinct_labels source=converted selected=2 labels=2 per_label=1] validate=[train=40 val=2 image_paths=86 multi_image_samples=42] |

## Summary

- Selected experiments: `6`
- Failures: `0`
- Record: `swap/records/dataset_subset_smoke_20260324_023730.md`
