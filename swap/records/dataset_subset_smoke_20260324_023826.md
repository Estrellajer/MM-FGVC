# Author-Style Dataset Subset Smoke Test

- Timestamp (UTC): `20260324_023826`
- Run mode: `subset_smoke`
- Model: `qwen2_vl`
- Methods: `sav`
- Support rule: `ref-data` when available, otherwise first `20` valid examples per label
- Eval rule: official labeled eval split when available; same-source tasks exclude support rows
- Validate datasets: `1`
- Execute `main.py`: `1`
- Dataset filter: `^(vizwiz|naturalbench_vqa|blink_art_style)$`

## Output Layout

- Subsets: `swap/subsets/20260324_023826`
- Logs: `swap/logs/20260324_023826`
- Metrics: `swap/outputs/20260324_023826`

## Result Table

| Experiment | Dataset Name | Evaluator | Train | Val | Validate | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| naturalbench_vqa:sav | naturalbench_vqa | raw | 80 | 4 | PASS | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=4 labels=2 per_label=2 exclude=30] validate=[train=80 val=4 image_paths=84 multi_image_samples=0] log=sav_naturalbench_vqa_qwen2_vl_subset.log |
| vizwiz:sav | vizwiz | raw | 40 | 4 | PASS | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=14-26 path_fix=40 label_fix=6] val=[mode=per_label source=converted selected=4 labels=2 per_label=2] validate=[train=40 val=4 image_paths=44 multi_image_samples=0] log=sav_vizwiz_qwen2_vl_subset.log |
| blink_art_style:sav | blink_art_style | raw | 40 | 2 | PASS | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=distinct_labels source=converted selected=2 labels=2 per_label=1] validate=[train=40 val=2 image_paths=86 multi_image_samples=42] log=sav_blink_art_style_qwen2_vl_subset.log |

## Summary

- Selected experiments: `3`
- Failures: `0`
- Record: `swap/records/dataset_subset_smoke_20260324_023826.md`
