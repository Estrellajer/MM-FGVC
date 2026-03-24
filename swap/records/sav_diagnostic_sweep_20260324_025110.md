# SAV Diagnostic Sweep

- Timestamp (UTC): `20260324_025110`
- Model: `qwen2_vl`
- Method: `sav`
- Goal: larger-scale, author-style subset diagnostics across all supported tasks
- Principle:
  - use author `ref-data` support when available
  - otherwise use balanced per-label support from converted official data
  - use labeled eval splits
  - keep train/eval disjoint for same-source tasks
  - cap very high-class datasets to tractable balanced subsets for diagnostics

## Output Layout

- Subsets: `swap/subsets/20260324_025110_sav_diag`
- Logs: `swap/logs/20260324_025110_sav_diag`
- Metrics: `swap/outputs/20260324_025110_sav_diag`

## Result Table

| Experiment | Dataset | Evaluator | Train | Val | Metric | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| naturalbench_vqa | naturalbench_vqa | raw | 80 | 40 | accuracy=0.8000 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=40 labels=4 per_label=6-14 restrict_labels=4 exclude=80] validate=[train=80 val=40 image_paths=120 multi_image_samples=0] log=sav_naturalbench_vqa_qwen2_vl_diag.log |
| pets | pets | raw | 175 | 70 | accuracy=1.0000 | PASS | train=[mode=per_label source=author_ref selected=175 labels=35 per_label=5 path_fix=700 label_fix=340] val=[mode=per_label source=converted selected=70 labels=35 per_label=2 restrict_labels=35] validate=[train=175 val=70 image_paths=245 multi_image_samples=0] log=sav_pets_qwen2_vl_diag.log |
| blink_art_style | blink_art_style | raw | 40 | 20 | accuracy=0.9000 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=per_label source=converted selected=20 labels=2 per_label=10 restrict_labels=2] validate=[train=40 val=20 image_paths=140 multi_image_samples=60] log=sav_blink_art_style_qwen2_vl_diag.log |

## Summary

- Completed runs: `3`
- Weak tasks (< 0.5): `0`
- Strong tasks (>= 0.8): `3`

### Lowest Scores

- `sav_naturalbench_vqa_qwen2_vl_diag` (naturalbench_vqa): `accuracy=0.8000`
- `sav_blink_art_style_qwen2_vl_diag` (blink_art_style): `accuracy=0.9000`
- `sav_pets_qwen2_vl_diag` (pets): `accuracy=1.0000`

### Highest Scores

- `sav_pets_qwen2_vl_diag` (pets): `accuracy=1.0000`
- `sav_blink_art_style_qwen2_vl_diag` (blink_art_style): `accuracy=0.9000`
- `sav_naturalbench_vqa_qwen2_vl_diag` (naturalbench_vqa): `accuracy=0.8000`
