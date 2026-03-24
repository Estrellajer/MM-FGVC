# SAV Diagnostic Sweep

- Timestamp (UTC): `20260324_025703`
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

- Subsets: `swap/subsets/20260324_025703_sav_diag`
- Logs: `swap/logs/20260324_025703_sav_diag`
- Metrics: `swap/outputs/20260324_025703_sav_diag`

## Result Table

| Experiment | Dataset | Evaluator | Train | Val | Metric | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| naturalbench_ret | naturalbench_ret | naturalbench_group | 40 | 40 | g_acc=0.8000 raw_acc=0.9500 q_acc=0.9000 i_acc=0.9000 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=grouped source=author_ref selected=40 labels=2 per_label=20 restrict_labels=2 exclude=1] validate=[train=40 val=40 image_paths=80 multi_image_samples=0] log=sav_naturalbench_ret_qwen2_vl_diag.log |
| naturalbench_vqa | naturalbench_vqa | raw | 80 | 40 | accuracy=0.8000 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=40 labels=4 per_label=6-14 restrict_labels=4 exclude=80] validate=[train=80 val=40 image_paths=120 multi_image_samples=0] log=sav_naturalbench_vqa_qwen2_vl_diag.log |
| sugarcrepe | sugarcrepe | pair | 40 | 40 | pair_accuracy=0.5000 raw_accuracy=0.7500 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=grouped source=converted selected=40 labels=2 per_label=20 restrict_labels=2 exclude=40] validate=[train=40 val=40 image_paths=80 multi_image_samples=0] log=sav_sugarcrepe_qwen2_vl_diag.log |
| pets | pets | raw | 175 | 70 | accuracy=1.0000 | PASS | train=[mode=per_label source=author_ref selected=175 labels=35 per_label=5 path_fix=700 label_fix=340] val=[mode=per_label source=converted selected=70 labels=35 per_label=2 restrict_labels=35] validate=[train=175 val=70 image_paths=245 multi_image_samples=0] log=sav_pets_qwen2_vl_diag.log |
| eurosat | eurosat | raw | 100 | 50 | accuracy=0.6400 | PASS | train=[mode=per_label source=author_ref selected=100 labels=10 per_label=10 path_fix=200 label_fix=160] val=[mode=per_label source=converted selected=50 labels=10 per_label=5 restrict_labels=10] validate=[train=100 val=50 image_paths=150 multi_image_samples=0] log=sav_eurosat_qwen2_vl_diag.log |
| flowers | flowers | raw | 204 | 102 | accuracy=0.9706 | PASS | train=[mode=per_label source=converted selected=204 labels=102 per_label=2] val=[mode=per_label source=converted selected=102 labels=102 per_label=1 restrict_labels=102] validate=[train=204 val=102 image_paths=306 multi_image_samples=0] log=sav_flowers_qwen2_vl_diag.log |
| cub | cub | raw | 400 | 200 | accuracy=0.7100 | PASS | train=[mode=per_label source=converted selected=400 labels=200 per_label=2] val=[mode=per_label source=converted selected=200 labels=200 per_label=1 restrict_labels=200] validate=[train=400 val=200 image_paths=600 multi_image_samples=0] log=sav_cub_qwen2_vl_diag.log |
| tinyimage | tinyimage | raw | 400 | 200 | accuracy=0.1050 | PASS | train=[mode=per_label source=converted selected=400 labels=200 per_label=2] val=[mode=per_label source=converted selected=200 labels=200 per_label=1 restrict_labels=200] validate=[train=400 val=200 image_paths=600 multi_image_samples=0] log=sav_tinyimage_qwen2_vl_diag.log |
| vizwiz | vizwiz | raw | 40 | 40 | accuracy=0.5500 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=40 labels=2 per_label=20 restrict_labels=2] validate=[train=40 val=40 image_paths=80 multi_image_samples=0] log=sav_vizwiz_qwen2_vl_diag.log |
| vlguard | vlguard | raw | 40 | 40 | accuracy=0.9500 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=40 labels=2 per_label=20 restrict_labels=2] validate=[train=40 val=40 image_paths=80 multi_image_samples=0] log=sav_vlguard_qwen2_vl_diag.log |
| mhalubench_val_v01 | mhalubench | raw | 40 | 40 | accuracy=0.9250 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=40 labels=2 per_label=20 restrict_labels=2] validate=[train=40 val=40 image_paths=80 multi_image_samples=0] log=sav_mhalubench_val_v01_qwen2_vl_diag.log |
| mhalubench_val_v02 | mhalubench | raw | 40 | 40 | accuracy=0.9250 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=40 labels=2 per_label=20 restrict_labels=2] validate=[train=40 val=40 image_paths=80 multi_image_samples=0] log=sav_mhalubench_val_v02_qwen2_vl_diag.log |
| blink_art_style | blink_art_style | raw | 40 | 20 | accuracy=0.9000 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=per_label source=converted selected=20 labels=2 per_label=10 restrict_labels=2] validate=[train=40 val=20 image_paths=140 multi_image_samples=60] log=sav_blink_art_style_qwen2_vl_diag.log |
| blink_counting | blink_counting | raw | 80 | 20 | accuracy=0.6000 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=20 labels=4 per_label=5 restrict_labels=4 exclude=80] validate=[train=80 val=20 image_paths=100 multi_image_samples=0] log=sav_blink_counting_qwen2_vl_diag.log |
| blink_forensic_detection | blink_forensic_detection | raw | 80 | 20 | accuracy=0.4000 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=20 labels=4 per_label=5 restrict_labels=4 exclude=80] validate=[train=80 val=20 image_paths=400 multi_image_samples=100] log=sav_blink_forensic_detection_qwen2_vl_diag.log |
| blink_functional_correspondence | blink_functional_correspondence | raw | 80 | 16 | accuracy=0.1875 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=16 labels=4 per_label=1-5 restrict_labels=4 exclude=80] validate=[train=80 val=16 image_paths=192 multi_image_samples=96] log=sav_blink_functional_correspondence_qwen2_vl_diag.log |
| blink_iq_test | blink_iq_test | raw | 80 | 20 | accuracy=0.0000 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=20 labels=4 per_label=5 restrict_labels=4 exclude=80] validate=[train=80 val=20 image_paths=100 multi_image_samples=0] log=sav_blink_iq_test_qwen2_vl_diag.log |
| blink_jigsaw | blink_jigsaw | raw | 40 | 20 | accuracy=0.7000 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=20 labels=2 per_label=10 restrict_labels=2 exclude=40] validate=[train=40 val=20 image_paths=180 multi_image_samples=60] log=sav_blink_jigsaw_qwen2_vl_diag.log |
| blink_multi-view_reasoning | blink_multi-view_reasoning | raw | 40 | 20 | accuracy=0.4000 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=20 labels=2 per_label=10 restrict_labels=2 exclude=40] validate=[train=40 val=20 image_paths=120 multi_image_samples=60] log=sav_blink_multi-view_reasoning_qwen2_vl_diag.log |
| blink_object_localization | blink_object_localization | raw | 40 | 20 | accuracy=0.4500 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=20 labels=2 per_label=10 restrict_labels=2 exclude=40] validate=[train=40 val=20 image_paths=60 multi_image_samples=0] log=sav_blink_object_localization_qwen2_vl_diag.log |
| blink_relative_depth | blink_relative_depth | raw | 40 | 20 | accuracy=0.6500 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=20 labels=2 per_label=10 restrict_labels=2 exclude=40] validate=[train=40 val=20 image_paths=60 multi_image_samples=0] log=sav_blink_relative_depth_qwen2_vl_diag.log |
| blink_relative_reflectance | blink_relative_reflectance | raw | 60 | 15 | accuracy=0.4000 | PASS | train=[mode=per_label source=converted selected=60 labels=3 per_label=20] val=[mode=per_label source=converted selected=15 labels=3 per_label=5 restrict_labels=3 exclude=60] validate=[train=60 val=15 image_paths=75 multi_image_samples=0] log=sav_blink_relative_reflectance_qwen2_vl_diag.log |
| blink_semantic_correspondence | blink_semantic_correspondence | raw | 80 | 20 | accuracy=0.1000 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=20 labels=4 per_label=5 restrict_labels=4 exclude=80] validate=[train=80 val=20 image_paths=200 multi_image_samples=100] log=sav_blink_semantic_correspondence_qwen2_vl_diag.log |
| blink_spatial_relation | blink_spatial_relation | raw | 40 | 20 | accuracy=0.7000 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=20 labels=2 per_label=10 restrict_labels=2 exclude=40] validate=[train=40 val=20 image_paths=60 multi_image_samples=0] log=sav_blink_spatial_relation_qwen2_vl_diag.log |
| blink_visual_correspondence | blink_visual_correspondence | raw | 80 | 20 | accuracy=0.3500 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=20 labels=4 per_label=5 restrict_labels=4 exclude=80] validate=[train=80 val=20 image_paths=200 multi_image_samples=100] log=sav_blink_visual_correspondence_qwen2_vl_diag.log |
| blink_visual_similarity | blink_visual_similarity | raw | 40 | 20 | accuracy=0.6000 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=20 labels=2 per_label=10 restrict_labels=2 exclude=40] validate=[train=40 val=20 image_paths=180 multi_image_samples=60] log=sav_blink_visual_similarity_qwen2_vl_diag.log |

## Summary

- Completed runs: `26`
- Weak tasks (< 0.5): `9`
- Strong tasks (>= 0.8): `8`

### Lowest Scores

- `sav_blink_iq_test_qwen2_vl_diag` (blink_iq_test): `accuracy=0.0000`
- `sav_blink_semantic_correspondence_qwen2_vl_diag` (blink_semantic_correspondence): `accuracy=0.1000`
- `sav_tinyimage_qwen2_vl_diag` (tinyimage): `accuracy=0.1050`
- `sav_blink_functional_correspondence_qwen2_vl_diag` (blink_functional_correspondence): `accuracy=0.1875`
- `sav_blink_visual_correspondence_qwen2_vl_diag` (blink_visual_correspondence): `accuracy=0.3500`
- `sav_blink_forensic_detection_qwen2_vl_diag` (blink_forensic_detection): `accuracy=0.4000`
- `sav_blink_multi-view_reasoning_qwen2_vl_diag` (blink_multi-view_reasoning): `accuracy=0.4000`
- `sav_blink_relative_reflectance_qwen2_vl_diag` (blink_relative_reflectance): `accuracy=0.4000`

### Highest Scores

- `sav_pets_qwen2_vl_diag` (pets): `accuracy=1.0000`
- `sav_flowers_qwen2_vl_diag` (flowers): `accuracy=0.9706`
- `sav_vlguard_qwen2_vl_diag` (vlguard): `accuracy=0.9500`
- `sav_mhalubench_val_v02_qwen2_vl_diag` (mhalubench): `accuracy=0.9250`
- `sav_mhalubench_val_v01_qwen2_vl_diag` (mhalubench): `accuracy=0.9250`
- `sav_blink_art_style_qwen2_vl_diag` (blink_art_style): `accuracy=0.9000`
- `sav_naturalbench_vqa_qwen2_vl_diag` (naturalbench_vqa): `accuracy=0.8000`
- `sav_naturalbench_ret_qwen2_vl_diag` (naturalbench_ret): `g_acc=0.8000`
