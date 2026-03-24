# SAV Head Selection Ablation

- Timestamp (UTC): `20260324_035047`
- Model: `qwen2_vl`
- Method: `sav`
- Goal: test whether explicit head selection is necessary, and whether random or non-adaptive choices can match `topk`
- Fixed `num_heads`: `20`
- Strategies:
  - `topk`
  - `firstk`
  - `all`
  - `random` with seeds: `11 22 33`
- Principle:
  - reuse author-style subset construction
  - use larger-than-smoke, compute-capped validation subsets
  - compare strategies on the same train/eval subsets

## Output Layout

- Subsets: `swap/subsets/20260324_035047_sav_head_ablation`
- Logs: `swap/logs/20260324_035047_sav_head_ablation`
- Metrics: `swap/outputs/20260324_035047_sav_head_ablation`

## Result Table

| Experiment | Dataset | Strategy | Train | Val | Metric | Run | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| vlguard | vlguard | topk | 40 | 80 | accuracy=0.9750 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=topk seed=42 log=sav_vlguard_topk_qwen2_vl_ablation.log |
| vlguard | vlguard | firstk | 40 | 80 | accuracy=0.7375 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=firstk seed=42 log=sav_vlguard_firstk_qwen2_vl_ablation.log |
| vlguard | vlguard | all | 40 | 80 | accuracy=0.9000 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=all seed=42 log=sav_vlguard_all_qwen2_vl_ablation.log |
| vlguard | vlguard | random_s11 | 40 | 80 | accuracy=0.6375 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=random seed=11 log=sav_vlguard_random_s11_qwen2_vl_ablation.log |
| vlguard | vlguard | random_s22 | 40 | 80 | accuracy=0.8250 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=random seed=22 log=sav_vlguard_random_s22_qwen2_vl_ablation.log |
| vlguard | vlguard | random_s33 | 40 | 80 | accuracy=0.9250 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=40] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=random seed=33 log=sav_vlguard_random_s33_qwen2_vl_ablation.log |
| mhalubench_val_v01 | mhalubench | topk | 40 | 80 | accuracy=0.7750 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=topk seed=42 log=sav_mhalubench_val_v01_topk_qwen2_vl_ablation.log |
| mhalubench_val_v01 | mhalubench | firstk | 40 | 80 | accuracy=0.2125 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=firstk seed=42 log=sav_mhalubench_val_v01_firstk_qwen2_vl_ablation.log |
| mhalubench_val_v01 | mhalubench | all | 40 | 80 | accuracy=0.1750 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=all seed=42 log=sav_mhalubench_val_v01_all_qwen2_vl_ablation.log |
| mhalubench_val_v01 | mhalubench | random_s11 | 40 | 80 | accuracy=0.2250 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=random seed=11 log=sav_mhalubench_val_v01_random_s11_qwen2_vl_ablation.log |
| mhalubench_val_v01 | mhalubench | random_s22 | 40 | 80 | accuracy=0.1625 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=random seed=22 log=sav_mhalubench_val_v01_random_s22_qwen2_vl_ablation.log |
| mhalubench_val_v01 | mhalubench | random_s33 | 40 | 80 | accuracy=0.6250 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20] val=[mode=per_label source=converted selected=80 labels=2 per_label=40 restrict_labels=2] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=random seed=33 log=sav_mhalubench_val_v01_random_s33_qwen2_vl_ablation.log |
| naturalbench_vqa | naturalbench_vqa | topk | 80 | 80 | accuracy=0.8250 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=80 labels=4 per_label=8-32 restrict_labels=4 exclude=80] validate=[train=80 val=80 image_paths=160 multi_image_samples=0] strategy=topk seed=42 log=sav_naturalbench_vqa_topk_qwen2_vl_ablation.log |
| naturalbench_vqa | naturalbench_vqa | firstk | 80 | 80 | accuracy=0.5500 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=80 labels=4 per_label=8-32 restrict_labels=4 exclude=80] validate=[train=80 val=80 image_paths=160 multi_image_samples=0] strategy=firstk seed=42 log=sav_naturalbench_vqa_firstk_qwen2_vl_ablation.log |
| naturalbench_vqa | naturalbench_vqa | all | 80 | 80 | accuracy=0.8125 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=80 labels=4 per_label=8-32 restrict_labels=4 exclude=80] validate=[train=80 val=80 image_paths=160 multi_image_samples=0] strategy=all seed=42 log=sav_naturalbench_vqa_all_qwen2_vl_ablation.log |
| naturalbench_vqa | naturalbench_vqa | random_s11 | 80 | 80 | accuracy=0.7875 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=80 labels=4 per_label=8-32 restrict_labels=4 exclude=80] validate=[train=80 val=80 image_paths=160 multi_image_samples=0] strategy=random seed=11 log=sav_naturalbench_vqa_random_s11_qwen2_vl_ablation.log |
| naturalbench_vqa | naturalbench_vqa | random_s22 | 80 | 80 | accuracy=0.6750 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=80 labels=4 per_label=8-32 restrict_labels=4 exclude=80] validate=[train=80 val=80 image_paths=160 multi_image_samples=0] strategy=random seed=22 log=sav_naturalbench_vqa_random_s22_qwen2_vl_ablation.log |
| naturalbench_vqa | naturalbench_vqa | random_s33 | 80 | 80 | accuracy=0.8250 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=grouped source=converted selected=80 labels=4 per_label=8-32 restrict_labels=4 exclude=80] validate=[train=80 val=80 image_paths=160 multi_image_samples=0] strategy=random seed=33 log=sav_naturalbench_vqa_random_s33_qwen2_vl_ablation.log |
| sugarcrepe | sugarcrepe | topk | 40 | 80 | pair_accuracy=0.6750 raw_accuracy=0.8375 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=grouped source=converted selected=80 labels=2 per_label=40 restrict_labels=2 exclude=40] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=topk seed=42 log=sav_sugarcrepe_topk_qwen2_vl_ablation.log |
| sugarcrepe | sugarcrepe | firstk | 40 | 80 | pair_accuracy=0.2750 raw_accuracy=0.6250 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=grouped source=converted selected=80 labels=2 per_label=40 restrict_labels=2 exclude=40] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=firstk seed=42 log=sav_sugarcrepe_firstk_qwen2_vl_ablation.log |
| sugarcrepe | sugarcrepe | all | 40 | 80 | pair_accuracy=0.6000 raw_accuracy=0.8000 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=grouped source=converted selected=80 labels=2 per_label=40 restrict_labels=2 exclude=40] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=all seed=42 log=sav_sugarcrepe_all_qwen2_vl_ablation.log |
| sugarcrepe | sugarcrepe | random_s11 | 40 | 80 | pair_accuracy=0.5500 raw_accuracy=0.7750 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=grouped source=converted selected=80 labels=2 per_label=40 restrict_labels=2 exclude=40] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=random seed=11 log=sav_sugarcrepe_random_s11_qwen2_vl_ablation.log |
| sugarcrepe | sugarcrepe | random_s22 | 40 | 80 | pair_accuracy=0.5000 raw_accuracy=0.7500 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=grouped source=converted selected=80 labels=2 per_label=40 restrict_labels=2 exclude=40] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=random seed=22 log=sav_sugarcrepe_random_s22_qwen2_vl_ablation.log |
| sugarcrepe | sugarcrepe | random_s33 | 40 | 80 | pair_accuracy=0.5500 raw_accuracy=0.7750 | PASS | train=[mode=per_label source=converted selected=40 labels=2 per_label=20] val=[mode=grouped source=converted selected=80 labels=2 per_label=40 restrict_labels=2 exclude=40] validate=[train=40 val=80 image_paths=120 multi_image_samples=0] strategy=random seed=33 log=sav_sugarcrepe_random_s33_qwen2_vl_ablation.log |
| eurosat | eurosat | topk | 100 | 100 | accuracy=0.6800 | PASS | train=[mode=per_label source=author_ref selected=100 labels=10 per_label=10 path_fix=200 label_fix=160] val=[mode=per_label source=converted selected=100 labels=10 per_label=10 restrict_labels=10] validate=[train=100 val=100 image_paths=200 multi_image_samples=0] strategy=topk seed=42 log=sav_eurosat_topk_qwen2_vl_ablation.log |
| eurosat | eurosat | firstk | 100 | 100 | accuracy=0.2600 | PASS | train=[mode=per_label source=author_ref selected=100 labels=10 per_label=10 path_fix=200 label_fix=160] val=[mode=per_label source=converted selected=100 labels=10 per_label=10 restrict_labels=10] validate=[train=100 val=100 image_paths=200 multi_image_samples=0] strategy=firstk seed=42 log=sav_eurosat_firstk_qwen2_vl_ablation.log |
| eurosat | eurosat | all | 100 | 100 | accuracy=0.6000 | PASS | train=[mode=per_label source=author_ref selected=100 labels=10 per_label=10 path_fix=200 label_fix=160] val=[mode=per_label source=converted selected=100 labels=10 per_label=10 restrict_labels=10] validate=[train=100 val=100 image_paths=200 multi_image_samples=0] strategy=all seed=42 log=sav_eurosat_all_qwen2_vl_ablation.log |
| eurosat | eurosat | random_s11 | 100 | 100 | accuracy=0.5500 | PASS | train=[mode=per_label source=author_ref selected=100 labels=10 per_label=10 path_fix=200 label_fix=160] val=[mode=per_label source=converted selected=100 labels=10 per_label=10 restrict_labels=10] validate=[train=100 val=100 image_paths=200 multi_image_samples=0] strategy=random seed=11 log=sav_eurosat_random_s11_qwen2_vl_ablation.log |
| eurosat | eurosat | random_s22 | 100 | 100 | accuracy=0.4900 | PASS | train=[mode=per_label source=author_ref selected=100 labels=10 per_label=10 path_fix=200 label_fix=160] val=[mode=per_label source=converted selected=100 labels=10 per_label=10 restrict_labels=10] validate=[train=100 val=100 image_paths=200 multi_image_samples=0] strategy=random seed=22 log=sav_eurosat_random_s22_qwen2_vl_ablation.log |
| eurosat | eurosat | random_s33 | 100 | 100 | accuracy=0.6500 | PASS | train=[mode=per_label source=author_ref selected=100 labels=10 per_label=10 path_fix=200 label_fix=160] val=[mode=per_label source=converted selected=100 labels=10 per_label=10 restrict_labels=10] validate=[train=100 val=100 image_paths=200 multi_image_samples=0] strategy=random seed=33 log=sav_eurosat_random_s33_qwen2_vl_ablation.log |
| tinyimage | tinyimage | topk | 400 | 200 | accuracy=0.1050 | PASS | train=[mode=per_label source=converted selected=400 labels=200 per_label=2] val=[mode=per_label source=converted selected=200 labels=200 per_label=1 restrict_labels=200] validate=[train=400 val=200 image_paths=600 multi_image_samples=0] strategy=topk seed=42 log=sav_tinyimage_topk_qwen2_vl_ablation.log |
| tinyimage | tinyimage | firstk | 400 | 200 | accuracy=0.0200 | PASS | train=[mode=per_label source=converted selected=400 labels=200 per_label=2] val=[mode=per_label source=converted selected=200 labels=200 per_label=1 restrict_labels=200] validate=[train=400 val=200 image_paths=600 multi_image_samples=0] strategy=firstk seed=42 log=sav_tinyimage_firstk_qwen2_vl_ablation.log |
| tinyimage | tinyimage | all | 400 | 200 | accuracy=0.3350 | PASS | train=[mode=per_label source=converted selected=400 labels=200 per_label=2] val=[mode=per_label source=converted selected=200 labels=200 per_label=1 restrict_labels=200] validate=[train=400 val=200 image_paths=600 multi_image_samples=0] strategy=all seed=42 log=sav_tinyimage_all_qwen2_vl_ablation.log |
| tinyimage | tinyimage | random_s11 | 400 | 200 | accuracy=0.2500 | PASS | train=[mode=per_label source=converted selected=400 labels=200 per_label=2] val=[mode=per_label source=converted selected=200 labels=200 per_label=1 restrict_labels=200] validate=[train=400 val=200 image_paths=600 multi_image_samples=0] strategy=random seed=11 log=sav_tinyimage_random_s11_qwen2_vl_ablation.log |
| tinyimage | tinyimage | random_s22 | 400 | 200 | accuracy=0.1400 | PASS | train=[mode=per_label source=converted selected=400 labels=200 per_label=2] val=[mode=per_label source=converted selected=200 labels=200 per_label=1 restrict_labels=200] validate=[train=400 val=200 image_paths=600 multi_image_samples=0] strategy=random seed=22 log=sav_tinyimage_random_s22_qwen2_vl_ablation.log |
| tinyimage | tinyimage | random_s33 | 400 | 200 | accuracy=0.3300 | PASS | train=[mode=per_label source=converted selected=400 labels=200 per_label=2] val=[mode=per_label source=converted selected=200 labels=200 per_label=1 restrict_labels=200] validate=[train=400 val=200 image_paths=600 multi_image_samples=0] strategy=random seed=33 log=sav_tinyimage_random_s33_qwen2_vl_ablation.log |
| blink_art_style | blink_art_style | topk | 40 | 40 | accuracy=0.6750 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=per_label source=converted selected=40 labels=2 per_label=20 restrict_labels=2] validate=[train=40 val=40 image_paths=200 multi_image_samples=80] strategy=topk seed=42 log=sav_blink_art_style_topk_qwen2_vl_ablation.log |
| blink_art_style | blink_art_style | firstk | 40 | 40 | accuracy=0.4750 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=per_label source=converted selected=40 labels=2 per_label=20 restrict_labels=2] validate=[train=40 val=40 image_paths=200 multi_image_samples=80] strategy=firstk seed=42 log=sav_blink_art_style_firstk_qwen2_vl_ablation.log |
| blink_art_style | blink_art_style | all | 40 | 40 | accuracy=0.6500 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=per_label source=converted selected=40 labels=2 per_label=20 restrict_labels=2] validate=[train=40 val=40 image_paths=200 multi_image_samples=80] strategy=all seed=42 log=sav_blink_art_style_all_qwen2_vl_ablation.log |
| blink_art_style | blink_art_style | random_s11 | 40 | 40 | accuracy=0.6000 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=per_label source=converted selected=40 labels=2 per_label=20 restrict_labels=2] validate=[train=40 val=40 image_paths=200 multi_image_samples=80] strategy=random seed=11 log=sav_blink_art_style_random_s11_qwen2_vl_ablation.log |
| blink_art_style | blink_art_style | random_s22 | 40 | 40 | accuracy=0.7000 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=per_label source=converted selected=40 labels=2 per_label=20 restrict_labels=2] validate=[train=40 val=40 image_paths=200 multi_image_samples=80] strategy=random seed=22 log=sav_blink_art_style_random_s22_qwen2_vl_ablation.log |
| blink_art_style | blink_art_style | random_s33 | 40 | 40 | accuracy=0.6500 | PASS | train=[mode=author_ref source=author_ref selected=40 labels=2 per_label=20 path_fix=120] val=[mode=per_label source=converted selected=40 labels=2 per_label=20 restrict_labels=2] validate=[train=40 val=40 image_paths=200 multi_image_samples=80] strategy=random seed=33 log=sav_blink_art_style_random_s33_qwen2_vl_ablation.log |
| blink_semantic_correspondence | blink_semantic_correspondence | topk | 80 | 35 | accuracy=0.1143 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=35 labels=4 per_label=5-10 restrict_labels=4 exclude=80] validate=[train=80 val=35 image_paths=230 multi_image_samples=115] strategy=topk seed=42 log=sav_blink_semantic_correspondence_topk_qwen2_vl_ablation.log |
| blink_semantic_correspondence | blink_semantic_correspondence | firstk | 80 | 35 | accuracy=0.2286 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=35 labels=4 per_label=5-10 restrict_labels=4 exclude=80] validate=[train=80 val=35 image_paths=230 multi_image_samples=115] strategy=firstk seed=42 log=sav_blink_semantic_correspondence_firstk_qwen2_vl_ablation.log |
| blink_semantic_correspondence | blink_semantic_correspondence | all | 80 | 35 | accuracy=0.2571 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=35 labels=4 per_label=5-10 restrict_labels=4 exclude=80] validate=[train=80 val=35 image_paths=230 multi_image_samples=115] strategy=all seed=42 log=sav_blink_semantic_correspondence_all_qwen2_vl_ablation.log |
| blink_semantic_correspondence | blink_semantic_correspondence | random_s11 | 80 | 35 | accuracy=0.1714 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=35 labels=4 per_label=5-10 restrict_labels=4 exclude=80] validate=[train=80 val=35 image_paths=230 multi_image_samples=115] strategy=random seed=11 log=sav_blink_semantic_correspondence_random_s11_qwen2_vl_ablation.log |
| blink_semantic_correspondence | blink_semantic_correspondence | random_s22 | 80 | 35 | accuracy=0.2000 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=35 labels=4 per_label=5-10 restrict_labels=4 exclude=80] validate=[train=80 val=35 image_paths=230 multi_image_samples=115] strategy=random seed=22 log=sav_blink_semantic_correspondence_random_s22_qwen2_vl_ablation.log |
| blink_semantic_correspondence | blink_semantic_correspondence | random_s33 | 80 | 35 | accuracy=0.2286 | PASS | train=[mode=per_label source=converted selected=80 labels=4 per_label=20] val=[mode=per_label source=converted selected=35 labels=4 per_label=5-10 restrict_labels=4 exclude=80] validate=[train=80 val=35 image_paths=230 multi_image_samples=115] strategy=random seed=33 log=sav_blink_semantic_correspondence_random_s33_qwen2_vl_ablation.log |

## Aggregate Summary

- Completed runs: `48`
- Mean primary score `topk`: `0.6030` (n=8)
- Mean primary score `firstk`: `0.3448` (n=8)
- Mean primary score `all`: `0.5412` (n=8)
- Mean primary score `random_mean`: `0.5103` (n=8)

## Per-Task Comparison

| Experiment | Dataset | Metric | topk | firstk | all | random mean | random std | best random | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mhalubench_val_v01 | mhalubench | accuracy | 0.7750 | 0.2125 | 0.1750 | 0.3375 | 0.2049 | 0.6250 | topk clearly beats random; selection helps over all-head voting; adaptive selection beats fixed first-k |
| vlguard | vlguard | accuracy | 0.9750 | 0.7375 | 0.9000 | 0.7958 | 0.1192 | 0.9250 | topk clearly beats random; selection helps over all-head voting; adaptive selection beats fixed first-k |
| sugarcrepe | sugarcrepe | pair_accuracy | 0.6750 | 0.2750 | 0.6000 | 0.5333 | 0.0236 | 0.5500 | topk clearly beats random; selection helps over all-head voting; adaptive selection beats fixed first-k |
| eurosat | eurosat | accuracy | 0.6800 | 0.2600 | 0.6000 | 0.5633 | 0.0660 | 0.6500 | topk clearly beats random; selection helps over all-head voting; adaptive selection beats fixed first-k |
| naturalbench_vqa | naturalbench_vqa | accuracy | 0.8250 | 0.5500 | 0.8125 | 0.7625 | 0.0637 | 0.8250 | topk clearly beats random; adaptive selection beats fixed first-k |
| blink_art_style | blink_art_style | accuracy | 0.6750 | 0.4750 | 0.6500 | 0.6500 | 0.0408 | 0.7000 | adaptive selection beats fixed first-k |
| blink_semantic_correspondence | blink_semantic_correspondence | accuracy | 0.1143 | 0.2286 | 0.2571 | 0.2000 | 0.0233 | 0.2286 | all heads beats topk |
| tinyimage | tinyimage | accuracy | 0.1050 | 0.0200 | 0.3350 | 0.2400 | 0.0779 | 0.3300 | all heads beats topk; adaptive selection beats fixed first-k |

## Automated Takeaways

- Tasks where adaptive `topk` head selection clearly helps over random mean:
  - `mhalubench_val_v01`: `topk=0.7750` vs `random_mean=0.3375` (gap `0.4375`)
  - `vlguard`: `topk=0.9750` vs `random_mean=0.7958` (gap `0.1792`)
  - `sugarcrepe`: `topk=0.6750` vs `random_mean=0.5333` (gap `0.1417`)
  - `eurosat`: `topk=0.6800` vs `random_mean=0.5633` (gap `0.1167`)
  - `naturalbench_vqa`: `topk=0.8250` vs `random_mean=0.7625` (gap `0.0625`)
- Tasks where using all heads beats `topk`:
  - `tinyimage`: `all=0.3350` vs `topk=0.1050` (gap `0.2300`)
  - `blink_semantic_correspondence`: `all=0.2571` vs `topk=0.1143` (gap `0.1429`)

## Curated Interpretation

- There is no single answer to whether head selection is necessary. The effect is strongly task-dependent.
- On `mhalubench`, `vlguard`, `sugarcrepe`, and `eurosat`, adaptive `topk` selection is a real driver of performance. Replacing it with `firstk`, `all`, or average random selection causes clear drops.
- On `naturalbench_vqa` and `blink_art_style`, `topk` is still competitive, but it is not uniquely necessary. `all` and the best random seed can get very close, and one random seed even slightly beats `topk` on `blink_art_style`.
- On `tinyimage` and `blink_semantic_correspondence`, the current `topk` rule appears misaligned with the task. Using all heads is much better than selecting only the heads that look best on the support set.
- The current SAV implementation should therefore be treated as two separable design choices:
  - head aggregation itself can work
  - the current train-hit-rate `topk` selector is sometimes helpful, sometimes neutral, and sometimes actively harmful
- The most promising next ablations are:
  - different head scoring rules, especially class-margin or leave-one-out scoring instead of raw train hit count
  - hybrid selectors that keep more heads on high-class tasks
  - task-aware selection for multi-image correspondence tasks instead of reusing the same selector everywhere
