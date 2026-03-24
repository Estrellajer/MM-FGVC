# SAV Weighted Voting Comparison

- Baseline SAV sweep: `/root/autodl-tmp/FGVC/swap/outputs/20260324_025703_sav_diag`
- SAV+WVote manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/manifest.tsv`
- Overlapping experiments: `26`
- Wins / ties / losses: `0 / 26 / 0`

## Aggregate

| Bucket | Count | SAV mean | SAV+WVote mean | Delta mean |
| --- | --- | --- | --- | --- |
| all | 26 | 0.5890 | 0.5890 | +0.0000 |
| fgvc | 5 | 0.6851 | 0.6851 | +0.0000 |
| non_fgvc | 21 | 0.5661 | 0.5661 | +0.0000 |

## Per Task

| Experiment | Dataset | Metric | SAV | SAV+WVote | Delta |
| --- | --- | --- | --- | --- | --- |
| blink_art_style | blink_art_style | accuracy | 0.9000 | 0.9000 | +0.0000 |
| blink_counting | blink_counting | accuracy | 0.6000 | 0.6000 | +0.0000 |
| blink_forensic_detection | blink_forensic_detection | accuracy | 0.4000 | 0.4000 | +0.0000 |
| blink_functional_correspondence | blink_functional_correspondence | accuracy | 0.1875 | 0.1875 | +0.0000 |
| blink_iq_test | blink_iq_test | accuracy | 0.0000 | 0.0000 | +0.0000 |
| blink_jigsaw | blink_jigsaw | accuracy | 0.7000 | 0.7000 | +0.0000 |
| blink_multi-view_reasoning | blink_multi-view_reasoning | accuracy | 0.4000 | 0.4000 | +0.0000 |
| blink_object_localization | blink_object_localization | accuracy | 0.4500 | 0.4500 | +0.0000 |
| blink_relative_depth | blink_relative_depth | accuracy | 0.6500 | 0.6500 | +0.0000 |
| blink_relative_reflectance | blink_relative_reflectance | accuracy | 0.4000 | 0.4000 | +0.0000 |
| blink_semantic_correspondence | blink_semantic_correspondence | accuracy | 0.1000 | 0.1000 | +0.0000 |
| blink_spatial_relation | blink_spatial_relation | accuracy | 0.7000 | 0.7000 | +0.0000 |
| blink_visual_correspondence | blink_visual_correspondence | accuracy | 0.3500 | 0.3500 | +0.0000 |
| blink_visual_similarity | blink_visual_similarity | accuracy | 0.6000 | 0.6000 | +0.0000 |
| cub | cub | accuracy | 0.7100 | 0.7100 | +0.0000 |
| eurosat | eurosat | accuracy | 0.6400 | 0.6400 | +0.0000 |
| flowers | flowers | accuracy | 0.9706 | 0.9706 | +0.0000 |
| mhalubench_val_v01 | mhalubench | accuracy | 0.9250 | 0.9250 | +0.0000 |
| mhalubench_val_v02 | mhalubench | accuracy | 0.9250 | 0.9250 | +0.0000 |
| naturalbench_ret | naturalbench_ret | g_acc | 0.8000 | 0.8000 | +0.0000 |
| naturalbench_vqa | naturalbench_vqa | accuracy | 0.8000 | 0.8000 | +0.0000 |
| pets | pets | accuracy | 1.0000 | 1.0000 | +0.0000 |
| sugarcrepe | sugarcrepe | pair_accuracy | 0.5000 | 0.5000 | +0.0000 |
| tinyimage | tinyimage | accuracy | 0.1050 | 0.1050 | +0.0000 |
| vizwiz | vizwiz | accuracy | 0.5500 | 0.5500 | +0.0000 |
| vlguard | vlguard | accuracy | 0.9500 | 0.9500 | +0.0000 |

## Dataset Groups

- `blink_art_style`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_counting`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_forensic_detection`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_functional_correspondence`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_iq_test`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_jigsaw`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_multi-view_reasoning`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_object_localization`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_relative_depth`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_relative_reflectance`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_semantic_correspondence`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_spatial_relation`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_visual_correspondence`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `blink_visual_similarity`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `cub`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `eurosat`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `flowers`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `mhalubench`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `2`
- `naturalbench_ret`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `naturalbench_vqa`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `pets`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `sugarcrepe`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `tinyimage`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `vizwiz`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
- `vlguard`: mean delta `+0.0000`, wins `0`, losses `0`, tasks `1`
