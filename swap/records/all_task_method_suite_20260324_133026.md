# Method Suite Results

- Model: `qwen2_vl`
- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/manifest.tsv`
- Datasets: `blink_art_style, blink_counting, blink_forensic_detection, blink_functional_correspondence, blink_iq_test, blink_jigsaw, blink_multi-view_reasoning, blink_object_localization, blink_relative_depth, blink_relative_reflectance, blink_semantic_correspondence, blink_spatial_relation, blink_visual_correspondence, blink_visual_similarity, cub, eurosat, flowers, mhalubench, naturalbench_ret, naturalbench_vqa, pets, sugarcrepe, tinyimage, vizwiz, vlguard`
- Methods: `SAV+WVote`

## Overview

| Dataset | SAV+WVote |
| --- | --- |
| blink_art_style | 0.9000 |
| blink_counting | 0.6000 |
| blink_forensic_detection | 0.4000 |
| blink_functional_correspondence | 0.1875 |
| blink_iq_test | 0.0000 |
| blink_jigsaw | 0.7000 |
| blink_multi-view_reasoning | 0.4000 |
| blink_object_localization | 0.4500 |
| blink_relative_depth | 0.6500 |
| blink_relative_reflectance | 0.4000 |
| blink_semantic_correspondence | 0.1000 |
| blink_spatial_relation | 0.7000 |
| blink_visual_correspondence | 0.3500 |
| blink_visual_similarity | 0.6000 |
| cub | 0.7100 |
| eurosat | 0.6400 |
| flowers | 0.9706 |
| mhalubench | 0.9250 |
| naturalbench_ret | 0.8000 |
| naturalbench_vqa | 0.8000 |
| pets | 1.0000 |
| sugarcrepe | 0.5000 |
| tinyimage | 0.1050 |
| vizwiz | 0.5500 |
| vlguard | 0.9500 |


## blink_art_style

- Best primary metric: `SAV+WVote` = `accuracy=0.9000`
- Macro-F1 of best run: `0.8990`
- Top confusions of best run:
  - `(A)` -> `(B)`: `2`
- Lowest-recall classes of best run:
  - `(A)`: recall `0.8000`, support `10`
  - `(B)`: recall `1.0000`, support `10`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.9000 | 0.8990 | 0.9000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_art_style_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_counting

- Best primary metric: `SAV+WVote` = `accuracy=0.6000`
- Macro-F1 of best run: `0.5929`
- Top confusions of best run:
  - `(C)` -> `(A)`: `2`
  - `(D)` -> `(A)`: `2`
  - `(A)` -> `(C)`: `1`
- Lowest-recall classes of best run:
  - `(C)`: recall `0.4000`, support `5`
  - `(D)`: recall `0.4000`, support `5`
  - `(A)`: recall `0.8000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.6000 | 0.5929 | 0.6000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_counting_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_forensic_detection

- Best primary metric: `SAV+WVote` = `accuracy=0.4000`
- Macro-F1 of best run: `0.4059`
- Top confusions of best run:
  - `(C)` -> `(B)`: `4`
  - `(B)` -> `(A)`: `3`
  - `(A)` -> `(B)`: `1`
- Lowest-recall classes of best run:
  - `(C)`: recall `0.2000`, support `5`
  - `(A)`: recall `0.4000`, support `5`
  - `(B)`: recall `0.4000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.4000 | 0.4059 | 0.4000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_forensic_detection_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_functional_correspondence

- Best primary metric: `SAV+WVote` = `accuracy=0.1875`
- Macro-F1 of best run: `0.1465`
- Top confusions of best run:
  - `(C)` -> `(B)`: `3`
  - `(A)` -> `(C)`: `2`
  - `(A)` -> `(D)`: `2`
- Lowest-recall classes of best run:
  - `(A)`: recall `0.0000`, support `5`
  - `(B)`: recall `0.0000`, support `1`
  - `(D)`: recall `0.2000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.1875 | 0.1465 | 0.1500 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_functional_correspondence_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_iq_test

- Best primary metric: `SAV+WVote` = `accuracy=0.0000`
- Macro-F1 of best run: `0.0000`
- Top confusions of best run:
  - `(D)` -> `(C)`: `5`
  - `(A)` -> `(D)`: `4`
  - `(B)` -> `(A)`: `2`
- Lowest-recall classes of best run:
  - `(A)`: recall `0.0000`, support `5`
  - `(B)`: recall `0.0000`, support `5`
  - `(C)`: recall `0.0000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.0000 | 0.0000 | 0.0000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_iq_test_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_jigsaw

- Best primary metric: `SAV+WVote` = `accuracy=0.7000`
- Macro-F1 of best run: `0.7000`
- Top confusions of best run:
  - `(A)` -> `(B)`: `3`
  - `(B)` -> `(A)`: `3`
- Lowest-recall classes of best run:
  - `(A)`: recall `0.7000`, support `10`
  - `(B)`: recall `0.7000`, support `10`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.7000 | 0.7000 | 0.7000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_jigsaw_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_multi-view_reasoning

- Best primary metric: `SAV+WVote` = `accuracy=0.4000`
- Macro-F1 of best run: `0.3939`
- Top confusions of best run:
  - `(A)` -> `(B)`: `7`
  - `(B)` -> `(A)`: `5`
- Lowest-recall classes of best run:
  - `(A)`: recall `0.3000`, support `10`
  - `(B)`: recall `0.5000`, support `10`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.4000 | 0.3939 | 0.4000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_multi-view_reasoning_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_object_localization

- Best primary metric: `SAV+WVote` = `accuracy=0.4500`
- Macro-F1 of best run: `0.4486`
- Top confusions of best run:
  - `(A)` -> `(B)`: `6`
  - `(B)` -> `(A)`: `5`
- Lowest-recall classes of best run:
  - `(A)`: recall `0.4000`, support `10`
  - `(B)`: recall `0.5000`, support `10`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.4500 | 0.4486 | 0.4500 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_object_localization_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_relative_depth

- Best primary metric: `SAV+WVote` = `accuracy=0.6500`
- Macro-F1 of best run: `0.6491`
- Top confusions of best run:
  - `(B)` -> `(A)`: `4`
  - `(A)` -> `(B)`: `3`
- Lowest-recall classes of best run:
  - `(B)`: recall `0.6000`, support `10`
  - `(A)`: recall `0.7000`, support `10`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.6500 | 0.6491 | 0.6500 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_relative_depth_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_relative_reflectance

- Best primary metric: `SAV+WVote` = `accuracy=0.4000`
- Macro-F1 of best run: `0.3944`
- Top confusions of best run:
  - `(B)` -> `(A)`: `4`
  - `(A)` -> `(C)`: `2`
  - `(A)` -> `(B)`: `1`
- Lowest-recall classes of best run:
  - `(B)`: recall `0.2000`, support `5`
  - `(A)`: recall `0.4000`, support `5`
  - `(C)`: recall `0.6000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.4000 | 0.3944 | 0.4000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_relative_reflectance_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_semantic_correspondence

- Best primary metric: `SAV+WVote` = `accuracy=0.1000`
- Macro-F1 of best run: `0.0913`
- Top confusions of best run:
  - `(D)` -> `(A)`: `4`
  - `(B)` -> `(A)`: `3`
  - `(A)` -> `(C)`: `2`
- Lowest-recall classes of best run:
  - `(B)`: recall `0.0000`, support `5`
  - `(D)`: recall `0.0000`, support `5`
  - `(A)`: recall `0.2000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.1000 | 0.0913 | 0.1000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_semantic_correspondence_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_spatial_relation

- Best primary metric: `SAV+WVote` = `accuracy=0.7000`
- Macro-F1 of best run: `0.6970`
- Top confusions of best run:
  - `(B)` -> `(A)`: `4`
  - `(A)` -> `(B)`: `2`
- Lowest-recall classes of best run:
  - `(B)`: recall `0.6000`, support `10`
  - `(A)`: recall `0.8000`, support `10`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.7000 | 0.6970 | 0.7000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_spatial_relation_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_visual_correspondence

- Best primary metric: `SAV+WVote` = `accuracy=0.3500`
- Macro-F1 of best run: `0.3333`
- Top confusions of best run:
  - `(B)` -> `(D)`: `3`
  - `(D)` -> `(A)`: `3`
  - `(C)` -> `(B)`: `2`
- Lowest-recall classes of best run:
  - `(B)`: recall `0.2000`, support `5`
  - `(C)`: recall `0.2000`, support `5`
  - `(D)`: recall `0.2000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.3500 | 0.3333 | 0.3500 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_visual_correspondence_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## blink_visual_similarity

- Best primary metric: `SAV+WVote` = `accuracy=0.6000`
- Macro-F1 of best run: `0.5833`
- Top confusions of best run:
  - `(A)` -> `(B)`: `6`
  - `(B)` -> `(A)`: `2`
- Lowest-recall classes of best run:
  - `(A)`: recall `0.4000`, support `10`
  - `(B)`: recall `0.8000`, support `10`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.6000 | 0.5833 | 0.6000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/blink_visual_similarity_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## cub

- Best primary metric: `SAV+WVote` = `accuracy=0.7100`
- Macro-F1 of best run: `0.6448`
- Top confusions of best run:
  - `American Three toed Woodpecker` -> `Red cockaded Woodpecker`: `1`
  - `Artic Tern` -> `Spotted Catbird`: `1`
  - `Bank Swallow` -> `Barn Swallow`: `1`
- Lowest-recall classes of best run:
  - `American Three toed Woodpecker`: recall `0.0000`, support `1`
  - `Artic Tern`: recall `0.0000`, support `1`
  - `Bank Swallow`: recall `0.0000`, support `1`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.7100 | 0.6448 | 0.7100 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/cub_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## eurosat

- Best primary metric: `SAV+WVote` = `accuracy=0.6400`
- Macro-F1 of best run: `0.6209`
- Top confusions of best run:
  - `Highway or Road` -> `River`: `3`
  - `Forest` -> `Sea or Lake`: `2`
  - `Pasture Land` -> `Herbaceous Vegetation Land`: `2`
- Lowest-recall classes of best run:
  - `Forest`: recall `0.2000`, support `5`
  - `Highway or Road`: recall `0.4000`, support `5`
  - `Permanent Crop Land`: recall `0.4000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.6400 | 0.6209 | 0.6400 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/eurosat_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## flowers

- Best primary metric: `SAV+WVote` = `accuracy=0.9706`
- Macro-F1 of best run: `0.9608`
- Top confusions of best run:
  - `globe-flower` -> `buttercup`: `1`
  - `sweet pea` -> `cape`: `1`
  - `windflower` -> `japanese anemone`: `1`
- Lowest-recall classes of best run:
  - `globe-flower`: recall `0.0000`, support `1`
  - `sweet pea`: recall `0.0000`, support `1`
  - `windflower`: recall `0.0000`, support `1`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.9706 | 0.9608 | 0.9706 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/flowers_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## mhalubench

- Best primary metric: `SAV+WVote` = `accuracy=0.9250`
- Macro-F1 of best run: `0.9246`
- Top confusions of best run:
  - `non-hallucination` -> `hallucination`: `3`
- Lowest-recall classes of best run:
  - `non-hallucination`: recall `0.8500`, support `20`
  - `hallucination`: recall `1.0000`, support `20`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.9250 | 0.9246 | 0.9250 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/mhalubench_val_v01_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |
| SAV+WVote | accuracy=0.9250 | 0.9246 | 0.9250 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/mhalubench_val_v02_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## naturalbench_ret

- Best primary metric: `SAV+WVote` = `g_acc=0.8000`
- Macro-F1 of best run: `0.9500`
- Top confusions of best run:
  - `No` -> `Yes`: `1`
  - `Yes` -> `No`: `1`
- Lowest-recall classes of best run:
  - `No`: recall `0.9500`, support `20`
  - `Yes`: recall `0.9500`, support `20`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | g_acc=0.8000 | 0.9500 | 0.9500 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/naturalbench_ret_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## naturalbench_vqa

- Best primary metric: `SAV+WVote` = `accuracy=0.8000`
- Macro-F1 of best run: `0.8071`
- Top confusions of best run:
  - `No` -> `Yes`: `3`
  - `Yes` -> `No`: `3`
  - `A` -> `B`: `2`
- Lowest-recall classes of best run:
  - `A`: recall `0.6667`, support `6`
  - `No`: recall `0.7857`, support `14`
  - `Yes`: recall `0.7857`, support `14`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.8000 | 0.8071 | 0.8095 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/naturalbench_vqa_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## pets

- Best primary metric: `SAV+WVote` = `accuracy=1.0000`
- Macro-F1 of best run: `1.0000`
- Lowest-recall classes of best run:
  - `Abyssinian`: recall `1.0000`, support `2`
  - `Bengal`: recall `1.0000`, support `2`
  - `Birman`: recall `1.0000`, support `2`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=1.0000 | 1.0000 | 1.0000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/pets_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## sugarcrepe

- Best primary metric: `SAV+WVote` = `pair_accuracy=0.5000`
- Macro-F1 of best run: `0.7494`
- Top confusions of best run:
  - `Yes` -> `No`: `6`
  - `No` -> `Yes`: `4`
- Lowest-recall classes of best run:
  - `Yes`: recall `0.7000`, support `20`
  - `No`: recall `0.8000`, support `20`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | pair_accuracy=0.5000 | 0.7494 | 0.7500 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/sugarcrepe_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## tinyimage

- Best primary metric: `SAV+WVote` = `accuracy=0.1050`
- Macro-F1 of best run: `0.0716`
- Top confusions of best run:
  - `African elephant` -> `lion`: `1`
  - `American alligator` -> `sea cucumber`: `1`
  - `American lobster` -> `volleyball`: `1`
- Lowest-recall classes of best run:
  - `African elephant`: recall `0.0000`, support `1`
  - `American alligator`: recall `0.0000`, support `1`
  - `American lobster`: recall `0.0000`, support `1`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.1050 | 0.0716 | 0.1050 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/tinyimage_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## vizwiz

- Best primary metric: `SAV+WVote` = `accuracy=0.5500`
- Macro-F1 of best run: `0.5455`
- Top confusions of best run:
  - `answerable` -> `unanswerable`: `11`
  - `unanswerable` -> `answerable`: `7`
- Lowest-recall classes of best run:
  - `answerable`: recall `0.4500`, support `20`
  - `unanswerable`: recall `0.6500`, support `20`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.5500 | 0.5455 | 0.5500 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/vizwiz_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |

## vlguard

- Best primary metric: `SAV+WVote` = `accuracy=0.9500`
- Macro-F1 of best run: `0.9500`
- Top confusions of best run:
  - `harmful` -> `unharmful`: `1`
  - `unharmful` -> `harmful`: `1`
- Lowest-recall classes of best run:
  - `harmful`: recall `0.9500`, support `20`
  - `unharmful`: recall `0.9500`, support `20`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV+WVote | accuracy=0.9500 | 0.9500 | 0.9500 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_133026_all_task_method_suite/vlguard_sav_wvote_qwen2_vl_20260324_133026.metrics.json` | - |
