# Method Suite Results

- Model: `qwen2_vl`
- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/manifest.tsv`
- Datasets: `blink_semantic_correspondence, cub, eurosat, naturalbench_vqa`
- Methods: `RSE, SAV, Zero-shot`

## Overview

| Dataset | RSE | SAV | Zero-shot |
| --- | --- | --- | --- |
| blink_semantic_correspondence | 0.2000 | 0.1000 | 0.0500 |
| cub | 0.8100 | 0.7100 | 0.9000 |
| eurosat | 0.5800 | 0.6400 | 0.6200 |
| naturalbench_vqa | 0.7250 | 0.8000 | 0.7500 |


## blink_semantic_correspondence

- Best primary metric: `RSE` = `accuracy=0.2000`
- Macro-F1 of best run: `0.2054`
- Top confusions of best run:
  - `(D)` -> `(A)`: `4`
  - `(A)` -> `(D)`: `3`
  - `(B)` -> `(A)`: `2`
- Lowest-recall classes of best run:
  - `(D)`: recall `0.0000`, support `5`
  - `(B)`: recall `0.2000`, support `5`
  - `(C)`: recall `0.2000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| RSE | accuracy=0.2000 | 0.2054 | 0.2000 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/blink_semantic_correspondence_rse_qwen2_vl_20260325_020639.metrics.json` | - |
| SAV | accuracy=0.1000 | 0.0913 | 0.1000 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/blink_semantic_correspondence_sav_qwen2_vl_20260325_020639.metrics.json` | - |
| Zero-shot | accuracy=0.0500 | 0.0167 | 0.0133 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/blink_semantic_correspondence_zero_shot_qwen2_vl_20260325_020639.metrics.json` | - |

## cub

- Best primary metric: `Zero-shot` = `accuracy=0.9000`
- Macro-F1 of best run: `0.8672`
- Top confusions of best run:
  - `Bank Swallow` -> `Cliff Swallow`: `1`
  - `Chestnut sided Warbler` -> `Myrtle Warbler`: `1`
  - `Chuck will Widow` -> `Bewick Wren`: `1`
- Lowest-recall classes of best run:
  - `Bank Swallow`: recall `0.0000`, support `1`
  - `Chestnut sided Warbler`: recall `0.0000`, support `1`
  - `Chuck will Widow`: recall `0.0000`, support `1`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.9000 | 0.8672 | 0.8911 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/cub_zero_shot_qwen2_vl_20260325_020639.metrics.json` | - |
| RSE | accuracy=0.8100 | 0.7640 | 0.8100 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/cub_rse_qwen2_vl_20260325_020639.metrics.json` | - |
| SAV | accuracy=0.7100 | 0.6448 | 0.7100 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/cub_sav_qwen2_vl_20260325_020639.metrics.json` | - |

## eurosat

- Best primary metric: `SAV` = `accuracy=0.6400`
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
| SAV | accuracy=0.6400 | 0.6209 | 0.6400 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/eurosat_sav_qwen2_vl_20260325_020639.metrics.json` | - |
| Zero-shot | accuracy=0.6200 | 0.5841 | 0.6200 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/eurosat_zero_shot_qwen2_vl_20260325_020639.metrics.json` | - |
| RSE | accuracy=0.5800 | 0.5623 | 0.5800 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/eurosat_rse_qwen2_vl_20260325_020639.metrics.json` | - |

## naturalbench_vqa

- Best primary metric: `SAV` = `accuracy=0.8000`
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
| SAV | accuracy=0.8000 | 0.8071 | 0.8095 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/naturalbench_vqa_sav_qwen2_vl_20260325_020639.metrics.json` | - |
| Zero-shot | accuracy=0.7500 | 0.7731 | 0.7738 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/naturalbench_vqa_zero_shot_qwen2_vl_20260325_020639.metrics.json` | - |
| RSE | accuracy=0.7250 | 0.7305 | 0.7321 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/naturalbench_vqa_rse_qwen2_vl_20260325_020639.metrics.json` | - |
