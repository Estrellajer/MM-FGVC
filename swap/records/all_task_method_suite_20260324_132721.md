# Method Suite Results

- Model: `qwen2_vl`
- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260324_132721_all_task_method_suite/manifest.tsv`
- Datasets: `eurosat, naturalbench_vqa`
- Methods: `SAV+WVote`

## Overview

| Dataset | SAV+WVote |
| --- | --- |
| eurosat | 0.6400 |
| naturalbench_vqa | 0.8000 |


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
| SAV+WVote | accuracy=0.6400 | 0.6209 | 0.6400 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_132721_all_task_method_suite/eurosat_sav_wvote_qwen2_vl_20260324_132721.metrics.json` | - |

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
| SAV+WVote | accuracy=0.8000 | 0.8071 | 0.8095 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_132721_all_task_method_suite/naturalbench_vqa_sav_wvote_qwen2_vl_20260324_132721.metrics.json` | - |
