# Method Suite Results

- Model: `qwen2_vl`
- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_065838_fgvc_method_suite/manifest.tsv`
- Datasets: `cub, eurosat, pets`
- Methods: `KeCO`

## Overview

| Dataset | KeCO |
| --- | --- |
| cub | 1.0000 |
| eurosat | 0.7900 |
| pets | 0.9800 |


## cub

- Best primary metric: `KeCO` = `accuracy=1.0000`
- Macro-F1 of best run: `1.0000`
- Lowest-recall classes of best run:
  - `Acadian Flycatcher`: recall `1.0000`, support `5`
  - `American Crow`: recall `1.0000`, support `5`
  - `American Goldfinch`: recall `1.0000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| KeCO | accuracy=1.0000 | 1.0000 | 1.0000 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_065838_fgvc_method_suite/cub_keco_qwen2_vl_20260325_065838.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260325_065838_fgvc_method_suite/cub_keco_qwen2_vl_20260325_065838.predictions.jsonl` |

## eurosat

- Best primary metric: `KeCO` = `accuracy=0.7900`
- Macro-F1 of best run: `0.7861`
- Top confusions of best run:
  - `Forest` -> `Highway or Road`: `2`
  - `Annual Crop Land` -> `Pasture Land`: `1`
  - `Annual Crop Land` -> `Permanent Crop Land`: `1`
- Lowest-recall classes of best run:
  - `Annual Crop Land`: recall `0.6000`, support `10`
  - `Forest`: recall `0.6000`, support `10`
  - `Pasture Land`: recall `0.7000`, support `10`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| KeCO | accuracy=0.7900 | 0.7861 | 0.7900 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_065838_fgvc_method_suite/eurosat_keco_qwen2_vl_20260325_065838.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260325_065838_fgvc_method_suite/eurosat_keco_qwen2_vl_20260325_065838.predictions.jsonl` |

## pets

- Best primary metric: `KeCO` = `accuracy=0.9800`
- Macro-F1 of best run: `0.9798`
- Top confusions of best run:
  - `basset hound` -> `german shorthaired`: `1`
  - `beagle` -> `english setter`: `1`
- Lowest-recall classes of best run:
  - `basset hound`: recall `0.8000`, support `5`
  - `beagle`: recall `0.8000`, support `5`
  - `Abyssinian`: recall `1.0000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| KeCO | accuracy=0.9800 | 0.9798 | 0.9800 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_065838_fgvc_method_suite/pets_keco_qwen2_vl_20260325_065838.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260325_065838_fgvc_method_suite/pets_keco_qwen2_vl_20260325_065838.predictions.jsonl` |
