# Method Suite Results

- Model: `qwen2_vl`
- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_065407_fgvc_method_suite/manifest.tsv`
- Datasets: `eurosat`
- Methods: `KeCO`

## Overview

| Dataset | KeCO |
| --- | --- |
| eurosat | 0.7900 |


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
| KeCO | accuracy=0.7900 | 0.7861 | 0.7900 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_065407_fgvc_method_suite/eurosat_keco_qwen2_vl_20260325_065407.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260325_065407_fgvc_method_suite/eurosat_keco_qwen2_vl_20260325_065407.predictions.jsonl` |
