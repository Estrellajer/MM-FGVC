# FGVC Method Suite Results

- Model: `qwen2_vl`
- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260324_124443_fgvc_method_suite/manifest.tsv`
- Datasets: `eurosat`
- Methods: `Zero-shot`

## Overview

| Dataset | Zero-shot |
| --- | --- |
| eurosat | 0.6000 |

## STV Ablation

| Dataset | STV | STV+QC | SAV-TV | SAV-TV+QC |
| --- | --- | --- | --- | --- |
| eurosat | - | - | - | - |

## eurosat

- Best accuracy: `Zero-shot` = `0.6000`
- Macro-F1 of best run: `0.5779`
- Top confusions of best run:
  - `Annual Crop Land` -> `Herbaceous Vegetation Land`: `2`
  - `Annual Crop Land` -> `Permanent Crop Land`: `2`
  - `Herbaceous Vegetation Land` -> `Forest`: `2`
- Lowest-recall classes of best run:
  - `Pasture Land`: recall `0.3000`, support `10`
  - `Residential Buildings`: recall `0.3000`, support `10`
  - `Annual Crop Land`: recall `0.4000`, support `10`

| Method | Acc | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| Zero-shot | 0.6000 | 0.5779 | 0.6000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124443_fgvc_method_suite/eurosat_zero_shot_qwen2_vl_20260324_124443.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124443_fgvc_method_suite/eurosat_zero_shot_qwen2_vl_20260324_124443.predictions.jsonl` |
