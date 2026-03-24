# FGVC Method Suite Results

- Model: `qwen2_vl`
- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/manifest.tsv`
- Datasets: `eurosat`
- Methods: `I2CL, MimIC, SAV, SAV-TV, SAV-TV+QC, STV, STV+QC, Zero-shot`

## Overview

| Dataset | I2CL | MimIC | SAV | SAV-TV | SAV-TV+QC | STV | STV+QC | Zero-shot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eurosat | 0.5900 | 0.6600 | 0.6600 | 0.5900 | 0.5900 | 0.6000 | 0.6000 | 0.6000 |

## STV Ablation

| Dataset | STV | STV+QC | SAV-TV | SAV-TV+QC |
| --- | --- | --- | --- | --- |
| eurosat | 0.6000 | 0.6000 | 0.5900 | 0.5900 |

## eurosat

- Best accuracy: `MimIC` = `0.6600`
- Macro-F1 of best run: `0.6403`
- Top confusions of best run:
  - `Pasture Land` -> `Annual Crop Land`: `3`
  - `Herbaceous Vegetation Land` -> `Pasture Land`: `2`
  - `Herbaceous Vegetation Land` -> `River`: `2`
- Lowest-recall classes of best run:
  - `Herbaceous Vegetation Land`: recall `0.2000`, support `10`
  - `Permanent Crop Land`: recall `0.4000`, support `10`
  - `Highway or Road`: recall `0.5000`, support `10`
- Most complementary method pairs:
  - `SAV` + `I2CL`: oracle `0.8500`, one-correct-only `0.4500`, disagreement `0.5700`
  - `SAV` + `SAV-TV`: oracle `0.8500`, one-correct-only `0.4500`, disagreement `0.5700`
  - `SAV` + `SAV-TV+QC`: oracle `0.8500`, one-correct-only `0.4500`, disagreement `0.5700`
  - `SAV` + `STV`: oracle `0.8500`, one-correct-only `0.4400`, disagreement `0.5600`
  - `SAV` + `STV+QC`: oracle `0.8500`, one-correct-only `0.4400`, disagreement `0.5600`

| Method | Acc | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| MimIC | 0.6600 | 0.6403 | 0.6600 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_mimic_qwen2_vl_20260324_124541.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_mimic_qwen2_vl_20260324_124541.predictions.jsonl` |
| SAV | 0.6600 | 0.6537 | 0.6600 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_sav_qwen2_vl_20260324_124541.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_sav_qwen2_vl_20260324_124541.predictions.jsonl` |
| STV | 0.6000 | 0.5762 | 0.6000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_stv_qwen2_vl_20260324_124541.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_stv_qwen2_vl_20260324_124541.predictions.jsonl` |
| STV+QC | 0.6000 | 0.5756 | 0.6000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_stv_qc_qwen2_vl_20260324_124541.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_stv_qc_qwen2_vl_20260324_124541.predictions.jsonl` |
| Zero-shot | 0.6000 | 0.5779 | 0.6000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_zero_shot_qwen2_vl_20260324_124541.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_zero_shot_qwen2_vl_20260324_124541.predictions.jsonl` |
| I2CL | 0.5900 | 0.5639 | 0.5900 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_i2cl_qwen2_vl_20260324_124541.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_i2cl_qwen2_vl_20260324_124541.predictions.jsonl` |
| SAV-TV | 0.5900 | 0.5639 | 0.5900 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_sav_tv_qwen2_vl_20260324_124541.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_sav_tv_qwen2_vl_20260324_124541.predictions.jsonl` |
| SAV-TV+QC | 0.5900 | 0.5639 | 0.5900 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_sav_tv_qc_qwen2_vl_20260324_124541.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_124541_fgvc_method_suite/eurosat_sav_tv_qc_qwen2_vl_20260324_124541.predictions.jsonl` |
