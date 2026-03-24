# FGVC Method Suite Results

- Model: `qwen2_vl`
- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/manifest.tsv`
- Datasets: `cub, eurosat, pets`
- Methods: `I2CL, MimIC, SAV, SAV-TV, SAV-TV+QC, STV, STV+QC, Zero-shot`

## Overview

| Dataset | I2CL | MimIC | SAV | SAV-TV | SAV-TV+QC | STV | STV+QC | Zero-shot |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cub | 0.9500 | 0.9800 | 0.9900 | 0.9500 | 0.9500 | 0.9500 | 0.9400 | 0.9500 |
| eurosat | 0.5900 | 0.6600 | 0.6600 | 0.5900 | 0.5900 | 0.6000 | 0.6000 | 0.6000 |
| pets | 0.9500 | 0.9500 | 0.9500 | 0.9500 | 0.9500 | 0.9600 | 0.9300 | 0.9500 |

## STV Ablation

| Dataset | STV | STV+QC | SAV-TV | SAV-TV+QC |
| --- | --- | --- | --- | --- |
| cub | 0.9500 | 0.9400 | 0.9500 | 0.9500 |
| eurosat | 0.6000 | 0.6000 | 0.5900 | 0.5900 |
| pets | 0.9600 | 0.9300 | 0.9500 | 0.9500 |

## cub

- Best accuracy: `SAV` = `0.9900`
- Macro-F1 of best run: `0.9899`
- Top confusions of best run:
  - `Black capped Vireo` -> `Bay breasted Warbler`: `1`
- Lowest-recall classes of best run:
  - `Black capped Vireo`: recall `0.8000`, support `5`
  - `Acadian Flycatcher`: recall `1.0000`, support `5`
  - `American Crow`: recall `1.0000`, support `5`
- Most complementary method pairs:
  - `SAV` + `I2CL`: oracle `1.0000`, one-correct-only `0.0600`, disagreement `0.0600`
  - `SAV` + `SAV-TV`: oracle `1.0000`, one-correct-only `0.0600`, disagreement `0.0600`
  - `SAV` + `SAV-TV+QC`: oracle `1.0000`, one-correct-only `0.0600`, disagreement `0.0600`
  - `SAV` + `STV`: oracle `1.0000`, one-correct-only `0.0600`, disagreement `0.0600`
  - `SAV` + `Zero-shot`: oracle `1.0000`, one-correct-only `0.0600`, disagreement `0.0600`

| Method | Acc | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| SAV | 0.9900 | 0.9899 | 0.9900 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_sav_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_sav_qwen2_vl_20260324_125034.predictions.jsonl` |
| MimIC | 0.9800 | 0.9375 | 0.9333 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_mimic_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_mimic_qwen2_vl_20260324_125034.predictions.jsonl` |
| I2CL | 0.9500 | 0.7767 | 0.7600 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_i2cl_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_i2cl_qwen2_vl_20260324_125034.predictions.jsonl` |
| SAV-TV | 0.9500 | 0.7767 | 0.7600 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_sav_tv_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_sav_tv_qwen2_vl_20260324_125034.predictions.jsonl` |
| SAV-TV+QC | 0.9500 | 0.7767 | 0.7600 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_sav_tv_qc_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_sav_tv_qc_qwen2_vl_20260324_125034.predictions.jsonl` |
| STV | 0.9500 | 0.7767 | 0.7600 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_stv_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_stv_qwen2_vl_20260324_125034.predictions.jsonl` |
| Zero-shot | 0.9500 | 0.7767 | 0.7600 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_zero_shot_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_zero_shot_qwen2_vl_20260324_125034.predictions.jsonl` |
| STV+QC | 0.9400 | 0.7686 | 0.7520 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_stv_qc_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/cub_stv_qc_qwen2_vl_20260324_125034.predictions.jsonl` |

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
| MimIC | 0.6600 | 0.6403 | 0.6600 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_mimic_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_mimic_qwen2_vl_20260324_125034.predictions.jsonl` |
| SAV | 0.6600 | 0.6537 | 0.6600 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_sav_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_sav_qwen2_vl_20260324_125034.predictions.jsonl` |
| STV | 0.6000 | 0.5762 | 0.6000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_stv_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_stv_qwen2_vl_20260324_125034.predictions.jsonl` |
| STV+QC | 0.6000 | 0.5756 | 0.6000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_stv_qc_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_stv_qc_qwen2_vl_20260324_125034.predictions.jsonl` |
| Zero-shot | 0.6000 | 0.5779 | 0.6000 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_zero_shot_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_zero_shot_qwen2_vl_20260324_125034.predictions.jsonl` |
| I2CL | 0.5900 | 0.5639 | 0.5900 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_i2cl_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_i2cl_qwen2_vl_20260324_125034.predictions.jsonl` |
| SAV-TV | 0.5900 | 0.5639 | 0.5900 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_sav_tv_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_sav_tv_qwen2_vl_20260324_125034.predictions.jsonl` |
| SAV-TV+QC | 0.5900 | 0.5639 | 0.5900 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_sav_tv_qc_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/eurosat_sav_tv_qc_qwen2_vl_20260324_125034.predictions.jsonl` |

## pets

- Best accuracy: `STV` = `0.9600`
- Macro-F1 of best run: `0.8451`
- Top confusions of best run:
  - `Egyptian Mau` -> `D. Russian Blue`: `1`
  - `beagle` -> `english setter`: `1`
  - `great pyrenees` -> `C. newfoundland`: `1`
- Lowest-recall classes of best run:
  - `great pyrenees`: recall `0.6000`, support `5`
  - `Egyptian Mau`: recall `0.8000`, support `5`
  - `beagle`: recall `0.8000`, support `5`
- Most complementary method pairs:
  - `SAV` + `STV+QC`: oracle `0.9900`, one-correct-only `0.1000`, disagreement `0.1100`
  - `I2CL` + `SAV`: oracle `0.9900`, one-correct-only `0.0800`, disagreement `0.0900`
  - `SAV` + `SAV-TV`: oracle `0.9900`, one-correct-only `0.0800`, disagreement `0.0900`
  - `SAV` + `SAV-TV+QC`: oracle `0.9900`, one-correct-only `0.0800`, disagreement `0.0900`
  - `SAV` + `Zero-shot`: oracle `0.9900`, one-correct-only `0.0800`, disagreement `0.0900`

| Method | Acc | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| STV | 0.9600 | 0.8451 | 0.8348 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_stv_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_stv_qwen2_vl_20260324_125034.predictions.jsonl` |
| I2CL | 0.9500 | 0.8689 | 0.8636 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_i2cl_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_i2cl_qwen2_vl_20260324_125034.predictions.jsonl` |
| MimIC | 0.9500 | 0.8715 | 0.8636 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_mimic_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_mimic_qwen2_vl_20260324_125034.predictions.jsonl` |
| SAV | 0.9500 | 0.9496 | 0.9500 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_sav_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_sav_qwen2_vl_20260324_125034.predictions.jsonl` |
| SAV-TV | 0.9500 | 0.8351 | 0.8261 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_sav_tv_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_sav_tv_qwen2_vl_20260324_125034.predictions.jsonl` |
| SAV-TV+QC | 0.9500 | 0.8351 | 0.8261 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_sav_tv_qc_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_sav_tv_qc_qwen2_vl_20260324_125034.predictions.jsonl` |
| Zero-shot | 0.9500 | 0.8689 | 0.8636 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_zero_shot_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_zero_shot_qwen2_vl_20260324_125034.predictions.jsonl` |
| STV+QC | 0.9300 | 0.7876 | 0.7750 | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_stv_qc_qwen2_vl_20260324_125034.metrics.json` | `/root/autodl-tmp/FGVC/swap/outputs/20260324_125034_fgvc_method_suite/pets_stv_qc_qwen2_vl_20260324_125034.predictions.jsonl` |
