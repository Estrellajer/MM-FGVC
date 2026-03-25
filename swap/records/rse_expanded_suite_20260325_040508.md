# RSE Improvement Suite

- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_040508_rse_expanded_suite/manifest.tsv`

## Overview

| Dataset | Zero-shot | SAV | RSE-CV | RSE-PCA |
| --- | --- | --- | --- | --- |
| eurosat | 0.6000 | 0.6600 | 0.5900 | 0.5900 |

## Aggregate Deltas vs RSE

| Method | Mean Delta vs RSE | Wins | Ties | Losses | Mean Delta vs Zero-shot | Mean Delta vs SAV |
| --- | --- | --- | --- | --- | --- | --- |
| RSE-CV | - | 0 | 0 | 0 | - | - |
| RSE-PCA | - | 0 | 0 | 0 | - | - |

## eurosat

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.6000 | - | - | - | - | - | - |
| SAV | accuracy=0.6600 | - | - | - | - | - | - |
| RSE-CV | accuracy=0.5900 | - | layer@26 (0.6100) | 8 | 0.5500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040508_rse_expanded_suite/eurosat_rse_cv_qwen2_vl_20260325_040508.diagnostics.json` |
| RSE-PCA | accuracy=0.5900 | - | layer@26 (0.6300) | 8 | 0.5200 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040508_rse_expanded_suite/eurosat_rse_pca_qwen2_vl_20260325_040508.diagnostics.json` |
