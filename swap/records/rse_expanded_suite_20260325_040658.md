# RSE Improvement Suite

- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/manifest.tsv`

## Overview

| Dataset | Zero-shot | SAV | RSE | RSE-ZScore | RSE-CV | RSE-Shrink | RSE-Adaptive | RSE-PCA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blink_semantic_correspondence | 0.0286 | 0.1143 | 0.2286 | 0.2000 | 0.2000 | 0.1714 | 0.2857 | 0.2286 |
| cub | 0.9500 | 0.9900 | 0.9700 | 0.9700 | 0.9700 | 0.9300 | 0.9600 | 0.9700 |
| eurosat | 0.6000 | 0.6600 | 0.5600 | 0.6300 | 0.5900 | 0.5300 | 0.6600 | 0.5900 |
| naturalbench_vqa | 0.7375 | 0.8250 | 0.7625 | 0.8000 | 0.7875 | 0.8000 | 0.8000 | 0.7875 |
| sugarcrepe | 0.2750 | 0.6750 | 0.4500 | 0.4750 | 0.4750 | 0.5000 | 0.4750 | 0.4750 |
| vlguard | 0.7125 | 0.9750 | 0.9250 | 1.0000 | 0.9625 | 0.9750 | 1.0000 | 1.0000 |

## Aggregate Deltas vs RSE

| Method | Mean Delta vs RSE | Wins | Ties | Losses | Mean Delta vs Zero-shot | Mean Delta vs SAV |
| --- | --- | --- | --- | --- | --- | --- |
| RSE-ZScore | +0.0298 | 4 | 1 | 1 | +0.1286 | -0.0274 |
| RSE-CV | +0.0148 | 4 | 1 | 1 | +0.1136 | -0.0424 |
| RSE-Shrink | +0.0017 | 3 | 0 | 3 | +0.1005 | -0.0555 |
| RSE-Adaptive | +0.0474 | 5 | 0 | 1 | +0.1462 | -0.0098 |
| RSE-PCA | +0.0258 | 4 | 2 | 0 | +0.1246 | -0.0314 |

## blink_semantic_correspondence

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.0286 | -0.2000 | - | - | - | - | - |
| SAV | accuracy=0.1143 | -0.1143 | - | - | - | - | - |
| RSE | accuracy=0.2286 | +0.0000 | layer@2 (0.3429) | 8 | 0.2750 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/blink_semantic_correspondence_rse_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-ZScore | accuracy=0.2000 | -0.0286 | layer@2 (0.3429) | 8 | 0.3250 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/blink_semantic_correspondence_rse_zscore_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-CV | accuracy=0.2000 | -0.0286 | layer@2 (0.3429) | 8 | 0.4000 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/blink_semantic_correspondence_rse_cv_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Shrink | accuracy=0.1714 | -0.0571 | attn@6 (0.3714) | 8 | 0.3625 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/blink_semantic_correspondence_rse_shrink_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Adaptive | accuracy=0.2857 | +0.0571 | layer@2 (0.3429) | 8 | 0.3250 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/blink_semantic_correspondence_rse_adaptive_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-PCA | accuracy=0.2286 | +0.0000 | attn@6 (0.3714) | 8 | 0.3500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/blink_semantic_correspondence_rse_pca_qwen2_vl_20260325_040658.diagnostics.json` |

## cub

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.9500 | -0.0200 | - | - | - | - | - |
| SAV | accuracy=0.9900 | +0.0200 | - | - | - | - | - |
| RSE | accuracy=0.9700 | +0.0000 | attn@26 (0.9500) | 8 | 0.9900 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/cub_rse_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-ZScore | accuracy=0.9700 | +0.0000 | attn@26 (0.9500) | 8 | 1.0000 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/cub_rse_zscore_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-CV | accuracy=0.9700 | +0.0000 | attn@26 (0.9500) | 8 | 1.0000 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/cub_rse_cv_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Shrink | accuracy=0.9300 | -0.0400 | head@27 (0.9400) | 8 | 0.8700 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/cub_rse_shrink_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Adaptive | accuracy=0.9600 | -0.0100 | attn@26 (0.9500) | 8 | 1.0000 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/cub_rse_adaptive_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-PCA | accuracy=0.9700 | +0.0000 | attn@27 (0.9400) | 8 | 1.0000 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/cub_rse_pca_qwen2_vl_20260325_040658.diagnostics.json` |

## eurosat

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.6000 | +0.0400 | - | - | - | - | - |
| SAV | accuracy=0.6600 | +0.1000 | - | - | - | - | - |
| RSE | accuracy=0.5600 | +0.0000 | layer@26 (0.6100) | 8 | 0.4700 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/eurosat_rse_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-ZScore | accuracy=0.6300 | +0.0700 | layer@26 (0.6100) | 8 | 0.4900 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/eurosat_rse_zscore_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-CV | accuracy=0.5900 | +0.0300 | layer@26 (0.6100) | 8 | 0.5500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/eurosat_rse_cv_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Shrink | accuracy=0.5300 | -0.0300 | layer@26 (0.5900) | 8 | 0.4700 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/eurosat_rse_shrink_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Adaptive | accuracy=0.6600 | +0.1000 | layer@26 (0.6100) | 8 | 0.4900 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/eurosat_rse_adaptive_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-PCA | accuracy=0.5900 | +0.0300 | layer@26 (0.6300) | 8 | 0.5200 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/eurosat_rse_pca_qwen2_vl_20260325_040658.diagnostics.json` |

## naturalbench_vqa

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.7375 | -0.0250 | - | - | - | - | - |
| SAV | accuracy=0.8250 | +0.0625 | - | - | - | - | - |
| RSE | accuracy=0.7625 | +0.0000 | attn@21 (0.8250) | 8 | 0.6875 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/naturalbench_vqa_rse_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-ZScore | accuracy=0.8000 | +0.0375 | attn@21 (0.8250) | 8 | 0.7500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/naturalbench_vqa_rse_zscore_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-CV | accuracy=0.7875 | +0.0250 | attn@21 (0.8250) | 8 | 0.7375 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/naturalbench_vqa_rse_cv_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Shrink | accuracy=0.8000 | +0.0375 | attn@19 (0.8375) | 8 | 0.6750 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/naturalbench_vqa_rse_shrink_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Adaptive | accuracy=0.8000 | +0.0375 | attn@21 (0.8250) | 8 | 0.7500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/naturalbench_vqa_rse_adaptive_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-PCA | accuracy=0.7875 | +0.0250 | head@21 (0.8375) | 8 | 0.7375 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/naturalbench_vqa_rse_pca_qwen2_vl_20260325_040658.diagnostics.json` |

## sugarcrepe

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | pair_accuracy=0.2750 | -0.1750 | - | - | - | - | - |
| SAV | pair_accuracy=0.6750 | +0.2250 | - | - | - | - | - |
| RSE | pair_accuracy=0.4500 | +0.0000 | layer@14 (0.8125) | 8 | 0.7250 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/sugarcrepe_rse_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-ZScore | pair_accuracy=0.4750 | +0.0250 | layer@14 (0.8125) | 8 | 0.7500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/sugarcrepe_rse_zscore_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-CV | pair_accuracy=0.4750 | +0.0250 | layer@14 (0.8125) | 8 | 0.7500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/sugarcrepe_rse_cv_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Shrink | pair_accuracy=0.5000 | +0.0500 | layer@14 (0.8125) | 8 | 0.7500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/sugarcrepe_rse_shrink_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Adaptive | pair_accuracy=0.4750 | +0.0250 | layer@14 (0.8125) | 8 | 0.7500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/sugarcrepe_rse_adaptive_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-PCA | pair_accuracy=0.4750 | +0.0250 | layer@14 (0.8125) | 8 | 0.7250 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/sugarcrepe_rse_pca_qwen2_vl_20260325_040658.diagnostics.json` |

## vlguard

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.7125 | -0.2125 | - | - | - | - | - |
| SAV | accuracy=0.9750 | +0.0500 | - | - | - | - | - |
| RSE | accuracy=0.9250 | +0.0000 | attn@15 (0.9875) | 8 | 0.9000 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/vlguard_rse_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-ZScore | accuracy=1.0000 | +0.0750 | attn@15 (0.9875) | 8 | 0.9500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/vlguard_rse_zscore_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-CV | accuracy=0.9625 | +0.0375 | attn@15 (0.9875) | 8 | 0.9500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/vlguard_rse_cv_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Shrink | accuracy=0.9750 | +0.0500 | attn@15 (0.9875) | 8 | 0.9500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/vlguard_rse_shrink_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-Adaptive | accuracy=1.0000 | +0.0750 | attn@15 (0.9875) | 8 | 0.9500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/vlguard_rse_adaptive_qwen2_vl_20260325_040658.diagnostics.json` |
| RSE-PCA | accuracy=1.0000 | +0.0750 | attn@15 (0.9875) | 8 | 0.9500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/vlguard_rse_pca_qwen2_vl_20260325_040658.diagnostics.json` |
