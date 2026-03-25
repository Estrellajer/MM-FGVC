# Method Suite Results

- Model: `qwen2_vl`
- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/manifest.tsv`
- Datasets: `blink_semantic_correspondence, cub, eurosat, naturalbench_vqa`
- Methods: `RSE-Combo, RSE-Fallback, RSE-Greedy, RSE-LOO, RSE-Route1, RSE-Route2, RSE-Top1, RSE-ZScore`

## Overview

| Dataset | RSE-Combo | RSE-Fallback | RSE-Greedy | RSE-LOO | RSE-Route1 | RSE-Route2 | RSE-Top1 | RSE-ZScore |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blink_semantic_correspondence | 0.1000 | 0.1000 | 0.2500 | 0.2000 | 0.2000 | 0.2000 | 0.2500 | 0.2000 |
| cub | 0.8650 | 0.8500 | 0.7950 | 0.7850 | 0.7950 | 0.7950 | 0.7600 | 0.8050 |
| eurosat | 0.6600 | 0.6400 | 0.6800 | 0.6800 | 0.6200 | 0.5800 | 0.6000 | 0.7000 |
| naturalbench_vqa | 0.7500 | 0.7500 | 0.7500 | 0.7750 | 0.7750 | 0.7750 | 0.7750 | 0.7750 |


## blink_semantic_correspondence

- Best primary metric: `RSE-Greedy` = `accuracy=0.2500`
- Macro-F1 of best run: `0.2595`
- Top confusions of best run:
  - `(A)` -> `(D)`: `3`
  - `(D)` -> `(C)`: `3`
  - `(B)` -> `(C)`: `2`
- Lowest-recall classes of best run:
  - `(D)`: recall `0.0000`, support `5`
  - `(A)`: recall `0.2000`, support `5`
  - `(B)`: recall `0.4000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| RSE-Greedy | accuracy=0.2500 | 0.2595 | 0.2500 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_greedy_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Top1 | accuracy=0.2500 | 0.2595 | 0.2500 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_top1_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-LOO | accuracy=0.2000 | 0.2222 | 0.2000 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_loo_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Route1 | accuracy=0.2000 | 0.2222 | 0.2000 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_route1_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Route2 | accuracy=0.2000 | 0.2222 | 0.2000 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_route2_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-ZScore | accuracy=0.2000 | 0.2121 | 0.2000 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_zscore_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Combo | accuracy=0.1000 | 0.0472 | 0.0400 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_combo_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Fallback | accuracy=0.1000 | 0.0508 | 0.0400 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_fallback_qwen2_vl_20260325_024615.metrics.json` | - |

## cub

- Best primary metric: `RSE-Combo` = `accuracy=0.8650`
- Macro-F1 of best run: `0.8245`
- Top confusions of best run:
  - `American Crow` -> `Gray Catbird`: `1`
  - `Bank Swallow` -> `Cliff Swallow`: `1`
  - `Black capped Vireo` -> `Bay breasted Warbler`: `1`
- Lowest-recall classes of best run:
  - `American Crow`: recall `0.0000`, support `1`
  - `Bank Swallow`: recall `0.0000`, support `1`
  - `Black capped Vireo`: recall `0.0000`, support `1`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| RSE-Combo | accuracy=0.8650 | 0.8245 | 0.8650 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_combo_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Fallback | accuracy=0.8500 | 0.8071 | 0.8458 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_fallback_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-ZScore | accuracy=0.8050 | 0.7642 | 0.8050 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_zscore_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Greedy | accuracy=0.7950 | 0.7412 | 0.7950 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_greedy_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Route1 | accuracy=0.7950 | 0.7440 | 0.7950 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_route1_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Route2 | accuracy=0.7950 | 0.7448 | 0.7950 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_route2_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-LOO | accuracy=0.7850 | 0.7395 | 0.7850 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_loo_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Top1 | accuracy=0.7600 | 0.7015 | 0.7600 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_top1_qwen2_vl_20260325_024615.metrics.json` | - |

## eurosat

- Best primary metric: `RSE-ZScore` = `accuracy=0.7000`
- Macro-F1 of best run: `0.6725`
- Top confusions of best run:
  - `Annual Crop Land` -> `Highway or Road`: `1`
  - `Annual Crop Land` -> `Pasture Land`: `1`
  - `Forest` -> `Permanent Crop Land`: `1`
- Lowest-recall classes of best run:
  - `Permanent Crop Land`: recall `0.0000`, support `5`
  - `Annual Crop Land`: recall `0.6000`, support `5`
  - `Forest`: recall `0.6000`, support `5`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| RSE-ZScore | accuracy=0.7000 | 0.6725 | 0.7000 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_zscore_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Greedy | accuracy=0.6800 | 0.6601 | 0.6800 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_greedy_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-LOO | accuracy=0.6800 | 0.6584 | 0.6800 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_loo_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Combo | accuracy=0.6600 | 0.6189 | 0.6600 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_combo_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Fallback | accuracy=0.6400 | 0.6099 | 0.6400 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_fallback_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Route1 | accuracy=0.6200 | 0.5885 | 0.6200 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_route1_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Top1 | accuracy=0.6000 | 0.5723 | 0.6000 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_top1_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Route2 | accuracy=0.5800 | 0.5538 | 0.5800 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_route2_qwen2_vl_20260325_024615.metrics.json` | - |

## naturalbench_vqa

- Best primary metric: `RSE-LOO` = `accuracy=0.7750`
- Macro-F1 of best run: `0.7915`
- Top confusions of best run:
  - `No` -> `Yes`: `4`
  - `Yes` -> `No`: `3`
  - `A` -> `B`: `1`
- Lowest-recall classes of best run:
  - `No`: recall `0.7143`, support `14`
  - `Yes`: recall `0.7857`, support `14`
  - `A`: recall `0.8333`, support `6`

| Method | Primary | Macro-F1 | Balanced Acc | Metrics | Predictions |
| --- | --- | --- | --- | --- | --- |
| RSE-LOO | accuracy=0.7750 | 0.7915 | 0.7917 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_loo_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Route1 | accuracy=0.7750 | 0.7915 | 0.7917 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_route1_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Route2 | accuracy=0.7750 | 0.7915 | 0.7917 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_route2_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Top1 | accuracy=0.7750 | 0.7891 | 0.7917 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_top1_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-ZScore | accuracy=0.7750 | 0.7915 | 0.7917 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_zscore_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Combo | accuracy=0.7500 | 0.7707 | 0.7738 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_combo_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Fallback | accuracy=0.7500 | 0.7731 | 0.7738 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_fallback_qwen2_vl_20260325_024615.metrics.json` | - |
| RSE-Greedy | accuracy=0.7500 | 0.7707 | 0.7738 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_greedy_qwen2_vl_20260325_024615.metrics.json` | - |
