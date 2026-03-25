# RSE Improvement Suite

- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/manifest.tsv`
- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/manifest.tsv`

## Overview

| Dataset | Zero-shot | SAV | RSE | RSE-LOO | RSE-Top1 | RSE-Greedy | RSE-ZScore | RSE-Fallback | RSE-Route1 | RSE-Route2 | RSE-Combo |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blink_semantic_correspondence | 0.0500 | 0.1000 | 0.2000 | 0.2000 | 0.2500 | 0.2500 | 0.2000 | 0.1000 | 0.2000 | 0.2000 | 0.1000 |
| cub | 0.9000 | 0.7100 | 0.8100 | 0.7850 | 0.7600 | 0.7950 | 0.8050 | 0.8500 | 0.7950 | 0.7950 | 0.8650 |
| eurosat | 0.6200 | 0.6400 | 0.5800 | 0.6800 | 0.6000 | 0.6800 | 0.7000 | 0.6400 | 0.6200 | 0.5800 | 0.6600 |
| naturalbench_vqa | 0.7500 | 0.8000 | 0.7250 | 0.7750 | 0.7750 | 0.7500 | 0.7750 | 0.7500 | 0.7750 | 0.7750 | 0.7500 |

## Aggregate Deltas vs RSE

| Method | Mean Delta vs RSE | Wins | Ties | Losses | Mean Delta vs Zero-shot | Mean Delta vs SAV |
| --- | --- | --- | --- | --- | --- | --- |
| RSE-LOO | +0.0313 | 2 | 1 | 1 | +0.0300 | +0.0475 |
| RSE-Top1 | +0.0175 | 3 | 0 | 1 | +0.0163 | +0.0337 |
| RSE-Greedy | +0.0400 | 3 | 0 | 1 | +0.0388 | +0.0563 |
| RSE-ZScore | +0.0413 | 2 | 1 | 1 | +0.0400 | +0.0575 |
| RSE-Fallback | +0.0062 | 3 | 0 | 1 | +0.0050 | +0.0225 |
| RSE-Route1 | +0.0188 | 2 | 1 | 1 | +0.0175 | +0.0350 |
| RSE-Route2 | +0.0088 | 1 | 2 | 1 | +0.0075 | +0.0250 |
| RSE-Combo | +0.0150 | 3 | 0 | 1 | +0.0138 | +0.0312 |

## blink_semantic_correspondence

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.0500 | -0.1500 | - | - | - | - | - |
| SAV | accuracy=0.1000 | -0.1000 | - | - | - | - | - |
| RSE | accuracy=0.2000 | +0.0000 | - | 8 | - | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/blink_semantic_correspondence_rse_qwen2_vl_20260325_020639.diagnostics.json` |
| RSE-LOO | accuracy=0.2000 | +0.0000 | attn@27 (0.3500) | 8 | 0.3250 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_loo_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Top1 | accuracy=0.2500 | +0.0500 | attn@27 (0.3500) | 1 | 0.3750 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_top1_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Greedy | accuracy=0.2500 | +0.0500 | attn@27 (0.3500) | 1 | 0.3750 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_greedy_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-ZScore | accuracy=0.2000 | +0.0000 | attn@27 (0.3500) | 8 | 0.3250 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_zscore_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Fallback | accuracy=0.1000 | -0.1000 | attn@27 (0.3500) | 8 | 0.3250 | 8 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_fallback_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Route1 | accuracy=0.2000 | +0.0000 | attn@27 (0.3500) | 8 | 0.3000 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_route1_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Route2 | accuracy=0.2000 | +0.0000 | attn@27 (0.3500) | 8 | 0.3000 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_route2_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Combo | accuracy=0.1000 | -0.1000 | attn@27 (0.3500) | 2 | 0.4125 | 7 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/blink_semantic_correspondence_rse_combo_qwen2_vl_20260325_024615.diagnostics.json` |

## cub

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.9000 | +0.0900 | - | - | - | - | - |
| SAV | accuracy=0.7100 | -0.1000 | - | - | - | - | - |
| RSE | accuracy=0.8100 | +0.0000 | - | 8 | - | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/cub_rse_qwen2_vl_20260325_020639.diagnostics.json` |
| RSE-LOO | accuracy=0.7850 | -0.0250 | attn@26 (0.7600) | 8 | 0.6950 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_loo_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Top1 | accuracy=0.7600 | -0.0500 | attn@26 (0.7600) | 1 | 0.6900 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_top1_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Greedy | accuracy=0.7950 | -0.0150 | attn@26 (0.7600) | 6 | 0.7350 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_greedy_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-ZScore | accuracy=0.8050 | -0.0050 | attn@26 (0.7600) | 8 | 0.6850 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_zscore_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Fallback | accuracy=0.8500 | +0.0400 | attn@26 (0.7600) | 8 | 0.6950 | 39 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_fallback_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Route1 | accuracy=0.7950 | -0.0150 | attn@26 (0.7600) | 8 | 0.7225 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_route1_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Route2 | accuracy=0.7950 | -0.0150 | attn@26 (0.7600) | 8 | 0.7325 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_route2_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Combo | accuracy=0.8650 | +0.0550 | attn@26 (0.7600) | 2 | 0.7250 | 47 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/cub_rse_combo_qwen2_vl_20260325_024615.diagnostics.json` |

## eurosat

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.6200 | +0.0400 | - | - | - | - | - |
| SAV | accuracy=0.6400 | +0.0600 | - | - | - | - | - |
| RSE | accuracy=0.5800 | +0.0000 | - | 8 | - | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/eurosat_rse_qwen2_vl_20260325_020639.diagnostics.json` |
| RSE-LOO | accuracy=0.6800 | +0.1000 | head@21 (0.6400) | 8 | 0.6000 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_loo_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Top1 | accuracy=0.6000 | +0.0200 | head@21 (0.6400) | 1 | 0.5700 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_top1_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Greedy | accuracy=0.6800 | +0.1000 | head@21 (0.6400) | 3 | 0.6400 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_greedy_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-ZScore | accuracy=0.7000 | +0.1200 | head@21 (0.6400) | 8 | 0.6100 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_zscore_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Fallback | accuracy=0.6400 | +0.0600 | head@21 (0.6400) | 8 | 0.6000 | 15 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_fallback_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Route1 | accuracy=0.6200 | +0.0400 | head@21 (0.6400) | 8 | 0.6200 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_route1_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Route2 | accuracy=0.5800 | +0.0000 | head@21 (0.6400) | 8 | 0.6200 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_route2_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Combo | accuracy=0.6600 | +0.0800 | head@21 (0.6400) | 4 | 0.6200 | 13 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/eurosat_rse_combo_qwen2_vl_20260325_024615.diagnostics.json` |

## naturalbench_vqa

| Method | Primary | Delta vs RSE | Best Component | Selected | Train Select Acc | Fallback Used | Diagnostics |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Zero-shot | accuracy=0.7500 | +0.0250 | - | - | - | - | - |
| SAV | accuracy=0.8000 | +0.0750 | - | - | - | - | - |
| RSE | accuracy=0.7250 | +0.0000 | - | 8 | - | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/naturalbench_vqa_rse_qwen2_vl_20260325_020639.diagnostics.json` |
| RSE-LOO | accuracy=0.7750 | +0.0500 | head@17 (0.8250) | 8 | 0.7375 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_loo_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Top1 | accuracy=0.7750 | +0.0500 | head@17 (0.8250) | 1 | 0.7625 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_top1_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Greedy | accuracy=0.7500 | +0.0250 | head@17 (0.8250) | 3 | 0.8000 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_greedy_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-ZScore | accuracy=0.7750 | +0.0500 | head@17 (0.8250) | 8 | 0.7500 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_zscore_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Fallback | accuracy=0.7500 | +0.0250 | head@17 (0.8250) | 8 | 0.7375 | 15 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_fallback_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Route1 | accuracy=0.7750 | +0.0500 | head@17 (0.8250) | 8 | 0.7625 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_route1_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Route2 | accuracy=0.7750 | +0.0500 | head@17 (0.8250) | 8 | 0.7625 | 0 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_route2_qwen2_vl_20260325_024615.diagnostics.json` |
| RSE-Combo | accuracy=0.7500 | +0.0250 | head@17 (0.8250) | 3 | 0.7875 | 6 | `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/naturalbench_vqa_rse_combo_qwen2_vl_20260325_024615.diagnostics.json` |
