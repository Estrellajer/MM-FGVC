# Representation-Generation Gap: Table Replacement for Figure 2

This note converts the original heatmap-based Figure 2 into a compact table-based presentation.

Scope:
- Data source: `swap/paper/outputs/20260325_095501_c1_main_results/manifest.tsv`
- Setting: `qwen2_vl`, `seed42`
- This matches the setup used by `swap/paper/scripts/generate_heatmap.py`

## Recommendation

Do not convert the heatmap into a dense matrix table. That keeps the visual burden but loses readability.

A better replacement is:
- one global summary row for F1: the gap exists
- one 6-row representative table for F2: peak components are task-dependent
- one sentence in the text for the full 26-task statistics
- optionally move the full 26-task table to appendix/supplement

## Main-Text Table A: Gap Summary

Suggested caption:

> **Representation-generation gap summary on 26 tasks.** Under the same setting as Figure 2 (`qwen2_vl`, seed 42), the best standalone representation component outperforms zero-shot on 24/26 tasks, while the oracle upper bound improves over zero-shot on 25/26 tasks.

| Setting | Best component > ZS | Oracle > ZS | Mean Δ(Best-ZS) | Mean Δ(Oracle-ZS) |
| --- | --- | --- | --- | --- |
| Qwen2-VL, 26 tasks, seed 42 | 24/26 | 25/26 | +41.8 pts | +60.2 pts |

Additional compact evidence if you want one more sentence:
- Peak level counts: `Attn=10`, `Layer=8`, `Head=5`, `MLP=3`
- Peak stage counts: `Early=1`, `Mid=6`, `Late=19`

## Main-Text Table B: Representative Peak Components

Suggested caption:

> **Representative task-dependent peaks across domains.** The most discriminative component varies by task, spanning different representation levels and decoder layers.

| Task | ZS | Best Comp. | Oracle | Peak (Level-Layer) | Δ(Best-ZS) |
| --- | --- | --- | --- | --- | --- |
| CUB-200 (FGVC) | 90.0 | 92.0 | 99.0 | Layer-24 | +2.0 |
| Flowers-102 (FGVC) | 89.2 | 99.0 | 100.0 | Head-0 | +9.8 |
| EuroSAT (Remote Sensing) | 62.0 | 62.0 | 96.0 | Layer-23 | +0.0 |
| VLGuard (Safety) | 72.5 | 100.0 | 100.0 | Layer-17 | +27.5 |
| NaturalBench (VQA) | 75.0 | 80.0 | 100.0 | MLP-21 | +5.0 |
| BLINK Depth (Reasoning) | 0.0 | 85.0 | 100.0 | Layer-21 | +85.0 |

Why this table works better than the heatmap in the main text:
- it preserves F1 with direct gap statistics
- it preserves F2 with explicit peak locations
- it is much smaller than a 6-panel figure
- it avoids spending space on per-layer color variation that the paper does not analyze in detail

## Suggested Paragraph for Section 5.5

You can replace the heatmap discussion with the paragraph below.

> To quantify the representation-generation gap, we compare zero-shot generation against the best standalone representation component selected from decoder layers and representation levels. Under the same setting as Figure 2 (`qwen2_vl`, seed 42), the best component already outperforms zero-shot on 24 of 26 tasks, with an average gain of 41.8 percentage points. Moreover, an oracle that selects the correct component prediction per sample improves over zero-shot on 25 of 26 tasks, with an average gain of 60.2 points. These results indicate that task-relevant discriminative information is often present in intermediate representations even when end-to-end generation fails to fully exploit it. Importantly, the peak component is task-dependent rather than fixed: Flowers peaks at `Head-0`, VLGuard at `Layer-17`, NaturalBench at `MLP-21`, and BLINK Depth at `Layer-21`, showing that different tasks rely on different representation levels and decoder depths.

Shorter version if space is very tight:

> The best standalone component beats zero-shot generation on 24/26 tasks (+41.8 pts on average), while the oracle upper bound improves on 25/26 tasks (+60.2 pts). Peak components are task-dependent, spanning different levels and layers, e.g., `Head-0` for Flowers, `Layer-17` for VLGuard, and `MLP-21` for NaturalBench.

## Appendix Table: Full 26-Task Peak Summary

Suggested caption:

> **Full peak-component summary for the Figure 2 setting.** `Best Comp.` denotes the validation accuracy of the best individual representation component; `RSEv2` is the final aggregated method.

| Task | ZS | Best Comp. | RSEv2 | Peak (Level-Layer) | Δ(Best-ZS) | Δ(Oracle-ZS) |
| --- | --- | --- | --- | --- | --- | --- |
| blink_art_style | 0.0 | 85.0 | 80.0 | attn-25 | +85.0 | +100.0 |
| blink_counting | 15.0 | 90.0 | 80.0 | attn-26 | +75.0 | +85.0 |
| blink_forensic_detection | 0.0 | 85.0 | 55.0 | attn-22 | +85.0 | +100.0 |
| blink_functional_correspondence | 0.0 | 68.8 | 68.8 | attn-16 | +68.8 | +93.8 |
| blink_iq_test | 10.0 | 40.0 | 25.0 | layer-27 | +30.0 | +80.0 |
| blink_jigsaw | 20.0 | 80.0 | 70.0 | attn-20 | +60.0 | +80.0 |
| blink_multi_view_reasoning | 0.0 | 85.0 | 75.0 | mlp-11 | +85.0 | +100.0 |
| blink_object_localization | 0.0 | 75.0 | 55.0 | attn-18 | +75.0 | +100.0 |
| blink_relative_depth | 0.0 | 85.0 | 80.0 | layer-21 | +85.0 | +100.0 |
| blink_relative_reflectance | 0.0 | 73.3 | 66.7 | mlp-27 | +73.3 | +100.0 |
| blink_semantic_correspondence | 0.0 | 45.0 | 30.0 | attn-14 | +45.0 | +100.0 |
| blink_spatial_relation | 75.0 | 90.0 | 85.0 | attn-27 | +15.0 | +25.0 |
| blink_visual_correspondence | 10.0 | 35.0 | 30.0 | head-22 | +25.0 | +80.0 |
| blink_visual_similarity | 40.0 | 85.0 | 65.0 | attn-21 | +45.0 | +60.0 |
| cub_small | 90.0 | 92.0 | 92.0 | layer-24 | +2.0 | +9.0 |
| eurosat_small | 62.0 | 62.0 | 64.0 | layer-23 | +0.0 | +34.0 |
| flowers_small | 89.2 | 99.0 | 99.0 | head-0 | +9.8 | +10.8 |
| mhalubench_val_v01 | 75.0 | 95.0 | 85.0 | head-23 | +20.0 | +22.5 |
| mhalubench_val_v02 | 77.5 | 95.0 | 85.0 | head-23 | +17.5 | +20.0 |
| naturalbench_ret | 70.0 | 100.0 | 100.0 | layer-20 | +30.0 | +30.0 |
| naturalbench_vqa | 75.0 | 80.0 | 75.0 | mlp-21 | +5.0 | +25.0 |
| pets_small | 100.0 | 100.0 | 100.0 | layer-26 | +0.0 | +0.0 |
| sugarcrepe | 35.0 | 80.0 | 55.0 | attn-23 | +45.0 | +65.0 |
| tinyimage_small | 64.0 | 65.5 | 61.0 | layer-25 | +1.5 | +21.0 |
| vizwiz | 0.0 | 77.5 | 70.0 | head-14 | +77.5 | +97.5 |
| vlguard | 72.5 | 100.0 | 97.5 | layer-17 | +27.5 | +27.5 |

## Writing Guidance

If you remove the heatmap, the safest restructuring is:

1. Keep the section title unchanged.
2. Replace “Figure 2 shows ...” with “Table X summarizes ...”.
3. Put the global 26-task statistics first to support F1.
4. Use the 6-row representative table to support F2.
5. Move the full 26-task table to appendix if reviewers ask for completeness.

## Important Caveat

These numbers are from the same single-seed setting used by the current heatmap (`qwen2_vl`, `seed42`), not from a 3-seed average. If you want the table to align exactly with the main results table, you should either:
- explicitly note that this is the Figure 2 setting, or
- rerun the summary using a multi-seed aggregation protocol
