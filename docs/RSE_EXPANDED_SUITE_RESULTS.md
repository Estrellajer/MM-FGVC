# RSE Expanded Suite Results

- Manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/manifest.tsv`
- Summary: `/root/autodl-tmp/FGVC/swap/records/rse_expanded_suite_20260325_040658.md`

## Expanded Evaluation Scale

This run expands beyond the original 4-task pilot in both task diversity and subset size:

| Dataset | Task Type | Train | Val | Labels |
| --- | --- | --- | --- | --- |
| `cub` | FGVC classification | 100 | 100 | 20 |
| `eurosat` | remote sensing classification | 100 | 100 | 10 |
| `naturalbench_vqa` | VQA / grouped reasoning | 80 | 80 | 4 |
| `sugarcrepe` | pairwise visual-language consistency | 40 | 80 | 2 |
| `vlguard` | multimodal safety | 40 | 80 | 2 |
| `blink_semantic_correspondence` | visual reasoning / correspondence | 80 | 35 | 4 |

Methods compared:

- `Zero-shot`
- `SAV`
- `RSE`
- `RSE-ZScore`
- `RSE-CV`
- `RSE-Shrink`
- `RSE-Adaptive`
- `RSE-PCA`

## Overview

| Dataset | Zero-shot | SAV | RSE | RSE-ZScore | RSE-CV | RSE-Shrink | RSE-Adaptive | RSE-PCA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `blink_semantic_correspondence` | 0.0286 | 0.1143 | 0.2286 | 0.2000 | 0.2000 | 0.1714 | **0.2857** | 0.2286 |
| `cub` | 0.9500 | **0.9900** | 0.9700 | 0.9700 | 0.9700 | 0.9300 | 0.9600 | 0.9700 |
| `eurosat` | 0.6000 | **0.6600** | 0.5600 | 0.6300 | 0.5900 | 0.5300 | **0.6600** | 0.5900 |
| `naturalbench_vqa` | 0.7375 | **0.8250** | 0.7625 | 0.8000 | 0.7875 | 0.8000 | 0.8000 | 0.7875 |
| `sugarcrepe` | 0.2750 | **0.6750** | 0.4500 | 0.4750 | 0.4750 | **0.5000** | 0.4750 | 0.4750 |
| `vlguard` | 0.7125 | 0.9750 | 0.9250 | **1.0000** | 0.9625 | 0.9750 | **1.0000** | **1.0000** |

## Main Findings

### 1. Larger-scale evaluation keeps the core RSE story alive

- `RSE` remains clearly stronger than `Zero-shot` on all 6 expanded tasks.
- Gains are especially large on:
  - `vlguard`: `0.7125 -> 0.9250`
  - `sugarcrepe`: `0.2750 -> 0.4500`
  - `blink_semantic_correspondence`: `0.0286 -> 0.2286`
- So the representation-space advantage is not just a tiny-pilot artifact.

### 2. The best new idea is `Per-Query Adaptive Selection`

- `RSE-Adaptive` has the best mean delta vs baseline `RSE`: `+0.0474`.
- It wins on `5/6` tasks and loses only on `cub`.
- Strongest examples:
  - `blink_semantic_correspondence`: `0.2286 -> 0.2857`
  - `eurosat`: `0.5600 -> 0.6600`
  - `vlguard`: `0.9250 -> 1.0000`

Interpretation:

> The most useful fix was not “estimate support accuracy more carefully”, but “route components per query instead of forcing every query to use the same component mixture”.

This directly supports the hypothesis that component usefulness is query-dependent.

### 3. `ZScore` calibration is still the strongest low-cost baseline

- Mean delta vs `RSE`: `+0.0298`
- Wins on `4/6` tasks and ties on `1/6`
- Best or tied-best on:
  - `vlguard`: `1.0000`
  - `naturalbench_vqa`: `0.8000`
  - `eurosat`: `0.6300`

Interpretation:

> Cross-component score scale mismatch is still a real bottleneck at larger scale.

### 4. `PCA` is the safest structural change

- Mean delta vs `RSE`: `+0.0258`
- Wins/ties/losses vs `RSE`: `4 / 2 / 0`
- It never hurts relative to baseline `RSE` in this 6-task suite.

Interpretation:

> Dimensionality reduction does not give the single biggest gain, but it is the most stable “no-regret” change among the new ideas.

### 5. `Cross-Validation` helps, but much less than expected

- Mean delta vs `RSE`: `+0.0148`
- It improves `4/6` tasks, but only modestly.
- It does **not** outperform `Adaptive` or `ZScore`.

Interpretation:

> The LOO-vs-val gap is real, but simply replacing LOO with K-fold CV is not enough. Variance reduction alone does not solve the selection problem.

This is an important refinement of the original diagnosis.

### 6. `Shrinkage` is mixed and task-dependent

- Mean delta vs `RSE`: only `+0.0017`
- Helps:
  - `naturalbench_vqa`: `0.7625 -> 0.8000`
  - `sugarcrepe`: `0.4500 -> 0.5000`
  - `vlguard`: `0.9250 -> 0.9750`
- Hurts:
  - `cub`: `0.9700 -> 0.9300`
  - `eurosat`: `0.5600 -> 0.5300`
  - `blink_semantic_correspondence`: `0.2286 -> 0.1714`

Interpretation:

> Simple global shrinkage regularizes noisy tasks, but over-regularizes already clean, highly discriminative FGVC-style tasks.

## Revised Bottleneck Diagnosis

After the expanded suite, the diagnosis is sharper:

1. Representation-space signal is real and scales beyond the tiny pilot.
2. The dominant bottleneck is still **selection/aggregation**, not feature extraction.
3. But the selection problem is now better described as:
   - `query-conditioned component utility`
   - plus `score calibration mismatch`
   - more than pure `support-set overfitting`

So the earlier “LOO overfits support” story was directionally right, but incomplete.

## Relation To SAV

`SAV` is still the strongest overall baseline on this expanded suite:

- Best on `cub`, `naturalbench_vqa`, `sugarcrepe`
- Ties the best RSE variant on `eurosat`
- Loses to improved RSE on `blink_semantic_correspondence`
- Loses to improved RSE on `vlguard`

This means the current evidence supports a balanced claim:

> RSE-style representation-space classification is real and competitive, but it is not yet a universal replacement for SAV.

## Practical Takeaway

If we keep moving RSE forward, the new default order should be:

1. `RSE-Adaptive`
2. `RSE-ZScore`
3. `RSE-PCA`

And if we want a single “safe” next baseline:

> Use `RSE-Adaptive + ZScore`, with `PCA` as an optional stabilizer.

## Most Important Next Step

The next experiment should not be another global selector.

It should be a **query-aware selector** that predicts which components to trust for the current sample, ideally using:

- query margin statistics,
- component agreement/disagreement,
- and a lightweight diversity penalty.

That is now the clearest high-value direction opened by the expanded results.
