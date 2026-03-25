# RSE Adaptive Combos and SugarCrepe Analysis

- Expanded suite summary: `/root/autodl-tmp/FGVC/swap/records/rse_expanded_suite_20260325_040658.md`
- Adaptive combo summary: `/root/autodl-tmp/FGVC/swap/records/rse_adaptive_combo_suite_20260325_051526.md`

## Important Clarification

The previously reported `RSE-Adaptive` configuration was already:

- `routing_mode=adaptive`
- `score_normalization=zscore`

So in practice:

> `RSE-Adaptive` in the expanded suite is already `RSE-Adaptive+ZScore`.

The new follow-up experiments therefore focused on the two missing comparisons:

1. `RSE-AdaptiveOnly`
2. `RSE-Adaptive+PCA`

## Adaptive Combo Results

| Dataset | RSE | AdaptiveOnly | Adaptive+ZScore | Adaptive+PCA |
| --- | --- | --- | --- | --- |
| `blink_semantic_correspondence` | 0.2286 | 0.2000 | **0.2857** | 0.2571 |
| `cub` | 0.9700 | 0.9600 | 0.9600 | **0.9700** |
| `eurosat` | 0.5600 | 0.5800 | **0.6600** | 0.6100 |
| `naturalbench_vqa` | 0.7625 | **0.8000** | **0.8000** | 0.7875 |
| `sugarcrepe` | 0.4500 | **0.5000** | 0.4750 | 0.4750 |
| `vlguard` | 0.9250 | 0.9750 | **1.0000** | **1.0000** |

## What This Means

### 1. `Adaptive+ZScore` is still the best overall adaptive configuration

- It remains the strongest mean performer.
- It is especially important on:
  - `blink`
  - `eurosat`
  - `vlguard`

This supports the claim that adaptive routing and score calibration are complementary.

### 2. `AdaptiveOnly` is not enough on the hardest tasks

- `blink`: `0.2000`, below `Adaptive+ZScore` (`0.2857`)
- `eurosat`: `0.5800`, well below `Adaptive+ZScore` (`0.6600`)

So the earlier gain was not “just adaptive routing”. The `zscore` calibration was doing real work.

### 3. `Adaptive+PCA` is safer than `AdaptiveOnly`, but not better than `Adaptive+ZScore`

- It never catastrophically fails.
- It helps on:
  - `blink`
  - `eurosat`
  - `vlguard`
- But it does not beat `Adaptive+ZScore` on the key tasks where adaptive routing matters most.

Practical conclusion:

> If we need one final adaptive configuration, use `Adaptive+ZScore`, not `Adaptive+PCA`.

## SugarCrepe: The Real Diagnosis

The original concern was:

> best component is `0.8125`, but ensemble pair accuracy is only `0.475`, so the ensemble must be drowning it in bad components.

After inspecting the diagnostics, that diagnosis is only partially correct.

### Observation 1: the selected components are **not** weak

For baseline `RSE` on `sugarcrepe`, the 8 selected components have raw validation accuracies:

- `0.75`
- `0.75`
- `0.7625`
- `0.7375`
- `0.7375`
- `0.7250`
- `0.7250`
- `0.7375`

There are **no** selected components below chance (`0.5`), and none are obviously random/noisy.

So the issue is **not** “7 junk components vote against 1 good component”.

### Observation 2: the comparison `0.8125 vs 0.475` is apples-to-oranges

`sugarcrepe` uses the `pair` evaluator:

- `raw_accuracy`: item-level accuracy
- `pair_accuracy`: a pair is correct only if **both** items in the pair are correct

So for `RSE`:

- `raw_accuracy = 0.725`
- `pair_accuracy = 0.450`

This gap is expected under pairwise evaluation.

That means:

> comparing the best component’s `0.8125` raw accuracy directly against ensemble `0.475` pair accuracy is not mathematically valid.

The evaluator definition is in [evaluators.py](/root/autodl-tmp/FGVC/src/evaluate/evaluators.py#L99).

### Observation 3: the real problem is selection mismatch, not random voting

The best raw component is still:

- `layer@14`, raw val accuracy `0.8125`

But it is not selected by:

- baseline `RSE`
- `RSE-Adaptive`
- `RSE-AdaptiveOnly`
- `RSE-Adaptive+PCA`

So the real `sugarcrepe` bottleneck is:

> the selection metric still misses the most pair-useful component.

### Observation 4: for SugarCrepe, `AdaptiveOnly` is actually best

Pair accuracies:

- `RSE = 0.4500`
- `RSE-ZScore = 0.4750`
- `RSE-AdaptiveOnly = 0.5000`
- `RSE-Adaptive+ZScore = 0.4750`
- `RSE-Adaptive+PCA = 0.4750`

This suggests that on pairwise binary consistency tasks:

- adaptive filtering helps,
- but extra calibration may slightly hurt pair-level consistency.

## Final Recommendation

For the paper / next phase:

1. Treat `RSE-Adaptive+ZScore` as the main final method.
2. Keep `RSE-AdaptiveOnly` as an important ablation, because it is best on `sugarcrepe`.
3. Do not frame `sugarcrepe` as “ensemble drowned a great component in random heads”.
4. Frame it as:

> pairwise evaluation amplifies small item-level errors, and current global selection still misses the most pair-useful component.
