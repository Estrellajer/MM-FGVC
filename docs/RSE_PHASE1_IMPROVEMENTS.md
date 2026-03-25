# RSE Phase-1 Improvements

- Base pilot manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_020639_all_task_method_suite/manifest.tsv`
- Improvement manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_024615_rse_phase1_improvements/manifest.tsv`
- Merged summary: `/root/autodl-tmp/FGVC/swap/records/rse_improvement_suite_20260325_024615.md`
- Improvement-only summary: `/root/autodl-tmp/FGVC/swap/records/rse_phase1_improvements_20260325_024615.md`

## What We Tried

On the same 4-task Phase-1 pilot, I ran 8 RSE variants:

1. `RSE-LOO`: replace FDR selection with leave-one-out support accuracy.
2. `RSE-Top1`: use only the single best LOO-ranked component.
3. `RSE-Greedy`: greedy forward subset selection on support accuracy.
4. `RSE-ZScore`: z-normalize per-component class scores before ensembling.
5. `RSE-Fallback`: enable confidence-gated zero-shot fallback.
6. `RSE-Route1`: query-time routing to the top-1 selected component.
7. `RSE-Route2`: query-time routing to the top-2 selected components.
8. `RSE-Combo`: `LOO + Greedy + ZScore + Route2 + Fallback`.

## Headline Results

| Dataset | Zero-shot | SAV | RSE | Best Variant | Score |
| --- | --- | --- | --- | --- | --- |
| blink_semantic_correspondence | 0.0500 | 0.1000 | 0.2000 | `RSE-Top1` / `RSE-Greedy` | `0.2500` |
| cub | 0.9000 | 0.7100 | 0.8100 | `RSE-Combo` | `0.8650` |
| eurosat | 0.6200 | 0.6400 | 0.5800 | `RSE-ZScore` | `0.7000` |
| naturalbench_vqa | 0.7500 | 0.8000 | 0.7250 | `RSE-LOO` / `RSE-Top1` / `RSE-Route1` / `RSE-Route2` / `RSE-ZScore` | `0.7750` |

## What Actually Helped

### 1. `LOO` selection helps, but it does **not** fully solve component selection

- Mean delta vs original `RSE`: `+0.0313`
- Wins/ties/losses vs original `RSE`: `2 / 1 / 1`
- Strong gains on:
  - `eurosat`: `0.58 -> 0.68`
  - `naturalbench_vqa`: `0.725 -> 0.775`
- But still fails to recover the true best val component on some tasks:
  - `naturalbench_vqa`: best val component is still `head@17 = 0.825`, but `RSE-LOO` does **not** select it.
  - `blink`: best val component is `attn@27 = 0.35`, but `RSE-Top1` still picks `attn@6 = 0.25`.

Takeaway: replacing FDR with LOO is directionally right, but the support-set proxy is still misaligned with true validation utility.

### 2. `Top1` validates the "too many mediocre components hurt" hypothesis

- Mean delta vs original `RSE`: `+0.0175`
- Biggest signal:
  - `blink`: `0.20 -> 0.25`
- But not universal:
  - `cub`: `0.81 -> 0.76`
  - `eurosat`: `0.58 -> 0.60`

Takeaway: over-aggregation is a real problem on some tasks, but "single best component" is not a universal answer.

### 3. `Greedy` subset selection is useful when redundancy is the main issue

- Mean delta vs original `RSE`: `+0.0400`
- Strong gains:
  - `blink`: `0.20 -> 0.25`
  - `eurosat`: `0.58 -> 0.68`
- Selected only `1` component on `blink`, `3` on `eurosat`, `6` on `cub`, `3` on `naturalbench`.

Takeaway: "less but cleaner" often beats a fixed 8-component ensemble, but greedy selection can still overfit to support accuracy and does not dominate on every task.

### 4. `ZScore` calibration is the strongest single low-cost change

- Mean delta vs original `RSE`: `+0.0413`
- Best no-fallback variant overall.
- Biggest gain:
  - `eurosat`: `0.58 -> 0.70`
- Also matches the best improved score on `naturalbench` (`0.775`).

Takeaway: score-scale mismatch across components is a real bottleneck. Calibration mattered more than query-time routing.

### 5. `Fallback` is highly task-dependent

- Mean delta vs original `RSE`: only `+0.0062`
- Helps when zero-shot is already strong:
  - `cub`: `0.81 -> 0.85`, with fallback used on `39/200` samples
  - `eurosat`: `0.58 -> 0.64`, with fallback used on `15/50` samples
- Hurts when zero-shot is weak:
  - `blink`: `0.20 -> 0.10`, with fallback used on `8/20` samples

Takeaway: fallback is a useful safety valve only if the underlying zero-shot model is trustworthy on that task.

### 6. `Route1 / Route2` did not emerge as a robust fix

- `Route1` mean delta vs original `RSE`: `+0.0188`
- `Route2` mean delta vs original `RSE`: `+0.0088`
- On `naturalbench`, routing ties the improved plateau (`0.775`).
- On `eurosat`, `Route2` falls back to baseline `RSE` (`0.58`).

Takeaway: query-time routing among a noisy selected pool is not enough. The selected pool itself still matters more.

### 7. `Combo` can be best, but it is brittle

- Mean delta vs original `RSE`: `+0.0150`
- Best result on `cub`: `0.865`
- Good on `eurosat`: `0.66`
- Bad on `blink`: `0.10`
- Mild on `naturalbench`: `0.75`

Takeaway: combining every idea does not produce a universally strong method. The gains mostly come from task-dependent interactions, especially with fallback.

## Main Scientific Conclusion

The overall story is now sharper:

1. There is real signal in representation space.
   - We repeatedly beat the original `RSE`, and on `eurosat` reached `0.70`, exceeding both zero-shot (`0.62`) and original `RSE` (`0.58`).
2. The main bottleneck is still **component selection / aggregation**, not feature extraction.
   - `LOO` helps but still misses the best validation component on some tasks.
   - `Top1` helps on `blink`, which directly supports the "too many weak components hurt" diagnosis.
3. Calibration matters more than routing.
   - `ZScore` is the most reliable single improvement.
4. Fallback is not universally safe.
   - It behaves more like a task-dependent mixture-of-experts with zero-shot, not a free lunch.

## Practical Recommendation For Phase 2

- Use `RSE-ZScore` as the default improved baseline.
- Keep `RSE-LOO` and `RSE-Greedy` as ablations for selection quality.
- Use `RSE-Combo` only when zero-shot is known to be strong enough to serve as a good fallback expert.
- Do **not** claim the selection problem is solved yet.

## Next Research Step

The strongest next step is not "more heuristics on top-k". It is a better selector:

- query-conditioned component routing that uses the query's own feature statistics,
- or a small meta-selector trained only on support features,
- or a validation-free selector that optimizes agreement, diversity, and margin jointly.

Current evidence supports a precise claim:

> Representation-space signal is real, but simple training-free selection rules are still the limiting factor.
