# RSEv2 Mahalanobis Results

- Base expanded-suite manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_040658_rse_expanded_suite/manifest.tsv`
- RSEv2 manifest: `/root/autodl-tmp/FGVC/swap/outputs/20260325_055700_rsev2_suite/manifest.tsv`
- Combined summary: `/root/autodl-tmp/FGVC/swap/records/rsev2_suite_20260325_055700.md`

## What Changed

`RSEv2` replaces the earlier multi-trick `RSE-Adaptive` pipeline with one cleaner formulation:

1. Extract multi-level decoder representations with one forward pass.
2. For each `(level, layer)` component, classify with a regularized Mahalanobis score:

   `score_c(x) = -0.5 * (x - mu_c)^T Sigma_reg^{-1} (x - mu_c)`

3. Aggregate all components with per-query confidence weights:

   `w_k(x) = max(margin_k(x), 0)`

In the current implementation, `Sigma_reg` uses automatic shrinkage toward a scaled identity target, computed in closed form without adding a new dependency.

## Main Result

On the same 6-task expanded suite, `RSEv2` is the new strongest RSE-family method overall.

| Dataset | SAV | RSE-Adaptive | RSEv2 |
| --- | --- | --- | --- |
| `cub` | 0.9900 | 0.9600 | **1.0000** |
| `eurosat` | 0.6600 | 0.6600 | **0.7600** |
| `naturalbench_vqa` | **0.8250** | 0.8000 | **0.8250** |
| `sugarcrepe` | 0.6750 | 0.4750 | **0.7250** |
| `vlguard` | 0.9750 | **1.0000** | 0.9750 |
| `blink_semantic_correspondence` | 0.1143 | **0.2857** | 0.2286 |

Aggregate comparison:

- Mean delta vs `RSE`: `+0.1029`
- Mean delta vs `RSE-Adaptive`: `+0.0555`
- Mean delta vs `SAV`: `+0.0457`
- Wins / ties / losses vs `RSE-Adaptive`: `4 / 0 / 2`
- Wins / ties / losses vs `SAV`: `4 / 2 / 0`

## Key Findings

### 1. The cleaner formulation actually worked

This was not just a stylistic simplification.

`RSEv2` improved over the old `RSE` on `5/6` tasks and matched it on the remaining one. More importantly, it also improved over `RSE-Adaptive` on the four tasks where representation-space classification matters most:

- `cub`: `0.9600 -> 1.0000`
- `eurosat`: `0.6600 -> 0.7600`
- `naturalbench_vqa`: `0.8000 -> 0.8250`
- `sugarcrepe`: `0.4750 -> 0.7250`

This is strong evidence that the previous gains from `ZScore + PCA + Shrinkage` were not three separate tricks after all. A large fraction of that benefit can be absorbed into a single regularized Mahalanobis classifier.

### 2. SugarCrepe is no longer a failure case

This is the clearest win.

- Previous best RSE-family result: `0.5000` (`RSE-Shrink`)
- Previous `RSE-Adaptive`: `0.4750`
- `SAV`: `0.6750`
- `RSEv2`: `0.7250`

And the raw item-level accuracy also rose to `0.8625`.

So the earlier pairwise collapse was not an unavoidable property of ensemble methods. It was largely a consequence of the old cosine-selection pipeline. Once the classifier moved to a regularized Mahalanobis geometry, the pairwise task became one of the best `RSE` success cases.

### 3. RSEv2 did not fully replace query-aware routing

Two tasks still resist the simplification:

- `blink_semantic_correspondence`: `RSEv2 0.2286 < RSE-Adaptive 0.2857`
- `vlguard`: `RSEv2 0.9750 < RSE-Adaptive 1.0000`

Interpretation:

> For FGVC, binary pair consistency, and moderate multi-class reasoning, the main problem was geometry/calibration. For harder correspondence-style or safety-style tasks, query-conditioned routing is still carrying signal that a pure all-component Mahalanobis ensemble does not capture.

## Diagnostics Notes

Three diagnostics are especially informative:

1. `eurosat`
   - Best RSEv2 component became `layer@25`, val accuracy `0.7600`
   - This is well above the old best `RSE-Adaptive` component range

2. `naturalbench_vqa`
   - Best RSEv2 component is `mlp@18`, val accuracy `0.8125`
   - Final ensemble reaches `0.8250`, matching `SAV`

3. `sugarcrepe`
   - Best RSEv2 component is `attn@24`, raw val accuracy `0.8625`
   - Unlike old RSE, the ensemble no longer misses the strongest pair-useful region of representation space

## Updated Takeaway

The previous concern was:

> `RSE-Adaptive` works, but it looks like a stack of engineering tricks.

After this run, the cleaner story is viable:

> A multi-level VLM ensemble becomes much stronger when every component is interpreted in the same regularized Mahalanobis geometry and combined by per-query confidence.

That is a clearer method story, with fewer moving parts and better average performance.

## Recommended Paper Positioning

If this line becomes the main paper method, the framing should now be:

1. `RSE` Phase 1 established the representation-generation gap.
2. `RSE-Adaptive` showed that query-conditioned routing matters.
3. `RSEv2` demonstrates that most of the practical gain can be recovered by a single statistically grounded classifier, without heuristic top-k selection.

The remaining open question is no longer “how many tricks do we need?” but:

> when is geometry enough, and when do we still need explicit query-aware routing?
