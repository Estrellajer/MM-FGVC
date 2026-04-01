# FGVC Simple Frozen-Feature Baselines

- Model: `qwen2_vl`
- Tasks: `eurosat_fgvc`, `pets_fgvc`, `cub_fgvc`
- Seed: `42`
- New methods: `Whitened NCM`, `Ridge Probe`
- New-method summary: `swap/paper/records/fgvc_review_simple_20260401_151500.md`
- Existing reference summary: `swap/paper/records/fgvc_baselines_20260401_122632.md`

## Focused Comparison

| Task | Zero-shot | SAV | Whitened NCM | Ridge Probe | RSEv2 |
| --- | --- | --- | --- | --- | --- |
| eurosat_fgvc | 0.6000 | 0.7100 | 0.6900 | 0.7100 | 0.7600 |
| pets_fgvc | 0.9500 | 0.9700 | 0.9700 | 0.9700 | 0.9600 |
| cub_fgvc | 0.9100 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| mean | 0.8200 | 0.8933 | 0.8867 | 0.8933 | 0.9067 |

## New Baseline Settings

- `Whitened NCM`: final decoder-block feature (`feature_level=layer`, `feature_index=-1`), support covariance whitening with `covariance_shrinkage=auto`, cosine nearest-class-mean scoring after whitening.
- `Ridge Probe`: final decoder-block feature (`feature_level=layer`, `feature_index=-1`), closed-form ridge regression on frozen features with `ridge_lambda=1.0`.

## Summary

- Both simple frozen-feature baselines are competitive, but neither exceeds `RSEv2` on the three-task FGVC mean.
- The clearest separation appears on `eurosat_fgvc`, where `RSEv2` stays above both `Whitened NCM` and `Ridge Probe`.
- On `pets_fgvc` and `cub_fgvc`, all strong frozen-feature methods are near saturation, so the additional reviewer-requested baselines mostly serve as a robustness check rather than a differentiating benchmark.
- This focused table is intended to answer the reviewer concern about missing simple frozen-feature baselines; the broader baseline picture, including `KeCO`, remains in the full FGVC summary.
