# smoke_review_baselines Summary

- Manifest: `/root/autodl-tmp/FGVC/swap/paper/outputs/_smoke_review_baselines/manifest.tsv`
- Timestamp: ``
- Models: `qwen2_vl`
- Tasks: `pets_small`

## qwen2_vl

| Task | Whitened NCM | Ridge Probe |
| --- | --- | --- |
| pets_small | 0.9714±0.0000 | 0.9571±0.0000 |

| Method | Mean Primary | Mean Fit (s) | Mean Pred (ms/sample) | Runs |
| --- | --- | --- | --- | --- |
| Whitened NCM | 0.9714±0.0000 | 13.3712 | 66.2028 | 1 |
| Ridge Probe | 0.9571±0.0000 | 10.9759 | 55.3399 | 1 |
