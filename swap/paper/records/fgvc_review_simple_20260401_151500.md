# fgvc_review_simple Summary

- Manifest: `/root/autodl-tmp/FGVC/swap/paper/outputs/20260401_151500_fgvc_review_simple/manifest.tsv`
- Timestamp: `20260401_151500`
- Models: `qwen2_vl`
- Tasks: `eurosat_fgvc, pets_fgvc, cub_fgvc`

## qwen2_vl

| Task | Whitened NCM | Ridge Probe |
| --- | --- | --- |
| eurosat_fgvc | 0.6900±0.0000 | 0.7100±0.0000 |
| pets_fgvc | 0.9700±0.0000 | 0.9700±0.0000 |
| cub_fgvc | 1.0000±0.0000 | 1.0000±0.0000 |

| Method | Mean Primary | Mean Fit (s) | Mean Pred (ms/sample) | Runs |
| --- | --- | --- | --- | --- |
| Whitened NCM | 0.8867±0.1396 | 6.4149 | 53.8393 | 3 |
| Ridge Probe | 0.8933±0.1302 | 6.0249 | 52.6957 | 3 |
