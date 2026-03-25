# Component Peak Summary

- Manifest: `/root/autodl-tmp/FGVC/swap/paper/outputs/20260325_100000_smoke_paper/manifest.tsv`
- Export CSV: `/root/autodl-tmp/FGVC/swap/paper/records/smoke_paper_20260325_100000_components.csv`

- Selected level totals: `attn`=84, `head`=84, `layer`=84, `mlp`=84
- Selected stage totals: `early`=120, `late`=108, `mid`=108

| Model | Task | Method | Best Level | Best Layer | Best Val Acc |
| --- | --- | --- | --- | --- | --- |
| qwen2_vl | cub_fgvc | RSEv2 | layer | 26 | 1.0000 |
| qwen2_vl | eurosat_fgvc | RSEv2 | head | 21 | 0.7700 |
| qwen2_vl | pets_fgvc | RSEv2 | head | 22 | 0.9800 |
