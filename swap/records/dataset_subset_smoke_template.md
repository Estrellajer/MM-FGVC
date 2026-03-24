# Dataset Subset Smoke Template

## Goal

Use small, representative subsets to verify that each dataset can be loaded, normalized, prompted, and optionally executed through `main.py`.

## Suggested Command

```bash
bash scripts/run_full_data_train.sh
```

## Suggested Deep Pass

```bash
METHODS='zero_shot sav' DATASET_FILTER='naturalbench|sugarcrepe|blink' bash scripts/run_full_data_train.sh
```

## Notes

- `DO_RUN=0` keeps the pass at dataset validation only.
- Generated Markdown reports are written to `swap/records/`.
- Generated subset files are written to `swap/subsets/`.

## Result Checklist

| Check | Status | Notes |
| --- | --- | --- |
| Subset generation succeeds | TODO |  |
| Dataset normalization succeeds | TODO |  |
| Image paths resolve locally | TODO |  |
| Zero-shot smoke run succeeds | TODO |  |
| SAV smoke run succeeds | TODO |  |
