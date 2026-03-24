# Swap Workspace

This directory is reserved for disposable batch-experiment artifacts.

## Layout

- `swap/subsets/`: generated smoke-test subsets cut from full annotations
- `swap/logs/`: per-run stdout/stderr logs
- `swap/outputs/`: evaluation metrics written by `main.py`
- `swap/records/`: Markdown experiment summaries

## Recommended Usage

Run the batch subset smoke test with:

```bash
bash scripts/run_full_data_train.sh
```

Useful overrides:

```bash
DATASET_FILTER='pets|blink_art_style' METHODS='zero_shot sav' bash scripts/run_full_data_train.sh
DO_RUN=0 bash scripts/run_full_data_train.sh
RUN_MODE=full_matrix bash scripts/run_full_data_train.sh
```
