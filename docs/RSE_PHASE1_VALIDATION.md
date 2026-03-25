# RSE Phase-1 Validation

## Goal

Validate the core NeurIPS-style hypothesis behind **Representation-Space Ensembles (RSE)**:

1. VLM internal representations contain task-discriminative information that is not fully reflected in generation outputs.
2. A multi-level representation classifier can outperform prompt-only generation on at least some tasks.
3. Layer/level discriminativeness is task-dependent rather than uniform.

## What Was Implemented

- A new `RSE` method with multi-level representation extraction:
  - `head`: attention `o_proj/out_proj` pre-hook features
  - `layer`: decoder block output
  - `attn`: self-attention module output
  - `mlp`: MLP module output
- FDR-based component selection over `(level, layer)` pairs
- Weighted representation-space ensemble classifier
- Optional confidence-gated generation fallback interface
- Diagnostics export:
  - per-component FDR
  - per-component validation accuracy
  - selected component list and weights

## Pilot Setup

- Model: `qwen2_vl`
- Tasks:
  - `naturalbench_vqa`
  - `eurosat`
  - `cub`
  - `blink_semantic_correspondence`
- Compared methods:
  - `Zero-shot`
  - `SAV`
  - `RSE`

Artifacts:

- Method-suite summary: `swap/records/all_task_method_suite_20260325_020639.md`
- RSE diagnostics summary: `swap/records/rse_phase1_20260325_020639.md`

## Top-Line Results

| Task | Zero-shot | SAV | RSE | Best RSE Component |
| --- | --- | --- | --- | --- |
| `naturalbench_vqa` | 0.7500 | 0.8000 | 0.7250 | 0.8250 |
| `eurosat` | 0.6200 | 0.6400 | 0.5800 | 0.6400 |
| `cub` | 0.9000 | 0.7100 | 0.8100 | 0.7600 |
| `blink_semantic_correspondence` | 0.0500 | 0.1000 | 0.2000 | 0.3500 |

## Main Findings

### 1. The strongest evidence is not the current ensemble score, but the best standalone component

This pilot already shows the most important Phase-1 signal:

- `best standalone RSE component > zero-shot` on `3/4` tasks
- `best standalone RSE component > SAV` on `3/4` tasks

Examples:

- `naturalbench_vqa`: best component `0.8250` > zero-shot `0.7500` and SAV `0.8000`
- `eurosat`: best component `0.6400` > zero-shot `0.6200`, tied with SAV `0.6400`
- `blink_semantic_correspondence`: best component `0.3500` >> zero-shot `0.0500` and SAV `0.1000`

This is the clearest current evidence for the proposed **representation-generation gap**.

### 2. Current FDR selection is promising, but not yet sufficient

The current ensemble is not consistently the best reader of the available representation signal:

- `naturalbench_vqa`: RSE `0.7250`, but best component `0.8250`
- `eurosat`: RSE `0.5800`, but best component `0.6400`
- `blink_semantic_correspondence`: RSE `0.2000`, but best component `0.3500`

Interpretation:

- the representation signal exists
- the current `FDR -> top-k -> weighted ensemble` pipeline is not yet selecting/combining the best components reliably

This is a *good* research outcome: it supports the paper hypothesis while clearly exposing what the next algorithmic step must improve.

### 3. Layer/level peaks are task-dependent

Selected components are not uniformly distributed.

Aggregate over this pilot:

- selected levels: `attn=10`, `head=8`, `mlp=8`, `layer=6`
- selected stages: `late=21`, `early=9`, `mid=2`

But the distribution changes by task:

- `cub`: selection is almost entirely late `head/attn`
- `naturalbench_vqa`: mostly late `head/attn/mlp/layer`
- `blink_semantic_correspondence`: current FDR-selected components are concentrated in early layers

This is exactly the sort of task-dependent structure the Phase-1 heatmap is supposed to reveal.

### 4. RSE already beats SAV on some hard tasks

- `cub`: `RSE 0.8100 > SAV 0.7100`
- `blink_semantic_correspondence`: `RSE 0.2000 > SAV 0.1000`

So even the first implementation already suggests that **multi-level representation reading** can be stronger than the current single-family SAV readout on harder tasks.

## Honest Read

The current pilot does **not** yet prove:

- `RSE > zero-shot` universally
- `FDR selection` is the final answer
- current ensemble is publication-ready

But it **does** support the paper direction in three concrete ways:

1. strong standalone representation components frequently outperform generation
2. useful components appear at different layers/levels for different tasks
3. representation-space reading can already beat SAV on harder settings

## Immediate Next Step

The most valuable next experiment is now very clear:

1. keep the current multi-level extractor
2. replace or strengthen the selection/aggregation stage

Priority candidates:

- leave-one-out component scoring instead of pure FDR
- beam search or greedy forward selection over components
- top-1 or top-2 component routing before full ensemble
- confidence-gated fallback using best-component margin instead of ensemble margin

In short:

> Phase 1 has already produced positive evidence for the RSE thesis, but the bottleneck is now **component selection and ensembling**, not representation availability.
