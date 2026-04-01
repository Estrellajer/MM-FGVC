# Paper Revision Checklist And Supplement (2026-04-01)

This note summarizes the current manuscript risks against reviewer-style objections and pairs each item with the smallest concrete revision action. It also records the experiment evidence currently available in the repo, including new runs completed on 2026-04-01.

## Evidence Used

- Current manuscript: `main.tex`
- Main new FGVC baseline run: `swap/paper/records/fgvc_baselines_20260401_122632.md`
- Reviewer-targeted simple FGVC baselines: `swap/paper/records/fgvc_review_simple_comparison_20260401_151500.md`
- Existing RSEv2 ablation run: `swap/paper/records/c2_ablations_20260326_144544.md`
- Existing efficiency run: `swap/paper/records/c3_efficiency_20260327_020551.md`
- New memory profile: `swap/paper/records/memory_profile_20260401_eurosat_qwen2.json`

## Revision Checklist

### 1. DPI / information-theoretic claim

- Status: still needs tone-down.
- Why: the disclaimer is now present in `main.tex`, but Proposition 3.1 and the conclusion still read as stronger than what DPI can support for label prediction.
- Required manuscript edits:
  - Convert Proposition 3.1 into a motivation statement or remark.
  - Replace language like "task-relevant information" with "more linearly accessible discriminative signal for closed-set classification."
  - Remove any wording that implies DPI proves classification superiority.
- No new experiment needed.

### 2. Mahalanobis optimality under Gaussian assumptions

- Status: still needs tone-down.
- Why: the proposition is conditionally correct, but nearby phrasing still over-generalizes to real LMM activations.
- Required manuscript edits:
  - Rephrase as a design motivation under a shared-covariance Gaussian approximation.
  - Remove phrases implying practical optimality in the actual Transformer feature space.
- No new experiment needed.

### 3. Stronger frozen-feature / FGVC baselines

- Status: experiment added; manuscript still needs update.
- Why: the new FGVC baseline run shows that a stronger alternative exists and must be discussed explicitly.
- Required manuscript edits:
  - Add a FGVC baseline table or appendix table including `KeCO`.
  - Replace any claim that implicitly treats SAV as the only strong FGVC comparator.
  - State honestly that HiRe is competitive but not dominant on all FGVC baselines.
- Supporting experiment:
  - `swap/paper/records/fgvc_baselines_20260401_122632.md`
  - `swap/paper/records/fgvc_review_simple_comparison_20260401_151500.md`
- Key result summary:
  - On `qwen2_vl` over `eurosat_fgvc / pets_fgvc / cub_fgvc`, mean accuracy is:
    - `KeCO`: `0.9367`
    - `RSEv2`: `0.9067`
    - `SAV`: `0.8933`
    - `Ridge Probe`: `0.8933`
    - `Whitened NCM`: `0.8867`
  - Per-task:
    - `eurosat_fgvc`: `KeCO 0.83 > RSEv2 0.76`
    - `pets_fgvc`: `KeCO 0.98 > RSEv2 0.96`
    - `cub_fgvc`: `KeCO = RSE = SAV = RSEv2 = 1.00`
- Takeaway:
  - The reviewer concern is partly real.
  - HiRe should not be positioned as clearly stronger than all reasonable FGVC baselines.
  - At the same time, the newly added simple frozen-feature baselines do not explain away the gains: `RSEv2` still stays above both `Ridge Probe` and `Whitened NCM` on the 3-task FGVC mean.

### 4. Support-set fairness / leakage concern

- Status: needs implementation/protocol clarification.
- Why: matched support budgets are stated, but the paper still does not explicitly say that covariance/prototypes are fit only from labeled support examples under a standard episodic few-shot protocol.
- Required manuscript edits:
  - Add 1-2 sentences in setup clarifying:
    - support labels are part of the allowed few-shot input;
    - no query labels or validation labels are used in fitting;
    - all few-shot methods use the same support budget.
- No new experiment needed.

### 5. Reproducibility and implementation details

- Status: still needs supplement-level detail and paper-code alignment.
- Why: the paper still omits details reviewer will need for reproduction, and the current wording around shrinkage does not match the implementation tightly enough.
- Required manuscript edits:
  - In supplement, specify:
    - how `alpha*` is computed in practice;
    - whether features are normalized before distance computation;
    - covariance stabilization details;
    - default `component_top_k`, `confidence_floor`, and storage choices.
  - Align the shrinkage description with the current implementation.
- Code-grounded facts to document:
  - `normalize_features=False` in `RSEv2`
  - Woodbury-style inverse with `pinv` fallback
  - CPU storage of component states
- No new experiment needed.

### 6. Margin weighting as a heuristic

- Status: still needs tone-down.
- Why: the intuitive explanation is fine, but phrases like "Soft MoE" and "closed-form gating" over-theorize what is still an empirical confidence proxy.
- Required manuscript edits:
  - Reframe margin weighting as a simple empirical confidence heuristic.
  - Keep the calibration intuition, but avoid implying formal confidence calibration guarantees.
- Supporting evidence already available:
  - `EqualWeight` ablation in `swap/paper/records/c2_ablations_20260326_144544.md`
- Key result summary:
  - `RSEv2`: `0.7225` mean
  - `RSEv2-EqualWeight`: `0.7108` mean
- Takeaway:
  - Weighting helps empirically, but should not be sold as a theory-backed confidence estimator.

### 7. Failure-case analysis

- Status: mostly resolved.
- Why: the current manuscript now contains failure panels and an explicit BLINK boundary discussion.
- Optional strengthening:
  - Add one small appendix table splitting BLINK by subtask, if space allows.
- No urgent new experiment required.

### 8. 4L components, redundancy, and active-set size

- Status: experiment evidence available; manuscript still needs update.
- Why: the paper introduces active proportion, but the current default method behaves much closer to an all-component ensemble than the text suggests.
- Required manuscript edits:
  - Add `Top1 / Top4 / Top8 / All` results or a short appendix table.
  - Add one sentence clarifying that the default configuration often keeps nearly the full component set active.
  - Remove wording that implies aggressive automatic pruning in the default setting.
- Supporting experiments:
  - `swap/paper/records/c2_ablations_20260326_144544.md`
  - Diagnostics mined from `swap/paper/outputs/**/*rsev2*.diagnostics.json`
- Key result summary:
  - In the ablation suite:
    - `RSEv2`: `0.7225`
    - `RSEv2-Top1`: `0.7000`
    - `RSEv2-Top4`: `0.7025`
    - `RSEv2-Top8`: `0.7058`
    - `RSEv2-All`: `0.7225`
  - Across `181` qwen2 paper diagnostics:
    - `152` runs have `mean_active_components = 112 / 112`
    - most remaining reductions come from explicit `Top1/4/8` ablations rather than the default model naturally pruning many components
- Takeaway:
  - The useful empirical story is not "the default method sparsifies heavily."
  - The stronger story is "using the full hierarchy is often robust, and aggressive top-k pruning is not consistently beneficial."

### 9. LoRA comparison

- Status: still needs tone-down.
- Why: if the LoRA numbers come from a reference paper rather than a matched in-house run, they should not be framed as a directly controlled upper bound.
- Required manuscript edits:
  - Replace "upper-bound" with "reported reference result from prior work" or equivalent.
  - Remove claims like "exceeding LoRA" unless the protocol match is explicitly documented.
- No new experiment requested.

### 10. Open-world / broad interaction claim

- Status: still needs tone-down.
- Why: the manuscript already narrows scope in analysis and limitations, but the conclusion still generalizes too broadly.
- Required manuscript edits:
  - Keep the claim centered on closed-set few-shot classification.
  - Soften "read, don't write" from a universal interaction prescription to a scoped recommendation for discriminative tasks.
- No new experiment needed.

### 11. Resource cost, especially memory

- Status: experiment added; manuscript still needs update.
- Why: timing alone is not enough because reviewers can reasonably ask whether storing hierarchical states makes HiRe memory-heavy.
- Required manuscript edits:
  - Add a small appendix table with peak GPU memory or reserved memory.
  - Clarify that component states are stored on CPU after fitting, while GPU peaks remain close to baseline model inference.
- Supporting experiments:
  - Time: `swap/paper/records/c3_efficiency_20260327_020551.md`
  - Memory: `swap/paper/records/memory_profile_20260401_eurosat_qwen2.json`
- Key result summary on `qwen2_vl / eurosat_fgvc / seed42`:
  - `zero_shot`
    - fit peak reserved: `15.45 GB`
    - predict peak reserved: `15.47 GB`
    - avg predict: `107.1 ms`
  - `SAV`
    - fit peak reserved: `15.51 GB`
    - predict peak reserved: `15.51 GB`
    - avg predict: `36.2 ms`
  - `RSEv2`
    - fit peak reserved: `15.56 GB`
    - predict peak reserved: `15.56 GB`
    - avg predict: `56.3 ms`
  - `KeCO`
    - fit peak reserved: `17.48 GB`
    - predict peak reserved: `17.50 GB`
    - avg predict: `90.9 ms`
- Takeaway:
  - In this setting, HiRe/RSEv2 does not incur a dramatic GPU-memory penalty relative to zero-shot or SAV.
  - KeCO is both stronger on mean FGVC accuracy and heavier on GPU memory, which is useful context for fair positioning.

## Compact Experiment Supplement

### A. New FGVC Baseline Run

- Run: `swap/paper/records/fgvc_baselines_20260401_122632.md`
- Scope:
  - model: `qwen2_vl`
  - tasks: `eurosat_fgvc, pets_fgvc, cub_fgvc`
  - methods: `Zero-shot, SAV, KeCO, RSE, STV, I2CL, MimIC, RSEv2`
- Main message:
  - `KeCO` is the strongest average FGVC baseline in this run.
  - `RSEv2` is still strong and the fastest fitted read-based method in this table, but it is not the undisputed FGVC leader.

### A2. Reviewer-Targeted Simple Frozen-Feature Baselines

- Run: `swap/paper/records/fgvc_review_simple_comparison_20260401_151500.md`
- Scope:
  - model: `qwen2_vl`
  - tasks: `eurosat_fgvc, pets_fgvc, cub_fgvc`
  - methods: `Whitened NCM, Ridge Probe`
- Main message:
  - The reviewer-requested simple frozen-feature baselines are now implemented and evaluated under the same FGVC protocol.
  - `RSEv2` remains above both new baselines on the 3-task mean (`0.9067` vs `0.8933 / 0.8867`), with the clearest gap on `eurosat_fgvc`.

### B. Existing RSEv2 Ablation Evidence

- Run: `swap/paper/records/c2_ablations_20260326_144544.md`
- Main message:
  - Multi-level Mahalanobis is stronger than head-only, equal-weight, and top-k-pruned variants on average.
  - The paper should present this as evidence for hierarchy + geometry, not as evidence of strong default sparsity.

### C. Existing Efficiency Evidence

- Run: `swap/paper/records/c3_efficiency_20260327_020551.md`
- Main message on `qwen2_vl`:
  - `RSEv2` has the best mean accuracy in the 6-task efficiency suite (`0.7225`) and the lowest fit time among trained/read-based comparators (`15.23 s`).
  - `SAV` remains slightly faster at prediction (`222 ms` vs `238 ms`), so the paper should avoid overstating inference-speed superiority unless clearly scoped.

### D. New Memory Profile

- Run: `swap/paper/records/memory_profile_20260401_eurosat_qwen2.json`
- Main message:
  - `RSEv2` reserved-memory peak is only slightly above `Zero-shot` / `SAV` in this profiling setup.
  - The extra memory concern is therefore real enough to discuss, but not large enough to undermine practical usability in this setting.

## Recommended Paper-Level Framing After These Results

- Best honest positioning:
  - HiRe is a strong training-free hierarchical reader for closed-set multimodal classification.
  - Its main strength is not "universally beating all alternatives," but combining broad task coverage, no gradient updates, strong read-based accuracy, and moderate resource cost.
- Best claims to avoid:
  - universal "read, don't write" language
  - LoRA as a clean upper bound
  - implying DPI proves label-level usefulness
  - implying the default method heavily prunes most components
