### Recommended Ideas (Session 2)

#### 🏆 Idea S2-1: SAV-Guided Task Vector Injection (SAV-TV)

**Hypothesis**: SAV-selected attention heads (ranked by classification accuracy on few-shot support set) are more relevant TV insertion locations than STV's sensitivity-based heads. For FGVC, context-sensitive heads ≠ class-discriminative heads.

**Minimum pilot** (CUB-200 50-class subset, Qwen-VL-7B, ~1.5 GPU-hours):

- Baseline: STV sensitivity-based 64-head selection
- Ours: SAV accuracy-based 20-head selection → inject same cluster bank TVs
- Success: ≥+1.5% top-1 accuracy on CUB-50

**Novelty**: 9/10 — connects two papers (MTV/STV + SAVs) from overlapping author groups that have never been combined.
**Risk**: LOW–MEDIUM | **Effort**: ~2 days implementation | **Contribution**: New method

**Differentiation**: SAVs use attention vectors as external features (no injection). Task vectors inject into heads (no accuracy-based selection). SAV-TV combines both. This is **not** incremental — the selection criterion (sensitivity vs. accuracy) fundamentally changes what information is injected.

---

#### 🥈 Idea S2-2: Query-Conditioned Cluster Selection (QC-TV)

**Hypothesis**: At inference, the cluster whose centroid activation is cosine-closest to the query's activation at the TV location encodes more query-relevant information than a fixed RL-selected cluster. Training-free.

**Minimum pilot** (CUB-200 50-class, ~1 GPU-hour):

- Compare: (a) STV fixed RL-selected cluster, (b) cosine-similarity-adaptive cluster selection
- Success: ≥+1.0% top-1 accuracy

**Novelty**: 8/10 — L2S (NeurIPS 2025) does input-dependent steering for MLLMs (safety), but uses a trained auxiliary network. This is training-free and applied to task-vector ICL for FGVC.
**Risk**: LOW | **Effort**: < 1 day | **Contribution**: Method improvement + empirical finding

---

#### 🥉 Idea S2-3: Class-Conditional Task Vector Banks (CC-TV)

**Hypothesis**: Building per-class activation cluster banks (instead of a global bank across all classes) encodes class-specific visual patterns for FGVC. At inference, select the bank for the top-K predicted classes.

**Minimum pilot** (CUB-200 50-class, ~2 GPU-hours):

- Per-class cluster banks (k=4 per class) vs. STV global bank
- Two-pass inference: zero-shot prediction → select class bank → TV re-ranking
- Success: ≥+2.0% top-1 accuracy

**Novelty**: 7/10 — most FGVC-specific idea; no paper uses class-conditional activation banks for TV injection.
**Risk**: MEDIUM (class-conditional selection may introduce chicken-and-egg noise)
**Effort**: ~2 days | **Contribution**: New FGVC-specific method

---

### Eliminated Ideas (Session 2)

| Idea                                            | Reason                                                       |
| ----------------------------------------------- | ------------------------------------------------------------ |
| Part-level task vectors                         | Requires part annotation + complex localization              |
| AU-level dimension pruning of TV heads          | SAVs already handles head-level; dimensions add complexity without clear motivation for 2h pilot |
| Attribute-guided cluster construction (CUB-312) | Needs attribute predictor at inference; too complex for pilot |
| Hierarchical class-structure TVs                | Weeks of implementation; not pilotable in 2h                 |
| Discriminative coreset × TV                     | KeCO (MM 2025) already covers coreset for ICL; incremental   |
| Sparse disentangled TVs                         | VS2 (2025) already does SAE-based sparse steering; insufficient novelty |

### Phase 3: Deep Novelty Check Results

| Idea   | Novelty Score | Closest Paper          | Key Differentiator                                           | Concurrent Work                                              |
| ------ | ------------- | ---------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| SAV-TV | **7/10**      | STV (arXiv:2511.08246) | Accuracy-based head selection for injection vs. sensitivity-based; SAVs use heads as features only | DP-MTV (arXiv:2603.04894, Mar 2026) — different focus (privacy), not a blocker |
| QC-TV  | **8/10**      | STV (arXiv:2511.08246) | Training-free cosine-nearest-centroid cluster selection; L2S/ATV require training | DP-MTV (same, not a blocker)                                 |
| CC-TV  | ~7/10         | STV / KeCO             | Class-conditional banks + two-pass inference                 | None found                                                   |

**Additional papers discovered during novelty check:**

- ATV (Kang et al., NeurIPS 2025 MI Workshop, arXiv:2506.03426) — query-conditioned task vectors for text LLMs, requires training
- Head Pursuit (Basile et al., NeurIPS 2025, arXiv:2510.21518) — concept-based head scoring + rescaling; related to SAV-TV but different domain
- M²IV/VLibrary (Fu et al., arXiv:2504.04633) — multimodal in-context vector bank retrieval; trained retriever; similar bank-retrieval spirit to QC-TV
- ELICIT (Wang et al., ICLR 2025, arXiv:2410.09343) — task vector library + trained retriever; text-only

---

### Phase 4: External Critical Review (GPT-4o, NeurIPS/ICML standard)

**Individual scores:**

- SAV-TV standalone: **4/10** — "Obvious combination of SAVs + STV; substituting one selection criterion for another lacks significant novelty without substantial empirical evidence"
- QC-TV standalone: **6/10** — "More novel due to training-free property; risk of 'nearest-neighbor retrieval is standard' objection"

**Combined SAV-TV + QC-TV paper: 7/10**

Narrative: *"STV reimagined without learned search — fully discriminative, training-free inference"*

- STV has two trained/searched components: (a) sensitivity-based head selection, (b) RL cluster selection
- The combined method replaces BOTH with support-set-based, training-free discriminative components
- This is principled: accuracy-based head selection directly measures task discrimination; cosine-nearest cluster selection aligns the task vector with the query's semantic neighborhood

**To reach clear accept at NeurIPS 2026 (reviewer's requirements):**

1. Evaluate on ≥5 FGVC benchmarks (CUB-200, Flowers, DTD, StanfordCars, FGVCAircraft)
2. Show results on ≥2 base MLLMs (Qwen-VL-7B, LLaVA-v1.6)
3. Add theoretical analysis: why accuracy-based head selection is a principled estimator of the optimal injection site
4. Ablation table: sensitivity-heads + RL-cluster vs. accuracy-heads + cosine-cluster vs. all combinations
5. CC-TV (class-conditional banks) as optional third component to differentiate from pure STV variants

---

### 🏆 FINAL RECOMMENDATION (Session 2): Combined DTV Paper

**Title direction**: "Discriminative Task Vectors: Training-Free Accuracy-Guided Injection and Query-Adaptive Cluster Selection for Fine-Grained Visual Recognition"

**Core contribution**: Replace STV's two trained/search components with purely discriminative, training-free alternatives:

1. **Component 1 (Where)**: SAV-style accuracy-based head selection instead of sensitivity-based
2. **Component 2 (What)**: QC-TV cosine-nearest cluster selection instead of RL-based
3. **Optional Component 3**: CC-TV class-conditional cluster banks

**Why this works as a single paper**:

- Unified narrative: "remove all training from STV without losing performance"
- Ablation is natural: show each component contributes
- Theoretical framing: discriminative vs. contextual as orthogonal dimensions of the "where/what" problem (echoes STV's own "where and what" title)

**Pilot Plan** (for manual execution — remote server has CUB-200 at `/root/autodl-tmp/CSV/data/cub200/` and CLIP infrastructure; Qwen-VL-7B needs to be set up separately):

| Idea   | Script             | Est. Time | Success Threshold        |
| ------ | ------------------ | --------- | ------------------------ |
| QC-TV  | `pilots/qc_tv.py`  | ~1h       | +1.0% over STV on CUB-50 |
| SAV-TV | `pilots/sav_tv.py` | ~1.5h     | +1.5% over STV on CUB-50 |
| CC-TV  | `pilots/cc_tv.py`  | ~2h       | +2.0% over STV on CUB-50 |

Sequential order (lowest-risk first): QC-TV → SAV-TV → CC-TV