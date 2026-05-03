# Datasheet for PhysBench-Diag

Following Gebru et al. (2018), *Datasheets for Datasets* (arxiv:1803.09010).

This document describes **PhysBench-Diag**, a diagnostic benchmark distilled from
PhysBench (Chow et al., 2024) for probing where in Vision-Language Model (VLM)
architectures physics-reasoning failures originate.

---

## 1. Motivation

### 1.1 For what purpose was the dataset created?

PhysBench-Diag was created to address a specific gap in VLM evaluation:
existing physics benchmarks (PhysBench, Physion, Physion++, ScienceQA) measure
**end-to-end accuracy** but do not localize where, in a multi-stage VLM
architecture (vision encoder → projector → LLM), physics-reasoning failures
arise. PhysBench-Diag enables **layer-wise probing** at four canonical sites
(encoder output, post-projector, decoder layers 8 and 16) by partitioning the
underlying PhysBench items into **quantitative** (numerical-prediction)
versus **qualitative** (relational/categorical) sub-tasks. This partition is
the operational substrate for testing whether vision-token compression at the
multimodal projector preferentially destroys quantitative-physics information.

### 1.2 Who created the dataset and on behalf of which entity?

The PhysBench-Diag *partition* and *probing protocol* were created by the
authors of this paper (single-author submission). The underlying **PhysBench**
items are from Chow et al. (2024) and are used unmodified except for the
quant/qual labeling described in §1.3.

### 1.3 Who funded the creation of the dataset?

No external funding for PhysBench-Diag itself. Compute resources were
provided by Worcester Polytechnic Institute Turing cluster (account `cngan`,
A100-80GB nodes). The underlying PhysBench dataset was funded per its
original release (cite Chow et al., 2024).

### 1.4 Any other comments?

PhysBench-Diag is a **derivative resource**, not a new image/video dataset.
We add: (a) a deterministic quant/qual sub_type → label mapping
(`src/optim/physbench_split.py`), (b) a 200-item validation split + 999-item
test split with the partition applied, (c) per-model probing JSONs across
seven VLMs as a reusable annotation layer.

---

## 2. Composition

### 2.1 What do the instances represent?

Each instance is a multimodal physics-reasoning item from PhysBench:
- One or more images (or short video frames extracted at frame 0) showing a
  physics scenario
- A natural-language question
- Four answer choices (A/B/C/D)
- The correct answer letter
- A `task_type` label ∈ {dynamics, properties, relationships, scenes}
- A `sub_type` label (19 fine-grained sub-categories per PhysBench taxonomy)
- A derived `quant_qual` label ∈ {quantitative, qualitative} based on
  `sub_type` (described in §4.2)

### 2.2 How many instances are there in total?

| Split | Total | Quantitative | Qualitative |
|---|---|---|---|
| Validation | 200 | 55 | 145 |
| Test | 999 | 274 | 725 |

### 2.3 Does the dataset contain all possible instances or is it a sample?

PhysBench-Diag is a **deterministic curated subset** of the public PhysBench
release. The 200 validation items and 999 test items are selected by
PhysBench's authors as their official splits. We apply the quant/qual
partition to all items where media files are resolvable on local disk.

In our probing experiments, 14 of 200 validation items were skipped because
their media references could not be resolved on local disk (these are noted
as `"status": "no_media"` in the per-model probing JSONs). Probing accuracies
in §3 of the paper therefore reflect n=186 validation items processed, not
the full 200.

### 2.4 What data does each instance consist of?

Raw inputs:
- `sample_id` (e.g., `val_0`, `val_13`)
- `image_paths` (list of file paths in PhysBench's image directory)
- `question` (natural-language string)
- `options` (list of 4 strings labeled A/B/C/D)
- `answer` (one letter A/B/C/D)
- `task_type` (one of 4 PhysBench taxonomy labels)
- `sub_type` (one of 19 fine-grained labels)

Probing-derived (per-model):
- 4 feature tensors per item (one per probe site: enc_out, post_proj,
  llm_8, llm_16), mean-pooled over the sequence dimension to a fixed-size
  vector per item
- These are stored as NumPy arrays in `cache/<run>/features/<model>/`

### 2.5 Is there a label or target associated with each instance?

Yes — three categorical labels are used as probe targets:
1. `answer` (4-way classification, A/B/C/D — letter position)
2. `task_type` (4-way classification: dynamics / properties / relationships / scenes)
3. `sub_type` (19-way classification — fine-grained physics sub-categories)

Plus the derived `quant_qual` slice label used for stratified analysis.

### 2.6 Is any information missing from individual instances?

For 14 of 200 validation items, image files referenced by `image_paths` are
not present in our local PhysBench mirror; those items are skipped during
probing. We do not modify the original PhysBench manifest.

### 2.7 Are relationships between individual instances made explicit?

No. PhysBench items are treated as i.i.d. for probing purposes.

### 2.8 Are there recommended data splits?

We use PhysBench's official **val** (200 items) and **test** (999 items)
splits. Probing analyses in this paper use the **val** split exclusively.
The **test** split is reserved for evaluation by future users of
PhysBench-Diag and is not used in our LOO regression.

### 2.9 Are there any errors, sources of noise, or redundancies?

Documented limitations:
- **14 of 200 val items have unresolvable media paths in our local mirror.**
  These are treated as missing-data, not errors, and are excluded from per-model
  probing. The `extraction_stats.errors=14` field in each
  `*_quant_qual_probe.json` records this.
- **PhysBench labeling reliability**: PhysBench's original paper documents
  inter-annotator agreement metrics; we inherit these and do not re-label.
- **Pixtral-12B was attempted but produced n=58 valid samples** due to a
  transformers 4.49.x integration bug (HF issue #28005); Pixtral results
  are NOT included in the final n=7 panel.

### 2.10 Is the dataset self-contained?

The dataset *manifest* (val.json + test.json) is self-contained. The
underlying images/videos are part of the original PhysBench release and must
be downloaded via PhysBench's distribution channel; PhysBench-Diag does not
re-distribute the media.

### 2.11 Does the dataset contain data that might be considered confidential?

No. PhysBench items are public physics-reasoning prompts; no PII, no
sensitive content.

### 2.12 Does the dataset contain data that might cause anxiety?

No.

---

## 3. Collection Process

### 3.1 How was the data acquired?

PhysBench items were collected by Chow et al. (2024); see their paper for
the original collection process. PhysBench-Diag adds only a labeling layer
(§4) and a per-model probing layer (the `*_quant_qual_probe.json` files).

### 3.2 What mechanisms or procedures were used to collect the data?

For the underlying PhysBench items: see Chow et al. (2024).
For the probing layer:
- VLMs were loaded via HuggingFace transformers 4.46.x / 4.49.x
- bitsandbytes 0.44 NF4 4-bit quantization (6 of 7 models)
- bfloat16 without quantization (Granite-Vision only, due to a transformers
  4.49 + bitsandbytes 0.44 dtype mismatch)
- Forward hooks at 4 probe sites per model
- Mean-pool over sequence dim → per-sample fixed-dim feature vector
- Stored as on-disk FeatureCache (NumPy arrays)

### 3.3 Were any of the data points subject to multiple labelings?

The `quant_qual` label is derived deterministically from PhysBench's
`sub_type` (no re-labeling, no human disagreement). The `task_type` and
`sub_type` labels are inherited from PhysBench unchanged.

### 3.4 If data were collected from individuals, was their consent obtained?

Not applicable — PhysBench items do not contain individual data.

### 3.5 Was there a data validation step?

Yes — see `tests/regression.py` which validates:
- All 4 probe sites populated for every cached probing JSON
- n_samples ≥ 50 per JSON before LOO regression input
- Per-target {answer, task_type, sub_type} accuracy stats present

---

## 4. Preprocessing / Cleaning / Labeling

### 4.1 Was any preprocessing/cleaning/labeling done?

Yes — three operations:

1. **Quant/qual partition** (`src/optim/physbench_split.py`):
   PhysBench's 19 `sub_type` labels are partitioned into:
   - **Quantitative** (numerical-prediction, e.g., "how many", "how fast",
     "what angle"): contains `sub_type` ∈ {object_count, velocity, ...}
   - **Qualitative** (relational/categorical, e.g., "is X above Y", "which
     direction"): everything else
   The exact mapping is in `src/optim/physbench_split.py:classify_quantitative()`.
   Out of 200 val items, 55 are quantitative and 145 are qualitative.

2. **Vision-language model preprocessing**:
   - Each VLM applies its own image preprocessing (resize, patch tokenization)
   - We use the model's `AutoProcessor` defaults
   - For Pixtral specifically (n=58 sample, dropped from final panel),
     we hard-cap to 1 image per item at 448×448 to mitigate transformers 4.49.x
     processor bugs

3. **Probing feature extraction**:
   - Forward pass with hooks at 4 named modules
   - Mean-pool the captured tensor over its sequence dimension
   - Cast to float32 CPU NumPy

### 4.2 Was the "raw" data saved in addition to the preprocessed/cleaned data?

The raw PhysBench items are not re-distributed (use the original PhysBench
release). The probing feature caches are saved to
`cache/week1_turing/features/<model>/` and can be regenerated from raw
PhysBench items by re-running `scripts/week1_quant_qual_probe.py`.

### 4.3 Is the software for preprocessing publicly available?

Yes:
- Quant/qual partition logic: `src/optim/physbench_split.py`
- Probing harness: `scripts/week1_quant_qual_probe.py`
- Permutation tests: `scripts/week1_permutation_check.py`
- LOO regression: `scripts/phys_lens_predict.py`
- All released under the same license as the repository (see §6.4)

---

## 5. Uses

### 5.1 Has the dataset been used for any tasks already?

In this paper, PhysBench-Diag was used to:
1. Probe seven VLMs at four sites (LLaVA-OneVision-7B, Phi-3.5-Vision,
   Granite-Vision-3.2-2B, Qwen3-VL-8B, Qwen2.5-VL-7B, InternVL3-8B,
   Gemma3-4B)
2. Test the H3 hypothesis (vision-token compression preferentially
   destroys quantitative-physics information at the multimodal projector)
3. Validate a closed-form architectural predictor (PhysLens-Predict)
   via leave-one-out regression

The pre-registered predictor *did not pass* the kill-gate at n=7
(median absolute error 0.24 > 0.20 threshold), so we report the predictor
as a **descriptive observation** rather than a validated predictor, per
our pre-registration (see `docs/PRE_REGISTRATION.md`, gate 3).

### 5.2 Is there a repository linking to papers using the dataset?

Not yet. The PhysBench-Diag release coincides with this paper.

### 5.3 What other tasks could the dataset be used for?

- Cross-VLM comparison of physics-reasoning at any architectural depth
- Validation of new compression/bottleneck metrics beyond our
  `log10(C) × Δprobe` formula
- Causal interventions (e.g., LoRA fine-tuning of the projector specifically)
- Multi-modal information-flow analysis

### 5.4 Is there anything that might affect future uses?

- The **n=200 validation split is small** for some statistical tests
  (e.g., per-`sub_type` analyses where some sub_types have <10 items)
- Use the **n=999 test split** for higher-power evaluations
- The quant/qual partition is **derived from sub_type labels**; if PhysBench
  updates its taxonomy, our partition needs to be re-derived

### 5.5 Are there tasks for which the dataset should not be used?

- **Not for behavior cloning or fine-tuning data leakage**: the val/test
  split aligns with PhysBench's official splits, but we make no claim about
  model-training data inclusion of these items.
- **Not for individual-level claims**: items are not annotated for
  difficulty, fluency, or other per-item attributes beyond what PhysBench
  provides.

---

## 6. Distribution

### 6.1 Will the dataset be distributed to third parties?

Yes — PhysBench-Diag's labeling and probing artifacts are released openly
alongside this paper.

### 6.2 How will the dataset be distributed?

Via the public GitHub repository `Sonica-B/VLAs` on the
`physics-steering` branch. Specifically:
- Quant/qual labels: derived deterministically from PhysBench items via
  `src/optim/physbench_split.py`
- Per-model probing JSONs: `results/week1_turing/*_quant_qual_probe.json`
  and `results/week1/*_quant_qual_probe.json`
- Permutation test JSONs: `results/week1_turing/*_permutation_check.json`
- Predictor output: `results/week1_turing/phys_lens_predict_weekb.json`
- Pre-registration: `docs/PRE_REGISTRATION.md`
- Reproduction recipe: `turing/setup_v2_env.sh` + `turing/requirements_v2.txt`

### 6.3 When will the dataset be distributed?

On NeurIPS 2026 paper acceptance. The current `physics-steering` branch
is the working version; a clean submission branch (`paper-submission-clean`)
will be created via `bash turing/cleanup_for_submission.sh` for the final
release.

### 6.4 Will the dataset be distributed under a copyright or other license?

The PhysBench-Diag *artifacts* (quant/qual labels, probing JSONs, code) are
released under the repository's license (TBD — recommend MIT or
CC-BY-4.0). The underlying PhysBench items remain under PhysBench's
original license (typically CC-BY-4.0 — verify and pass through).

### 6.5 Have any third parties imposed IP-based or other restrictions?

No additional restrictions beyond the inherited PhysBench license.

### 6.6 Do any export controls or other regulatory restrictions apply?

No.

---

## 7. Maintenance

### 7.1 Who is supporting/hosting/maintaining the dataset?

The paper author. Maintained on GitHub `Sonica-B/VLAs`.

### 7.2 How can the owner/curator be contacted?

Via GitHub issues on the repository.

### 7.3 Is there an erratum?

If errors are found post-release, an `ERRATA.md` file will be added at the
repository root.

### 7.4 Will the dataset be updated?

Yes — additional VLMs may be added to the panel as the open-weight
ecosystem grows. Each addition will be tagged as a new release.

### 7.5 If the dataset relates to people, are there applicable limits?

Not applicable.

### 7.6 Will older versions continue to be supported?

Yes — git tags will preserve each version. The `paper-submission-clean`
branch is the immutable artifact for the NeurIPS 2026 submission.

### 7.7 If others want to extend/augment/build on the dataset, is there a mechanism?

Yes — pull requests welcome on the GitHub repository. The probing harness
(`scripts/week1_quant_qual_probe.py`) is designed to accept new VLMs by
adding to `MODEL_REGISTRY`; see `CLAUDE.md` for contribution conventions.

---

## Appendix A: PhysBench taxonomy → quant/qual mapping

The mapping is implemented in `src/optim/physbench_split.py`. Key sub_types:

**Quantitative** (numerical-prediction):
- velocity, acceleration, mass, distance, count, angle, time, force,
  torque, frequency, energy

**Qualitative** (relational/categorical):
- direction, ordering, comparison, identity, presence, configuration,
  collision_outcome, motion_type

The exact predicate is `classify_quantitative(item)` — refer to the source.

---

## Appendix B: Per-model probing protocol consistency

| Model | Quantization | Resolution policy | n_processed |
|---|---|---|---|
| Qwen3-VL-8B | bnb-NF4 | native (FULL_RESOLUTION=1) | 200 |
| Qwen2.5-VL-7B | bnb-NF4 | native | 200 |
| InternVL3-8B | bnb-NF4 | native | 200 |
| Gemma3-4B | bnb-NF4 | native | 200 |
| LLaVA-OneVision-7B | bnb-NF4 | native | 186 |
| Phi-3.5-Vision | bnb-NF4 | native | 186 |
| Granite-Vision-3.2-2B | **bf16 (no quant)** | native | 186 |
| Pixtral-12B | bnb-NF4 + 1-image-cap | 448×448 | **58** (excluded from panel) |

The bf16 protocol for Granite-Vision is documented in
`scripts/week1_quant_qual_probe.py:load_granite_vision()` — bnb-NF4 triggers
a dtype mismatch in some Linears at transformers 4.49 + bnb 0.44.

---

## Appendix C: Citation

If you use PhysBench-Diag, please cite both this paper and the original
PhysBench:

```bibtex
@inproceedings{boyane2026physbench_diag,
  title={PhysBench-Diag: A Diagnostic Benchmark for VLM Physics Reasoning},
  author={Boyane, Shreyaa},
  booktitle={NeurIPS 2026 Datasets and Benchmarks Track},
  year={2026},
  url={https://github.com/Sonica-B/VLAs}
}

@article{chow2024physbench,
  title={PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding},
  author={Chow, Wei and others},
  journal={arXiv preprint arXiv:2401.16937},
  year={2024}
}
```
