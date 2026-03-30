# 4-Week Execution Plan: "Where Does Physics Live in Vision Encoders?"
## Target: NeurIPS 2026 (Abstract: May 4, Paper: May 6 AOE)

---

## Overview

| Phase | Weeks | Goal |
|---|---|---|
| Phase 1: Probing | Week 1–2 | Physics saliency maps, R² degradation across pipeline |
| Phase 2: Ablation | Week 3 | Identify leverage components via factorial LoRA |
| Phase 3: Viz + Writing | Week 4 | Before/after fine-tuning, full paper draft |

---

## WEEK 1 (Mar 30 – Apr 5): Foundation & Data Pipeline

**Goal:** Environment setup, dataset acquisition, activation extraction pipeline working for 1 model.

### Day 1-2 (Mon-Tue): Environment & Data

- Set up compute environment (A100 GPU access, CUDA 12.x, PyTorch 2.x, transformers ≥4.45, peft)
- Install and verify all 3 VLM models load correctly in 4-bit or bf16:
  - Qwen2.5-VL-7B (`Qwen/Qwen2.5-VL-7B-Instruct`)
  - InternVL 2.5-8B (`OpenGVLab/InternVL2_5-8B`)
  - LLaVA-OneVision-7B (`lmms-lab/llava-onevision-qwen2-7b-ov`)
- Download Physion++ dataset from S3 (~50GB), verify `.pkl` metadata structure
- Parse Physion++ metadata: extract per-object mass, friction, elasticity, positions, velocities
- Confirm: `physion_loader.py` can iterate all scenarios and return structured physics labels

### Day 3-4 (Wed-Thu): Activation Extraction & Patch Labeling

- Implement patch-level activation extraction for Qwen2.5-VL-7B at 4 pipeline stages:
  1. **Stage 1:** Vision encoder output (pre-projection) — shape `[B, N_patches, D_enc]`
  2. **Stage 2:** Post-projection (cross-modal token space) — shape `[B, N_patches, D_llm]`
  3. **Stage 3:** LLM layer 8 hidden state — shape `[B, N_patches, D_llm]`
  4. **Stage 4:** LLM layer 16 hidden state — shape `[B, N_patches, D_llm]`
- Use `torch.nn.Module.register_forward_hook` to capture intermediate activations
- Implement patch-to-physics-label assignment:
  - **Object-level** (mass, friction, elasticity): assign labels via segmentation mask overlap — each patch gets the label of the object covering >50% of its area
  - **Scene-level dynamics** (stability): compute per-patch "physics activity score" from temporal derivatives of object positions across frames
- Store extracted activations in HDF5 format: `results/activations/{model}/{scenario_id}.h5`
  - Dataset structure: `stage_1`, `stage_2`, `stage_3`, `stage_4`, each `[N_patches, D]`
  - Attributes: `physics_labels`, `patch_coords`, `scenario_id`

### Day 5-6 (Fri-Sat): Probe Training & First Saliency Maps

- Implement linear ridge regression probe (`src/probing/linear_probe.py`):
  - Fit `Ridge(alpha=1.0)` on `[N_samples × N_patches, D]` → predict physics property
  - Report per-patch R², aggregate to 14×14 grid
- Implement MLP probe (`src/probing/mlp_probe.py`):
  - Architecture: `D → 512 → 1` with ReLU, trained with Adam, lr=1e-3
- Train probes on Qwen2.5-VL-7B activations for **mass prediction** as proof-of-concept
- Compute R² scores at each pipeline stage — expect degradation from Stage 1 → Stage 4
- Generate first physics saliency map prototype:
  - 14×14 grid of per-patch R² values
  - Bilinear upsample to original image resolution
  - Overlay as semi-transparent heatmap (colormap: `inferno`)
- Acceptance criteria: R²(mass, Stage 1) > 0.15 (above chance)

### Day 7 (Sun): Debug, Document, Commit

- Debug any activation extraction shape mismatches
- Document findings in `docs/WEEK1_RESULTS.md`
- Push all working code to git with descriptive commits
- Buffer day for catching up on any delayed tasks

---

## WEEK 2 (Apr 6 – Apr 12): Full Probing Study (Phase 1 Complete)

**Goal:** Physics saliency maps for all 3 VLMs, all physics variables, all pipeline stages.

### Day 8-9 (Mon-Tue): Multi-Model Extraction

- Run activation extraction for InternVL 2.5-8B and LLaVA-OneVision-7B
- Adapt `activation_extractor.py` for architectural differences:
  - InternVL: `InternVisionModel` has different block names than ViT-L
  - LLaVA: SigLIP encoder uses `vision_model.encoder.layers[i]`
- Verify activation shapes: `[B, N_patches, D]` at each stage for each model

### Day 10-11 (Wed-Thu): Full Probe Matrix

- Train probes for ALL physics variables × ALL models × ALL pipeline stages:
  - Variables: `{mass, friction, elasticity, stability}` — 4 total
  - Models: `{qwen, internvl, llava}` — 3 total
  - Stages: `{enc_out, post_proj, llm_8, llm_16}` — 4 total
  - Total: 48 probe training runs (linear + MLP = 96 runs)
- Compute **R² matrix**: shape `[3 models, 4 variables, 4 stages]`
- Generate R² degradation curves: x=pipeline stage, y=R², one line per variable per model
- Statistical significance testing: bootstrap 95% CIs over 1000 resamples

### Day 12-13 (Fri-Sat): Saliency Maps & Metrics

- Generate complete physics saliency maps for all 48 conditions
- Compute **spatial precision**: fraction of top-20% activated patches that overlap GT object masks
- Compute **spatial recall**: fraction of GT object mask covered by top-20% activated patches
- Compute **information loss ratio**: `(R²_stage1 - R²_stage2) / R²_stage1` per variable per model
- Key comparison: Qwen (unfrozen ViT) vs LLaVA (frozen SigLIP) — does training the encoder produce sharper spatial physics?
- Create publication-quality figure drafts (matplotlib, 300 DPI)

### Day 14 (Sun): Phase 1 Checkpoint

- Phase 1 complete — all results collected
- Document findings, update `results/` tables as CSV
- Begin writing Phase 1 section of paper (Introduction, Method, Phase 1 Results)

---

## WEEK 3 (Apr 13 – Apr 19): Component Ablation (Phase 2)

**Goal:** All 5 ablation conditions trained and evaluated for all 3 VLMs.

### Day 15-16 (Mon-Tue): LoRA Setup & Training Data

- Implement `src/models/lora_wrapper.py` for all 5 conditions:
  - **Condition A — Encoder-only:** LoRA rank 16 on last 6 ViT blocks, Q/V matrices only
  - **Condition B — Projection-only:** LoRA on all MLP projector linear layers
  - **Condition C — LLM-only:** LoRA rank 16 on first 8 LLM transformer layers, Q/V matrices
  - **Condition D — Encoder+Projection:** Conditions A + B combined
  - **Condition E — Full model baseline:** LoRA rank 16 on all Q/V across full model
- Target ~16M trainable parameters per condition (adjust rank if needed)
- Generate Physion++ physics QA training set from simulation metadata:
  - QA types: stability prediction, property ordering (heavier/lighter), dynamic outcome prediction
  - Target: ~50K QA pairs per model training

### Day 17-18 (Wed-Thu): Training Qwen + InternVL

- Train Conditions A–E on Qwen2.5-VL-7B:
  - Batch size: 4 (gradient accumulation ×8 = effective 32)
  - LR: 1e-4, cosine schedule, 3 epochs on 50K QA pairs
  - ~8 hrs per condition → parallelize on 5 GPUs if available
- Train Conditions A–E on InternVL 2.5-8B
- Monitor training curves via W&B; validate no overfitting on held-out 10% split

### Day 19-20 (Fri-Sat): Training LLaVA + Evaluation

- Train Conditions A–E on LLaVA-OneVision-7B
- Evaluate ALL 15 trained models on:
  - **PhysBench** — overall score + per-category breakdown
  - **GRASP Level 2** — physical reasoning subset
  - **ConservationBench** — conservation law adherence
- Compute per-physics-concept accuracy (mass/stability/friction/elasticity)
- Additional ablations: LoRA rank ∈ {8, 16, 32}, ViT layers ∈ {last 2, last 6, all}

### Day 21 (Sun): Phase 2 Checkpoint

- Phase 2 complete — identify winning condition per model
- **Key hypothesis to validate:** Condition D (Encoder+Projection) beats Condition C (LLM-only)
  - If true: physics is bottlenecked at the projection, not in the LLM's priors
  - If false: physical reasoning is primarily a language-space phenomenon
- Begin writing Phase 2 section of paper

---

## WEEK 4 (Apr 20 – Apr 26): Before/After Visualization + Paper Writing (Phase 3 + Submission)

**Goal:** Complete Phase 3, write full paper, prepare for NeurIPS submission.

### Day 22-23 (Mon-Tue): Phase 3 — Before/After Analysis

- Regenerate physics saliency maps for best fine-tuned models from Phase 2 (condition A–E winners)
- Create before/after comparison visualizations:
  - Side-by-side: original model saliency vs. fine-tuned model saliency
  - Quantify: Δ spatial precision, Δ spatial recall, Δ R² per stage
- Key question: does Encoder+Projection fine-tuning produce sharper, more physically grounded saliency?
- Attention weight analysis: extract LLM cross-attention, compare weight on physics-relevant vs. non-physics patches

### Day 24-25 (Wed-Thu): Paper Writing Sprint

- Write full paper draft (target: 9 content pages + supplementary):
  - **Abstract** (250 words)
  - **1. Introduction** — motivation, contributions, 4 RQs
  - **2. Related Work** — physics in VLMs, probing representations, LoRA fine-tuning
  - **3. Method** — Physion++ setup, probing framework, LoRA ablation design
  - **4. Phase 1: Physics Saliency Probing** — results + figures
  - **5. Phase 2: Component Ablation** — results + tables
  - **6. Phase 3: Fine-tuning Effects** — before/after analysis
  - **7. Discussion** — implications for VLM architecture design
  - **8. Conclusion**
- Create all publication-quality figures:
  - Figure 1: Method overview diagram
  - Figure 2: R² degradation curves (all 3 models, all 4 variables)
  - Figure 3: Physics saliency map grid (3 models × 4 variables)
  - Figure 4: Ablation bar chart (5 conditions × 3 benchmarks × 3 models)
  - Figure 5: Before/after saliency comparison

### Day 26 (Fri): Internal Review

- Internal review pass: check all claims are supported by numbers
- Anonymize: remove lab name, compute cluster name, author affiliations
- Format for NeurIPS 2026 LaTeX template
- Prepare supplementary: additional ablation tables, failure case saliency maps

### Day 27-28 (Sat-Sun, Apr 27-28): Final Polish

- Final revision pass: tighten prose, verify all figures referenced in text
- Cross-check: every number in text matches tables and figures exactly
- Proofread entire paper (spellcheck, grammar, citation completeness)
- Upload to OpenReview:
  - Abstract submission: **May 4 AOE**
  - Full paper submission: **May 6 AOE**
- Buffer: 1 full week before final deadline

---

## CONTINGENCY BUFFER (Apr 28 – May 4)

- Address any experiments that ran over schedule
- Run additional ablations if reviewers would likely request them (e.g., larger models, more LoRA ranks)
- Additional qualitative examples and failure case analysis
- Final abstract submission: **May 4 AOE**
- Final paper submission: **May 6 AOE**

---

## Resource Requirements

| Resource | Spec | Needed For |
|---|---|---|
| GPU | 4–8× A100 80GB | VLM inference, probe training, LoRA fine-tuning |
| Storage | ~500GB SSD | Physion++ data (~50GB) + activations (~200GB) + checkpoints (~250GB) |
| RAM | 128GB+ | Large batch activation extraction |
| Compute time | ~500 GPU-hrs | 15 LoRA training runs + extraction + evaluation |
