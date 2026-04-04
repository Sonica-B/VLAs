# Option C: Where Does Physics Live in Vision-Language Models?
## Revised Experiment Design — NeurIPS 2026

### The Story Arc
1. **Observation**: VLM vision encoders encode physics (P2P showed 85% probing accuracy)
2. **Question**: Where exactly does this information live, and where is it lost?
3. **Diagnostic Finding 1**: Physics is scene-global, not patch-local (novel)
4. **Diagnostic Finding 2**: The merger/projection layer is the primary bottleneck (contradicts Hidden in Plain Sight)
5. **Intervention**: QLoRA at the merger recovers physics understanding
6. **Proof**: PhysBench scores improve with merger-targeted fine-tuning but NOT with LLM-only fine-tuning

### Research Questions & Hypotheses

**H1**: Physics properties (mass, friction, bounciness) are encoded in vision encoder representations at above-chance levels (replication of P2P)

**H2**: Physics information is scene-global — global-pooled representations decode physics as well as or better than patch-level representations (novel finding)

**H3**: The merger/projection layer is the primary bottleneck where physics information is lost — NOT the LLM (contradicts "Hidden in Plain Sight")

**H4**: Targeted QLoRA fine-tuning at the merger layer recovers physics understanding, as measured by PhysBench accuracy improvements

### Datasets

**Physion++ (Diagnostic Probing)**
- 10 physics scenarios across 4 domains: mass (3), friction (3), bounciness (2), deformation (2)
- 166 trials total, 800 .pkl files with ground-truth physics properties
- Properties: mass, dynamic_friction, static_friction, bounciness per object
- Video frames extractable from _img.mp4 files
- Used for: linear probing of intermediate representations

**PhysBench (Downstream Evaluation)**
- 10,002 multiple-choice physics questions across 4 domains
- Domains: object properties, relationships, scene understanding, dynamics
- Published baselines: GPT-4o 49.5%, various open VLMs
- Used for: measuring real-world physics understanding before/after intervention

### Experiment Matrix

#### Phase 1: Setup & Baselines (Current)
- [x] Extract Physion++ data, verify structure
- [ ] Download PhysBench evaluation data
- [ ] Run Qwen2.5-VL-7B baseline on PhysBench (4-bit quantized)
- [ ] Run InternVL2-2B baseline on PhysBench

#### Phase 2: Diagnostic Probing
| Experiment | Model | What | GPU | Time Est |
|---|---|---|---|---|
| 2A | Qwen2.5-VL-7B (4-bit) | Physion++ probing: mass, friction, bounciness | 5070 Ti | 3 hrs |
| 2B | InternVL2-2B | Physion++ probing: mass, friction, bounciness | 5070 Ti | 2 hrs |
| 2C | Qwen2.5-VL-7B | Global vs patch-level pooling comparison | 5070 Ti | 1 hr |
| 2D | Both models | Per-scenario analysis (mass vs friction vs bounciness) | 5070 Ti | 2 hrs |
| 2E | Both models | Layer-by-layer degradation curves (encoder → merger → LLM) | 5070 Ti | 3 hrs |

**Probing Protocol:**
- Extract activations at: encoder layers (every 4th), post-merger, LLM layers 0,4,8,12,16,20,24,28
- For each extraction point: train Ridge regression (α=1.0) to predict mass/friction/bounciness
- Pooling strategies: global average, CLS token (if available), max pooling
- Metric: R² on held-out 20% test split
- Cross-validate across scenarios to test generalization

#### Phase 3: Intervention (QLoRA Ablation)

| Condition | Target | Rank | Training Data | GPU | Time Est | Priority |
|---|---|---|---|---|---|---|
| Baseline | None (frozen) | — | — | 5070 Ti | 15 min | P0 |
| A | Merger/projection ONLY | 8 | Physics QA | 5070 Ti / A100 | 4-5 hrs | P1 |
| B | LLM layers 0-7 ONLY | 8 | Physics QA | 5070 Ti / A100 | 4-5 hrs | P1 |
| C | Encoder last 6 blocks ONLY | 8 | Physics QA | A100 | 6-8 hrs | P1 |
| D | Merger + Encoder | 8 | Physics QA | A100 | 6-8 hrs | P2 |
| E | Full QLoRA (all components) | 8 | Physics QA | A100 | 6-8 hrs | P2 |

**Training Data for QLoRA:**
- Primary: Physics QA generated from Physion++ metadata (mass comparisons, friction predictions, bounciness questions)
- Secondary: PhysBench train split if available
- Format: image + question → multiple choice answer
- ~5000 training examples target

**Evaluation after each condition:**
1. PhysBench accuracy (overall + per-domain)
2. R² probing BEFORE and AFTER fine-tuning at each extraction point
3. Degradation curve comparison (pre vs post fine-tuning)

#### Phase 4: Analysis & Paper

**Key Analyses:**
- If Condition A (merger) >> Condition B (LLM) → H3 proven: merger is the bottleneck
- If Condition D (merger+encoder) ≈ Condition A (merger) → encoder already good, merger is the problem
- If Condition A (merger) >> Baseline but Condition B (LLM) ≈ Baseline → strongest evidence for H3
- Before/after degradation curves → show fine-tuning flattens the degradation at merger (H4)
- Cross-model consistency (Qwen vs InternVL2) → finding generalizes

**Expected Results (based on prior work):**
- Encoder R² for mass/friction: 0.6-0.85 (P2P showed ~85% classification)
- Post-merger R² drop: 30-50% (our hypothesis)
- LLM R² further drop: 10-20% (contradicts Hidden in Plain Sight)
- Condition A recovery: +5-15% on PhysBench
- Condition B recovery: +1-3% on PhysBench (minimal)

### Compute Requirements Summary

| Experiment | GPU | Time | Priority |
|---|---|---|---|
| PhysBench baseline (inference) | 5070 Ti | 15 min | P0 - do now |
| Physion++ probing Qwen (800 trials) | 5070 Ti | 3 hrs | P0 - do now |
| Physion++ probing InternVL2 | 5070 Ti | 2 hrs | P0 - do now |
| QLoRA Condition A (merger) | 5070 Ti or A100 | 4-5 hrs | P1 |
| QLoRA Condition B (LLM) | 5070 Ti or A100 | 4-5 hrs | P1 |
| QLoRA Condition C (encoder) | A100 recommended | 6-8 hrs | P1 |
| QLoRA Conditions D, E | A100 recommended | 6-8 hrs each | P2 |
| All post-training probing + eval | 5070 Ti | 30 min each | P1 |

**Total compute budget:**
- 5070 Ti (16GB, available now): ~15 hrs for P0 tasks
- A100 (80GB, needed later): ~30 hrs for P1/P2 tasks

### Hardware Notes

**RTX 5070 Ti (16GB VRAM) — Current:**
- Qwen2.5-VL-7B in 4-bit: ~5GB VRAM → fits comfortably
- InternVL2-2B: ~4GB VRAM → fits easily
- QLoRA training on 7B 4-bit: ~12-14GB → tight but feasible for merger-only (Condition A)
- Batch size 1-2 for training, 4-8 for inference

**A100 (80GB) — Future:**
- All conditions fit comfortably
- Can run larger batch sizes (8-16)
- Enables Condition C (encoder fine-tuning) and D/E without memory pressure
- Can explore rank-16 or rank-32 if rank-8 shows promise

### Risk Mitigation

| Risk | Mitigation |
|---|---|
| PhysBench scores don't improve with merger QLoRA | Try rank-16/32; add more training data; try merger+encoder (Condition D) |
| Probing doesn't show merger bottleneck | The finding itself is publishable — "physics is preserved through the merger" contradicts our hypothesis but is still novel |
| 5070 Ti OOM during QLoRA | Reduce batch size to 1; use gradient checkpointing; defer to A100 |
| Physion++ probing R² too low | Increase training data via data augmentation; try nonlinear probes (MLP) as secondary analysis |
| Results don't generalize across models | Focus paper on Qwen2.5-VL; present InternVL2 as supplementary |

### Timeline

| Week | Tasks |
|---|---|
| Week 1 (current) | Phase 1: Data extraction, PhysBench baseline, experiment design |
| Week 2 | Phase 2: Full probing on both models, degradation curves |
| Week 3 | Phase 3A: QLoRA Conditions A & B (merger vs LLM) |
| Week 4 | Phase 3B: QLoRA Conditions C, D, E (if A100 available) |
| Week 5 | Phase 4: Analysis, figures, paper draft |
| Week 6 | Paper revision, additional experiments if needed |

### File Structure

```
VLAs/
├── data/
│   ├── physion_readout/          # Extracted Physion++ data
│   │   └── readout_data_v1/     # 10 scenarios, 166 trials
│   └── physbench/               # PhysBench evaluation data
├── scripts/
│   ├── run_physbench_eval.py    # PhysBench evaluation (baseline + post-training)
│   ├── run_physion_probing.py   # Physion++ probing pipeline
│   └── run_week2_lora_ablation.py  # QLoRA training conditions
├── results/
│   ├── physbench/               # PhysBench evaluation results
│   └── probing/                 # Probing R² results
├── docs/
│   └── OPTION_C_PLAN.md         # This document
└── configs/                     # Training/eval configs
```
