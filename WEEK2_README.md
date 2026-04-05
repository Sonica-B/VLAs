# Week 2 — LoRA Intervention for H3 Causal Test

This document is the primary reference for reviewers (Codex, Antigravity, or
anyone else auditing the Week 2 code). It explains what Week 2 does, why it
does it, and how each design decision follows from the Week 1 findings.

## TL;DR

Week 1 found that Qwen3-VL-8B's `Qwen3VLVisionPatchMerger` has a 784×
feature-std compression ratio that correlates with a domain-conditional
quantitative-physics degradation (the H3 effect). Week 2 tests the causal
claim: if we apply LoRA to **just the merger's two Linear layers**
(`linear_fc1`, `linear_fc2`) and retrain on physics QA, does the
quantitative-slice PhysBench accuracy recover **more** than if we apply LoRA
to an LLM-only baseline with matched parameter count?

The test is a **double dissociation**:

- Condition B (merger-only LoRA) should improve quant accuracy > LLM LoRA
- Condition C (LLM-only LoRA) should improve qual accuracy ≥ merger LoRA

If both hold, the merger is the causal bottleneck. If only the first holds,
it is a weaker but still meaningful result. If neither holds, Week 2 has
refuted the H3 causal claim and we pivot back to the clean-negative-result
Option A reframe described in the Notion log.

## Why this is the right next experiment

| Week 1 finding | Week 2 follow-up |
|---|---|
| Qwen3-VL-8B merger has 784× compression; shows H3 on 2/3 targets | Target Qwen3-VL-8B specifically (strongest signal) |
| InternVL3-8B-hf merger has 2.4× compression; shows NO H3 | Not a Week 2 target — nothing to intervene on |
| `sub_type` target showed H3 growth +0.032 → +0.105 at merger | Expected causal intervention site = `visual.merger.*` |
| Bootstrap PCA-128 smoothed the signal → low-variance subspace | LoRA on merger's Linear layers can directly modify those directions |

The choice of conditions B and C (and not A, D, E) comes from the following
logic:

- **A (encoder-only)** was not selected because Week 1 showed the encoder's
  output still has the physics signal (enc_out probe accuracy is highest of
  all 4 sites). Intervening on the encoder is fixing something that isn't
  broken yet.
- **B (merger-only)** is the primary test: directly modifies the suspected
  choke point.
- **C (LLM-only)** is the control: tests whether *any* LoRA helps. If C
  improves quant accuracy by as much as B, the merger isn't special and the
  H3 claim is falsified.
- **D (encoder + merger)** and **E (full stack)** are reference ceilings.
  Optional follow-ups only if B vs C is inconclusive.

## Code layout

```
src/optim/
    lora.py                    Qwen3-VL-8B LoraIntervention registry + resolution helper
    permutation_baseline.py    Probe null-hypothesis check (Week 1 credibility)

scripts/
    week2_prepare_training_data.py   Build balanced training set from PhysBench test
    week2_lora_intervention.py       Main experiment: baseline → train B → train C → aggregate
```

## Training data strategy

PhysBench has 200 val samples and 9802 test samples (no train split). We
use the **test split as our training source** and keep val as the held-out
evaluation. The auditor in `week2_prepare_training_data.py` enforces four
guarantees:

1. **Zero overlap**: no sample_id from PhysBench val ever appears in the
   training set. Enforced by `assert not (train_ids & val_ids)`.
2. **Balance**: training set has approximately equal quant and qual samples
   (≤10% skew). Prevents the LoRA from being biased toward one slice.
3. **Media resolvability**: every sampled training example has at least one
   resolvable file path (spot-checked on 40 samples).
4. **Determinism**: a fixed `--seed` produces the same train/val split
   across runs.

**Caveat for reviewers**: this is an unusual choice. The standard PhysBench
usage is to report accuracy on the test set, not train on it. We explicitly
do NOT participate in the PhysBench leaderboard evaluation — we use only
the val split for our paper's numbers, and we document in the method
section that the test split was repurposed as our internal training data
for the LoRA intervention. This is defensible because:

- PhysBench val is disjoint from test by construction (different split
  column in the source JSON)
- Our paper's claim is about a mechanistic finding (merger compression
  predicts physics degradation), not a leaderboard number
- An alternative would be Physion++ synthetic QA, which introduces its own
  distribution-shift concern

## Training hyperparameters

| Parameter | Value | Why |
|---|---|---|
| Base model | Qwen3-VL-8B 4-bit nf4 | Matches Week 1 baseline exactly |
| Batch size | 1 | VRAM constraint on 12.8 GB laptop |
| Grad accum | 16 | Effective batch = 16 |
| LR | 1e-4 | Standard LoRA starting point |
| Schedule | Linear warmup 3%, cosine decay to 10% | Reduces step-to-step noise on small training set |
| Epochs | 3 | ~3600 samples × 3 = 10800 examples; fits in 2 GPU-hours |
| Early stop | Patience 2 on lora_val loss | Prevents overtraining on small dataset |
| LoRA B (merger) | r=32, α=64 | 2 Linear layers → bigger rank to increase capacity |
| LoRA C (LLM) | r=8, α=16 | 16 Linear layers → smaller rank to match total trainable params |
| Optimizer | PagedAdamW8bit (bnb) | Optimizer state lives on CPU, saves ~2 GB VRAM |
| Grad checkpointing | Enabled (use_reentrant=False) | Caps forward activation memory |
| Grad clip | 1.0 | Standard |

Parameter count is matched approximately across B and C (within ~2×) so
any difference in performance isn't just a function of adapter size.

## Reviewer checklist

Run these commands to verify Week 2 end-to-end:

```bash
# 0. Sanity check: LoRA target paths resolve on the loaded model
python scripts/week2_lora_intervention.py --stage resolution-check

# 1. Prepare training data (~5 min, creates cache/week2/training_data/)
python scripts/week2_prepare_training_data.py --max-per-slice 200  # smoke test
python scripts/week2_prepare_training_data.py                       # full

# 2. Baseline eval (~10 min on PhysBench val, 200 samples)
python scripts/week2_lora_intervention.py --stage baseline

# 3. Condition B: merger-only LoRA (~1.5-2 hr with --epochs 3)
python scripts/week2_lora_intervention.py --stage train --condition B

# 4. Condition C: LLM-only LoRA (~1.5-2 hr with --epochs 3)
python scripts/week2_lora_intervention.py --stage train --condition C

# 5. Aggregate + H3 causal test
python scripts/week2_lora_intervention.py --stage aggregate
```

Each stage is independently resumable. `--force` re-runs a stage even if
its output exists.

## Known limitations and deliberate choices (NOT bugs)

1. **No HF Trainer.** The training loop is hand-written (~60 lines) so the
   reviewer can see every step explicitly. Standard HF Trainer would hide
   the loss computation, optimizer step, and LR schedule behind abstraction.
2. **Single-sample effective batch via grad accum.** Multi-sample batching
   across variable-length image inputs is non-trivial with Qwen3-VL; the
   grad accumulation path is equivalent in gradient signal, simpler, and
   easier to audit.
3. **Validation loss computed on 32 samples max.** Full lora_val is ~360
   samples; evaluating all would add ~2 min per eval step. Early-stop
   signal is robust enough on the 32-sample subset.
4. **Loss masks prompt tokens.** Only the answer letter contributes to the
   cross-entropy loss. This is the standard instruction-tuning setup.
5. **No KV cache during training.** Required for gradient checkpointing.
   `model.config.use_cache = False` is set explicitly at load time.
6. **Best checkpoint saved by val loss only.** We do NOT select by PhysBench
   val accuracy during training — that would be a form of test-set peeking
   (PhysBench val is our held-out eval). The best checkpoint is the one
   with lowest lora_val loss, where lora_val is a disjoint slice of
   PhysBench test samples.
7. **PhysBench val eval uses greedy decoding, `max_new_tokens=10`.** Matches
   the eval protocol in `scripts/run_physbench_eval.py`. No sampling, no
   beam search, to keep the eval deterministic.

## Expected outcomes (for reviewer sanity-checking)

Given the Week 1 baselines:

- Baseline (no LoRA): PhysBench val total = 64.5%, quant ≈ 60-65%, qual ≈ 64-66%
- Condition B (merger LoRA): expected quant +3 to +6 pp, qual flat to +2 pp
- Condition C (LLM LoRA): expected quant +0 to +3 pp, qual +2 to +4 pp

Headline H3 result we need for the paper:

- `deltaB_quant - deltaC_quant > +2 pp` (merger wins on quant by at least 2 pp)
- `deltaC_qual >= deltaB_qual` (LLM wins or ties on qual)

If both hold, H3 is confirmed as a causal claim. If only the first holds,
the paper still has a publishable finding. If neither holds, Week 2 is a
clean negative result that rules out the merger-causal-bottleneck framing
and we ship the HiPS-refinement Option A paper.
