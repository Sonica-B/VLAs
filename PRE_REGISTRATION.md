# Pre-Registration — PhysLens (Phase 5)

**Committed:** 2026-04-16
**Branch:** `physics-steering`
**Purpose:** Lock hypotheses, alpha values, kill-criteria, and analysis choices
**BEFORE** any Colab A100 runs begin, so no post-hoc selection is possible.

This document is deliberately read-only after the first Colab run starts.
Any changes require a signed-off addendum at the bottom with timestamp and
rationale, not an edit of the original text.

---

## 1. Research Questions (frozen)

**RQ-A (steering):** Does Subspace-Contrast Activation Steering (SCAS) at
the VLM merger recover quantitative-physics accuracy on PhysBench beyond
what random perturbations of equal magnitude would achieve?

**RQ-B (predictor):** Does a scalar derived from merger compression ratio +
per-stage probing gap predict the direction and magnitude of quantitative
physics degradation across unseen VLM architectures?

**RQ-C (benchmark):** Can a released 4-stage × N-model activation cache +
probing protocol (PhysBench-Diag) let a third party reproduce the
diagnostic in under one hour on a single A100?

---

## 2. Pre-Registered Alpha Values (RQ-A only)

SCAS alpha sweep is **restricted** to the following three values per
model. No other alpha is reported as a primary result.

| alpha | Motivation |
|---|---|
| 3.0 | Earlier plan's nominal default; included to honor Phase 4 spec |
| 5.0 | Previously observed inflection on Qwen3-VL val (headline number) |
| √C − 1 (per-model) | Theory-derived optimal (√(compression_ratio) − 1) |

Where √C − 1 is model-specific:
- InternVL3-8B (C=2.4×):  √2.4 − 1 ≈ 0.55
- Gemma4-E4B (C=114×):    √114 − 1 ≈ 9.68
- Qwen2.5-VL-7B (C=270×): √270 − 1 ≈ 15.43
- Qwen3-VL-8B (C=784×):   √784 − 1 = 27.00

Exploratory alphas (0, 1, 10, 15, 20, 30, 50) **may be run** for plots but
**may not** be reported as primary numbers. If the working window excludes
the pre-registered alphas, this is a negative result for RQ-A, not an
invitation to re-select.

---

## 3. Pre-Registered Controls

### 3.1 Random-direction control (REQUIRED, RQ-A)

For each model evaluated with SCAS:
- Generate a random unit vector in V_low (the K=64 low-variance subspace).
- Run identical inference pipeline at alpha=5.0 (matching current headline).
- Repeat with 20 independent seeds (numpy RNG seeds 0..19).
- Report the full seed distribution + median + 95% bootstrap CI.

**Decision rule:** If median random Δ_quant ≥ (observed SCAS Δ_quant × 0.5),
**SCAS is indistinguishable from noise regularization** and must be
demoted to an appendix observation. The paper headline cannot claim
"SCAS amplifies physics signal."

### 3.2 Qualitative-contrast control (REQUIRED, RQ-A)

For contrast-SCAS (PhysLens-OC): compute a "qualitative contrast" vector
(qual_centroid − quant_centroid — inverted direction), amplify at same
alphas. Prediction: should hurt quantitative accuracy. If it helps,
the contrast interpretation fails.

### 3.3 Leakage-free PCA (ALREADY ENFORCED)

PCA basis is computed on training-split features (1793 samples from
`lora_train_clean.jsonl`) only. Val/test are **never** used for PCA.
Commit `2550266` already implements this. Any deviation is a bug, not
a design choice.

---

## 4. Multiple-Comparison Correction

For each model × split, the Bonferroni-corrected significance threshold
on SCAS Δ_quant is:

    p_corrected = p_raw × (3 alphas × 2 methods) = p_raw × 6

For the 8-model LOO regression, we report Spearman rank correlation
with 95% bootstrap CI (not p-value) to avoid per-model multiplicity.

---

## 5. Primary Outcome Measures

### RQ-A (steering)

- **Primary:** Δ_quant on PhysBench test set (9802 samples, n_quant ≈ 2696).
- **Secondary:** Δ_qual (expected ≈ 0 under mechanistic theory).
- **Report:** Point estimate + 95% bootstrap CI (1000 resamples) +
  Bonferroni-corrected p-value.

### RQ-B (predictor)

- **Primary:** Leave-one-out (LOO) Spearman correlation between predicted
  and empirical H3 hit-rate across 8 model families.
- **Report:** LOO median absolute error + Spearman ρ + 95% CI.

### RQ-C (benchmark)

- **Primary:** Wall-clock time for a fresh user on a rented A100 to run
  the probing protocol on a new VLM from a clean checkout.
- **Report:** Timing breakdown per stage + artifact sizes.

---

## 6. Kill Criteria (Decision Gates)

These are **pre-committed** and override any post-hoc framing.

### Gate 1 (after Colab random-direction control, ETA 2026-04-17)

- **If median random Δ_quant on Qwen3-VL val ≥ +1.8pp (= 50% of observed
  SCAS Δ_quant):** SCAS is downgraded from headline. Paper reframes as
  diagnostic + benchmark (PhysLens-Predict + PhysBench-Diag), target
  NeurIPS D&B.
- **Else:** SCAS survives; proceed to test-set evaluation.

### Gate 2 (after test-set SCAS run, ETA 2026-04-20)

- **If test-set Δ_quant at pre-registered α has 95% CI crossing 0
  (Bonferroni-corrected):** Same demotion as Gate 1.
- **Else:** SCAS is paper section 4; benchmark is section 5.

### Gate 3 (after 8-model LOO, ETA 2026-04-27)

- **If LOO median |error| on H3 hit-rate > 20pp:** Predictor is anecdotal;
  report as "observation" not "predictor"; no claim of generalization.
- **Else:** Predictor is section 3.

### Gate 4 (DeepStack analysis, ETA 2026-04-21)

- **If Qwen3-VL per-stage R² is flat across stages:** DeepStack genuinely
  fixed the bottleneck. Paper reframes as "classical mergers break
  physics; DeepStack does not — a positive architecture-design finding."
  **This is still publishable and possibly stronger** than the original
  framing. No demotion.
- **Else:** Qwen3 joins the main curve with its own three evidence lines.

---

## 7. Scope Boundaries (what will NOT be claimed)

Regardless of results, the paper will NOT claim:

1. That SCAS solves physics reasoning in VLMs.
2. That the compression predictor is causal (only predictive in-distribution).
3. Generalization to continuous physics benchmarks not in this study
   unless we transfer to QuantiPhy.
4. That PEM failure is a failure of the general "trained bypass" concept.
   The paper reports PEM as a negative result specific to our training
   protocol and gate-supervision absence.
5. Architecture superiority of any specific VLM family. We describe
   trade-offs, not rankings.

---

## 8. Data & Code Release

On acceptance:

- Full 4-stage × 8-model activation cache on HuggingFace Datasets.
- Probing scripts + Colab notebook in a public mirror of this repo.
- Steering vectors (per-model) as supplementary.
- The exact `PRE_REGISTRATION.md` committed at the start of Phase 5
  (this file, unchanged) as Appendix A of the paper.

---

## 9. Addenda (append-only, timestamped)

_None yet. Any change to the above sections after 2026-04-17 00:00 UTC
must be added here with rationale, not edited above._
