# Information-Bottleneck Theory for PhysLens-Predict

**Status:** Draft skeleton (2026-04-19). Fills in during D5–D6 (Apr 23–25).
**Purpose:** Prove that compression ratio is a normative (not just empirical) predictor of physics degradation. This section is the single biggest reviewer-lift for Main Track Outstanding Paper.

---

## 1. Motivation

Empirically, we observe that VLMs with higher merger compression ratios show larger quantitative-physics degradation between encoder output and post-projection. We want to prove this is NOT coincidence — that a spatial-compression channel, under reasonable assumptions, MUST lose more of the physics-carrying low-variance subspace V_low than the semantics-carrying high-variance subspace V_high.

The Information-Bottleneck (IB) framework (Tishby & Zaslavsky 2015; HiPS COLM 2025) gives us the machinery.

---

## 2. Setup

### Variables

- **X** ∈ ℝ^{N_patches × d_enc}: vision-encoder output (pre-merger) — one vector per patch.
- **T** ∈ ℝ^{M × d_enc}: post-merger representation — M < N_patches.
- **Y**: physics label space, split as:
  - **Y_quant**: quantitative physics (mass, friction, velocity magnitudes)
  - **Y_qual**: qualitative physics (collide/not, stable/not, support relations)

### Merger as IB channel

The merger M: X → T is a stochastic channel (deterministic if pooling, stochastic if token-merging). Compression ratio C = N_patches / M.

### PCA decomposition

On X's feature distribution (per-patch):
- **V_high** = span of top-k eigenvectors (k chosen so V_high explains 95% variance)
- **V_low** = span of remaining (d_enc − k) eigenvectors

**Empirical observation (Week 1):** I(T; Y_quant) → physics labels correlate with projections onto V_low, while I(T; Y_qual) correlates with projections onto V_high.

---

## 3. Key Theorems (to prove)

### Theorem 1 — IB capacity ordering

Under the data-processing inequality:
```
I(X; Y) ≥ I(T; Y)
```
with equality iff the merger M preserves the sufficient statistic for Y. The **information gap** Δ_Y = I(X; Y) − I(T; Y) is always non-negative.

### Theorem 2 — Subspace variance and compression

**Claim:** For a linear spatial-compression channel with pool factor p (so C = p^2 for 2D pooling):
```
Var(T_v) ≈ Var(X_v) / p^2     if X_v is spatially uncorrelated
Var(T_v) ≈ Var(X_v)           if X_v is spatially perfectly correlated
```
where T_v is the projection onto eigenvector v. Thus low-variance (uncorrelated-across-patches) components are attenuated by 1/C, while high-variance (spatially smooth) components survive.

**Proof sketch:** Central limit theorem for the pooled-patch sum; spatial auto-correlation determines variance-preservation factor.

### Theorem 3 — Physics-info loss scales with compression

Assuming Y_quant lives in V_low and Y_qual in V_high:
```
Δ_quant = I(X; Y_quant) − I(T; Y_quant) ~ log C  (up to Gaussian constant)
Δ_qual  = I(X; Y_qual) − I(T; Y_qual)  ≈ 0       (first-order)
```
**Corollary:** The PhysLens-Predict score `log₁₀(C) × max(gap, 0)` is a Gaussian-approximation lower bound on Δ_quant.

### Theorem 4 — Optimal SCAS amplification (if V_low were physics-rich)

For a naive amplifier f → f + α · P_low · f to fully restore Var(T_v) to Var(X_v):
```
α* = √C − 1
```
**Empirical check (Week A):** Qwen3-VL-8B C=784 gives α*≈27. Our observed working window [α=3, α=10] is far below this. Two possible explanations:
1. V_low variance in Qwen3-VL is NOT physics-rich at the merger output — it's been largely destroyed already.
2. The linear-Gaussian assumption is too strong — the effective α* is much smaller because the LLM's downstream readout is sub-linear in activation magnitude.

**This explains why zero-shot SCAS fails.** The predictor is descriptive (says which models are bottlenecked) but does not justify zero-shot restoration. It justifies a **trained** intervention (PPFT) that can modify the signal at the right architectural point, not just amplify what's already lost.

---

## 4. Gaussian Approximation Details

Assume post-PCA-whitened features X ∈ ℝ^d with X ~ N(0, diag(λ_1, ..., λ_d)), λ_i decreasing.

For a linear pool channel T = A X with A ∈ ℝ^{M × N}, A A^T = (1/p) I_M:

```
Cov(T) = A · diag(λ) · A^T
```

Under spatial-i.i.d. assumption on low-variance components (true for V_low eigenvectors by construction of "uncorrelated directions"):

```
Var(T_v, v ∈ V_low) = (1/p) · Var(X_v)
```

For high-variance components with spatial structure:

```
Var(T_v, v ∈ V_high) ≈ Var(X_v) · (1 − 1/p) · ρ_v + (1/p)
```

where ρ_v is the average spatial auto-correlation along v. Typically ρ_v → 1 for top-k components, so Var is preserved.

### Physics-info bound

Under Gaussian assumption, I(T; Y_quant) for Y_quant linearly decodable from V_low features:

```
I(T; Y_quant) = (1/2) log (1 + SNR_T)
SNR_T = Var(signal_V_low in T) / Var(noise)
SNR_X = Var(signal_V_low in X) / Var(noise)
```

Under spatial i.i.d. and linear pool:

```
SNR_T = SNR_X / p^2 = SNR_X / C
```

So:

```
Δ_quant = I(X; Y_quant) − I(T; Y_quant) = (1/2) log((1 + SNR_X) / (1 + SNR_X/C))
        ≈ (1/2) log C       (for SNR_X >> C)
```

**This gives the logarithmic scaling that matches the empirical `log₁₀(C)` factor in PhysLens-Predict.**

---

## 5. Predicted vs Empirical α*

Under IB derivation, the SCAS amplify coefficient that maximizes I(T_steered; Y_quant) is:

```
α* = √C − 1
```

**Empirical predictions per model (to validate in paper):**

| Model | Compression C | Predicted α* | Observed working range |
|---|---|---|---|
| InternVL3-8B | 2.4× | 0.55 | (not run — low compression, SCAS should have minimal effect) |
| Gemma4-E4B | 114× | 9.68 | TBD — add to Week B if time |
| Qwen2.5-VL-7B | 270× | 15.43 | TBD |
| Qwen3-VL-8B | 784× | 27.00 | observed [3, 10]; fails at > 15 |

**The observed window [3, 10] is much smaller than predicted α*=27.** This is itself a finding:

> "Our empirical observation that SCAS's working window is far smaller than the IB-predicted optimal α* suggests that the LLM's sensitivity to V_low perturbations is **sub-linear in the Gaussian-channel model**. This motivates trained interventions (PPFT) over zero-shot amplification."

---

## 6. Cross-Domain Extension (Optional — D8 or post-submission)

The IB argument is architecture-agnostic: any content→merger→LLM pipeline with spatial compression will exhibit the same subspace-selective information loss.

**Prediction:** Audio-language models with audio-to-token pooling (Qwen2-Audio, AudioLM) should show the same compression-ratio predictor for audio events that live in V_low of the audio-encoder output.

**Test (stretch):** Run PhysLens-Predict probing on Qwen2-Audio on AudioBench. If compression-ratio predicts audio-event-degradation, the paper becomes "bottleneck theory for content-language models" — broader than physics.

---

## 7. To Prove in §3.2 of the Paper

- **Prop. 3.1:** Under spatial-i.i.d. low-variance assumption, Var(T_v, v ∈ V_low) = (1/C) · Var(X_v).
- **Prop. 3.2:** Under Gaussian assumption, Δ_quant = I(X;Y) − I(T;Y) = (1/2) log C for high-SNR limit.
- **Prop. 3.3:** PhysLens-Predict score `log₁₀(C) × gap` is a Gaussian lower bound on Δ_quant normalized by a probeable architecture signal.
- **Prop. 3.4 (corollary):** Optimal zero-shot amplification α* = √C − 1; empirical window being smaller evidence of sub-linear LLM readout.

---

## 8. What Goes in Supplementary

- Full Gaussian derivation (1–2 pages of LaTeX)
- Python simulation validating Theorem 2 on synthetic pool channels
- Per-model predicted vs observed α* table with 95% CI
- Alternative SNR assumptions (what breaks when SNR_X ≈ C)

---

## 9. Honest Caveats

- Gaussian assumption on feature distribution is approximate; transformer activations are heavy-tailed.
- Spatial-i.i.d. assumption on V_low is empirical (validated by permutation test in Week 1), not derived.
- LLM downstream readout is non-linear; the α*=√C−1 prediction has error bars.
- n=8 architectures is small-sample to empirically validate log-scaling — we report LOO-CV and bootstrap but do not claim asymptotic guarantees.

---

## 10. Writing checklist (when filling this in D5–D6)

- [ ] §3.2 formalization with named propositions
- [ ] Proof sketches in main text (2 paragraphs max per theorem)
- [ ] Full proofs in supplementary
- [ ] Synthetic simulation plot: Δ_Y vs C on toy Gaussian data
- [ ] Per-model predicted-vs-observed α* table
- [ ] "Why SCAS zero-shot fails" paragraph (Theorem 4 corollary)
- [ ] Cross-domain extension paragraph (Qwen2-Audio AudioBench if we do it)
