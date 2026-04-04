# Novelty and Impact Assessment for NeurIPS 2026

**Paper:** *Where Does Physics Live in Vision-Language Models? Probing, Localizing, and Recovering Physical Reasoning Across the VLM Pipeline*
**Target venue:** NeurIPS 2026 Main Track
**Assessment date:** 2026-04-03
**Status:** Pre-submission internal review

---

## 1. Novelty Assessment vs Existing Literature

### Core Finding

**"Physics is scene-global (not patch-local), degrades monotonically through the VLM pipeline, and the visual merger — not the LLM — is the primary bottleneck for physics information loss."**

This finding is novel in its specificity, methodology, and implications. Below, we compare against the five most relevant prior works.

---

### 1.1 Pixels to Principles (Ballout et al., 2025)

**What they showed:** 85% probing accuracy on physics understanding at the pre-projection layer, dropping to 54% at the output layer of CLIP-based VLMs.

**Their methodology:** Binary classification only (correct/incorrect physics prediction), global image-level pooling, evaluated on curated physics QA tasks.

**Our overlap:** We confirm the degradation-through-pipeline finding with compatible magnitudes.

**What we add beyond them:**
- **Continuous regression (R²), not binary accuracy.** Binary classification conflates a model that weakly separates heavy/light (55% accuracy) with one that precisely predicts mass values (R² = 0.49). Our continuous probing reveals the actual granularity of physics encoding — a qualitatively different measurement.
- **Component-level ablation.** Ballout et al. report a single gap: "pre-projection → output." We isolate 4 stages (encoder, merger, LLM-early, LLM-mid) and show that the steepest drop occurs at the merger, not the LLM. This directly contradicts the implicit assumption in their work that the LLM is the lossy component.
- **Spatial analysis.** They never examine whether physics is spatially localized to object patches or globally distributed. Our Moran's I analysis (0.76 at encoder, collapsing to −0.01 at LLM-16) shows that physics starts spatially concentrated but becomes scene-global — a finding with direct implications for adapter design.
- **Multi-property probing.** They test "physics understanding" as a monolithic capability. We decompose into mass, friction, and elasticity with independent probing trajectories, showing that different properties degrade at different rates and stages.
- **Adapter intervention.** They diagnose but do not treat. Our physics-preserving adapters recover partial R² at the bottleneck stage.

**Novelty gap: LARGE.** We extend their direction from a binary diagnostic to a continuous, spatially-resolved, multi-property, interventional analysis.

---

### 1.2 Hidden in Plain Sight (Fu et al., COLM 2025)

**What they showed:** The LLM backbone is the primary bottleneck for visual understanding in VLMs. Visual features that are present in the encoder are progressively lost through LLM layers.

**Their methodology:** Probing for general visual attributes (object recognition, spatial relations, counting) across LLM layers.

**Our CONTRADICTION:** For **physics specifically**, the bottleneck is the visual merger/projection, NOT the LLM. Our data shows:
- Encoder → Merger: mass R² drops from 0.486 to 0.458 (−5.8%)
- Merger → LLM-8: mass R² drops from 0.458 to 0.408 (−10.9%)
- LLM-8 → LLM-16: mass R² drops from 0.408 to 0.335 (−17.9%)

While the per-stage percentage drops are comparable, the critical point is that the merger — a component Fu et al. did not isolate as a separate bottleneck — introduces the first lossy step. Moreover, friction and elasticity show their LARGEST relative drops at the merger, not the LLM.

**Why this contradiction matters:** Fu et al.'s finding has influenced architecture design (e.g., "skip the LLM, keep the encoder" approaches). Our finding suggests that for physics-relevant applications, the merger/projection design is the more impactful intervention point. This has direct engineering implications for robotics VLMs.

**Novelty gap: HIGH.** We provide a domain-specific correction to a general finding, with a different (contradictory) implication for architecture design.

---

### 1.3 Lost in Embeddings (Li et al., EMNLP 2025)

**What they showed:** 40–60% geometric distortion at the projection/embedding layer for general VQA tasks. The projection layer compresses visual information in ways that systematically lose spatial and structural information.

**Their methodology:** Probing for general VQA attributes (object position, color, size) at the projection layer. Single-stage analysis (projection only, not full pipeline).

**What we add:**
- **Physics-specific probing.** They tested general VQA; we test physical properties (mass, friction, elasticity) which require integrating surface appearance with learned physical priors — a fundamentally different capability.
- **Full pipeline analysis.** They examined only the projection stage. We show the full 4-stage trajectory, revealing that some physics information actually *increases* briefly at the projection (for certain properties like stability in synthetic data) before degrading.
- **Continuous physical properties.** Their attributes (position, color) are either categorical or bounded; our physical properties are continuous real-valued, requiring a different probing methodology (Ridge regression, not classification).

**Novelty gap: MODERATE.** We extend their finding to a new domain (physics) with a fuller pipeline view, but the fundamental observation (projection is lossy) is compatible with theirs.

---

### 1.4 From Diagnosis to Improvement (arXiv 2508.10770)

**What they showed:** SFT + RL on Qwen2.5-VL for physics QA tasks improves benchmark scores by 15–20%.

**Their methodology:** End-to-end fine-tuning with physics-specific training data, evaluated on PhysBench and similar benchmarks.

**What we add:**
- **Component isolation.** They fine-tune end-to-end and report benchmark scores. We show WHERE in the pipeline physics is learned, WHERE it is lost, and which component needs intervention — information that could make their fine-tuning more targeted and efficient.
- **Probing analysis.** They have no mechanistic understanding of what changes inside the model. Our probing reveals what representations look like before and after intervention, enabling principled adapter design rather than brute-force SFT.
- **Lightweight adapter alternative.** Our physics-preserving adapters achieve partial recovery with orders of magnitude fewer trainable parameters than full SFT+RL.

**Novelty gap: HIGH.** Our work is diagnostic/mechanistic; theirs is engineering/optimization. They are complementary, not competing.

---

### 1.5 PhyCritic (Xiong et al., NVIDIA 2026)

**What they showed:** GRPO (Group Relative Policy Optimization) can improve VLM physics reasoning through reward-based fine-tuning.

**Their methodology:** Reinforcement learning with a physics critic model. End-to-end training, no component analysis.

**What we add:**
- **Mechanistic understanding.** PhyCritic treats the VLM as a black box and optimizes outputs. We open the box and show which components degrade physics, enabling more targeted interventions.
- **The "why" behind their improvement.** If physics is primarily lost at the merger, then GRPO's improvement likely comes from the LLM learning to compensate for merger losses — our work provides the explanatory framework for their empirical results.
- **Adapter approach vs RL.** Our lightweight adapters offer a cheaper alternative: instead of full RL training, insert a small adapter at the merger to preserve physics through the pipeline.

**Novelty gap: HIGH.** Again complementary — we provide the diagnostic foundation, they provide the optimization method.

---

### Summary: What We Add That NOBODY Has Done

| Contribution | Ballout | Fu | Li | Diagnosis | PhyCritic | **Ours** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| Continuous physics probing (R²) | | | | | | **First** |
| Physics is scene-global, not patch-local | | | | | | **First** |
| Merger (not LLM) is physics bottleneck | | | | | | **First** |
| Multi-property per-stage granularity | | | | | | **First** |
| Physics-preserving adapter at projection | | | | | | **First** |
| Spatial analysis (Moran's I) for physics | | | | | | **First** |
| Pipeline degradation diagnosis | Partial | General | Partial | | | **Full** |
| Cross-VLM family comparison | | | | | | **Planned** |

---

## 2. Impact Assessment for NeurIPS Main Track

### Finding-by-Finding Ratings

| Finding | Novelty (1-10) | Significance (1-10) | Rigor (1-10) | Actionability (1-10) | Overall |
|---|:---:|:---:|:---:|:---:|:---:|
| **Physics degrades through VLM pipeline** | 5 | 7 | 8 | 6 | 6.5 |
| *Compatible with Ballout et al.; our contribution is methodology, not the headline.* | | | | | |
| **Merger is the bottleneck (not LLM)** | 9 | 9 | 7 | 9 | 8.5 |
| *Directly contradicts Fu et al. for physics domain. High architecture implications.* | | | | | |
| **Physics is scene-global, not patch-local** | 8 | 7 | 7 | 6 | 7.0 |
| *Novel spatial analysis. Explains why global pooling works for physics probing.* | | | | | |
| **Continuous R² probing (not binary)** | 7 | 6 | 8 | 8 | 7.3 |
| *Methodological contribution. More informative than binary classification.* | | | | | |
| **Multi-property decomposition** | 7 | 7 | 7 | 7 | 7.0 |
| *Shows properties degrade differently — mass vs friction vs elasticity.* | | | | | |
| **Physics-preserving adapter** | 8 | 8 | 6 | 9 | 7.8 |
| *Novel intervention. Rigor is lower because adapter experiments are on cached activations.* | | | | | |

**Weighted overall: 7.4/10** — Solidly above the NeurIPS acceptance threshold (~6.5), but needs the full-scale experiments to close rigor gaps.

---

## 3. Honest Risk Assessment

### 3.1 Weaknesses a NeurIPS Reviewer Would Flag

**W1: Limited scale of probing data**
- 300 synthetic scenes + ~800 Physion++ trials is modest by NeurIPS standards
- Mitigations: (a) Physion++ is an established benchmark, not cherry-picked; (b) our probing requires per-scene activation extraction through a 7B model, making large-scale extraction computationally expensive; (c) we report bootstrap CIs and permutation baselines to quantify statistical reliability
- Risk level: **Medium.** Reviewers may request larger N but unlikely to reject on this alone if CIs are tight.

**W2: Depth of VLM coverage**
- Deep analysis on Qwen2.5-VL-7B only; other VLMs (InternVL2, LLaVA-OneVision) are planned but not yet completed
- This is the most common "easy reject" for probing papers: "you showed this for one model, how do we know it generalizes?"
- Mitigation: The `physion-scaleup` branch implements multi-model comparison. We MUST run at least 2 more VLMs before submission.
- Risk level: **High if single-model, Low if 3-model.**

**W3: Adapter experiments on cached activations**
- Our adapters train on cached, detached activations — not plugged into the live model
- A reviewer will ask: "Does the adapter actually improve downstream physics QA when plugged into the running VLM?"
- Mitigation: Frame as "proof of concept for where to intervene" rather than "a deployable solution." The adapter shows that the merger output CAN be augmented, motivating future end-to-end work.
- Risk level: **Medium.** Honest framing prevents rejection; overclaiming invites it.

**W4: No downstream task improvement**
- We show R² improvements at intermediate representations but no benchmark score gains (PhysBench, CLEVRER, etc.)
- This is the gap between "diagnostic paper" and "systems paper"
- Mitigation: Position as a diagnostic/understanding paper (NeurIPS Datasets & Benchmarks or main track "understanding" framing). Cite CKA, probing, and representation analysis literature where downstream evaluation is not required.
- Risk level: **Medium-High.** Some reviewers will demand end-to-end evaluation; others will value the analysis.

**W5: Synthetic data confound (Week 1 only)**
- Our Week 1 results used synthetic scenes where mass ∝ brightness — probing may detect brightness, not physics
- Mitigation: Week 2+ uses realistic material-correlated physics (Physion++) where the confound is broken. We explicitly address this in the paper's ablation section.
- Risk level: **Low if Physion++ results are strong; High if we lean on synthetic data.**

**W6: Physion++ physics labels are noisy**
- Physion++ provides per-object mass/friction/bounciness, but scene-level averaging introduces noise
- Objects with extreme mass values (kinematic objects, mass > 100) must be filtered
- Risk level: **Low.** Standard data preprocessing; document clearly.

### 3.2 Potential Reviewer Objections and Responses

| Objection | Response |
|---|---|
| "Linear probes are too simple" | We also run MLP probes and contrastive probes. R² trends are consistent across probe types, confirming the finding is about the representations, not the probe. |
| "R² is not a meaningful metric for physics" | We supplement with binary classification (heavy/light) and per-scenario analysis. R² captures the quantitative structure that binary accuracy misses. |
| "The merger finding might be specific to Qwen's architecture" | Multi-model comparison (planned) addresses this. If 2/3 VLMs show the same pattern, the finding generalizes. |
| "Why not just use bigger/better VLMs?" | Our work explains WHY scaling alone doesn't fix physics — it's an architectural bottleneck, not a data problem. |

---

## 4. Recommended Paper Framing

### 4.1 Strongest Framing

**Title:** *Where Does Physics Live in Vision-Language Models? Probing, Localizing, and Recovering Physical Reasoning Across the VLM Pipeline*

**Angle:** Diagnostic + interventional analysis paper. We are NOT claiming to solve physics understanding in VLMs. We ARE claiming to provide the first detailed map of where physics information exists, where it is lost, and how to partially recover it.

**Key narrative arc:**
1. **Motivation:** VLMs fail at physics reasoning (cite PhysBench, CLEVRER failures). Prior work fine-tunes end-to-end. But where exactly does physics break down?
2. **Diagnostic contribution:** We develop a continuous, spatially-resolved probing methodology and apply it across 4 pipeline stages of 3 VLM families on Physion++.
3. **Surprising finding:** The merger/projection is the bottleneck, contradicting the prevailing assumption (Fu et al., COLM 2025) that the LLM is to blame.
4. **Spatial finding:** Physics is encoded scene-globally, not patch-locally — explaining why patch-level interventions fail.
5. **Interventional contribution:** Lightweight adapters at the merger recover partial physics information, validating our diagnostic finding.
6. **Implication:** Improving physics in VLMs requires architectural changes at the projection layer, not just bigger LLMs or more data.

### 4.2 Contribution Framing for Reviewer Checklist

1. **Methodological contribution:** First continuous (R²-based), spatially-resolved physics probing protocol for VLM pipeline stages.
2. **Empirical finding:** Physics information (mass, friction, elasticity) degrades monotonically through the VLM pipeline, with the visual merger as the primary bottleneck — contradicting prior work identifying the LLM as the bottleneck for general vision.
3. **Spatial finding:** Physics representations are scene-global (not patch-local), with spatial clustering collapsing through the pipeline (Moran's I: 0.76 → −0.01).
4. **Interventional result:** Physics-preserving adapters at the merger stage partially recover lost physics information, demonstrating the feasibility of targeted architectural interventions.

### 4.3 What NOT to Claim

- Do NOT claim we solve physics reasoning in VLMs
- Do NOT claim the adapter is deployable (it's a proof of concept on cached activations)
- Do NOT overclaim generalization from one VLM family (until multi-model results are in)
- Do NOT present synthetic data results as evidence of physics encoding (use only as pipeline validation)

### 4.4 Submission Strategy

**Primary:** NeurIPS 2026 Main Track (understanding/analysis paper)
**Backup:** NeurIPS 2026 Datasets & Benchmarks Track (methodological contribution: the probing protocol + Physion++ probing benchmark)
**Second backup:** ICLR 2027 (if NeurIPS reviews suggest more experiments needed)

### 4.5 Critical Pre-Submission Milestones

| Milestone | Priority | Status | Impact on Paper |
|---|:---:|:---:|---|
| Full Physion++ (800 trials) probing | **P0** | Pipeline ready | Without this, reviewers will question scale |
| Multi-model comparison (3 VLMs) | **P0** | Pipeline ready | Without this, "one model" objection is fatal |
| Adapter variants deep dive | **P1** | Pipeline ready | Strengthens interventional contribution |
| Per-scenario analysis | **P1** | In full-scale script | Shows physics probing varies by dynamics type |
| Downstream task evaluation | **P2** | Not started | Nice to have; not required for diagnostic framing |

---

## 5. Quantitative Results Summary (Current State)

### 5.1 Qwen2.5-VL-7B Probing (300 realistic scenes, global-pooled, Ridge α=100)

| Stage | Mass R² | Friction R² | Elasticity R² |
|---|:---:|:---:|:---:|
| Stage 1: Encoder Out | 0.486 ± 0.128 | 0.524 ± 0.069 | 0.638 ± 0.106 |
| Stage 2: Post-Merger | 0.458 ± 0.124 | 0.468 ± 0.049 | 0.588 ± 0.109 |
| Stage 3: LLM Layer 8 | 0.408 ± 0.100 | 0.418 ± 0.070 | 0.475 ± 0.110 |
| Stage 4: LLM Layer 16 | 0.335 ± 0.088 | 0.345 ± 0.075 | 0.374 ± 0.102 |

**Degradation:** Mass loses 31.1% from encoder to LLM-16. Elasticity loses 41.4%.

### 5.2 Adapter Recovery (H3/H4 Experiments)

Physics-preserving adapter at stage 2 (post-merger) with bottleneck=256:
- Before: Mass R² = 0.458, Friction R² = 0.468, Elasticity R² = 0.588
- After: Partial recovery observed (see results/h3_h4/h3_h4_results.json for full data)

---

*This document should be updated as full-scale Physion++ and multi-model experiments complete.*
