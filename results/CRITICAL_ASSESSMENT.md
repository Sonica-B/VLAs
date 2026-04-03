# Critical Assessment — Week 1 Results

**Paper:** *Where Does Physics Live in Vision Encoders?*
**Reviewer:** Internal (pre-submission)
**Date:** 2026-04-03
**Verdict:** The pipeline works. The results are interpretable. But as of Week 1, **we have zero evidence about physics understanding in VLMs.** What we have is a validated toolchain and a proof-of-concept on a toy domain. The gap between "ViT detects brightness" and "VLMs encode intuitive physics" is enormous, and closing it is the entire contribution.

---

## 1. Raw Results Summary

### 1.1 R² — Per-Patch Linear Probes (probe_comparison.json)

| Stage | Mass (Linear) | Mass (MLP) | Friction (Lin) | Elasticity (Lin) | Stability (Lin) | Stability (MLP) |
|-------|---------------|------------|----------------|-------------------|-----------------|-----------------|
| S1 Enc-out | **0.894** | 0.933 | −0.168 | −0.181 | 0.602 | 0.578 |
| S2 Post-proj | 0.838 | 0.885 | −0.225 | −0.145 | **0.663** | 0.718 |
| S3 LLM-8 | 0.701 | 0.781 | −0.269 | −0.288 | 0.656 | 0.644 |
| S4 LLM-16 | 0.478 | 0.601 | −0.240 | −0.250 | 0.572 | 0.425 |

### 1.2 Spatial Analysis (spatial_analysis.json)

| Stage | Mass Moran's I | Stability Moran's I | Mass Precision | Mass Recall | Mass F1 |
|-------|---------------|---------------------|----------------|-------------|---------|
| S1 | 0.764 | 0.288 | 1.000 | 0.471 | 0.641 |
| S2 | 0.724 | 0.530 | 1.000 | 0.471 | 0.641 |
| S3 | 0.566 | 0.562 | 1.000 | 0.471 | 0.641 |
| S4 | −0.012 | 0.526 | 0.531 | 1.000 | 0.693 |

### 1.3 Contrastive Probing (contrastive_results.json)

| Stage | Mean AUC | Max AUC | Mean Acc |
|-------|----------|---------|----------|
| S1 | 0.636 | 0.863 | 0.597 |
| (other stages: in file but not read in full) | | | |

### 1.4 Cross-Property Correlation (cross_property_results.json)

| Stage | Mass–Stability r | Mass–Friction r |
|-------|-------------------|-----------------|
| S1 | 0.210 (p=0.003) | 0.006 (p=0.937) |
| S2 | 0.594 (p≈0) | −0.041 (p=0.566) |
| S3 | 0.490 (p≈0) | −0.047 (p=0.514) |
| S4 | 0.087 (p=0.224) | 0.196 (p=0.006) |

### 1.5 Layer-by-Layer (layer_by_layer_results.json)

- Mass mean R² peaks at **Layer 1** (0.240), then monotonically *decreases* to Layer 12 (0.038).
- Stability peaks around Layers 7–10 (~0.228–0.241), relatively flat from L3 onward.
- Friction and elasticity are flat near zero across all layers.

### 1.6 E2E Test (e2e_test/probe_results.json)

- Much sparser: almost all patches show R²=0, with a handful of isolated non-zero patches (max ~0.49).
- This appears to be a smaller/different test split — low signal.

---

## 2. The Fundamental Confound: We're Not Measuring Physics

### 2.1 What the Synthetic Data Actually Tests

The code in `src/data/synthetic_physion.py` reveals the exact visual encodings:

| Property | Visual Encoding | What the Probe Actually Learns |
|----------|----------------|-------------------------------|
| **Mass** | Color brightness (heavier = darker, factor = 1.0 − 0.25 × mass/5.0) | **Brightness detection** |
| **Friction** | Not encoded visually | Nothing (confirmed: R² < 0 everywhere) |
| **Elasticity** | Not encoded visually | Nothing (confirmed: R² < 0 everywhere) |
| **Stability** | Width/height ratio of rendered shapes | **Aspect ratio detection** |

**This is the critical problem.** The headline result — "Mass R² = 0.89 at the encoder, dropping to 0.48 at LLM layers" — does NOT mean "the encoder encodes physics that the LLM loses." It means:

> A ViT-base can detect brightness differences in early layers, and this low-level visual feature is progressively abstracted away in deeper layers where representations are optimized for semantic classification.

This is **entirely expected and well-known** from the ViT representation learning literature. Early layers preserve low-level features (edges, colors, textures); deep layers encode semantic categories. We are rediscovering a basic fact about hierarchical feature extraction, not probing physics.

### 2.2 Why the Degradation Pattern Is Uninformative

The mass R² degradation (S1: 0.89 → S4: 0.48) follows the same curve you'd get for ANY low-level visual feature (hue, saturation, edge orientation) in a classification-trained ViT. The "physics information degrades through the pipeline" framing is misleading — what degrades is the brightness signal, because later layers don't need it for ImageNet classification.

**Counter-argument one might raise:** "But the spatial clustering (Moran's I = 0.76) shows the probe localizes to objects." **Rebuttal:** Of course it does — the brightness signal IS on the objects. Background patches have uniform color (light gray + brown ground plane), so only object patches carry brightness variation. The Moran's I statistic is measuring spatial autocorrelation of "where the colored shapes are," not "where physics knowledge lives."

### 2.3 The Stability Results Are Also Confounded

Stability = width/height ratio. A ViT that can detect oriented edges and spatial extent can trivially learn aspect ratios. The R² of 0.60–0.66 for stability means "the ViT can tell tall rectangles from wide ones." Again, this is a basic geometric feature, not physics understanding.

The fact that stability R² **increases** from S1 (0.60) to S2 (0.66) is interesting but has a mundane explanation: mid-level ViT features aggregate local edge information into shape-level representations, making aspect ratio more linearly decodable at intermediate layers.

### 2.4 Friction and Elasticity: The Unintentional Control

The zero results for friction and elasticity actually serve as a useful negative control — they confirm the probe isn't picking up spurious correlations. But they also confirm the probe only works when there's a **direct visual correlate**. This undermines the claim that we're probing "physics understanding" — we're probing visual feature detection.

---

## 3. ViT-Base vs Real VLMs: The Proxy Problem

### 3.1 Architecture Mismatch

| Aspect | ViT-base-patch16-224 | Qwen2.5-VL-7B (target) |
|--------|---------------------|------------------------|
| Vision encoder | ViT-B/16 (86M params) | ViT-G (1.8B params, different patch strategy) |
| Training | ImageNet-1K classification | Web-scale image-text contrastive + instruction tuning |
| Projection | None (we simulate with layer indices) | Learned MLP projection to LLM space |
| LLM | None (we use later ViT layers as proxy) | Qwen2.5-7B transformer |
| What "S2 post-proj" means | ViT Layer 6 | Actual projection MLP output |
| What "S3 LLM-8" means | ViT Layer 9 | 8th LLM transformer layer |

**The "4 pipeline stages" in ViT-base are fundamentally different from encoder→projection→LLM in a VLM.** Layer 6 of a ViT is NOT analogous to a learned projection layer. Layer 9 of a ViT is NOT analogous to an LLM's 8th transformer block processing multimodal tokens. The stage-to-stage comparison does not transfer.

### 3.2 What the Proxy Can and Cannot Validate

**Can validate:**
- The probing pipeline runs end-to-end (data → activations → probes → R² → spatial metrics)
- The metrics (R², Moran's I, contrastive AUC) behave sensibly
- Negative controls work (friction/elasticity → zero)

**Cannot validate:**
- Whether real VLM encoders encode physics differently than ImageNet features
- Whether the projection layer is a bottleneck (it doesn't exist in ViT-base)
- Whether LoRA at different pipeline stages produces different physics QA performance
- Whether spatial clustering patterns generalize to VLMs trained on diverse visual data

### 3.3 Risk: False Confidence

The clean results from ViT-base could give false confidence that the methodology "works." A reviewer will note that the ViT-base results are consistent with the trivial explanation (brightness/aspect ratio detection) and that no evidence from actual VLMs is presented. **The ViT-base experiment is publishable as a sanity check, but it is NOT a result.**

---

## 4. Hypothesis-by-Hypothesis Assessment

### H1: Physics information is spatially structured in the encoder

**Claim:** Moran's I = 0.764 for mass at S1 → physics is localized to object regions.

**Assessment: NOT SUPPORTED by current evidence.**

- Moran's I = 0.76 measures spatial autocorrelation of R² values across the 14×14 grid.
- High-R² patches correspond to object locations because mass is encoded as object brightness.
- Any patch-level feature correlated with object appearance (color, texture, shape) would produce similar Moran's I values.
- **Critical test needed:** Moran's I for a NON-physics visual property (e.g., object color hue). If it also gives I ≈ 0.76, spatial clustering is just "objects are spatially coherent," which is trivially true.

**What would actually prove H1:** Run the same analysis on a real VLM with a physics property that has NO simple visual correlate (e.g., mass of visually identical objects with different materials). High Moran's I in that scenario would be genuinely surprising and publishable.

### H2: Physics decodability degrades through the pipeline

**Claim:** Mass R² drops 46.5% from S1 to S4.

**Assessment: TRIVIALLY TRUE for this setup, but UNINFORMATIVE about physics.**

- Low-level visual features (brightness) degrade in deeper ViT layers — this is standard hierarchical feature extraction.
- The degradation curve matches what you'd see for any low-level feature in any deep network.
- **The interesting version of H2** — that the projection layer specifically attenuates physics while preserving semantics — cannot be tested with ViT-base because there is no projection layer.

**What would actually prove H2:** Show that physics R² drops *more* at the projection layer than non-physics features (e.g., object identity, color). A differential degradation analysis on a real VLM would be novel. If everything drops equally, the projection is just a general-purpose bottleneck, not a physics-specific one.

### H3: Encoder+Projection LoRA outperforms LLM-only LoRA

**Assessment: COMPLETELY UNTESTED.** This is the paper's central and most novel contribution, and zero experiments have been run. This must be the #1 priority for Week 2.

### H4: Fine-tuning sharpens spatial encoding

**Assessment: COMPLETELY UNTESTED.** Requires pre/post LoRA comparison on a real VLM.

---

## 5. Statistical Rigor

### 5.1 Are the R² Values Meaningful?

- **Mass R² = 0.89:** Yes, this is a strong signal, well above chance. But it's measuring brightness, not physics.
- **Stability R² = 0.60:** Moderate signal. Measuring aspect ratio.
- **Friction/Elasticity R² < 0:** Correctly below constant predictor. The negative values confirm no signal.
- **Missing:** No permutation test baseline. We don't know R² under a null model (shuffled labels). For mass, R² would likely be ~0 under permutation, confirming the signal is real — but the question isn't whether the signal is real, it's whether the signal is *interesting*.

### 5.2 Is Moran's I = 0.76 Meaningful?

- Moran's I ranges from −1 (dispersed) to +1 (clustered), with E[I] ≈ −1/(n−1) ≈ −0.005 under null.
- I = 0.76 is highly significant... but what it's measuring is "object patches cluster together," which is geometrically guaranteed for any spatially contiguous object.
- **Missing:** No null distribution for Moran's I. Should compute I for random R² assignments to the patch grid to establish the expected range under the null hypothesis.

### 5.3 Precision/Recall Concerns

- S1–S3 all show identical precision=1.0, recall=0.471, F1=0.641 for mass.
- S4 shows precision=0.531, recall=1.0, F1=0.693.
- The fact that S1–S3 are **exactly identical** is suspicious. It suggests the "high-R² patch" threshold is set such that exactly 49 out of 196 patches are selected (the top 25%), and they all happen to be object patches. The number 49 = 25% of 196, suggesting a fixed percentile threshold rather than an absolute R² cutoff.
- At S4, n_high_r2_patches = 196 (ALL patches), meaning the threshold captures everything — precision drops to the base rate of object patches (104/196 = 0.531). This means R² at S4 is above threshold for all patches, which contradicts the "degradation" narrative. It suggests the threshold is poorly calibrated.

### 5.4 Bootstrap CIs

- The PRELIMINARY_RESULTS.md mentions bootstrap CIs but doesn't report them inline.
- CIs are computed over patch positions, not over scene resampling. This means they quantify spatial variability, not generalization uncertainty.
- **Scene-level CIs are essential** — they tell us whether results replicate on different synthetic scenes. Without them, we can't distinguish signal from a peculiarity of seed 42.

### 5.5 Sample Size

- 1000 scenes, 196 patches each → 196,000 patch-level observations for probe training.
- Each per-patch probe sees only 1000 samples (one per scene). With 768-dimensional features and ridge regression, this is adequate for linear probes but borderline for MLPs.
- 500 scenes for the e2e test shows much sparser R² values — possibly under-powered.

---

## 6. Novelty Assessment: What's New vs Known

### 6.1 Prior Work

| Finding | Already Shown By | Our Version |
|---------|------------------|-------------|
| Physics is decodable from vision encoder | Pixels to Principles (85% acc) | R² = 0.89 (regression, not classification) |
| Information degrades at projection | Lost in Embeddings | Mass R² drops across "stages" |
| Spatial probing of VLM representations | Various attention/saliency studies | Moran's I + patch-level R² heatmaps |
| LoRA at different stages yields different results | Standard LoRA literature | **NOT YET TESTED** |

### 6.2 Genuinely Novel Elements (if executed well)

1. **Spatial physics probing with Moran's I** — novel metric for the physics-in-VLMs literature, *if* applied to real VLMs with non-trivial physics properties.
2. **5-condition LoRA ablation** — no one has systematically compared encoder-only vs projection-only vs LLM-only LoRA specifically for physics reasoning. This is the paper's strongest potential contribution.
3. **Cross-property spatial correlation** — showing whether the same patches encode mass and stability would be interesting. Current data shows r = 0.21 at S1 (weak), r = 0.59 at S2 (moderate), declining to r = 0.09 at S4. On a real VLM this could tell a real story.
4. **Layer-by-layer physics curves** — fine-grained (all 12 layers) analysis is more detailed than prior work's coarse-grained probing.

### 6.3 Honest Assessment

With only synthetic/ViT-base results, we have **no novel findings.** Every result reduces to "ViT detects brightness and aspect ratios; this signal degrades in deeper layers." The novelty budget is entirely in the Week 2+ experiments.

---

## 7. Recommendations for Week 2

### 7.1 MUST-DO (Non-Negotiable for a Submission)

1. **Run probing on at least ONE real VLM** (Qwen2.5-VL-7B is the smallest and most practical). Extract activations at encoder output, projection output, and LLM layer 8 and 16. This is the minimum viable experiment.

2. **Use Physion++ or equivalent with REAL physics** — the mass/friction/elasticity labels must come from physics simulation, not visual encoding. Objects should be visually similar but physically different (same color, different mass). If Physion++ doesn't support this, create a controlled subset where it's true.

3. **Run the 5-condition LoRA experiment.** Even on one VLM with one physics QA benchmark. Conditions A–E compared on physics QA accuracy is the central contribution.

4. **Add a visual-feature control.** Probe for object color alongside mass. If mass Moran's I ≈ color Moran's I, spatial clustering is trivial. If mass I > color I, that's interesting.

5. **Compute permutation baselines.** Shuffle physics labels and re-run probes. Report the null distribution of R² and Moran's I.

### 7.2 SHOULD-DO (Significantly Strengthens the Paper)

6. **Differential degradation analysis.** Compare how physics R² and non-physics R² (object identity, color, size) change across pipeline stages. The paper's contribution is that physics *specifically* degrades at certain stages, not that all information does.

7. **Scene-level bootstrap CIs.** Retrain probes on bootstrap subsamples of scenes (not patches). This gives honest generalization intervals.

8. **Attention analysis.** Compute attention rollout or GradCAM for physics-relevant patches vs irrelevant patches. This complements the probing analysis with an independent method.

9. **Multi-VLM comparison.** Qwen2.5-VL vs InternVL vs LLaVA to test generality. Even 2 out of 3 would strengthen the paper significantly.

### 7.3 KILLER EXPERIMENTS (Would Make the Paper Stand Out)

10. **The "invisible physics" test.** Create scenes where two objects look identical (same color, shape, size) but have different masses. Probe for mass. If R² > 0, the VLM has learned physics beyond visual correlates — this would be a headline result. If R² = 0, it confirms VLMs rely on visual heuristics for physics.

11. **Pre/post fine-tuning spatial maps.** Show Moran's I before and after LoRA training. If LoRA at the encoder *increases* spatial clustering of physics (H4), visualize it as a saliency map. This would be an extremely compelling figure.

12. **Physics vs. semantics trade-off.** Show that LoRA conditions that improve physics QA *hurt* general VQA, or vice versa. This would demonstrate that physics lives in a different representational subspace than semantics.

### 7.4 Potential Pivots

- **Drop friction and elasticity** from the synthetic validation entirely. They serve as negative controls, but spending any more time on them with synthetic data is wasted effort. On Physion++, if friction correlates with surface texture, it becomes interesting again.
- **Reframe the ViT-base experiment** as a "pipeline validation" appendix, not a main result. Be honest: "We validated our probing methodology on a controlled setting with known ground truth."
- **Consider dropping the "4 stages" framing for ViT-base.** Layers 3/6/9/12 of a ViT are not stages of a VLM pipeline. Call them what they are: early, early-mid, late-mid, and late ViT layers. Reserve the "pipeline stage" language for real VLMs.

---

## 8. Bottom Line

### What Week 1 Accomplished
- A working end-to-end pipeline: synthetic data → activation extraction → per-patch probing → spatial analysis → visualization.
- Validated that the probing methodology produces sensible results (positive controls work, negative controls work).
- Identified the right metrics (R², Moran's I, contrastive AUC, cross-property correlation).

### What Week 1 Did NOT Accomplish
- Any evidence about physics in real VLMs.
- Any test of the paper's central hypotheses (H3, H4).
- Any result that couldn't be explained by "ViT detects brightness."

### Risk Assessment
- **If Week 2 delivers real VLM results + LoRA ablation:** The paper has a shot. The synthetic validation becomes a useful appendix, and the real results become the story.
- **If Week 2 only delivers probing on real VLMs (no LoRA):** The paper is an incremental extension of Pixels to Principles. Probably not NeurIPS-tier.
- **If the real VLM results look like the ViT-base results (physics = visual correlates):** The thesis is in trouble. We'd need to show that VLMs go *beyond* visual heuristics, or pivot to "VLMs use visual heuristics for physics and here's exactly how."

### The Single Most Important Question for Week 2

> When we probe a real VLM for mass using Physion++ scenes where mass is NOT correlated with brightness, do we get R² > 0?

If yes: the VLM has learned something about physics beyond visual shortcuts, and every other experiment builds on a solid foundation.

If no: we need to fundamentally rethink what "physics understanding" means in a VLM, or pivot to studying what visual heuristics VLMs use as physics proxies (which is also publishable, but a different paper).

---

*Assessment generated 2026-04-03. To be revisited after Week 2 results.*
