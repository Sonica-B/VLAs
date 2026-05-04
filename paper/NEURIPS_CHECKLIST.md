# NeurIPS 2026 Paper Checklist

> **Mandatory.** Papers without this checklist are desk-rejected. Includes 16 questions; each requires a [Yes]/[No]/[N/A] answer with a 1-2 sentence justification. The checklist follows the references in the camera-ready and does NOT count toward the 9-page limit.

This file is the answer-key for our submission. Convert each block to LaTeX using the official `\answerYes{}`, `\answerNo{}`, `\answerNA{}` macros from `neurips_2026.sty`.

---

## 1. Claims

**Question:** Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

**Answer:** [Yes]

**Justification:** The abstract states: (a) we release PhysBench-Diag as an evaluation resource; (b) we pre-register a closed-form predictor with a kill-gate; (c) the predictor *fails* its primary kill-gate at n=10 (verified: median |error| = 0.225 > 0.20 threshold; Spearman ρ = 0.327, p = 0.357); (d) post-hoc mechanism-stratified analysis reveals a clean categorical separation (verified: spatial-merge mean H3 = 0.667, learned-resampler = 0.000, no-compression = 0.222). Every numeric claim traces to `results/week1_turing/phys_lens_predict_weekb.json` or `results/week1_turing/<model>_permutation_check.json` (Section 3 + Appendix B). Limitations are flagged honestly (Section 7).

---

## 2. Limitations

**Answer:** [Yes]

**Justification:** Section 7 ("Limitations") enumerates seven explicit limitations: L1 sample size n=10 with wide bootstrap CIs, L2 linear probes only (nonlinear bottlenecks invisible), L3 single benchmark (PhysBench-Diag only), L4 H3 hit-rate definition asymmetry between baseline (manual) and active (auto-extracted) values, L5 quantization protocol asymmetry (Granite-Vision in bf16, others NF4), L6 excluded models (Pixtral, Molmo) with documented integration bugs, L7 mechanism taxonomy is coarse 3-way classification.

---

## 3. Theory assumptions and proofs

**Answer:** [N/A]

**Justification:** The paper makes no formal theoretical claim. The information-bottleneck framing in §2 and §6.2 is offered as motivating intuition, not formal proof. The predictor `score = log₁₀(C) × Δprobe` is a closed-form heuristic with explicit kill-gate validation; we make no claim of analytical optimality.

---

## 4. Experimental result reproducibility

**Answer:** [Yes]

**Justification:** Section 8 ("Reproducibility Statement") provides the exact env spec (torch 2.4.1+cu124, transformers 4.49.0, bnb 0.44.1, full pin in `turing/requirements_v2.txt`), the SLURM scripts that ran each model (`turing/{10,12,14,16,17}_probe_*.sh`), the permutation runner (`turing/15_permutation_active.sh`), the predictor entrypoint (`scripts/phys_lens_predict.py`), the figure regeneration (`scripts/figures/generate_paper_figures.py`), and a regression test suite (`tests/regression.py`, 93/93 passing). Total compute documented: ~3 GPU-hours + ~30 CPU-min on A100-80GB.

---

## 5. Open access to data and code

**Answer:** [Yes]

**Justification:** PhysBench-Diag *labels* (deterministic from PhysBench v2 sub_type), all 10 per-model probing JSONs, all 6 permutation_check JSONs, the predictor output JSON, the figure-generation script, and the SLURM/probing pipeline are released openly at `github.com/Sonica-B/VLAs` (branch `paper-submission-clean` for the immutable submission artifact, `physics-steering` for the working version). The repository is a public GitHub repository accessible without PI request. The underlying PhysBench v2 images/videos are obtained from the original PhysBench release (CC-BY-4.0). License of our derivative artifacts: TBD (recommended: MIT for code, CC-BY-4.0 for labels).

---

## 6. Experimental setting/details

**Answer:** [Yes]

**Justification:** Section 4 specifies the model panel (Table 1), probing protocol (mean-pool sequence dim, sklearn LogisticRegression with 5-fold CV, 4 sites per model), the predictor formula and locked thresholds (per `docs/PRE_REGISTRATION.md`), the LOO regression protocol (1-D linear fit, predict held-out, clip to [0,1]), and the quantization protocol. The 14-of-200 missing-media items are documented (`extraction_stats.errors=14` per probe JSON) with sample IDs traceable in `*_quant_qual_probe.json`.

---

## 7. Experiment statistical significance

**Answer:** [Yes]

**Justification:** Permutation tests with n=200 random label shuffles compute one-sided p-values for every (model × target × site × slice) probe (Section 4.3, results in §5.1). Spearman correlation includes p-value (one-sided, 0.357) and 95% bootstrap CI ([-0.643, 0.887], n=10) (§5.2). LOO median |error| is reported with per-model breakdown in Table 2. The exploratory mechanism-stratification t-test (§5.3) is flagged as exploratory because the grouping was not pre-registered; we report (t=2.91, p=0.027, df=3) with the explicit caveat.

---

## 8. Experiments compute resources

**Answer:** [Yes]

**Justification:** Section 8 documents: hardware (NVIDIA A100-80GB SXM4), per-model probing time (5–30 min depending on model size), per-model permutation test time (3–5 min on CPU, sklearn lbfgs on PCA-128 features), aggregator runtime (~1 min), total project compute (~3 GPU-hours active probing + ~30 CPU-min for permutation tests). Models that did not make it to the final panel (Pixtral, Molmo, Idefics3 backup) consumed an additional ~2 GPU-hours collectively. Storage: probing feature caches ~500 MB total.

---

## 9. Code of ethics

**Answer:** [Yes]

**Justification:** The research uses publicly-released open-weight VLMs and a publicly-released physics benchmark (PhysBench v2). No human subjects research. No PII. No scraped data. The pre-registered predictor failure is reported honestly per the locked kill-gate in `docs/PRE_REGISTRATION.md`; we explicitly demote the predictor rather than reframe results post-hoc, in adherence to scientific integrity norms. The release of PhysBench-Diag (a derivative resource) preserves the original PhysBench license.

---

## 10. Broader impacts

**Answer:** [Yes]

**Justification:** Positive: the mechanism-vs-ratio finding informs VLM developers' architectural choices (Section 6.3) — deployment use-cases requiring quantitative-physics accuracy may benefit from learned token-reduction over deterministic spatial-merge. Negative: probing analyses could be used to pre-screen VLMs for jailbreak vectors or to identify architectural vulnerabilities; we mitigate this risk by releasing only standard probing tools (linear probes on activations) that do not enable novel attack capabilities beyond what existing probing literature already provides.

---

## 11. Safeguards

**Answer:** [N/A]

**Justification:** No new high-risk model is released. The PhysBench-Diag dataset is a *labels-and-probing-results* derivative of an existing public benchmark (PhysBench v2 by Chow et al., 2024) — it does not contain new images, videos, or generative model weights. The probing-feature caches are plain NumPy arrays of mean-pooled activations, posing no misuse risk.

---

## 12. Licenses for existing assets

**Answer:** [Yes]

**Justification:** Each VLM in the panel is cited with its release paper (Table 1 + References). Licenses: Qwen3-VL-8B (Apache-2.0), Qwen2.5-VL-7B (Apache-2.0), Gemma3-4B (Gemma terms), InternVL3-8B (MIT), LLaVA-OneVision-7B (Apache-2.0), Phi-3.5-Vision (MIT), Granite-Vision-3.2-2B (Apache-2.0), Idefics3-8B-Llama3 (Apache-2.0), Idefics2-8B (Apache-2.0), BLIP-2 OPT-2.7B (MIT). PhysBench v2: CC-BY-4.0. PyTorch: BSD-3. transformers: Apache-2.0. bitsandbytes: MIT. All cited and license-respected.

---

## 13. New assets

**Answer:** [Yes]

**Justification:** New released assets: (a) PhysBench-Diag *partition* (deterministic quant/qual labels derived from PhysBench v2 `sub_type`); (b) per-model probing JSONs (10 models × 4 sites × 3 targets × 3 slices); (c) per-model permutation_check JSONs (6 active models × 200 shuffles); (d) predictor output JSON. All documented via Datasheet (`docs/DATASHEET_PHYSBENCH_DIAG.md`, Gebru et al., 2018 template, 7 sections + 3 appendices). Croissant ML metadata is provided as `paper/croissant_metadata.json` (per `paper/CROISSANT_METADATA.md`).

---

## 14. Crowdsourcing and research with human subjects

**Answer:** [N/A]

**Justification:** No human subjects research, no crowdsourcing. All annotations are deterministic functions of PhysBench's existing taxonomy.

---

## 15. Institutional review board (IRB) approvals

**Answer:** [N/A]

**Justification:** No human subjects research; IRB review not applicable.

---

## 16. Declaration of LLM usage

**Answer:** [Yes]

**Justification:** LLMs (Anthropic Claude) were used as a coding-assistance tool during development of the probing harness, SLURM scripts, and figure-generation code. No LLM was used to derive the predictor formula, pre-register the kill-gate threshold, design the experimental protocol, or interpret the empirical results. All scientific claims, hypotheses, and conclusions are author-originated. LLM-assisted code was reviewed and validated by the author; the regression suite (`tests/regression.py`, 93 tests) verifies code correctness independently.

---

## ✓ Pre-submission verification checklist

Before final submission, verify:
- [ ] All 16 answers above are filled with current paper section/file references
- [ ] Justifications match the actual paper content (no orphan claims)
- [ ] Repo URL is anonymized for double-blind (replace `Sonica-B/VLAs` with `anonymous-NeurIPS2026/physbench-diag`)
- [ ] Croissant JSON-LD validates (`croissant-validator paper/croissant_metadata.json`)
- [ ] All artifact URLs resolve without authentication
- [ ] Page count: 9 pages MAX content (refs/checklist/appendix uncounted) — verify in final PDF
- [ ] Style: `\usepackage{neurips_2026}` (no options at submission time = anonymized)
- [ ] Fonts: pdflatex-generated, only Type-1 / Embedded TrueType (verify with `pdffonts`)
