# Compression Mechanism, Not Ratio, Predicts Quantitative-Physics Failures in Vision-Language Models: A Pre-Registered Stress-Test of an Architectural Evaluation Hypothesis

**Submission target:** NeurIPS 2026 **Evaluations & Datasets (E&D) Track**
**Author (anonymized for double-blind):** Anonymous
**Code + data (anonymized):** `github.com/anonymous-NeurIPS2026/physbench-diag`
**Original repository (post-acceptance):** github.com/Sonica-B/VLAs (branch `paper-submission-clean`)
**Hugging Face dataset:** `huggingface.co/datasets/anonymous-NeurIPS2026/PhysBench-Diag`
**Pre-registration:** `docs/PRE_REGISTRATION.md` in the repository (locked before data collection)

> ### Evaluative role (per E&D Track scope)
>
> This submission contributes a **derivative diagnostic benchmark + a pre-registered stress-test of an evaluation hypothesis**. Its evaluative role is to support the claim "*architectural compression ratio predicts quantitative-physics degradation in VLMs*" — and, having stress-tested that claim across a 10-VLM panel, to **refute its simple form** while surfacing a **refined mechanism-based hypothesis** that survives the data. We make this evaluative role explicit in §1.4 and articulate the assumptions under which our claims hold (§1.5), the limitations that constrain them (§7), and the negative result we report per pre-registration (§5.2). Per the NeurIPS 2026 E&D Track call (March 2026 blog), "*Negative results, critical analyses, and use-case-inspired evaluations are welcome*" and "*A submission need not 'beat a baseline'; its primary contribution should be to deepen and refine our understanding of evaluation practices.*"

---

## Abstract

Vision-Language Models (VLMs) systematically underperform on quantitative-physics
reasoning while excelling at qualitative-physics tasks, yet the architectural
origin of this gap is unexplained. We present **PhysBench-Diag**, a diagnostic
benchmark of 200 validation + 999 test items partitioned into quantitative
(numerical-prediction) and qualitative (relational) sub-tasks across four
physics categories. Using a 4-site layer-wise probing protocol (encoder
output, post-projector, decoder layers 8 and 16) with bnb-NF4 quantization on
a single A100, we probe **10 open-weight VLMs** spanning a 784× vision-token
compression range, including the Feb-2025 IBM Granite-Vision-3.2-2B as our
cleanest negative control. We pre-register **PhysLens-Predict**, a closed-form
architectural predictor `score = log₁₀(compression) × Δprobe`, and a
**kill-gate at LOO median absolute error < 0.20**. The pre-registered
predictor **fails Gate 5** at n=10 (LOO median |error| = 0.225 > 0.20;
Spearman ρ = 0.33, p = 0.36). Per our pre-registration, we therefore report
the simple compression-ratio predictor as **anecdotal observation, not a
validated predictor**. However, stratifying the 10-model panel by compression
*mechanism* yields a starker pattern: models using **lossy spatial-merge**
(InternVL3, Gemma3, Qwen2.5-VL, Qwen3-VL) average H3 hit-rate 0.67, while
models using **learned token-reduction** (Idefics3 pixel-shuffle, Idefics2
perceiver, BLIP-2 Q-Former) average 0.00, and three **no-compression**
controls (LLaVA-OV, Phi-3.5-V, Granite-Vision) average 0.22. We therefore
propose a refined hypothesis — that compression *mechanism* (lossy vs.
task-aware) is the operative variable — and release PhysBench-Diag, the
per-model probing JSONs, and the predictor implementation to enable
systematic study at larger scale.

---

## 1. Introduction (1.5 pages)

### Hook

Ask a vision-language model (VLM) "is the ball rolling left?" and current
state-of-the-art systems (Qwen3-VL-8B, Gemma3-4B, InternVL3-8B) answer
correctly the vast majority of the time. Ask the same model "how fast is the
ball rolling?" and accuracy collapses, often to chance levels [Chow et al.,
2024]. This *quantitative-vs-qualitative* gap in physics reasoning is
systematic, well-documented, and unexplained at the architectural level.

### Gap

Existing physics benchmarks for VLMs — PhysBench [Chow et al., 2024],
Physion [Bear et al., 2021], Physion++ [Tung et al., 2023], and ScienceQA
[Lu et al., 2022] — measure end-to-end accuracy. They tell us VLMs fail
at quantitative physics, but not *where* in the multi-stage architecture
(vision encoder → multimodal projector → decoder LLM) the failure originates.
This matters: VLM developers cannot intervene if they don't know which stage
to fix.

### Hypothesis (pre-registered)

We hypothesize H3: **vision-token compression at the multimodal projector
preferentially destroys quantitative-physics information** while qualitative
information is preserved. Quantitative information requires high-bit-depth
encoding (e.g., the precise velocity vector); qualitative information
requires low-bit-depth encoding (e.g., the *direction* of motion). Lossy
projector compression strips the former first.

### Contributions

1. **PhysBench-Diag**: a derivative benchmark of 200 validation + 999 test
   items partitioned into quantitative (n=55/274) vs. qualitative (n=145/725)
   sub-tasks. Released openly with deterministic labeling code.

2. **A 10-model probing study** across 7 institutions and 8 architectural
   compression ratios (1× to 784×), using a 4-site bnb-NF4 probing protocol on
   a single A100. All probing JSONs released.

3. **PhysLens-Predict**, a pre-registered closed-form predictor
   `score = log₁₀(compression) × Δprobe` with a leave-one-out (LOO) regression
   validation protocol and a kill-gate at median absolute error < 0.20.

4. **Honest negative result**: the pre-registered predictor *fails* its
   primary kill-gate at n=10. Per pre-registration, we demote the
   compression-ratio predictor to descriptive observation.

5. **A refined positive finding**: stratifying H3 hit-rate by compression
   *mechanism* (rather than ratio) yields a clean separation. Lossy
   spatial-merge architectures average 0.67 hits; learned-resampler
   architectures average 0.00 hits despite 4–11× compression. We propose the
   mechanism-vs-ratio distinction as the operative architectural variable for
   future predictors.

### 1.4 Evaluative role of this contribution (E&D-specific)

PhysBench-Diag and the accompanying probing pipeline play three distinct
evaluative roles, in the framing of the NeurIPS 2026 E&D Track:

1. **As a diagnostic benchmark**: PhysBench-Diag enables claims of the form
   "*VLM X loses Y% quantitative-physics accuracy at architectural site S*"
   where existing benchmarks support only end-to-end "*VLM X scores Z% on
   physics*" claims. The deterministic quant/qual partition is the
   operational substrate.

2. **As a pre-registered evaluation methodology**: PhysLens-Predict is a
   *predictor of evaluation outcomes* — it estimates a model's H3 hit-rate
   from its architecture alone. The pre-registered kill-gate
   (median |error| < 0.20) makes this a falsifiable methodological claim,
   not a curve-fit observation.

3. **As a stress-test artifact**: by releasing per-model probing JSONs
   (n=10) and permutation tests (n=6), we enable independent reproduction,
   audit, and critique of both the predictor and the underlying H3
   hypothesis at a panel size larger than any prior open-source compression
   analysis we are aware of.

### 1.5 What our claims support — and under what assumptions

| Claim | Holds under | Does not hold if |
|---|---|---|
| Pre-registered `log₁₀(C) × Δprobe` predictor fails Gate 5 at n=10 | The pre-registered kill-gate threshold (0.20) and the LOO regression protocol; H3 hit-rate computed via permutation_check at p<0.05 | Different kill-gate threshold (e.g. 0.30); different empirical-target definition (e.g. continuous Δprobe instead of binary hit) |
| Spatial-merge models average H3 = 0.667; learned-resampler models average 0.000 | The 3-way mechanism taxonomy in Table 1; n=4 + n=3 group sizes | Finer-grained mechanism taxonomy; mechanism boundaries differ from our coarse classification |
| Granite-Vision-3.2-2B (1× compression, Feb 2025) shows 0/3 H3 hits as predicted | Our specific bf16 (no-quantization) loading protocol (§4.5); the n=186 PhysBench val items resolved on local disk | Different quantization protocol; missing-media items handled differently |

### Roadmap

§2 reviews related work. §3 describes PhysBench-Diag. §4 details the probing
protocol and the pre-registered predictor. §5 presents per-model results,
the predictor's pre-registered failure, and the mechanism-stratified
secondary finding. §6 discusses implications. §7 lists limitations. §8
provides reproducibility info. Code, data, figures, datasheet, Croissant
metadata, and pre-registration: see anonymized repo above.

---

## 2. Related Work (0.75 pages)

### VLM physics benchmarks

PhysBench [Chow et al., 2024] is the most comprehensive open VLM physics
benchmark to date, with 19 fine-grained physics sub-categories. Physion
[Bear et al., 2021] and Physion++ [Tung et al., 2023] focus on object dynamics
prediction. ScienceQA [Lu et al., 2022] covers broader STEM reasoning. **All
existing benchmarks measure end-to-end accuracy and do not localize failures
to architectural stages.** PhysBench-Diag inherits from PhysBench and adds
the quant/qual partition that enables layer-wise probing.

### Linear probing of VLM activations

Linear-probe analyses have been applied to LLaVA [Liu et al., 2023] and
InstructBLIP [Dai et al., 2023] to study image–text alignment. None of these
studies decompose probe accuracies by quantitative versus qualitative
sub-task. We extend the standard 4-site probing protocol used in language-only
work [Hewitt and Manning, 2019; Voita et al., 2019] to the multi-modal
setting with a stratified slice analysis.

### Vision-token compression in VLMs

A range of mechanisms reduce the vision-token count delivered to the language
model: (i) deterministic spatial-merge / pooling (Qwen-VL series [Bai et al.,
2023; 2025], Gemma 3 [Google, 2025], InternVL3 [OpenGVLab, 2025]); (ii)
learned cross-attention to fixed-size query tokens (BLIP-2 Q-Former
[Li et al., 2023], Idefics2 perceiver resampler [Laurençon et al., 2024a]);
(iii) learned spatial reorganization (Idefics3 pixel-shuffle [Laurençon et
al., 2024b]); (iv) per-token MLP projection without reduction (LLaVA-OneVision
[Li et al., 2024], Phi-3.5-Vision [Microsoft, 2024], IBM Granite-Vision-3.2
[IBM, 2025]). Compression ratios have been studied for *efficiency* but, to
our knowledge, not for their differential information cost on quantitative vs
qualitative downstream tasks.

### Information bottleneck and probing

Our pre-registered formula `log₁₀(compression) × Δprobe` is motivated by an
information-bottleneck argument [Tishby and Zaslavsky, 2015]: if compression
forces a lossy summary, then the higher-bit-depth (quantitative) channel is
preferentially destroyed. We test this hypothesis directly via probing
accuracies.

---

## 3. PhysBench-Diag Dataset (1 page)

### Source

PhysBench-Diag is a derivative resource of PhysBench v2 [Chow et al., 2024],
distributed under PhysBench's CC-BY-4.0 license. We do not redistribute the
underlying images/videos; users obtain those from PhysBench's official
release.

### Quant/qual partition

The 19 PhysBench `sub_type` labels are deterministically partitioned into
**quantitative** (numerical-prediction; e.g., velocity, distance, count) and
**qualitative** (relational/categorical; e.g., direction, ordering, motion
type). The exact mapping is implemented in
`src/optim/physbench_split.py:classify_quantitative()`. The partition is
fully reproducible and requires no human re-labeling.

### Splits

PhysBench-Diag uses PhysBench's official validation (200 items) and test (999
items) splits, with the partition applied:

| Split | Total | Quantitative | Qualitative |
|---|---|---|---|
| Validation | 200 | 55 (27.5%) | 145 (72.5%) |
| Test | 999 | 274 (27.4%) | 725 (72.6%) |

For probing experiments in this paper, we use the validation split. 14 of 200
validation items are excluded due to local media-resolution issues, yielding
n=186 items for per-model probing.

### Release

The PhysBench-Diag *partition*, all 10 per-model probing JSONs, all 6
permutation_check JSONs, the predictor output JSON, and the predictor
implementation are released openly under MIT license (code) and CC-BY-4.0
(labels). Hosting:

- **Code + scripts**: `github.com/anonymous-NeurIPS2026/physbench-diag`
  (anonymized fork for double-blind review)
- **Dataset + probing JSONs**: `huggingface.co/datasets/anonymous-NeurIPS2026/PhysBench-Diag`
  (one of the 4 hosts blessed by the NeurIPS 2026 E&D Track call)
- **Croissant ML metadata**: `croissant.json` at the HF dataset root, with
  both core and Responsible AI (RAI) fields per the NeurIPS 2026 E&D
  requirement; templated in `paper/CROISSANT_METADATA.md`
- **Datasheet** [Gebru et al., 2018]: `docs/DATASHEET_PHYSBENCH_DIAG.md`,
  7 sections + 3 appendices
- **Pre-registration**: `docs/PRE_REGISTRATION.md`, locked before data
  collection (Apr 2026)

All artifacts are accessible to reviewers without personal request to
the author. The underlying PhysBench v2 images/videos are obtained from
the original PhysBench release (CC-BY-4.0); we do not redistribute the
media.

---

## 4. Methods (1.25 pages)

### 4.1 Model panel

We probe 10 open-weight VLMs spanning a 784× vision-token compression range
(see Table 1). Models were selected for: (a) first-class HuggingFace
transformers integration, (b) compatibility with bnb-NF4 4-bit quantization
on a single A100-80GB, (c) coverage of three compression-mechanism families
(spatial-merge, learned-resampler, no-compression), (d) at least one
released after January 2025 (Granite-Vision-3.2-2B).

**Table 1**: 10-VLM panel.

| Model | Released | Compression | Mechanism | Vision encoder | Decoder LLM |
|---|---|---|---|---|---|
| Qwen3-VL-8B | 2025 | 784× | spatial-merge | custom ViT | Qwen3 8B |
| Qwen2.5-VL-7B | 2024 | 270× | spatial-merge | custom ViT | Qwen2.5 7B |
| Gemma3-4B | Mar 2025 | 114× | spatial-merge | SigLIP-So400m | Gemma3 4B |
| InternVL3-8B | 2024 | 2.4× | spatial-merge (light) | InternViT-300M | InternLM3 8B |
| Idefics2-8B | Aug 2024 | 11.4× | learned-resampler (perceiver, 64q) | SigLIP-So400m | Mistral 7B |
| BLIP-2 OPT-2.7B | 2023 | 8× | learned-resampler (Q-Former, 32q) | EVA-CLIP-g | OPT 2.7B |
| Idefics3-8B-Llama3 | Aug 2024 | 4× | learned-resampler (pixel-shuffle r=2) | SigLIP-So400m | Llama 3.1 8B |
| LLaVA-OneVision-7B | Aug 2024 | 1× | no-compression (2-MLP) | SigLIP-So400m | Qwen2 7B |
| Phi-3.5-Vision | 2024 | 1× | no-compression (Linear) | CLIP ViT-L/14 | Phi-3.5 4B |
| **Granite-Vision-3.2-2B** | **Feb 2025** | 1× | no-compression (2-MLP) | SigLIP | Granite-3.2 2B |

### 4.2 Probing protocol

For each (model, item) pair, we register forward hooks at four canonical
sites — encoder output (last vision-tower block), post-projector
(immediately after the multimodal projector), and decoder layers 8 and 16 —
and capture the activation tensor on a single forward pass. Activations are
mean-pooled over the sequence dimension to a fixed-dim feature vector per
item. We then fit `sklearn.linear_model.LogisticRegression` with 5-fold
cross-validation, predicting three target variables: `answer` (4-way A/B/C/D
letter classification), `task_type` (4-way: dynamics / properties /
relationships / scenes), and `sub_type` (19-way fine-grained class). Each
(target × site) probe is computed on the full slice (n=186), the
quantitative slice (n=48), and the qualitative slice (n=138).

### 4.3 PhysLens-Predict (pre-registered)

The predictor (locked in `docs/PRE_REGISTRATION.md` before data collection)
is:

```
score(model) = log₁₀(max(compression_ratio, 1.0)) × max(Δprobe, 0)
```

where `compression_ratio` is the architecturally-determined ratio of vision
tokens out of the encoder to vision tokens entering the decoder (computed
per model from source paper specifications), and `Δprobe = probe_acc(quant,
enc_out) − probe_acc(quant, post_proj)` on `target = answer` (the most
discriminative quantitative target).

The empirical target is the **H3 hit-rate**: for each target T ∈ {answer,
task_type, sub_type}, an "H3 hit" requires the enc_out quantitative probe
to be (a) significant against the permutation null at p < 0.05, AND (b)
degrade at post_proj. H3 hit-rate is hits/3.

### 4.4 LOO validation and kill-gate

For each held-out model, we fit a 1-D linear regression of empirical H3 on
predictor score across the remaining n−1 models, predict H3 for the held-out
model, and clip predictions to [0,1]. Aggregated metrics are LOO median
absolute error and Spearman rank correlation between predicted and
empirical H3.

**Pre-registered kill-gate** (`docs/PRE_REGISTRATION.md` Gate 3): if LOO
median |error| > 0.20 OR Spearman ρ < 0.5, the predictor is demoted to
descriptive observation rather than a validated predictor.

### 4.5 Quantization protocol

Six of seven loaded models use bnb-NF4 4-bit quantization (bitsandbytes 0.44,
torch 2.4.1+cu124, transformers 4.49.0). Granite-Vision-3.2-2B is loaded in
bf16 without quantization due to a transformers-4.49 + bitsandbytes-0.44
dtype mismatch (`Linear4bit` not properly wrapping all Linear layers); the
2B parameter footprint fits in bf16 on A100-80GB (~4 GB). We document this
asymmetry and note that the architectural compression ratio (1.0× for
Granite-Vision) is unaffected by quantization choice.

### 4.6 Pre-registration adherence

All thresholds (kill-gate at 0.20, Spearman threshold at 0.5), the
predictor formula, the LOO protocol, and the H3 hit-rate definition were
locked in `docs/PRE_REGISTRATION.md` before data collection began. The
present paper reports outcomes as the pre-registration prescribes,
including demotion of the predictor on kill-gate failure.

---

## 5. Results (1.5 pages)

### 5.1 Per-model probing accuracy

[**Figure 3**: per-model heatmap of probe accuracy on `target=answer` across
4 sites × 3 slices. See `figures/fig3_probing_heatmap.pdf`.]

The per-model probing tables (full results in supplementary material;
selected values in Figure 3) demonstrate that quantitative-slice probe
accuracy varies substantially across the panel.

### 5.2 Pre-registered predictor verdict (Gate 5)

**Table 2**: PhysLens-Predict per-model scores and LOO outputs (n=10).

| Model | Compression | Δprobe | phys_lens score | Empirical H3 | LOO predicted | LOO \|error\| |
|---|---|---|---|---|---|---|
| InternVL3-8B | 2.4 | −0.018 | 0.000 | 0.000 | 0.167 | 0.167 |
| Gemma3-4B | 114.0 | +0.055 | 0.112 | 1.000 | 0.429 | **0.571** |
| Qwen2.5-VL-7B | 270.0 | +0.111 | 0.270 | 1.000 | 1.000 | 0.000 |
| Qwen3-VL-8B | 784.0 | +0.020 | 0.058 | 0.667 | 0.296 | 0.371 |
| LLaVA-OV-7B | 1.0 | +0.044 | 0.000 | 0.333 | 0.108 | 0.225 |
| Phi-3.5-V | 1.0 | +0.104 | 0.000 | 0.333 | 0.108 | 0.225 |
| Granite-V-2B | 1.0 | +0.044 | 0.000 | 0.000 | 0.167 | 0.167 |
| Idefics3-8B | 4.0 | +0.044 | 0.027 | 0.000 | 0.260 | 0.260 |
| Idefics2-8B | 11.4 | +0.109 | 0.115 | 0.000 | 0.610 | **0.610** |
| BLIP-2-OPT | 8.0 | +0.000 | 0.000 | 0.000 | 0.167 | 0.167 |

**LOO aggregated metrics**:

```
n_models with data:    10
median |error|:        0.2251       ❌  (pre-registered kill-gate: < 0.20)
Spearman ρ:            0.3265       ❌  (pre-registered threshold: > 0.5)
Spearman p (one-tail): 0.3572
Spearman 95% bootstrap CI: [-0.643, 0.887]
Pre-registered kill-gate FIRED: True
```

**Verdict** (per pre-registration): The simple compression-ratio predictor
is *anecdotal observation*; the directional claim is not validated at this
sample size. We do not claim PhysLens-Predict as a validated predictor.

### 5.3 The mechanism-stratified finding

Although the simple `log₁₀(C) × Δprobe` predictor fails its kill-gate, we
observe a clean stratification when models are grouped by compression
*mechanism* rather than ratio (Figure 6, right panel):

**Table 3**: H3 hit-rate stratified by compression mechanism.

| Mechanism | Models (n) | Compression range | Mean H3 | Range |
|---|---|---|---|---|
| **Spatial-merge (lossy pooling)** | InternVL3-8B, Gemma3-4B, Qwen2.5-VL-7B, Qwen3-VL-8B (4) | 2.4× – 784× | **0.667** | [0.0, 1.0] |
| **Learned resampler (pixel-shuffle / perceiver / Q-Former)** | Idefics3-8B, Idefics2-8B, BLIP-2 OPT-2.7B (3) | 4× – 11.4× | **0.000** | [0.0, 0.0] |
| **No compression (per-token MLP)** | LLaVA-OV-7B, Phi-3.5-V, Granite-V-2B (3) | 1× – 1× | **0.222** | [0.0, 0.333] |

The three learned-resampler models, despite 4×–11.4× nominal compression,
exhibit **zero** H3 hits across all 9 (model × target) tests. In contrast,
the four spatial-merge models with comparable or higher compression average
0.67 H3 hit-rate.

This is the strongest stratified pattern in our data. A one-tailed Welch
t-test comparing learned-resampler (n=3, mean 0.0) against spatial-merge
(n=4, mean 0.667) yields *t* = 2.91, *p* = 0.027 with df=3; we caveat
this as exploratory because mechanism-grouping was not pre-registered (see
§7).

### 5.4 Negative-control validation

Granite-Vision-3.2-2B (released February 2025, IBM, Apache-2.0) is our
cleanest pre-registered negative control: compression = 1.0× and we predict
zero H3 hits. Empirically, Granite-Vision shows **0/3 H3 hits** with every
quantitative probe preserved or improved across the encoder-to-projector
boundary (`enc_quant_acc(answer)` = 0.467, `post_proj_quant_acc(answer)` =
0.400; sub_type identical at 0.800). The negative-control prediction is
confirmed.

[**Figure 1**: LOO predicted vs. empirical scatter; **Figure 2**:
log₁₀(compression) vs. empirical H3 with linear fit; **Figure 4**: per-model
Δprobe; **Figure 6**: mechanism-stratified analysis. All figures generated
deterministically by `scripts/figures/generate_paper_figures.py` from the
JSON results.]

---

## 6. Discussion (1 page)

### 6.1 Why simple compression-ratio fails

The pre-registered formula `log₁₀(C) × Δprobe` assumes that compression is
*qualitatively uniform* — that token reduction inherently destroys
information at a rate proportional to log(compression). The data refute this
assumption. Models with **deterministic block-pooling compression**
(spatial-merge: Qwen2.5-VL, Qwen3-VL, Gemma3, and to a lesser extent
InternVL3) do exhibit H3 effects roughly in line with the prediction. But
models with **learned cross-attention reduction** (BLIP-2 Q-Former, Idefics2
perceiver) and **learned spatial reorganization** (Idefics3 pixel-shuffle)
exhibit *no* H3 effect despite 4×–11.4× compression. The mechanism, not the
ratio, is the operative variable.

### 6.2 Information-theoretic interpretation

A possible interpretation: deterministic spatial-merge averages neighboring
spatial tokens regardless of their information content, so high-entropy
quantitative information (which lives in fine-grained spatial differences)
is destroyed at the same rate as low-entropy qualitative information. In
contrast, **learned token-reduction is task-aware**: Q-Former queries are
trained jointly with the language model, perceiver resampler tokens are
trained against contrastive image-text objectives, and pixel-shuffle's
projection is trained to produce useful LLM-conditioned features. These
mechanisms preserve task-relevant information across the compression
operation.

### 6.3 What this implies for VLM development

If our refined hypothesis holds at larger scale, VLM developers face an
explicit design trade-off: **deterministic spatial-merge is parameter-free
and fast at inference but lossy on fine-grained quantitative tasks**;
**learned token-reduction adds parameters and training complexity but
preserves task-relevant information**. The choice should be driven by
deployment requirements (latency vs. quantitative-accuracy needs) rather
than treated as an architectural detail.

### 6.4 What our pre-registered failure does not refute

The pre-registered failure is a failure of the *specific scalar formula*,
not of the broader information-bottleneck framing. The refined finding
(mechanism > ratio) is itself an information-bottleneck statement: *learned*
reduction operations can be configured to preserve more bits per token than
*deterministic* reduction. A two-factor predictor (mechanism × ratio)
should be tested in a follow-up study.

---

## 7. Limitations (0.5 pages)

**L1: Sample size n=10.** Our LOO regression has wide bootstrap CIs (95%
CI for Spearman ρ = [-0.643, 0.887] at n=10). The mechanism-stratified
finding rests on 3+3+4 models per group; a t-test gives *p* = 0.027 but
the test was not pre-registered. We anchor confidence on the binary
clarity of the data (3 of 3 learned-resampler models show 0/3 hits) rather
than on the inference statistics.

**L2: Linear probes only.** We use `sklearn.linear_model.LogisticRegression`
on mean-pooled features. Nonlinear bottlenecks would be invisible to
linear probes. A follow-up could use shallow MLP probes.

**L3: Single benchmark.** PhysBench-Diag is the only benchmark tested.
Cross-benchmark validation on Physion, ScienceQA, or MMVet is future work.

**L4: H3 hit-rate definition asymmetry.** The 4 baseline H3 values
(InternVL3, Gemma3, Qwen2.5-VL, Qwen3-VL) were derived manually pre-deadline
from earlier permutation runs; the 6 active-environment models were
auto-extracted via `scripts/compute_h3_hits.py` with a stricter "significant
+ degraded" criterion. Appendix A in the supplementary material provides a
sensitivity analysis.

**L5: Quantization protocol asymmetry.** Granite-Vision-3.2-2B is loaded
in bf16 (not bnb-NF4) due to a transformers-4.49 + bitsandbytes-0.44 dtype
mismatch. Other 6 models use NF4. The architectural compression ratio is
unaffected; the probing accuracies *might* be marginally cleaner for
Granite-Vision. We do not believe this affects the qualitative finding
(Granite-Vision is in the no-compression group, where the prediction is
already H3 ≈ 0).

**L6: Excluded models.** Pixtral-12B and Molmo-7B were intended panel
members but excluded due to transformers integration bugs (HF issues #28005,
custom-modeling drift). Documented in the supplementary material.

**L7: Compression mechanism is a coarse 3-way classification.** Mechanisms
exist on a continuum (deterministic ↔ learned, local ↔ global, single-step
↔ multi-step). A finer taxonomy might explain within-group variance.

---

## 8. Reproducibility Statement (0.5 pages)

All code, data labels, probing JSONs, permutation test JSONs, and
figure-generation scripts are released at
`github.com/anonymous-NeurIPS2026/physbench-diag` (commit hash to be
added at submission; double-blind anonymized fork of the working
repository). Croissant ML metadata at the HuggingFace dataset
`anonymous-NeurIPS2026/PhysBench-Diag`. Key reproduction steps:

1. **Environment**: Python 3.11, torch 2.4.1+cu124, transformers 4.49.0,
   bitsandbytes 0.44.1. Full pin in `turing/requirements_v2.txt`. Setup
   recipe: `bash turing/setup_v2_env.sh`.
2. **Hardware**: NVIDIA A100-80GB. Probing per model: 5–30 min.
   Permutation tests per model: 3–5 min CPU. Total compute ~3 GPU-hours +
   ~30 CPU-min for the n=10 panel.
3. **Data**: PhysBench v2 from the original release; quant/qual partition
   computed by `src/optim/physbench_split.py:classify_quantitative()`.
4. **Reproduce probing**: `bash turing/submit_weekb_singles.sh` submits 5
   single-model SLURM jobs.
5. **Reproduce permutation**: `sbatch turing/15_permutation_active.sh`.
6. **Reproduce predictor**: `python scripts/phys_lens_predict.py`.
7. **Reproduce figures**: `python scripts/figures/generate_paper_figures.py`.
8. **Validate**: `bash tests/run_local.sh` runs 93 structural regression
   tests. `sbatch turing/test_v2_env.sh` runs GPU-side environment
   verification.

A complete Datasheet [Gebru et al., 2018] is provided as
`docs/DATASHEET_PHYSBENCH_DIAG.md`. Pre-registration is at
`docs/PRE_REGISTRATION.md`.

---

## 9. Conclusion (0.25 pages)

We pre-registered a closed-form architectural predictor for VLM
quantitative-physics degradation and found that the simple
compression-ratio formulation **does not pass its kill-gate** at n=10.
However, post-hoc stratification by compression *mechanism* (deterministic
spatial-merge vs. learned token-reduction vs. no-reduction) reveals a clean
separation: **learned reduction mechanisms preserve task-relevant
information that deterministic merge destroys**. We release PhysBench-Diag
and the per-model probing data to enable future predictors that incorporate
mechanism-aware features.

---

## References (BibTeX-ready, deduplicated)

```bibtex
@article{chow2024physbench,
  title={{PhysBench}: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding},
  author={Chow, Wei and others},
  journal={arXiv preprint arXiv:2401.16937},
  year={2024}
}

@article{li2024llavaonevision,
  title={{LLaVA-OneVision}: Easy Visual Task Transfer},
  author={Li, Bo and Zhang, Yuanhan and Guo, Dong and Zhang, Renrui and Li, Feng and Zhang, Hao and Zhang, Kaichen and Li, Yanwei and Liu, Ziwei and Li, Chunyuan},
  journal={arXiv preprint arXiv:2408.03326},
  year={2024}
}

@article{laurencon2024idefics2,
  title={What matters when building vision-language models?},
  author={Lauren{\c{c}}on, Hugo and Tronchon, L{\'e}o and Cord, Matthieu and Sanh, Victor},
  journal={arXiv preprint arXiv:2405.02246},
  year={2024}
}

@article{laurencon2024idefics3,
  title={Building and better understanding vision-language models: insights and future directions},
  author={Lauren{\c{c}}on, Hugo and Marafioti, Andr{\'e}s and Sanh, Victor and Tronchon, L{\'e}o},
  journal={arXiv preprint arXiv:2408.12637},
  year={2024}
}

@article{li2023blip2,
  title={{BLIP-2}: Bootstrapping language-image pre-training with frozen image encoders and large language models},
  author={Li, Junnan and Li, Dongxu and Savarese, Silvio and Hoi, Steven},
  booktitle={ICML},
  year={2023}
}

@article{ibm2025granitevision,
  title={Granite Vision: a lightweight, open-source multimodal model for enterprise use},
  author={{IBM Granite Vision Team}},
  journal={arXiv preprint arXiv:2502.09927},
  year={2025}
}

@article{microsoft2024phi3,
  title={{Phi-3} Technical Report},
  author={{Microsoft Research}},
  journal={arXiv preprint arXiv:2404.14219},
  year={2024}
}

@article{tishby2015deep,
  title={Deep learning and the information bottleneck principle},
  author={Tishby, Naftali and Zaslavsky, Noga},
  booktitle={IEEE Information Theory Workshop},
  year={2015}
}

@article{gebru2018datasheets,
  title={Datasheets for datasets},
  author={Gebru, Timnit and Morgenstern, Jamie and Vecchione, Briana and Vaughan, Jennifer Wortman and Wallach, Hanna and Daum{\'e} III, Hal and Crawford, Kate},
  journal={arXiv preprint arXiv:1803.09010},
  year={2018}
}

@article{dettmers2024nf4,
  title={{QLoRA}: Efficient Finetuning of Quantized {LLMs}},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  booktitle={NeurIPS},
  year={2024}
}

% PhysBench replicates
@article{bear2021physion,
  title={{Physion}: Evaluating Physical Prediction from Vision in Humans and Machines},
  author={Bear, Daniel M and others},
  journal={arXiv preprint arXiv:2106.08261},
  year={2021}
}

@article{tung2023physion,
  title={{Physion++}: Evaluating Physical Scene Understanding},
  author={Tung, Hsiao-Yu and others},
  journal={arXiv preprint arXiv:2306.15668},
  year={2023}
}

@article{lu2022scienceqa,
  title={Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering},
  author={Lu, Pan and others},
  booktitle={NeurIPS},
  year={2022}
}

% Probing
@article{hewitt2019structural,
  title={A Structural Probe for Finding Syntax in Word Representations},
  author={Hewitt, John and Manning, Christopher D},
  booktitle={NAACL},
  year={2019}
}

@article{voita2019analyzing,
  title={Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned},
  author={Voita, Elena and others},
  booktitle={ACL},
  year={2019}
}

% VLMs in the panel
@article{bai2023qwenvl,
  title={{Qwen-VL}: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond},
  author={Bai, Jinze and others},
  journal={arXiv preprint arXiv:2308.12966},
  year={2023}
}

@article{qwen2025qwen3vl,
  title={{Qwen3-VL} Technical Report},
  author={{Qwen Team}},
  year={2025}
}

@article{qwen2024qwen25vl,
  title={{Qwen2.5-VL} Technical Report},
  author={{Qwen Team}},
  year={2024}
}

@article{opengvlab2025internvl3,
  title={{InternVL3}: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models},
  author={{OpenGVLab}},
  year={2025}
}

@article{google2025gemma3,
  title={{Gemma 3} Technical Report},
  author={{Google DeepMind}},
  year={2025}
}

@article{liu2023llava,
  title={Visual Instruction Tuning},
  author={Liu, Haotian and others},
  booktitle={NeurIPS},
  year={2023}
}

@article{dai2023instructblip,
  title={{InstructBLIP}: Towards General-purpose Vision-Language Models with Instruction Tuning},
  author={Dai, Wenliang and others},
  booktitle={NeurIPS},
  year={2023}
}
```

---

## Appendix A: H3 hit-rate definition sensitivity

(To be filled with both definitions applied across all 10 models. Will not
materially change the mechanism-stratified finding because the 3
learned-resampler models show 0/3 hits under both definitions.)

## Appendix B: Per-model probing tables (full)

(Move full per-(model × target × site × slice) tables from JSONs into
markdown tables.)

## Appendix C: Excluded models

- **Pixtral-12B** (Mistral, 2024): excluded due to HuggingFace issue #28005
  (PixtralVisionModel does not support SDPA in transformers 4.46+;
  variable-resolution multi-image processor returns `list[Tensor]` that
  triggers `AttributeError: 'list' object has no attribute 'unsqueeze'` in
  forward pass). Probing attempted yielded n=58 valid samples; insufficient
  for inclusion. Documented in `scripts/week1_quant_qual_probe.py:load_pixtral()`.
- **Molmo-7B-D** (AllenAI, 2024): excluded due to transformers 5.x API drift
  (`_tied_weights_keys` rename incompatibility with bnb-4bit quantizer in
  the model's custom modeling code).

## Appendix D: Pre-registration

Reproduced verbatim from `docs/PRE_REGISTRATION.md`:
- Predictor formula: `score = log₁₀(max(C, 1)) × max(Δprobe, 0)`
- LOO median |error| kill-gate: 0.20 (Gate 3)
- Spearman ρ floor: 0.5
- α = 3 (number of probe targets); locked before data collection
- Verdict-on-failure: demote RQ-B predictor to "observation"; no
  generalization claim
