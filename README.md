# Q-LENS

**A Pre-Registered Probing Stress-Test of Vision-Language Quantitative-Physics Reasoning.**

This repository accompanies a NeurIPS 2026 Evaluations & Datasets (E&D) Track
submission. It contains a deterministic quantitative/qualitative partition over
PhysBench v2 (`PhysBench-Diag`), a pre-registered closed-form predictor
(`PhysLens-Predict`) of where in 10 open-weight VLMs quantitative-physics
information is preferentially lost, and a sensitivity-analysis suite that
audits both the pre-registered predictor and an exploratory follow-up.

[arXiv (placeholder)](#) · [OpenReview (placeholder)](#) ·
**License:** code MIT, labels CC-BY-4.0

---

## Status

NeurIPS 2026 **Evaluations & Datasets (E&D) Track** — submission **#3628**
(under double-blind review, anonymous mirror at
[anonymous.4open.science](https://anonymous.4open.science/)).

The submitted paper PDF is on OpenReview. This repository hosts:
the locked pre-registration document, the deterministic partition function,
all per-model probing JSONs, the predictor implementation, the H3 and subtype
sensitivity scripts, the power analysis, and the Croissant ML metadata for
`PhysBench-Diag`. Every numeric claim in the paper traces to a script in this
repo.

---

## The 60-second story

1. We pre-register a closed-form predictor of which VLMs lose quantitative-physics
   decodability across the multimodal projector. The predictor and its
   leave-one-out kill-gates (median |error| < 0.20 AND Spearman ρ > 0.5) are
   locked in `PRE_REGISTRATION.md` before any data is collected.
2. At n=10 the predictor **fails both kill-gates** (observed median |error| =
   0.225, Spearman ρ = 0.33, 95% bootstrap CI [-0.64, 0.89]). Per the
   pre-registration we demote the compression-ratio hypothesis to descriptive
   observation, refusing to re-tune.
3. Stratifying the same panel by compression *mechanism* (spatial-merge /
   learned-resampler / no-compression) initially yields a nominal separation
   favoring spatial-merge.
4. **A unified-definition sensitivity analysis collapses that gap** (from +0.444
   to +0.056). The post-hoc finding does not survive its own audit.
5. The contribution is therefore **methodological**: a worked example of
   pre-registration plus sensitivity analysis catching both a primary and a
   secondary false claim before they enter the literature.

---

## Reproducing every paper claim from a clone

Each of the steps below runs from a fresh clone on a normal Linux/Mac/Windows
laptop (no GPU, no SLURM, no cluster account needed). Outputs land in
`results/`. Expected runtimes are wall-clock on a 2024 laptop.

### 0. Install (≈ 2 min)

```bash
conda create -n q_lens python=3.11 -y
conda activate q_lens
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 1. Pre-registered predictor + kill-gate failure (≈ 5 sec)

```bash
python scripts/phys_lens_predict.py
```

Reads the 10 per-model probing JSONs in `results/week1_turing/` and writes
`results/week1_turing/phys_lens_predict_weekb.json`. Expected key fields:
`loo_regression.median_abs_error = 0.225`, `loo_regression.spearman_rho = 0.327`,
`loo_regression.kill_gate_fired = true`.

### 2. H3 sensitivity (the headline-collapsing analysis, ≈ 2 sec)

```bash
python scripts/h3_sensitivity_unified.py
```

Re-computes H3 hit-rate under a unified definition for all 10 models and
re-stratifies. Expected output: spatial-merge mean **0.667 → 0.500**,
learned-resampler mean **0.000 → 0.444**, no-compression mean **0.222 → 0.222**.
Gap-to-next-highest collapses **+0.444 → +0.056**. Saves
`results/h3_sensitivity_unified.json`.

### 3. Power analysis (≈ 1 sec)

```bash
python scripts/power_analysis.py
```

Computes Spearman-ρ MDE at α=0.05, power=0.80 for n in {10..50}. Documents
why n=10 was at the edge of detectability for ρ > 0.5. Saves
`results/power_analysis.json`.

### 4. Subtype-whitelist sensitivity (≈ 5 sec, requires PhysBench data)

```bash
# (one-time) Place PhysBench v2 val.json + test.json under data/physbench/
python scripts/subtype_sensitivity.py
```

Re-classifies under an expanded whitelist `{size, mass, number, distance,
temperature} ∪ {collision, depth, throwing}` (the three expansion candidates
that exist as native PhysBench sub_types). Reports whether the kill-gate
failure and the mechanism trend are robust to this redefinition.

### 5. Regenerate the partition file (≈ 5 sec, requires PhysBench data)

```bash
python scripts/generate_physbench_diag_partition.py
```

Materializes `results/physbench_diag_partition.json` (the file the Croissant
metadata advertises). Validates val counts (200 items, 55 quant / 145 qual)
against the paper.

### 6. Regenerate paper figures (≈ 30 sec)

```bash
python scripts/figures/generate_paper_figures.py
```

Outputs `figures/fig{1..6}_*.pdf` from the JSONs above.

---

## Code structure

```
VLAs/
├── src/                  Importable modules (probing, optim, data splits, evaluation)
├── scripts/              Runnable scripts (every paper claim has an entry-point here)
│   └── patches/          Documentation for the Granite-Vision NF4 ablation patch
├── turing/               SLURM wrappers used to produce the per-model probing JSONs
│                         on a Turing A100 cluster. **Reviewers do NOT need to run
│                         these** — every probing JSON they produce is committed
│                         under `results/week1_turing/`.
├── tests/                pytest unit + regression tests
├── results/              Probing JSONs + predictor LOO + sensitivity outputs
├── figures/              Final paper figures (PDF, vector)
├── configs/              Hydra configs for models / probing / ablation
└── data/                 (gitignored) PhysBench v2 raw items — see "Where the data lives"
```

E&D-mandatory artifacts at the repo root:

- `PRE_REGISTRATION.md` — locked pre-registration document (predictor formula,
  kill-gate thresholds, n=10 panel — committed before data collection).
- `DATASHEET_PHYSBENCH_DIAG.md` — Gebru et al. datasheet for the partition.
- `CROISSANT_METADATA.md` — Croissant ML metadata template for the released
  artifacts on Hugging Face Datasets.

---

## Where the data lives

| Asset | In repo? | Reviewer action |
|---|---|---|
| 10 × per-model probing JSONs | ✅ `results/week1_turing/<model>_quant_qual_probe.json` | None — used directly by `phys_lens_predict.py` and `h3_sensitivity_unified.py` |
| 6 × per-model permutation tests | ✅ `results/week1_turing/<model>_permutation_check.json` | None |
| Predictor LOO output | ✅ `results/week1_turing/phys_lens_predict_weekb.json` | None |
| All 6 paper figures | ✅ `figures/fig{1..6}_*.pdf` | None |
| PhysBench v2 raw items (val + test) | ❌ — license doesn't permit redistribution | Download from the [PhysBench](https://github.com/USC-GVL/PhysBench) release into `data/physbench/` if you want to re-run scripts 4 and 5 above |
| Probing feature caches (~500 MB) | ❌ — too large + regenerable | Re-running them requires a GPU; the resulting probe JSONs are already committed |

---

## Key files

- `PRE_REGISTRATION.md` — the locked pre-registration (paper §2 references it as
  immutable).
- `DATASHEET_PHYSBENCH_DIAG.md` — datasheet for the partition.
- `CROISSANT_METADATA.md` — Croissant ML metadata template.
- `results/week1_turing/phys_lens_predict_weekb.json` — predictor LOO output
  (canonical kill-gate verdict for the paper).
- `results/h3_sensitivity_unified.json` — the headline-collapsing sensitivity
  result (paper §5.3 / §7.L4).
- `results/power_analysis.json` — Spearman-ρ MDE table (paper §7.L1).
- `results/subtype_sensitivity.json` — robustness to expanded whitelist
  (paper §7.L7).
- `src/optim/physbench_split.py` — deterministic 5-subtype partition function.
- `scripts/phys_lens_predict.py` — predictor implementation (closed-form score,
  LOO regression, kill-gate check).
- `scripts/h3_sensitivity_unified.py` — re-computes H3 under unified definition
  for all 10 models.
- `scripts/patches/granite_nf4_patch.md` — env-gated patch for the planned
  Granite-Vision bf16 → NF4 ablation.

---

## Citation

```bibtex
@inproceedings{anonymous2026qlens,
  title     = {Q-LENS: A Pre-Registered Probing Stress-Test of
               Vision-Language Quantitative-Physics Reasoning},
  author    = {Anonymous},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Evaluations and Datasets Track},
  year      = {2026},
  note      = {Submission \#3628; under double-blind review}
}
```

---

## License

- **Code** (`src/`, `scripts/`, `turing/`, `tests/`, `configs/`): MIT.
- **Labels + sensitivity outputs** (`results/*.json`, `CROISSANT_METADATA.md`):
  CC-BY-4.0.
- Underlying PhysBench v2 images and prompts come from
  [Chow et al., 2024](https://github.com/USC-GVL/PhysBench) and inherit that
  benchmark's terms; this repository contributes only the partition function,
  per-model probing measurements, and the sensitivity-analysis methodology.
