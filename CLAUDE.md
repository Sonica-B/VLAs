# PhysLens / VLAs project notes

## NeurIPS 2026 — Critical deadlines (Anywhere on Earth)

| Track | Abstract | Full paper | Notification |
|---|---|---|---|
| **Main Conference** | **May 4 '26** | **May 6 '26** | Sep 24 '26 |
| **Evaluations & Datasets (E&D, formerly D&B)** | **May 4 '26** | **May 6 '26** | Sep 24 '26 |
| **Position Papers** | May 4 '26 | May 6 '26 | Sep 24 '26 |
| Competitions | — | May 15 '26 | Jun 15 '26 |

### Workshops
- Application open: Apr 20 '26
- Application deadline: Jun 6 '26
- Acceptance: Jul 11 '26
- Suggested workshop contributions submission: Aug 29 '26
- Mandatory accept/reject notif: Sep 29 '26

### What we're targeting
- **Primary**: NeurIPS 2026 **Evaluations & Datasets (E&D)** Track — formerly D&B,
  renamed in 2026 with EXPANDED scope to welcome:
    - Negative results, critical analyses, evaluation-methodology audits
    - Pre-registered studies and stress-tests of evaluation practices
    - Reproduction and auditing of prior evaluations
  Per blog.neurips.cc 2026-03-23: "A submission need not 'beat a baseline'; its primary
  contribution should be to deepen and refine our understanding of evaluation practices."
- **Page limit**: 9 pages MAX including figures (refs + checklist + appendix uncounted).
- **Mandatory artifacts**: Croissant ML metadata, dataset hosted on
  HF/Kaggle/Dataverse/OpenML, code accessible to reviewers without PI request,
  16-question NeurIPS Paper Checklist (DESK REJECT if missing).
- **Deadlines**: Abstract May 4 AOE, full paper May 6 AOE, notification Sep 24.

## Environment
- **Production env**: `vla_physics_v2` (created May 2-3 '26)
  - torch 2.4.1+cu124 (matched to A100 driver 12.8 max)
  - bitsandbytes 0.44.1, transformers 4.46.x, accelerate <1.0
  - Lives at `/home/ssboyane/miniconda3/envs/vla_physics_v2/`
- **Old env (deprecated)**: `vla_physics` — has torch 2.11+cu130, broken on A100 driver
- **Cluster**: Turing (WPI) — A100 80GB, account=cngan, partition=quick (max 12hr)

## Week B status (PhysLens-Predict n=4 → n=7 expansion)
- Existing n=4: Qwen3-VL-8B, Qwen2.5-VL-7B, InternVL3-8B, Gemma4-E4B (done)
- Week B targets: LLaVA-OneVision-7B, Pixtral-12B, Phi-3.5-Vision (in progress)
- Molmo-7B-D dropped (transformers 5.x API drift)
- Probe scripts: `scripts/week1_quant_qual_probe.py`, `scripts/discover_probe_sites.py`
- Predictor: `scripts/phys_lens_predict.py` → outputs `phys_lens_predict_weekb.json`
- Gate 5 PASS thresholds: `loo_regression.median_abs_error < 0.20`, `spearman_rho > 0.5`, `kill_gate_fired == false`

## SLURM job options
- Array: `bash turing/submit_weekb_parallel.sh` (uses `08b_weekb_array.sh` + `09_weekb_aggregate.sh`)
- Singles (preferred when cluster congested): `bash turing/submit_weekb_singles.sh`
  (uses `10_probe_llava_ov.sh` + `11_probe_pixtral.sh` + `12_probe_phi35v.sh` + `09_weekb_aggregate.sh`)

## Conventions
- All SLURM scripts use `bash turing/foo.sh` (not `./` — no execute bit on git checkout from Windows).
- conda activation pattern: `source ~/miniconda3/etc/profile.d/conda.sh; conda activate vla_physics_v2`
- Always use `python -m pip install ...` (never bare `pip` — pip can be bound to wrong python).
- HF_TOKEN should be set in shell env before sbatch (passed through via `--export=ALL`).
