#!/usr/bin/env python3
"""
PhysLens-Predict — Compression-derived predictor of physics degradation.

Goal: given a VLM's (compression_ratio, per-stage probing gap), predict its
H3 hit-rate (fraction of permutation tests where S_quant is more degraded
at post-merger than at encoder-out, p<0.05).

This is the RQ-B scaffold. The heavy lifting is done by week1 probing
(already on disk for 4 models). This script:
  1. Loads the 8-model probing outputs from results/week1/*.json
  2. Computes the PhysLens score for each model
  3. Runs leave-one-out (LOO) regression + Spearman correlation
  4. Writes predicted vs empirical per-model to a JSON

Checkpoint-safe: reads only; overwrites output atomically.

Usage:
    python scripts/phys_lens_predict.py
    python scripts/phys_lens_predict.py --models qwen3-vl-8b qwen2.5-vl-7b internvl3-8b gemma4-e4b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Per-model config (extend as Week B runs complete).
#
# compression_ratio = (encoder_seq_len / post_proj_seq_len) for a
# fixed-resolution reference input. Values copied from Phase 4 log.
# ---------------------------------------------------------------------------

MODEL_COMPRESSION = {
    "internvl3-8b":   2.4,
    "gemma4-e4b":   114.0,
    "qwen2.5-vl-7b": 270.0,
    "qwen3-vl-8b":  784.0,
    # --- Week B additions (2026-04-19, verified 2026-05-03) ---
    # All three Week B models use plain per-token projections (MLP / Linear)
    # with NO token-count reduction. Verified by scripts/discover_probe_sites.py
    # on a 448x448 dummy: enc_seq / post_seq = 1.0x for both LLaVA-OV and
    # Pixtral. Phi-3.5 uses CLIP-L → img_projection per token, also 1.0x by
    # architecture (HD-transform GROWS tokens via crops, doesn't compress).
    #
    # These three are PERFECT NEGATIVE CONTROLS for PhysLens-Predict: if the
    # log10(compression) × Δprobe formula is real, they should all land near
    # H3 hit-rate = 0 while the high-compression baselines (qwen3 784x,
    # qwen2.5 270x, gemma4 114x) land high.
    #
    # Sources:
    #   LLaVA-OneVision (Li et al., 2024, arxiv:2408.03326) — SigLIP-SO400M
    #     384x384 → 729 tokens → 2-MLP → 729 tokens (1.0x).
    #   Pixtral 12B (Mistral AI, 2024, arxiv:2410.07073) — Pixtral-ViT
    #     variable res → MLP per token (1.0x). Discovery confirmed.
    #   Phi-3.5-Vision (Microsoft, 2024, arxiv:2404.14219) — CLIP ViT-L/14
    #     336x336 → 576 tokens per crop → img_projection per token (1.0x).
    "llava-onevision-7b":  1.0,
    "phi3.5-vision":       1.0,
    "pixtral-12b":         1.0,
    "molmo-7b":            None,  # dropped from Week B (transformers 5.x API drift)
    # --- Pixtral replacement (2026-05-03) ---
    # Idefics3-8B-Llama3: SigLIP-SO400M-patch14 @ 364x364 → 676 tokens →
    # pixel-shuffle (r=2) → 169 tokens. Compression = 676 / 169 = 4.0x.
    # MID-COMPRESSION data point that fills the 2.4x → 114x gap in our LOO
    # regression — strengthens predictor's interpolation power.
    # Source: Laurençon et al., 2024, arxiv:2408.12637 (Idefics3 paper),
    #         Section 3 (Vision encoder + pixel-shuffle r=2).
    "idefics3-8b":         4.0,
    # Granite-Vision-3.2-2B (IBM, Feb 2025) — Pixtral replacement, true 2025 entry.
    # Architecture: SigLIP vision encoder + 2-layer MLP projector + Granite-3.2 2B LM
    # via LlavaNextForConditionalGeneration. Per-tile token count is preserved
    # (no merger / pooling / pixel-shuffle), so compression = 1.0x.
    # Adds a 3rd negative-control data point (alongside LLaVA-OV and Phi-3.5),
    # strengthening the "no compression bottleneck → no H3 effect" claim.
    # Requires transformers >= 4.49 (see turing/upgrade_env_for_2025.sh).
    # Source: IBM Granite Vision team, 2025, arxiv:2502.09927.
    "granite-vision-3.2-2b": 1.0,
    # --- n=10 expansion (2026-05-03): mid-compression panel-fillers ---
    # Idefics2-8B: SigLIP 729 patches → perceiver resampler 64 query tokens
    # = 11.4x compression. arxiv 2405.02246 (Laurençon et al., 2024).
    "idefics2-8b":           11.4,
    # BLIP-2 OPT-2.7B: EVA-CLIP-g 257 patches → Q-Former 32 query tokens
    # = 8.03x compression. arxiv 2301.12597 (Li et al., 2023).
    "blip2-opt-2.7b":        8.0,
}


# Empirical H3 hit-rate per model (from Week 1 permutation tests).
# Values from logs: hit = stage probe_acc_quant - baseline > threshold with p<0.05.
# Week B entries will be filled in after Week B probing completes (D3/D4).
MODEL_H3_HITS = {
    "internvl3-8b":   0.0,   # 0/3
    "gemma4-e4b":     1.0,   # 3/3
    "qwen2.5-vl-7b":  1.0,   # 3/3
    "qwen3-vl-8b":    2 / 3,  # 2/3
    # Week B additions — measured 2026-05-03 via scripts/compute_h3_hits.py
    # Definition: for each target T in {answer, task_type, sub_type}, hit =
    #   (enc_out quantitative significant, p<0.05) AND (enc_out_acc > post_proj_acc).
    # H3 hit-rate = hits / 3.
    # NB: this is a CONSERVATIVE operational definition that auto-extracts from
    # permutation_check.json. The 4 hardcoded baseline values above were derived
    # by manual tally with a different (richer) criterion -- the predictor
    # therefore mixes definitions across rows. Documented in paper limitations.
    "llava-onevision-7b":      1 / 3,  # 1/3 (sub_type only)
    "phi3.5-vision":           1 / 3,  # 1/3
    "pixtral-12b":             None,   # dropped (transformers 4.46.x bugs)
    "molmo-7b":                None,   # dropped (transformers 5.x API drift)
    # --- Pixtral replacement (2026-05-03) ---
    "granite-vision-3.2-2b":   0.0,    # 0/3 (true negative control: no H3 anywhere)
    # --- n=10 expansion (2026-05-03): MEASURED via permutation_check + compute_h3_hits.py ---
    # ALL THREE mid-compression models surprise: empirical H3 = 0/3 despite
    # compression in [4, 11] range. See perm-active.2000258.out for full per-target
    # breakdown. This is the predictor's stress test:
    #   Idefics3 (4x):     enc_quant=0.667 -> post=0.733 (NEGATIVE delta -- info GREW)
    #   Idefics2 (11.4x):  enc_quant=0.600 -> post=0.867 (negative delta -- info GREW)
    #   BLIP-2  (8x):      enc_quant=0.533 -> post=0.733 (negative delta -- info GREW)
    # Possible interpretations: (a) Q-Former / perceiver / pixel-shuffle compression
    # is QUALITATIVELY DIFFERENT from spatial-merge compression (Qwen-style); these
    # learned-token mechanisms preserve task-relevant info. (b) Sample size n=186
    # could yield noisy permutation null. Document in paper Section 4.3 (Discussion).
    "idefics3-8b":             0.0,    # 0/3 hits (perm-active.2000258.out)
    "idefics2-8b":             0.0,    # 0/3 hits
    "blip2-opt-2.7b":          0.0,    # 0/3 hits
}


# ---------------------------------------------------------------------------
# Score computation.
# ---------------------------------------------------------------------------

@dataclass
class PhysLensScore:
    model: str
    compression: Optional[float]
    enc_minus_postproj_gap: Optional[float]  # R² or acc diff (whichever exists)
    phys_lens_score: Optional[float]
    empirical_h3: Optional[float]


def compute_enc_postproj_gap(quant_probe_json: Dict) -> Optional[float]:
    """Compute (enc_out accuracy - post_proj accuracy) on the quantitative slice.

    Looks inside the multi_model_summary.json structure:
        probe_results[target="answer"][site][slice="quantitative"]["mean_acc"]
    """
    try:
        pr = quant_probe_json["probe_results"]["answer"]
        enc = pr["enc_out"]["quantitative"]["mean_acc"]
        post = pr["post_proj"]["quantitative"]["mean_acc"]
        return float(enc - post)
    except (KeyError, TypeError):
        return None


def compute_phys_lens_score(
    compression: Optional[float], gap: Optional[float],
) -> Optional[float]:
    """Canonical PhysLens score.

    Formula (v0, pre-registered in Phase 5):
        score = log10(max(compression, 1)) * max(gap, 0)

    Interpretation: larger compression + larger enc-vs-post_proj drop
    ⇒ larger expected H3 hit-rate.
    """
    if compression is None or gap is None:
        return None
    return float(np.log10(max(compression, 1.0)) * max(gap, 0.0))


# ---------------------------------------------------------------------------
# LOO regression.
# ---------------------------------------------------------------------------

def leave_one_out_regression(scores: Dict[str, float], targets: Dict[str, float]) -> Dict:
    """For each model, train a linear regression on the other N-1 models
    predicting H3 hit-rate from PhysLens score, and predict on the held-out.

    Returns:
        dict with per-model predicted, empirical, abs error; overall
        median absolute error and Spearman correlation.
    """
    from scipy.stats import spearmanr

    names = [k for k in scores if scores[k] is not None and k in targets and targets[k] is not None]
    if len(names) < 3:
        return {"error": f"need >= 3 models with complete data, have {len(names)}"}

    loo = {}
    preds = []
    empiricals = []
    for held_out in names:
        train_x = np.array([scores[m] for m in names if m != held_out])
        train_y = np.array([targets[m] for m in names if m != held_out])
        if len(train_x) < 2:
            continue
        # Simple 1-D linear fit: y = a*x + b.
        a, b = np.polyfit(train_x, train_y, 1)
        pred = float(a * scores[held_out] + b)
        # Clip to [0, 1] since H3 hit-rate is a fraction.
        pred_clipped = max(0.0, min(1.0, pred))
        emp = float(targets[held_out])
        loo[held_out] = {
            "predicted": pred_clipped,
            "predicted_raw": pred,
            "empirical": emp,
            "abs_error": abs(pred_clipped - emp),
            "train_a": float(a),
            "train_b": float(b),
        }
        preds.append(pred_clipped)
        empiricals.append(emp)

    median_err = float(np.median([loo[m]["abs_error"] for m in loo]))

    # Spearman ρ with bootstrap CI.
    if len(preds) >= 3:
        rho, p = spearmanr(preds, empiricals)
        # Simple bootstrap.
        rng = np.random.default_rng(42)
        n = len(preds)
        rhos = []
        for _ in range(1000):
            idx = rng.integers(0, n, size=n)
            if len(set(idx.tolist())) >= 2:
                try:
                    r, _ = spearmanr(np.array(preds)[idx], np.array(empiricals)[idx])
                    if not np.isnan(r):
                        rhos.append(r)
                except Exception:
                    pass
        if rhos:
            ci = (float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975)))
        else:
            ci = (None, None)
    else:
        rho, p = None, None
        ci = (None, None)

    # Pre-registered kill gate: median |error| > 20pp = 0.20 fails predictor RQ-B.
    kill_fired = median_err > 0.20

    return {
        "per_model_loo": loo,
        "median_abs_error": median_err,
        "spearman_rho": float(rho) if rho is not None else None,
        "spearman_p": float(p) if p is not None else None,
        "spearman_bootstrap_ci95": list(ci),
        "n_models": len(names),
        "kill_gate_fired": bool(kill_fired),
        "verdict": (
            "Predictor is ANECDOTAL (LOO median |error| > 0.20) — "
            "demote RQ-B to 'observation' per PRE_REGISTRATION.md gate 3"
            if kill_fired else
            "Predictor survives gate 3"
        ),
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODEL_COMPRESSION.keys()))
    ap.add_argument("--week1-dir", type=Path, default=Path("results/week1"))
    ap.add_argument("--output", type=Path, default=Path("results/week4/phys_lens_predict.json"))
    ap.add_argument("--write-empty", action="store_true",
                    help="Write the output even if fewer than 3 models have complete data.")
    args = ap.parse_args()

    scores: Dict[str, PhysLensScore] = {}

    # Try to load multi-model summary first (lump), else per-model probe files.
    summary_path = args.week1_dir / "multi_model_summary.json"
    multi = None
    if summary_path.exists():
        try:
            multi = json.loads(summary_path.read_text())
        except Exception:
            multi = None

    for m in args.models:
        comp = MODEL_COMPRESSION.get(m)
        gap = None
        # Prefer multi_model_summary.json if present.
        if multi and "probe_results_by_model" in multi:
            per = multi["probe_results_by_model"].get(m, {})
            gap = compute_enc_postproj_gap(per)
        # Fallback: per-model JSON.
        if gap is None:
            per_path = args.week1_dir / f"{m}_quant_qual_probe.json"
            if per_path.exists():
                try:
                    per = json.loads(per_path.read_text())
                    gap = compute_enc_postproj_gap(per)
                except Exception:
                    pass
        score = compute_phys_lens_score(comp, gap)
        scores[m] = PhysLensScore(
            model=m, compression=comp, enc_minus_postproj_gap=gap,
            phys_lens_score=score, empirical_h3=MODEL_H3_HITS.get(m),
        )

    # Build LOO.
    score_map = {k: v.phys_lens_score for k, v in scores.items()}
    target_map = {k: v.empirical_h3 for k, v in scores.items()}
    have_data = [k for k, v in scores.items() if v.phys_lens_score is not None and v.empirical_h3 is not None]

    output = {
        "per_model_scores": {m: vars(s) for m, s in scores.items()},
        "n_models_with_data": len(have_data),
        "loo_regression": None,
    }

    if len(have_data) >= 3:
        loo = leave_one_out_regression(score_map, target_map)
        output["loo_regression"] = loo

    atomic_write_json(args.output, output)

    # Console summary.
    print(f"\n{'='*70}")
    print(f" PhysLens-Predict v0 — per-model scores")
    print(f"{'='*70}")
    print(f"  {'model':<22} {'compression':>12} {'enc-post gap':>14} "
          f"{'phys_lens':>12} {'empirical H3':>14}")
    print(f"  {'-'*76}")
    def fmt(x, width, precision=3, sign=False):
        if x is None:
            return f"{'n/a':>{width}}"
        if sign:
            return f"{x:>+{width}.{precision}f}"
        return f"{x:>{width}.{precision}f}"

    for m, s in scores.items():
        print(f"  {m:<22} {fmt(s.compression, 12, 1)} "
              f"{fmt(s.enc_minus_postproj_gap, 14, 4, sign=True)} "
              f"{fmt(s.phys_lens_score, 12, 3)} "
              f"{fmt(s.empirical_h3, 14, 3)}")
    print(f"  {'-'*76}")

    if output["loo_regression"]:
        loo = output["loo_regression"]
        print(f"\n  LOO median |error|: {loo['median_abs_error']:.4f}")
        print(f"  Spearman rho: {loo.get('spearman_rho')}")
        print(f"  Kill gate fired: {loo['kill_gate_fired']}")
        print(f"  Verdict: {loo['verdict']}")
    else:
        print(f"\n  Not enough complete models for LOO ({len(have_data)}/8 need ≥3)")

    print(f"\n  wrote {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
