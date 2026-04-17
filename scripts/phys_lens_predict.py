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
    # Week B additions (placeholders until extracted):
    "minicpm-v-2.6":    None,
    "glm-4.5v":         None,
    "llava-onevision-7b": None,
    "cogvlm2":          None,
}


# Empirical H3 hit-rate per model (from Week 1 permutation tests).
# Values from logs: hit = stage probe_acc_quant - baseline > threshold with p<0.05.
MODEL_H3_HITS = {
    "internvl3-8b":   0.0,   # 0/3
    "gemma4-e4b":     1.0,   # 3/3
    "qwen2.5-vl-7b":  1.0,   # 3/3
    "qwen3-vl-8b":    2 / 3,  # 2/3
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
