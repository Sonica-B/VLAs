#!/usr/bin/env python3
"""
Week 3 Phase 5: SCAS random-direction control — CHECKPOINT-SAFE.

This is the pre-registered control for the SCAS headline result. For each
seed, generate a random unit vector in the K=64 low-variance subspace V_low
and use it as the steering direction at alpha=5. If the distribution of
random-direction Δ_quant includes the observed SCAS Δ_quant, SCAS is
indistinguishable from stochastic perturbation in V_low.

Design choices (frozen by PRE_REGISTRATION.md):
  - 20 seeds (0..19) by default
  - alpha=5.0 (matching current SCAS headline)
  - low_var_k=64 (matching current SCAS config)
  - PCA on training split (leakage-free)
  - Eval on PhysBench val (200 samples) unless --eval-split=test

Checkpoint contract:
  - Each seed writes a standalone JSON to
    results/week4/random_control/<model>/seed_<NN>.json
  - On start, the script scans existing seeds and skips completed ones
  - Writes are atomic (tmp file + fsync + rename)
  - Safe to Ctrl-C and resume; safe to run on Colab with runtime disconnects

Usage:
  # Default: 20 seeds on Qwen3-VL-8B val, alpha=5
  python scripts/week3_scas_random_control.py

  # Custom: fewer seeds, different model
  python scripts/week3_scas_random_control.py \\
      --model qwen3-vl-8b --seeds 5 --alpha 5.0

  # Test-set run (production)
  python scripts/week3_scas_random_control.py --eval-split test --seeds 20

  # Quick smoke (20 samples, 3 seeds) — confirms plumbing on Colab first
  python scripts/week3_scas_random_control.py --seeds 3 --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

# Reuse the existing SCAS eval path — same model loading, same hook, same
# PhysBench pipeline. We only swap the steering vector for a random one.
from scripts.week3_scas_sweep import (  # noqa: E402
    MODEL_LOADERS, MERGER_PATHS, load_model, evaluate_with_steering,
)
from src.optim.vram import (  # noqa: E402
    snapshot_vram, format_vram_delta, hard_cleanup,
)
from src.optim.resilience import configure_traceback_logging  # noqa: E402
from src.optim.steering import compute_steering_vector  # noqa: E402


# ---------------------------------------------------------------------------
# Atomic JSON write (survives kill-9 mid-write).
# ---------------------------------------------------------------------------

def atomic_write_json(path: Path, data: Dict) -> None:
    """Write JSON atomically. Never leaves a partial file on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Build a random steering_info dict compatible with make_steering_hook.
# ---------------------------------------------------------------------------

def build_random_steering_info(
    sv_info_real: Dict, seed: int, method: str = "contrast",
) -> Dict:
    """Replace the physics steering direction with a random unit vector in V_low.

    We generate the random vector INSIDE V_low (same subspace SCAS operates
    on) so it's a fair control: same subspace, same norm, only the direction
    within V_low is random.

    Args:
        sv_info_real: output of compute_steering_vector(method="contrast")
                      with "components" still present (not stripped)
        seed: numpy RNG seed
        method: "contrast" (single direction, fair control for contrast-SCAS)
                or "amplify" (full K-dim basis, unused here)

    Returns:
        Dict with same schema as sv_info_real but "vector" replaced by
        a random unit vector in V_low.
    """
    rng = np.random.default_rng(seed)
    components = sv_info_real["components"]  # [D, D] PCA basis
    low_var_k = sv_info_real["low_var_k"]
    # Take the SAME V_low SCAS uses (last K eigenvectors).
    low_var_basis = components[-low_var_k:]  # [K, D]
    k, d = low_var_basis.shape

    # Generate random coefficients in V_low, map back to full space, normalize.
    coeffs = rng.standard_normal(k)  # [K]
    vec = coeffs @ low_var_basis  # [D] — lives entirely in V_low
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        # Degenerate — retry with a different seed offset.
        coeffs = rng.standard_normal(k)
        vec = coeffs @ low_var_basis
        norm = np.linalg.norm(vec)
    unit = vec / norm

    return {
        "vector": unit.astype(np.float32),
        "method": method,  # use "contrast" so the hook does dir . v * v (same as real SCAS)
        "low_var_k": low_var_k,
        "feature_dim": d,
        "explained_variance_low": sv_info_real.get("explained_variance_low"),
        "seed": int(seed),
        "is_random_control": True,
    }


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b", choices=list(MODEL_LOADERS.keys()))
    ap.add_argument("--seeds", type=int, default=20,
                    help="Number of random-direction seeds (0..seeds-1)")
    ap.add_argument("--alpha", type=float, default=5.0,
                    help="Amplification factor (matches SCAS headline)")
    ap.add_argument("--low-var-k", type=int, default=64)
    ap.add_argument("--data-dir", type=Path, default=Path("data/physbench"))
    ap.add_argument("--cache-dir", type=Path, default=Path("cache/week1"))
    ap.add_argument("--output-dir", type=Path, default=Path("results/week4/random_control"))
    ap.add_argument("--log-dir", type=Path, default=Path("logs"))
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--pca-split", default="train",
                    help="PCA source split (train = leakage-free)")
    ap.add_argument("--eval-split", default="val",
                    help="'val' (200) or 'test' (9802)")
    ap.add_argument("--include-baseline", action="store_true",
                    help="Also run alpha=0 as a sanity check (skipped if already on disk)")
    args = ap.parse_args()

    logger = configure_traceback_logging(
        args.log_dir, f"week3_scas_random_control_{args.model}",
    )
    logger.info("=" * 70)
    logger.info(f"SCAS random-direction control")
    logger.info(f"  model={args.model}  alpha={args.alpha}  low_var_k={args.low_var_k}")
    logger.info(f"  seeds=0..{args.seeds - 1}  eval_split={args.eval_split}")
    logger.info(f"  output_dir={args.output_dir}")
    logger.info("=" * 70)

    out_model_dir = args.output_dir / args.model
    out_model_dir.mkdir(parents=True, exist_ok=True)

    # --- Build the REAL steering vector once to get V_low + PCA components ---
    logger.info("Computing REAL steering vector (for PCA basis reuse)...")
    t0 = time.time()
    sv_real = compute_steering_vector(
        cache_dir=str(args.cache_dir),
        model_name=args.model,
        split=args.pca_split,
        method="contrast",
        low_var_k=args.low_var_k,
        site="post_proj",
    )
    logger.info(
        f"  K={sv_real['low_var_k']}  D={sv_real['feature_dim']}  "
        f"V_low variance fraction={sv_real['explained_variance_low']:.6e}  "
        f"[computed in {time.time() - t0:.1f}s]"
    )

    # --- Checkpoint scan: skip seeds already on disk ---
    pending_seeds = []
    for seed in range(args.seeds):
        seed_path = out_model_dir / f"seed_{seed:02d}.json"
        if seed_path.exists():
            try:
                existing = json.loads(seed_path.read_text())
                if existing.get("completed"):
                    logger.info(f"  seed {seed:02d} already completed; skipping")
                    continue
            except Exception:
                pass  # corrupt JSON, re-run
        pending_seeds.append(seed)
    logger.info(f"Pending seeds: {pending_seeds}")

    if not pending_seeds and not args.include_baseline:
        logger.info("Nothing to do. All seeds complete.")
        return 0

    # --- Load model once (expensive) ---
    before = snapshot_vram()
    model, processor = load_model(args.model)
    logger.info(format_vram_delta(before, snapshot_vram()))

    try:
        # Optional: run baseline (alpha=0) once for reference.
        if args.include_baseline:
            baseline_path = out_model_dir / "baseline_alpha0.json"
            if not baseline_path.exists():
                logger.info("\n--- baseline (alpha=0, no steering) ---")
                t_eval = time.time()
                result = evaluate_with_steering(
                    model, processor, args.model, args.data_dir,
                    steering_info=None, alpha=0.0, logger=logger,
                    max_samples=args.max_samples, split=args.eval_split,
                )
                result["elapsed_s"] = time.time() - t_eval
                result["completed"] = True
                result["kind"] = "baseline"
                atomic_write_json(baseline_path, result)
                logger.info(f"  baseline saved: {baseline_path}")

        # Per-seed random-direction runs.
        for seed in pending_seeds:
            seed_path = out_model_dir / f"seed_{seed:02d}.json"
            logger.info(f"\n--- seed {seed:02d} (random direction in V_low, alpha={args.alpha}) ---")

            # Build random steering vector (cheap).
            sv_random = build_random_steering_info(sv_real, seed=seed, method="contrast")

            # Write a "started" marker immediately so a crashed run doesn't get
            # mistaken for completed.
            atomic_write_json(seed_path, {
                "model": args.model,
                "seed": seed,
                "alpha": args.alpha,
                "low_var_k": args.low_var_k,
                "pca_split": args.pca_split,
                "eval_split": args.eval_split,
                "started": True,
                "completed": False,
                "start_time": time.time(),
            })

            # Evaluate.
            t_eval = time.time()
            result = evaluate_with_steering(
                model, processor, args.model, args.data_dir,
                steering_info=sv_random, alpha=args.alpha, logger=logger,
                max_samples=args.max_samples, split=args.eval_split,
            )
            result["elapsed_s"] = time.time() - t_eval
            result["model"] = args.model
            result["seed"] = seed
            result["low_var_k"] = args.low_var_k
            result["pca_split"] = args.pca_split
            result["eval_split"] = args.eval_split
            result["kind"] = "random_control"
            result["completed"] = True
            result["end_time"] = time.time()
            atomic_write_json(seed_path, result)
            logger.info(
                f"  seed {seed:02d}: acc_quant={result['acc_quant']:.4f} "
                f"acc_qual={result['acc_qual']:.4f} "
                f"acc_all={result['acc_all']:.4f}  "
                f"[{result['elapsed_s']:.0f}s]"
            )

        # --- Final aggregate ---
        aggregate_random_results(out_model_dir, args.model, args.alpha, logger)
    finally:
        hard_cleanup(model, processor)

    return 0


def aggregate_random_results(
    out_model_dir: Path, model_name: str, alpha: float, logger,
) -> None:
    """Scan all seed JSONs in out_model_dir and write a summary with kill-gate verdict."""
    seed_files = sorted(out_model_dir.glob("seed_*.json"))
    completed = []
    for f in seed_files:
        try:
            d = json.loads(f.read_text())
            if d.get("completed"):
                completed.append(d)
        except Exception:
            pass

    if not completed:
        logger.warning("No completed seeds to aggregate.")
        return

    acc_quants = np.array([d["acc_quant"] for d in completed if d.get("acc_quant") is not None])
    acc_quals = np.array([d["acc_qual"] for d in completed if d.get("acc_qual") is not None])
    acc_alls = np.array([d["acc_all"] for d in completed if d.get("acc_all") is not None])

    # Compare against SCAS observed delta (+3.64pp on Qwen3-VL val).
    # Read baseline if present; else use the value in PRE_REGISTRATION.md.
    baseline_path = out_model_dir / "baseline_alpha0.json"
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        base_quant = baseline.get("acc_quant", 0.7091)
        base_qual = baseline.get("acc_qual", 0.6207)
    else:
        # Values from Phase 4 (Qwen3-VL-8B val baseline, leakage-free).
        base_quant = 0.7091
        base_qual = 0.6207

    deltas_quant = acc_quants - base_quant
    deltas_qual = acc_quals - base_qual

    def boot_ci(arr: np.ndarray, n: int = 1000, alpha: float = 0.05):
        if len(arr) == 0:
            return (None, None)
        rng = np.random.default_rng(42)
        samples = rng.choice(arr, size=(n, len(arr)), replace=True)
        means = samples.mean(axis=1)
        lo = float(np.quantile(means, alpha / 2))
        hi = float(np.quantile(means, 1 - alpha / 2))
        return (lo, hi)

    med_dq = float(np.median(deltas_quant))
    mean_dq = float(np.mean(deltas_quant))
    ci_dq = boot_ci(deltas_quant)
    med_dl = float(np.median(deltas_qual))

    # Pre-registered kill gate: median random Δ_quant >= 1.8pp (half of SCAS observed).
    observed_scas_dq = 0.0364  # +3.64pp headline
    kill_threshold = observed_scas_dq * 0.5
    kill_fired = med_dq >= kill_threshold

    summary = {
        "model": model_name,
        "alpha": alpha,
        "n_seeds": len(completed),
        "baseline_acc_quant": base_quant,
        "baseline_acc_qual": base_qual,
        "random_acc_quant_median": float(np.median(acc_quants)),
        "random_acc_quant_mean": float(np.mean(acc_quants)),
        "random_acc_quant_std": float(np.std(acc_quants, ddof=1)) if len(acc_quants) > 1 else 0.0,
        "random_delta_quant_median": med_dq,
        "random_delta_quant_mean": mean_dq,
        "random_delta_quant_ci95": list(ci_dq),
        "random_delta_qual_median": med_dl,
        "observed_scas_delta_quant": observed_scas_dq,
        "kill_threshold_delta_quant": kill_threshold,
        "kill_gate_fired": bool(kill_fired),
        "verdict": (
            "SCAS is INDISTINGUISHABLE from random V_low perturbation "
            "— DEMOTE from headline per PRE_REGISTRATION.md gate 1"
            if kill_fired else
            "SCAS median delta EXCEEDS random controls — headline survives this gate"
        ),
    }

    out_path = out_model_dir / "random_control_summary.json"
    atomic_write_json(out_path, summary)

    logger.info("")
    logger.info("=" * 70)
    logger.info(f"RANDOM-CONTROL SUMMARY: {model_name} alpha={alpha}")
    logger.info("=" * 70)
    logger.info(f"  n_seeds = {summary['n_seeds']}")
    logger.info(f"  baseline acc_quant = {base_quant:.4f}")
    logger.info(f"  random acc_quant: median={np.median(acc_quants):.4f} "
                f"mean={np.mean(acc_quants):.4f} std={summary['random_acc_quant_std']:.4f}")
    logger.info(f"  random delta_quant: median={med_dq:+.4f} mean={mean_dq:+.4f} "
                f"95% CI [{ci_dq[0]:+.4f}, {ci_dq[1]:+.4f}]")
    logger.info(f"  observed SCAS delta_quant = {observed_scas_dq:+.4f}")
    logger.info(f"  kill threshold = {kill_threshold:+.4f}")
    logger.info(f"  KILL GATE FIRED: {kill_fired}")
    logger.info(f"  VERDICT: {summary['verdict']}")
    logger.info(f"  summary: {out_path}")


if __name__ == "__main__":
    sys.exit(main())
