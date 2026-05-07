#!/usr/bin/env python3
"""
Minimum-detectable-effect (MDE) analysis for the PhysLens-Predict
Spearman ρ kill-gate.

Pre-registration §6 Gate 3 sets a Spearman ρ floor of 0.5. The Week-B
n=10 LOO regression yielded ρ = 0.327, p (one-tail) = 0.357, with a 95%
bootstrap CI that crosses zero. A natural reviewer question is:

    "Was the n=10 panel ever statistically capable of detecting ρ = 0.5
    one-sidedly at α=0.05, 80% power?"

This script answers that question two ways and writes both to disk:

    (1) ANALYTICAL — Fisher z-transformation of Spearman's ρ. We use the
        standard normal-approximation formula (Bonett & Wright, 2000):
            z'_obs = atanh(ρ_obs)  ~  N(atanh(ρ_true), 1/(n − 3))
        Solving for the smallest |ρ| that yields power ≥ 0.80 against
        the null ρ = 0 (one-sided OR two-sided, configurable).

    (2) SIMULATION — resample n iid (x_i, y_i) pairs from a bivariate
        normal with target ρ, compute Spearman's ρ, and tally the
        proportion of trials whose ρ exceeds the n-specific critical
        value (computed from the null permutation distribution at
        α=0.05). With 5000 trials per ρ × n cell, MDE is the smallest ρ
        whose empirical power ≥ 0.80.

Output: results/power_analysis.json with structure:

    {
      "panel_sizes": [10, 15, 20, 25, 30, 40, 50],
      "alpha": 0.05,
      "target_power": 0.80,
      "mde_one_sided_analytical": [...],
      "mde_two_sided_analytical": [...],
      "mde_one_sided_simulation": [...],
      "mde_two_sided_simulation": [...],
      "n_required_for_rho_0.5_80pct_power_one_sided": int,
      "n_required_for_rho_0.5_80pct_power_two_sided": int,
      "narrative": "..."
    }

This documents the underpower critique referenced in paper §7 (Limitations
L1) and supports the E&D Track's request for "*critical analyses [...] of
evaluation practices*".

Usage:
    python scripts/power_analysis.py
    python scripts/power_analysis.py --simulations 0   # skip simulation, fast
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

# Ensure UTF-8 stdout/stderr (Windows cp1252 default chokes on Greek + arrows).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_PANEL_SIZES = [10, 15, 20, 25, 30, 40, 50]
DEFAULT_ALPHA = 0.05
DEFAULT_POWER = 0.80
DEFAULT_TARGET_RHO = 0.5  # the pre-registered Spearman threshold
DEFAULT_SIMS = 5000


def atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# (1) Analytical MDE via Fisher z-transformation.
# ---------------------------------------------------------------------------

def _phi_inv(p: float) -> float:
    """Standard-normal quantile (Acklam 2003 rational approximation).

    Avoids a SciPy dependency for the analytical path so this script is
    runnable on a minimal install (numpy + Python stdlib).
    """
    if p <= 0.0 or p >= 1.0:
        raise ValueError(f"phi_inv requires 0<p<1, got {p}")
    # Acklam's rational approximation, |relative error| < 1.15e-9
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)


def fisher_mde_for_rho(
    n: int, alpha: float, power: float, two_sided: bool = False,
) -> float:
    """Minimum |ρ_true| such that a Spearman test on n pairs has the given
    power to reject H0: ρ=0 at significance alpha.

    Uses Fisher z-approximation. The variance term we use is `1/(n−3)`,
    which is technically the Pearson form. For Spearman, the more
    accurate variance is `1.06/(n−3)` (Bonett & Wright 2000), so we apply
    that adjustment.

    Returns ρ in [0, 1]. If the requested power cannot be reached for
    any ρ (e.g. n too small), returns 1.0 as a sentinel.
    """
    if n <= 3:
        return 1.0  # uncomputable
    sigma_z = math.sqrt(1.06 / (n - 3))  # Bonett-Wright Spearman correction
    # Critical value of z' under the null (one-sided or two-sided).
    if two_sided:
        z_crit = _phi_inv(1.0 - alpha / 2.0) * sigma_z
    else:
        z_crit = _phi_inv(1.0 - alpha) * sigma_z
    z_power = _phi_inv(power)  # z s.t. Phi(z) = power
    # Required mean under H1: z_mu = z_crit + z_power * sigma_z
    z_mu = z_crit + z_power * sigma_z
    rho = math.tanh(z_mu)
    # Numerical clip; tanh(z_mu) is already in (-1, 1) but be safe.
    return float(min(max(rho, 0.0), 0.999999))


def fisher_n_required_for_rho(
    target_rho: float, alpha: float, power: float, two_sided: bool = False,
    n_max: int = 1000,
) -> Optional[int]:
    """Smallest n such that Fisher-z power ≥ target_power against ρ=target_rho.

    Returns None if no n in [4, n_max] suffices.
    """
    if not (0 < target_rho < 1):
        return None
    z_target = math.atanh(target_rho)
    z_alpha = _phi_inv(1.0 - (alpha / 2.0 if two_sided else alpha))
    # Solve for n such that:
    #   z_target / sigma_z >= z_alpha + Phi^-1(power)
    # sigma_z = sqrt(1.06 / (n-3))
    # => sqrt((n-3)/1.06) * z_target >= z_alpha + z_power
    z_power = _phi_inv(power)
    rhs = z_alpha + z_power
    if rhs <= 0:
        return 4  # trivially achieved
    # (n-3)/1.06 >= (rhs/z_target)^2
    n_min = 1.06 * (rhs / z_target) ** 2 + 3.0
    n_required = math.ceil(n_min)
    if n_required <= n_max:
        return int(n_required)
    return None


# ---------------------------------------------------------------------------
# (2) Simulation MDE.
# ---------------------------------------------------------------------------

def _simulate_mde(
    n: int, alpha: float, power: float, two_sided: bool = False,
    rho_grid: Optional[List[float]] = None,
    n_sims: int = 5000, seed: int = 42,
) -> Optional[float]:
    """Find smallest ρ in `rho_grid` whose simulated power ≥ `power`.

    Procedure per ρ:
      - Generate n iid pairs from bivariate normal with correlation ρ.
      - Compute Spearman rank correlation.
      - Test against the (one-sided / two-sided) α=0.05 critical region
        derived from a parametric null (ρ=0) simulation with n pairs.

    Skips entirely if numpy / scipy are unavailable.
    """
    try:
        import numpy as np
        from scipy.stats import spearmanr
    except ImportError:
        return None

    rng = np.random.default_rng(seed)

    # Step 1: derive the null critical value via simulation under ρ=0.
    null_rhos = []
    for _ in range(max(n_sims, 5000)):
        x = rng.standard_normal(n)
        y = rng.standard_normal(n)
        r, _ = spearmanr(x, y)
        if r is not None and not np.isnan(r):
            null_rhos.append(r)
    null_rhos = np.array(null_rhos)
    if two_sided:
        crit = float(np.quantile(np.abs(null_rhos), 1.0 - alpha))
    else:
        crit = float(np.quantile(null_rhos, 1.0 - alpha))

    # Step 2: power scan.
    if rho_grid is None:
        rho_grid = [round(0.05 * k, 3) for k in range(1, 20)]  # 0.05..0.95
    for rho_true in rho_grid:
        # Cholesky of [[1, rho], [rho, 1]] = [[1, 0], [rho, sqrt(1-rho^2)]]
        L00 = 1.0
        L11 = math.sqrt(max(1.0 - rho_true ** 2, 0.0))
        hits = 0
        for _ in range(n_sims):
            z1 = rng.standard_normal(n)
            z2 = rng.standard_normal(n)
            x = z1
            y = rho_true * z1 + L11 * z2
            r, _ = spearmanr(x, y)
            if r is None or np.isnan(r):
                continue
            if two_sided:
                detected = abs(r) >= crit
            else:
                detected = r >= crit
            if detected:
                hits += 1
        emp_power = hits / n_sims
        if emp_power >= power:
            return float(rho_true)
    return None  # not detectable in the grid


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-sizes", type=int, nargs="+",
                    default=DEFAULT_PANEL_SIZES,
                    help="Panel sizes n to compute MDE for")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="Significance level (default 0.05)")
    ap.add_argument("--power", type=float, default=DEFAULT_POWER,
                    help="Target statistical power (default 0.80)")
    ap.add_argument("--target-rho", type=float, default=DEFAULT_TARGET_RHO,
                    help="Pre-registered ρ floor (default 0.5)")
    ap.add_argument("--simulations", type=int, default=DEFAULT_SIMS,
                    help="Simulations per (n, ρ) cell. 0 = skip simulation.")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results" / "power_analysis.json")
    args = ap.parse_args()

    # ---- Analytical (Fisher z) MDE per panel size ----
    mde_one = []
    mde_two = []
    for n in args.panel_sizes:
        mde_one.append(round(fisher_mde_for_rho(n, args.alpha, args.power, False), 4))
        mde_two.append(round(fisher_mde_for_rho(n, args.alpha, args.power, True), 4))

    # ---- Sample-size requirement to reach target ρ at given power ----
    n_req_one = fisher_n_required_for_rho(
        args.target_rho, args.alpha, args.power, two_sided=False,
    )
    n_req_two = fisher_n_required_for_rho(
        args.target_rho, args.alpha, args.power, two_sided=True,
    )

    # ---- Simulation MDE per panel size ----
    sim_mde_one: List[Optional[float]] = [None] * len(args.panel_sizes)
    sim_mde_two: List[Optional[float]] = [None] * len(args.panel_sizes)
    sim_status = "skipped"
    if args.simulations > 0:
        rho_grid = [round(0.025 * k, 4) for k in range(2, 41)]  # 0.05..1.0 step 0.025
        try:
            import numpy  # noqa: F401
            import scipy  # noqa: F401
            sim_status = "ok"
            for i, n in enumerate(args.panel_sizes):
                sim_mde_one[i] = _simulate_mde(
                    n, args.alpha, args.power, False,
                    rho_grid=rho_grid, n_sims=args.simulations, seed=42,
                )
                sim_mde_two[i] = _simulate_mde(
                    n, args.alpha, args.power, True,
                    rho_grid=rho_grid, n_sims=args.simulations, seed=43,
                )
        except ImportError as e:
            sim_status = f"skipped (missing dep: {e.name})"

    # ---- Narrative ----
    pre_reg_n = 10
    pre_reg_idx = (
        args.panel_sizes.index(pre_reg_n) if pre_reg_n in args.panel_sizes else -1
    )
    if pre_reg_idx >= 0:
        n10_mde_one = mde_one[pre_reg_idx]
    else:
        n10_mde_one = fisher_mde_for_rho(pre_reg_n, args.alpha, args.power, False)
    n10_observed_rho = 0.327  # from results/week1_turing/phys_lens_predict_weekb.json

    narrative = (
        f"At n=10, a one-sided Spearman test at α={args.alpha} has "
        f"{int(args.power*100)}% power to detect "
        f"ρ ≥ {round(n10_mde_one, 3)} (Fisher-z analytical). "
        f"The pre-registered Gate-3 threshold is ρ > {args.target_rho}. "
        f"To reliably detect ρ = {args.target_rho} at "
        f"{int(args.power*100)}% power one-sided, the panel needs "
        f"n ≥ {n_req_one} models. The Week-B observation of ρ = "
        f"{n10_observed_rho} is therefore not statistically distinguishable "
        f"from the null at the panel size used; per pre-registration the "
        f"predictor is demoted to an observation regardless, but reviewers "
        f"should not interpret 'fail Gate 3' as evidence against ρ ≥ "
        f"{args.target_rho} in the population — only as failure to reject "
        f"H0 at this n."
    )

    output = {
        "schema_version": "1.0.0",
        "generator": "scripts/power_analysis.py",
        "panel_sizes": args.panel_sizes,
        "alpha": args.alpha,
        "target_power": args.power,
        "target_rho": args.target_rho,
        "method_analytical": "Fisher z (Bonett-Wright Spearman correction, sigma_z = sqrt(1.06/(n-3)))",
        "method_simulation": "bivariate-normal resample, "
                              f"{args.simulations} sims/cell, "
                              f"empirical Spearman vs simulated null",
        "simulation_status": sim_status,
        "mde_one_sided_analytical": mde_one,
        "mde_two_sided_analytical": mde_two,
        "mde_one_sided_simulation": sim_mde_one,
        "mde_two_sided_simulation": sim_mde_two,
        f"n_required_for_rho_{args.target_rho}_80pct_power_one_sided": n_req_one,
        f"n_required_for_rho_{args.target_rho}_80pct_power_two_sided": n_req_two,
        "weekb_n": pre_reg_n,
        "weekb_observed_rho": n10_observed_rho,
        "narrative": narrative,
    }
    atomic_write_json(args.output, output)

    # ---- Console summary ----
    print()
    print("=" * 78)
    print(" Spearman ρ Minimum Detectable Effect (MDE) by panel size")
    print(f"  α={args.alpha}, power={args.power}")
    print("=" * 78)
    print(f"  {'n':>4}  {'MDE 1-sided':>12}  {'MDE 2-sided':>12}  "
          f"{'sim 1-sided':>12}  {'sim 2-sided':>12}")
    print(f"  {'-'*60}")
    for i, n in enumerate(args.panel_sizes):
        s1 = "n/a" if sim_mde_one[i] is None else f"{sim_mde_one[i]:.3f}"
        s2 = "n/a" if sim_mde_two[i] is None else f"{sim_mde_two[i]:.3f}"
        print(f"  {n:>4}  {mde_one[i]:>12.4f}  {mde_two[i]:>12.4f}  "
              f"{s1:>12}  {s2:>12}")
    print(f"  {'-'*60}")
    print()
    print(f"  To detect ρ = {args.target_rho} at {int(args.power*100)}% power,")
    print(f"      one-sided n ≥ {n_req_one}")
    print(f"      two-sided n ≥ {n_req_two}")
    print()
    print(f"  Week-B n=10, observed ρ={n10_observed_rho:.3f}, MDE one-sided "
          f"≈ {n10_mde_one:.3f}.")
    print(f"  Implication: the n=10 panel was UNDERPOWERED to detect "
          f"ρ ≥ {args.target_rho}.")
    print()
    print(f"  wrote {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
