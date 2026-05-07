#!/usr/bin/env python3
"""
H3 sensitivity analysis: re-compute H3 hit-rate for all 10 Q-LENS panel
VLMs under a SINGLE unified definition, then re-run the PhysLens-Predict LOO
regression and re-stratify by mechanism group.

Why this exists
---------------
The paper records H3 hit-rates produced under TWO different procedures:
  - 4 spatial-merge baselines (qwen3-vl-8b, qwen2.5-vl-7b, internvl3-8b,
    gemma4-e4b): MANUALLY tallied (richer, opaque criterion baked into
    scripts/phys_lens_predict.py:MODEL_H3_HITS — see comment block there).
  - 6 Week-B / n=10 active models (llava-onevision-7b, phi3.5-vision,
    granite-vision-3.2-2b, idefics3-8b, idefics2-8b, blip2-opt-2.7b):
    AUTO-extracted via scripts/compute_h3_hits.py from per-model
    permutation_check.json files. Hit = (enc_out quantitative significant,
    p<0.05 vs permutation null) AND (enc_acc > post_acc on PCA-128 split).

The 4 manually-counted models happen to be exactly the spatial-merge group,
so the headline mechanism finding (spatial-merge mean H3 = 0.667 vs
learned-resampler 0.000 vs no-compression 0.222) could in principle be
DRIVEN BY THE DEFINITION ASYMMETRY rather than by mechanism. This script
tests that.

Strategy
--------
The auto definition cannot be applied to the 4 baselines from local data
because their JSONs lack `real_acc_pca128_split` and the empirical
permutation null. We therefore use a deterministic PROXY criterion that
mirrors the spirit of the auto rule but reads only fields present in
ALL 10 `_quant_qual_probe.json` files (k-fold mean accuracy + class-prior
chance):

  Unified hit (per target T):
      enc_q  := probe_results[T]["enc_out"]["quantitative"]["mean_acc"]
      post_q := probe_results[T]["post_proj"]["quantitative"]["mean_acc"]
      chance := probe_results[T]["enc_out"]["quantitative"]["chance"]

      strict_hit = (enc_q > chance + EPSILON_STRICT) AND (enc_q > post_q)
      lax_hit    = (enc_q > chance)                  AND (enc_q > post_q)

  H3_unified = #hits / 3

EPSILON_STRICT = 0.05 reflects a "meaningfully above chance" threshold,
consistent with PCA-128 permutation null p95 being ~0.05–0.13 above
null_mean across models (sampled).

We report BOTH variants (strict and lax) so the reader can see how
the verdict depends on the strictness bar. We also report the original
(mixed-definition) H3 unchanged for reference.

Constraint: this script only READS files. It writes
results/h3_sensitivity_unified.json and prints a summary.

Usage:
    python scripts/h3_sensitivity_unified.py
"""

from __future__ import annotations

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

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------- #
# Panel + bookkeeping (mirrors scripts/phys_lens_predict.py)
# --------------------------------------------------------------------------- #

# 10-model panel actually used in the predictor LOO
# (pixtral-12b and molmo-7b are excluded from analysis, so we mirror that).
PANEL_10 = [
    "internvl3-8b",
    "gemma4-e4b",
    "qwen2.5-vl-7b",
    "qwen3-vl-8b",
    "llava-onevision-7b",
    "phi3.5-vision",
    "granite-vision-3.2-2b",
    "idefics3-8b",
    "idefics2-8b",
    "blip2-opt-2.7b",
]

MODEL_COMPRESSION = {
    "internvl3-8b":          2.4,
    "gemma4-e4b":          114.0,
    "qwen2.5-vl-7b":       270.0,
    "qwen3-vl-8b":         784.0,
    "llava-onevision-7b":    1.0,
    "phi3.5-vision":         1.0,
    "granite-vision-3.2-2b": 1.0,
    "idefics3-8b":           4.0,
    "idefics2-8b":          11.4,
    "blip2-opt-2.7b":        8.0,
}

# Original H3 values from scripts/phys_lens_predict.py:MODEL_H3_HITS
# (mixed-definition: 4 manual + 6 auto). Used here ONLY for delta reporting.
ORIGINAL_H3 = {
    "internvl3-8b":         0.0,
    "gemma4-e4b":           1.0,
    "qwen2.5-vl-7b":        1.0,
    "qwen3-vl-8b":          2.0 / 3.0,
    "llava-onevision-7b":   1.0 / 3.0,
    "phi3.5-vision":        1.0 / 3.0,
    "granite-vision-3.2-2b":0.0,
    "idefics3-8b":          0.0,
    "idefics2-8b":          0.0,
    "blip2-opt-2.7b":       0.0,
}

# Mechanism group assignment (paper §6).
MECHANISM = {
    "qwen3-vl-8b":           "spatial-merge",
    "qwen2.5-vl-7b":         "spatial-merge",
    "internvl3-8b":          "spatial-merge",
    "gemma4-e4b":            "spatial-merge",
    "idefics3-8b":           "learned-resampler",  # pixel-shuffle (r=2)
    "idefics2-8b":           "learned-resampler",  # perceiver
    "blip2-opt-2.7b":        "learned-resampler",  # Q-Former
    "llava-onevision-7b":    "no-compression",
    "phi3.5-vision":         "no-compression",
    "granite-vision-3.2-2b": "no-compression",
}

# How to locate each model's _quant_qual_probe.json. The 4 baselines
# live in results/week1/ (canonical Turing-side results); the 6 active
# models in results/week1_turing/.
PROBE_LOCATIONS = {
    "internvl3-8b":          "results/week1/internvl3-8b_quant_qual_probe.json",
    "gemma4-e4b":            "results/week1/gemma4-e4b_quant_qual_probe.json",
    "qwen2.5-vl-7b":         "results/week1/qwen2.5-vl-7b_quant_qual_probe.json",
    "qwen3-vl-8b":           "results/week1/qwen3-vl-8b_quant_qual_probe.json",
    "llava-onevision-7b":    "results/week1_turing/llava-onevision-7b_quant_qual_probe.json",
    "phi3.5-vision":         "results/week1_turing/phi3.5-vision_quant_qual_probe.json",
    "granite-vision-3.2-2b": "results/week1_turing/granite-vision-3.2-2b_quant_qual_probe.json",
    "idefics3-8b":           "results/week1_turing/idefics3-8b_quant_qual_probe.json",
    "idefics2-8b":           "results/week1_turing/idefics2-8b_quant_qual_probe.json",
    "blip2-opt-2.7b":        "results/week1_turing/blip2-opt-2.7b_quant_qual_probe.json",
}

TARGETS = ("answer", "task_type", "sub_type")
EPSILON_STRICT = 0.05  # strict variant requires enc_acc > chance + 0.05

# --------------------------------------------------------------------------- #
# Probe-JSON helpers — read fields that are present in EVERY _quant_qual_probe
# --------------------------------------------------------------------------- #


def _resolve_probe_json(model: str) -> Path:
    """Return absolute path to the canonical _quant_qual_probe.json for the
    model, falling back gracefully across both possible locations."""
    candidates = [
        PROJECT_ROOT / PROBE_LOCATIONS[model],
        PROJECT_ROOT / f"results/week1/{model}_quant_qual_probe.json",
        PROJECT_ROOT / f"results/week1_turing/{model}_quant_qual_probe.json",
    ]
    for p in candidates:
        if p.exists():
            # Some week1_turing copies are pointer-stubs (1-line redirects)
            # to the canonical week1 path. Detect & follow.
            try:
                head = p.read_text(encoding="utf-8")[:200].strip()
                if head.startswith("/"):
                    follow = Path(head.splitlines()[0])
                    if follow.exists():
                        return follow
                if head.startswith("{"):
                    return p
            except Exception:
                pass
    raise FileNotFoundError(f"no readable quant_qual_probe.json for {model}")


def _site_quant_acc(probe_data: Dict, target: str, site: str) -> Optional[float]:
    try:
        return float(
            probe_data["probe_results"][target][site]["quantitative"]["mean_acc"]
        )
    except (KeyError, TypeError, ValueError):
        return None


def _site_quant_chance(probe_data: Dict, target: str, site: str) -> Optional[float]:
    try:
        return float(
            probe_data["probe_results"][target][site]["quantitative"]["chance"]
        )
    except (KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Unified H3 computation (proxy: k-fold mean_acc, applicable to all 10 models)
# --------------------------------------------------------------------------- #


def compute_unified_h3(
    probe_data: Dict, epsilon: float = EPSILON_STRICT
) -> Tuple[float, float, Dict]:
    """Return (strict_rate, lax_rate, per_target_breakdown).

    Hit definition (per target T):
        enc_q  > chance + epsilon  AND  enc_q > post_q  ->  strict hit
        enc_q  > chance            AND  enc_q > post_q  ->  lax hit
    """
    breakdown = {}
    strict_hits = 0
    lax_hits = 0
    for t in TARGETS:
        enc_acc = _site_quant_acc(probe_data, t, "enc_out")
        post_acc = _site_quant_acc(probe_data, t, "post_proj")
        chance = _site_quant_chance(probe_data, t, "enc_out")
        if enc_acc is None or post_acc is None or chance is None:
            breakdown[t] = {
                "strict_hit": False,
                "lax_hit": False,
                "reason": "missing field(s)",
                "enc_acc": enc_acc,
                "post_acc": post_acc,
                "chance": chance,
            }
            continue
        degraded = enc_acc > post_acc
        above_chance_strict = enc_acc > chance + epsilon
        above_chance_lax = enc_acc > chance
        s_hit = degraded and above_chance_strict
        l_hit = degraded and above_chance_lax
        if s_hit:
            strict_hits += 1
        if l_hit:
            lax_hits += 1
        breakdown[t] = {
            "strict_hit": s_hit,
            "lax_hit": l_hit,
            "enc_acc": round(enc_acc, 4),
            "post_acc": round(post_acc, 4),
            "chance": round(chance, 4),
            "delta_enc_post": round(enc_acc - post_acc, 4),
            "delta_enc_chance": round(enc_acc - chance, 4),
            "degraded": degraded,
            "above_chance_strict": above_chance_strict,
            "above_chance_lax": above_chance_lax,
        }
    return (strict_hits / len(TARGETS), lax_hits / len(TARGETS), breakdown)


# --------------------------------------------------------------------------- #
# PhysLens-Predict score + LOO (mirrors scripts/phys_lens_predict.py exactly)
# --------------------------------------------------------------------------- #


def compute_enc_postproj_gap(probe_data: Dict) -> Optional[float]:
    """enc_out - post_proj on the 'answer' target's quantitative slice mean_acc."""
    enc = _site_quant_acc(probe_data, "answer", "enc_out")
    post = _site_quant_acc(probe_data, "answer", "post_proj")
    if enc is None or post is None:
        return None
    return float(enc - post)


def phys_lens_score(comp: float, gap: float) -> float:
    return float(np.log10(max(comp, 1.0)) * max(gap, 0.0))


def loo_regression(
    scores: Dict[str, float], targets: Dict[str, float]
) -> Dict:
    """Mirrors scripts/phys_lens_predict.py:leave_one_out_regression."""
    from scipy.stats import spearmanr

    names = [
        k
        for k in scores
        if scores[k] is not None and k in targets and targets[k] is not None
    ]
    if len(names) < 3:
        return {"error": f"need >=3 models, have {len(names)}"}

    per_model = {}
    preds, emps = [], []
    for held_out in names:
        train_x = np.array([scores[m] for m in names if m != held_out])
        train_y = np.array([targets[m] for m in names if m != held_out])
        a, b = np.polyfit(train_x, train_y, 1)
        pred = float(a * scores[held_out] + b)
        pred_clipped = max(0.0, min(1.0, pred))
        emp = float(targets[held_out])
        per_model[held_out] = {
            "predicted": pred_clipped,
            "predicted_raw": pred,
            "empirical": emp,
            "abs_error": abs(pred_clipped - emp),
            "train_a": float(a),
            "train_b": float(b),
        }
        preds.append(pred_clipped)
        emps.append(emp)

    median_err = float(np.median([per_model[m]["abs_error"] for m in per_model]))

    rho, pval = (None, None)
    ci = (None, None)
    if len(preds) >= 3:
        rho, pval = spearmanr(preds, emps)
        rng = np.random.default_rng(42)
        rhos = []
        for _ in range(1000):
            idx = rng.integers(0, len(preds), size=len(preds))
            if len(set(idx.tolist())) < 2:
                continue
            try:
                r, _ = spearmanr(np.array(preds)[idx], np.array(emps)[idx])
                if not math.isnan(r):
                    rhos.append(r)
            except Exception:
                pass
        if rhos:
            ci = (float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975)))

    kill_fired = median_err > 0.20
    return {
        "per_model_loo": per_model,
        "median_abs_error": median_err,
        "spearman_rho": float(rho) if rho is not None else None,
        "spearman_p": float(pval) if pval is not None else None,
        "spearman_bootstrap_ci95": list(ci),
        "n_models": len(names),
        "kill_gate_fired": bool(kill_fired),
    }


# --------------------------------------------------------------------------- #
# Mechanism stratification
# --------------------------------------------------------------------------- #


def mechanism_means(h3_map: Dict[str, float]) -> Dict[str, Dict]:
    groups: Dict[str, List[Tuple[str, float]]] = {
        "spatial-merge": [],
        "learned-resampler": [],
        "no-compression": [],
    }
    for m, h in h3_map.items():
        if h is None:
            continue
        groups[MECHANISM[m]].append((m, h))
    out = {}
    for g, items in groups.items():
        vals = [v for _, v in items]
        out[g] = {
            "n": len(vals),
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=0)) if vals else None,
            "values": items,
        }
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def atomic_write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main() -> int:
    out_path = PROJECT_ROOT / "results" / "h3_sensitivity_unified.json"

    per_model: Dict[str, Dict] = {}
    enc_post_gap: Dict[str, Optional[float]] = {}

    for m in PANEL_10:
        try:
            probe_path = _resolve_probe_json(m)
        except FileNotFoundError as e:
            per_model[m] = {"error": str(e)}
            continue
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
        s_rate, l_rate, breakdown = compute_unified_h3(probe)
        gap = compute_enc_postproj_gap(probe)
        enc_post_gap[m] = gap
        per_model[m] = {
            "probe_path": str(probe_path.relative_to(PROJECT_ROOT)),
            "compression": MODEL_COMPRESSION[m],
            "mechanism": MECHANISM[m],
            "h3_original": ORIGINAL_H3[m],
            "h3_unified_strict": s_rate,
            "h3_unified_lax": l_rate,
            "delta_strict": s_rate - ORIGINAL_H3[m],
            "delta_lax": l_rate - ORIGINAL_H3[m],
            "per_target": breakdown,
            "enc_minus_postproj_gap_answer": gap,
            "phys_lens_score": (
                phys_lens_score(MODEL_COMPRESSION[m], gap) if gap is not None else None
            ),
        }

    # ---- Mechanism group means under each H3 definition ------------------- #
    h3_orig_map = {m: ORIGINAL_H3[m] for m in PANEL_10}
    h3_strict_map = {m: per_model[m]["h3_unified_strict"] for m in PANEL_10}
    h3_lax_map = {m: per_model[m]["h3_unified_lax"] for m in PANEL_10}
    group_orig = mechanism_means(h3_orig_map)
    group_strict = mechanism_means(h3_strict_map)
    group_lax = mechanism_means(h3_lax_map)

    # ---- LOO regression: original vs unified ------------------------------ #
    score_map = {
        m: per_model[m]["phys_lens_score"]
        for m in PANEL_10
        if per_model[m].get("phys_lens_score") is not None
    }
    loo_orig = loo_regression(score_map, h3_orig_map)
    loo_strict = loo_regression(score_map, h3_strict_map)
    loo_lax = loo_regression(score_map, h3_lax_map)

    # ---- Verdict logic ---------------------------------------------------- #
    def _gap_pp(g_means: Dict[str, Dict]) -> float:
        sm = g_means["spatial-merge"]["mean"] or 0.0
        lr = g_means["learned-resampler"]["mean"] or 0.0
        nc = g_means["no-compression"]["mean"] or 0.0
        return sm - max(lr, nc)

    orig_gap = _gap_pp(group_orig)
    strict_gap = _gap_pp(group_strict)
    lax_gap = _gap_pp(group_lax)

    # Mechanism finding "survives" if spatial-merge mean is still highest
    # AND the gap to the next-highest group is at least 0.20 (20pp).
    def _verdict(g_means: Dict[str, Dict]) -> Dict:
        sm = g_means["spatial-merge"]["mean"]
        lr = g_means["learned-resampler"]["mean"]
        nc = g_means["no-compression"]["mean"]
        if sm is None or lr is None or nc is None:
            return {"survives": False, "reason": "missing group"}
        sm_top = sm > lr and sm > nc
        gap = sm - max(lr, nc)
        return {
            "survives": bool(sm_top and gap >= 0.20),
            "spatial_merge_top": bool(sm_top),
            "gap_pp": float(gap),
            "criterion": "spatial-merge top AND gap >= 0.20",
        }

    verdict_strict = _verdict(group_strict)
    verdict_lax = _verdict(group_lax)

    # Kill-gate: mirrors scripts/phys_lens_predict.py — fires if median |err| > 0.20
    kill_orig = loo_orig.get("kill_gate_fired", None)
    kill_strict = loo_strict.get("kill_gate_fired", None)
    kill_lax = loo_lax.get("kill_gate_fired", None)

    summary = {
        "metadata": {
            "panel_n": len(PANEL_10),
            "panel": PANEL_10,
            "epsilon_strict": EPSILON_STRICT,
            "definition_note": (
                "Unified H3 uses k-fold mean_acc available in every "
                "_quant_qual_probe.json. Strict variant: hit if "
                "enc_q > chance+0.05 AND enc_q > post_q. Lax variant: hit "
                "if enc_q > chance AND enc_q > post_q. The auto reference "
                "(scripts/compute_h3_hits.py) uses real_acc_pca128_split + "
                "permutation null p<0.05 — those fields are absent from "
                "the 4-baseline probe JSONs, hence the proxy."
            ),
        },
        "per_model": per_model,
        "enc_post_gap_answer": enc_post_gap,
        "mechanism_means": {
            "original_mixed_definition": group_orig,
            "unified_strict": group_strict,
            "unified_lax": group_lax,
        },
        "mechanism_gap_pp": {
            "original": orig_gap,
            "unified_strict": strict_gap,
            "unified_lax": lax_gap,
        },
        "loo_regression": {
            "original_mixed_definition": loo_orig,
            "unified_strict": loo_strict,
            "unified_lax": loo_lax,
        },
        "verdict": {
            "mechanism_finding_survives_strict": verdict_strict,
            "mechanism_finding_survives_lax": verdict_lax,
            "kill_gate_fired_original": kill_orig,
            "kill_gate_fired_unified_strict": kill_strict,
            "kill_gate_fired_unified_lax": kill_lax,
        },
    }

    atomic_write_json(out_path, summary)

    # ----------------- Console summary ----------------- #
    print()
    print("=" * 78)
    print(" H3 sensitivity (unified definition) — per-model deltas")
    print("=" * 78)
    print(
        f"  {'model':<24} {'mech':<18} {'orig':>6} "
        f"{'strict':>7} {'lax':>5} {'d-str':>7} {'d-lax':>6}"
    )
    print("  " + "-" * 76)
    for m in PANEL_10:
        r = per_model[m]
        if "error" in r:
            print(f"  {m:<24} {'ERROR':<18} {r['error']}")
            continue
        print(
            f"  {m:<24} {r['mechanism']:<18} "
            f"{r['h3_original']:>6.3f} "
            f"{r['h3_unified_strict']:>7.3f} "
            f"{r['h3_unified_lax']:>5.3f} "
            f"{r['delta_strict']:>+7.3f} "
            f"{r['delta_lax']:>+6.3f}"
        )

    print()
    print("=" * 78)
    print(" Mechanism group means (mean H3)")
    print("=" * 78)
    for grp in ("spatial-merge", "learned-resampler", "no-compression"):
        o = group_orig[grp]["mean"]
        s = group_strict[grp]["mean"]
        l = group_lax[grp]["mean"]
        n_o = group_orig[grp]["n"]
        print(f"  {grp:<22} (n={n_o:>2}):  orig={o:.3f}  strict={s:.3f}  lax={l:.3f}")
    print(f"  spatial-merge minus next-highest:  orig={orig_gap:+.3f}  "
          f"strict={strict_gap:+.3f}  lax={lax_gap:+.3f}")

    print()
    print("=" * 78)
    print(" LOO regression (median |error| / Spearman ρ / kill-gate)")
    print("=" * 78)
    for label, loo in [
        ("original (mixed)", loo_orig),
        ("unified-strict", loo_strict),
        ("unified-lax", loo_lax),
    ]:
        if "error" in loo:
            print(f"  {label:<18} ERROR: {loo['error']}")
            continue
        med = loo["median_abs_error"]
        rho = loo["spearman_rho"]
        rho_s = f"{rho:+.3f}" if rho is not None else "n/a"
        kg = loo["kill_gate_fired"]
        print(
            f"  {label:<18}  median|err|={med:.4f}  ρ={rho_s}  "
            f"kill_gate={'FIRED' if kg else 'not fired'}"
        )

    print()
    print("=" * 78)
    print(" Verdicts")
    print("=" * 78)
    print(f"  Mechanism finding survives (strict): "
          f"{verdict_strict['survives']}  (gap={verdict_strict.get('gap_pp', 0):+.3f})")
    print(f"  Mechanism finding survives (lax):    "
          f"{verdict_lax['survives']}  (gap={verdict_lax.get('gap_pp', 0):+.3f})")
    print(f"  Kill-gate (orig / strict / lax):    "
          f"{kill_orig} / {kill_strict} / {kill_lax}")
    print()
    print(f"  wrote {out_path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
