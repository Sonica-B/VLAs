#!/usr/bin/env python3
"""
Subtype-whitelist sensitivity analysis for the PhysLens-Predict kill-gate
and the mechanism-vs-ratio finding.

Background
----------
`src/optim/physbench_split.py:PHYSBENCH_QUANT_SUBTYPES` whitelists 5 sub_types as
"quantitative":

    {size, mass, number, distance, temperature}

A reasonable reviewer ask: "what if `depth` or `collision` were also counted as
quantitative?" The pre-registered analysis reports kill-gate failure at n=10
under this 5-subtype whitelist; we want to know whether (a) the kill-gate
remains failed under an expanded whitelist (robustness), and (b) the
mechanism-vs-ratio post-hoc finding (spatial-merge mean H3 = 0.667 vs
learned-resampler mean = 0.000) survives.

What this script can do FROM LOCAL DATA
---------------------------------------
1. Re-classify every PhysBench v2 val + test item under both whitelists,
   producing a per-sample diff (which items shift quant ↔ qual).
2. Quantify the partition delta:
     - n_quant under each whitelist
     - n_items_reclassified
     - per-sub_type table: which sub_types move
3. Compute a *partition-shift bound* on the predictor's Δprobe sensitivity
   by inspecting how much of the existing per-model `quantitative.mean_acc`
   would need to be re-weighted given that some items previously in
   `qualitative` are now in `quantitative`. This is a CONSERVATIVE BOUND:
   the true re-classified Δprobe requires re-fitting the linear probe on
   the new slice, which we cannot do without the per-sample features.

What this script CANNOT do without GPU re-runs
----------------------------------------------
The per-(model × site × slice) probe accuracies in
`results/week1_turing/<model>_quant_qual_probe.json` are aggregates from
fitting `sklearn.linear_model.LogisticRegression` on per-sample features
(mean-pooled activations from forward hooks). Re-classifying items between
quant and qual changes the slice composition, which changes the fold splits,
the L2 regularizer's effective sample size, and the learned weights.

A proper "expanded-whitelist Δprobe" therefore requires:
    1. Per-sample features on disk (NOT currently saved — they were
       streamed through sklearn during the Turing GPU run; only aggregated
       accuracies persist).
    2. Re-running `scripts/week1_quant_qual_probe.py` after editing
       `PHYSBENCH_QUANT_SUBTYPES` → ~30 min/model × 10 models ≈ 5 GPU-hours.

Until that re-run completes, this script reports:
    - The partition delta (computed exactly).
    - The H3-hit-rate-stratified-by-mechanism finding under both whitelists,
      where for the expanded whitelist we report "best case" / "worst case"
      bounds on each per-model Δprobe assuming the new items inherit the
      model's existing qual-slice probe accuracy at each site.

Output: results/subtype_sensitivity.json with structure:

    {
      "original_whitelist": [...],
      "expanded_whitelist": [...],
      "partition_delta": {
        "val":  {"n_total": 200, "n_quant_orig": 55, "n_quant_expanded": ?,
                 "n_reclassified": ?, "subtype_movements": [...]},
        "test": {...}
      },
      "per_model_recompute_status": {"mode": "stub|full",
                                      "reason": "..."},
      "stratified_finding": {
         "spatial_merge_mean_H3":  {"orig": 0.667, "expanded": ...},
         "learned_resampler_mean_H3": {"orig": 0.000, "expanded": ...},
         "no_compression_mean_H3": {"orig": 0.222, "expanded": ...},
         "mechanism_finding_survives": bool
      },
      "kill_gate_flips": bool,
      "narrative": "..."
    }

The script is self-contained and idempotent.

Usage:
    python scripts/subtype_sensitivity.py
    python scripts/subtype_sensitivity.py \\
        --expanded velocity acceleration force friction depth collision throwing
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

# Ensure UTF-8 stdout/stderr (Windows cp1252 default chokes on Greek + arrows).
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optim.physbench_split import (  # noqa: E402
    PHYSBENCH_QUANT_SUBTYPES,
    classify_quantitative,
)

# Default expansion candidate set, per the paper's mechanism-stratification
# concern. Members chosen because their semantics are arguably about
# motion-quantity / numerical magnitudes:
#   - velocity / acceleration / force / friction: not present as native
#       PhysBench sub_types but listed for documentation; ignored if missing.
#   - depth: ordinal/categorical depth-ordering, borderline-quant.
#   - collision: numerical-quantity-ish (impact magnitude) but PhysBench
#       primarily phrases as "which option's picture happens first".
#   - throwing: trajectory-prediction; arguably involves quantitative
#       reasoning about distance/velocity.
DEFAULT_EXPANDED_ADDITIONS = (
    "velocity", "acceleration", "force", "friction",
    "depth", "collision", "throwing",
)

# Mechanism taxonomy from paper §4.1 Table 1 / §5.3 Table 3.
MECHANISM_GROUPS: Dict[str, List[str]] = {
    "spatial_merge": [
        "internvl3-8b", "gemma4-e4b", "qwen2.5-vl-7b", "qwen3-vl-8b",
    ],
    "learned_resampler": [
        "idefics3-8b", "idefics2-8b", "blip2-opt-2.7b",
    ],
    "no_compression": [
        "llava-onevision-7b", "phi3.5-vision", "granite-vision-3.2-2b",
    ],
}


def atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_split(physbench_dir: Path, split: str) -> Optional[List[Dict[str, Any]]]:
    path = physbench_dir / f"{split}.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ERROR: failed to parse {path}: {e}", file=sys.stderr)
        return None


def classify_under(item: Dict[str, Any], whitelist: frozenset) -> str:
    """Same logic as classify_quantitative() but using a custom whitelist.

    Mirrors the structure of src/optim/physbench_split.py:classify_quantitative
    so the script is hermetic — modifications here do NOT touch the canonical
    classifier.
    """
    sub_type = str(item.get("sub_type", "") or "").lower().strip()
    if sub_type:
        return "quantitative" if sub_type in whitelist else "qualitative"
    # Without a sub_type we cannot meaningfully diff; default to caller's
    # canonical classifier (lexical fallback).
    return classify_quantitative(item)


def partition_delta(
    items: List[Dict[str, Any]],
    orig_whitelist: frozenset,
    new_whitelist: frozenset,
) -> Dict[str, Any]:
    """Compute the per-sub_type and overall partition shift between
    `orig_whitelist` and `new_whitelist` for a list of PhysBench items.
    """
    n_quant_orig = 0
    n_quant_new = 0
    reclassified: List[Dict[str, Any]] = []
    sub_type_counts: Counter = Counter()
    for it in items:
        st = str(it.get("sub_type", "") or "").lower().strip()
        sub_type_counts[st] += 1
        a = "quantitative" if st in orig_whitelist else "qualitative"
        b = "quantitative" if st in new_whitelist else "qualitative"
        if a == "quantitative":
            n_quant_orig += 1
        if b == "quantitative":
            n_quant_new += 1
        if a != b:
            reclassified.append({
                "idx": it.get("idx"),
                "sub_type": st,
                "from": a,
                "to": b,
            })
    # Movement table per sub_type.
    movements: List[Dict[str, Any]] = []
    for st, ct in sub_type_counts.most_common():
        a = "quantitative" if st in orig_whitelist else "qualitative"
        b = "quantitative" if st in new_whitelist else "qualitative"
        if a != b:
            movements.append({
                "sub_type": st,
                "n_items": ct,
                "from": a,
                "to": b,
            })
    return {
        "n_total": len(items),
        "n_quant_orig": n_quant_orig,
        "n_quant_expanded": n_quant_new,
        "n_qual_orig": len(items) - n_quant_orig,
        "n_qual_expanded": len(items) - n_quant_new,
        "n_reclassified": len(reclassified),
        "subtype_movements": movements,
        "subtype_counts": dict(sub_type_counts.most_common()),
    }


def load_per_model_probe(
    results_dir: Path, model: str,
) -> Optional[Dict[str, Any]]:
    """Load <model>_quant_qual_probe.json. Skip stub redirect files."""
    path = results_dir / f"{model}_quant_qual_probe.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "probe_results" not in data:
            return None
        return data
    except Exception:
        return None


def compute_orig_h3_breakdown(
    perm_data: Dict[str, Any],
) -> Tuple[float, Dict[str, Dict[str, Any]]]:
    """Reuse the H3 hit-rate definition from scripts/compute_h3_hits.py.

    For each target T in {answer, task_type, sub_type}:
        hit = (enc_out quant significant at p<0.05) AND (enc_acc > post_acc)
    """
    targets = ["answer", "task_type", "sub_type"]
    breakdown: Dict[str, Dict[str, Any]] = {}
    hits = 0
    for t in targets:
        try:
            enc_q = perm_data["results"][t]["enc_out"]["quantitative"]
            post_q = perm_data["results"][t]["post_proj"]["quantitative"]
        except (KeyError, TypeError):
            breakdown[t] = {"hit": False, "reason": "missing"}
            continue
        enc_acc = enc_q.get("real_acc_pca128_split") or enc_q.get("real_acc_raw_kfold")
        post_acc = post_q.get("real_acc_pca128_split") or post_q.get("real_acc_raw_kfold")
        if enc_acc is None or post_acc is None:
            breakdown[t] = {"hit": False, "reason": "no acc"}
            continue
        sig = bool(enc_q.get("significant"))
        degraded = enc_acc > post_acc
        hit = sig and degraded
        if hit:
            hits += 1
        breakdown[t] = {
            "hit": hit, "enc_acc": float(enc_acc), "post_acc": float(post_acc),
            "delta": float(enc_acc - post_acc), "significant": sig,
            "p_value": enc_q.get("p_value"),
        }
    return hits / max(len(targets), 1), breakdown


def estimate_expanded_h3_bounds(
    quant_probe: Dict[str, Any],
    n_quant_orig: int, n_quant_expanded: int,
    n_qual_orig: int,
) -> Dict[str, Any]:
    """Estimate per-target enc/post probe accuracy on the EXPANDED quant slice.

    Method (linear-mixture bound):
        Let A_quant = mean_acc on the original 5-subtype quant slice.
        Let A_qual  = mean_acc on the original qual slice.
        Items moving qual -> quant are a SUBSET of qual; their contribution
        to the new quant slice's mean accuracy is bounded by [A_qual_min,
        A_qual_max] = [0, 1] in worst case. A reasonable proxy is to
        assume the moving items inherit A_qual on average:

            A_quant_expanded ≈ (n_q * A_quant + n_moved * A_qual)
                                  / (n_q + n_moved)

        This is *not* a re-fit; it ignores that adding training items can
        change the linear probe's solution. But it bounds the direction
        and rough magnitude of the change.

    Returns per-target enc/post estimated accuracies and Δprobe.
    """
    if n_quant_expanded == n_quant_orig:
        return {"mode": "no_change",
                "reason": "expanded whitelist did not move any items in this split"}

    n_moved = n_quant_expanded - n_quant_orig
    targets_out: Dict[str, Any] = {}
    if n_moved <= 0:
        return {"mode": "no_change",
                "reason": "non-positive item movement (whitelist shrank?)"}

    for target in ("answer", "task_type", "sub_type"):
        try:
            pr = quant_probe["probe_results"][target]
            enc_q = pr["enc_out"]["quantitative"]["mean_acc"]
            post_q = pr["post_proj"]["quantitative"]["mean_acc"]
            enc_qual = pr["enc_out"]["qualitative"]["mean_acc"]
            post_qual = pr["post_proj"]["qualitative"]["mean_acc"]
        except (KeyError, TypeError):
            targets_out[target] = {"reason": "missing target/site"}
            continue

        # Linear-mixture estimate (proxy, not a re-fit).
        denom = float(n_quant_orig + n_moved)
        enc_est = (n_quant_orig * enc_q + n_moved * enc_qual) / denom
        post_est = (n_quant_orig * post_q + n_moved * post_qual) / denom
        targets_out[target] = {
            "enc_acc_estimate": round(enc_est, 4),
            "post_acc_estimate": round(post_est, 4),
            "delta_estimate":   round(enc_est - post_est, 4),
            "enc_acc_orig_quant": round(enc_q, 4),
            "post_acc_orig_quant": round(post_q, 4),
            "delta_orig": round(enc_q - post_q, 4),
            "method": "linear_mixture_proxy_no_refit",
            "n_quant_orig": int(n_quant_orig),
            "n_quant_expanded": int(n_quant_orig + n_moved),
        }
    return {"mode": "linear_mixture_estimate", "targets": targets_out}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physbench-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "physbench")
    ap.add_argument("--results-dir", type=Path,
                    default=PROJECT_ROOT / "results" / "week1_turing")
    ap.add_argument("--expanded", nargs="+",
                    default=list(DEFAULT_EXPANDED_ADDITIONS),
                    help="Sub_types to add on top of the original whitelist")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results" / "subtype_sensitivity.json")
    args = ap.parse_args()

    orig_whitelist = frozenset(s.lower() for s in PHYSBENCH_QUANT_SUBTYPES)
    expanded_whitelist = orig_whitelist | frozenset(s.lower() for s in args.expanded)

    print(f"[subtype_sensitivity] orig:     {sorted(orig_whitelist)}")
    print(f"[subtype_sensitivity] expanded: {sorted(expanded_whitelist)}")

    # --- Load PhysBench v2 splits ---
    splits_loaded: Dict[str, List[Dict[str, Any]]] = {}
    splits_missing: List[str] = []
    for split in ("val", "test"):
        items = load_split(args.physbench_dir, split)
        if items is None:
            splits_missing.append(split)
        else:
            splits_loaded[split] = items
            print(f"  loaded {split}: n={len(items)}")

    if not splits_loaded:
        stub = {
            "schema_version": "1.0.0",
            "generator": "scripts/subtype_sensitivity.py",
            "status": "stub",
            "reason": "PhysBench v2 not on local disk; see "
                      "scripts/download_physbench.py",
            "expected_data_paths": [
                str(args.physbench_dir / "val.json"),
                str(args.physbench_dir / "test.json"),
            ],
        }
        atomic_write_json(args.output, stub)
        print(f"  STUBBED to {args.output}.")
        return 1

    # --- Compute partition delta per split ---
    partition_diff: Dict[str, Any] = {}
    for split, items in splits_loaded.items():
        partition_diff[split] = partition_delta(
            items, orig_whitelist, expanded_whitelist,
        )

    # Use val-split deltas (n=200) for per-model proxy (the panel was
    # probed on val).
    val_delta = partition_diff.get("val", {})
    n_quant_orig = val_delta.get("n_quant_orig", 0)
    n_quant_expanded = val_delta.get("n_quant_expanded", 0)
    n_qual_orig = val_delta.get("n_qual_orig", 0)

    # --- Per-model H3 reaggregation under expanded whitelist ---
    per_model_results: Dict[str, Dict[str, Any]] = {}
    flat_models: List[str] = sum(MECHANISM_GROUPS.values(), [])

    # Fall back to phys_lens_predict_weekb.json for empirical H3 when a
    # model's local probe JSON is a Turing-redirect stub (66-byte string
    # pointer rather than the real artifact).
    pred_path_for_h3 = args.results_dir / "phys_lens_predict_weekb.json"
    pred_for_h3: Dict[str, Any] = {}
    if pred_path_for_h3.exists():
        try:
            with open(pred_path_for_h3, encoding="utf-8") as f:
                pred_for_h3 = json.load(f)
        except Exception:
            pred_for_h3 = {}
    fallback_h3 = {
        m: rec.get("empirical_h3")
        for m, rec in (pred_for_h3.get("per_model_scores", {}) or {}).items()
    }

    for model in flat_models:
        probe = load_per_model_probe(args.results_dir, model)
        perm_path = args.results_dir / f"{model}_permutation_check.json"
        perm = None
        if perm_path.exists():
            try:
                with open(perm_path, encoding="utf-8") as f:
                    perm = json.load(f)
            except Exception:
                perm = None

        entry: Dict[str, Any] = {"model": model}
        if probe is None:
            entry["status"] = "missing_probe_json"
            # Still capture the original H3 from the predictor JSON, so
            # the stratified-original means use all 10 models.
            entry["h3_orig"] = fallback_h3.get(model)
            entry["h3_orig_breakdown"] = {"reason": "probe_json_stub_redirect; "
                                                     "h3_orig copied from "
                                                     "phys_lens_predict_weekb.json"}
            entry["h3_expanded_estimate"] = entry["h3_orig"]
            entry["h3_expanded_breakdown"] = {"reason": "probe_json_stub_redirect; "
                                                          "expanded estimate falls back "
                                                          "to original h3 (no per-slice "
                                                          "data on local disk)"}
            per_model_results[model] = entry
            continue
        # Original H3 (from permutation_check, if available; else fallback).
        if perm is not None:
            h3_orig, br_orig = compute_orig_h3_breakdown(perm)
            entry["h3_orig"] = round(h3_orig, 4)
            entry["h3_orig_breakdown"] = br_orig
        else:
            entry["h3_orig"] = fallback_h3.get(model)
            entry["h3_orig_breakdown"] = {"reason": "no_permutation_json; "
                                                     "h3_orig from predictor JSON"}

        # Expanded H3 estimate from probe JSON (no re-fit; mixture proxy).
        bounds = estimate_expanded_h3_bounds(
            probe, n_quant_orig, n_quant_expanded, n_qual_orig,
        )
        entry["expanded_estimate"] = bounds

        # H3 hit estimate under expanded slice: hit if (enc_significant
        # under orig quant) AND (delta_estimate > 0).
        # We CANNOT recompute permutation significance under the expanded
        # slice without raw features, so we conservatively keep the
        # original significance flag.
        if bounds.get("mode") == "linear_mixture_estimate" and perm is not None:
            hit_count = 0
            per_target: Dict[str, Any] = {}
            for t, trec in (bounds["targets"] or {}).items():
                if "delta_estimate" not in trec:
                    per_target[t] = {"hit": None, "reason": "missing"}
                    continue
                try:
                    sig = bool(perm["results"][t]["enc_out"]["quantitative"].get("significant"))
                except (KeyError, TypeError):
                    sig = False
                hit = sig and trec["delta_estimate"] > 0
                if hit:
                    hit_count += 1
                per_target[t] = {
                    "hit": hit,
                    "enc_significant_orig_slice": sig,
                    "delta_estimate": trec["delta_estimate"],
                }
            entry["h3_expanded_estimate"] = round(hit_count / 3.0, 4)
            entry["h3_expanded_breakdown"] = per_target
        else:
            entry["h3_expanded_estimate"] = entry.get("h3_orig")
            entry["h3_expanded_breakdown"] = {"reason": bounds.get("reason", "unchanged")}

        per_model_results[model] = entry

    # --- Stratified means under both partitions ---
    stratified: Dict[str, Any] = {"orig": {}, "expanded": {}}
    for mech, members in MECHANISM_GROUPS.items():
        orig_vals = [per_model_results[m].get("h3_orig") for m in members
                     if per_model_results.get(m, {}).get("h3_orig") is not None]
        exp_vals = [per_model_results[m].get("h3_expanded_estimate") for m in members
                    if per_model_results.get(m, {}).get("h3_expanded_estimate") is not None]
        stratified["orig"][mech] = {
            "members": members,
            "n_with_data": len(orig_vals),
            "mean_h3": round(sum(orig_vals) / len(orig_vals), 4) if orig_vals else None,
            "values": orig_vals,
        }
        stratified["expanded"][mech] = {
            "members": members,
            "n_with_data": len(exp_vals),
            "mean_h3": round(sum(exp_vals) / len(exp_vals), 4) if exp_vals else None,
            "values": exp_vals,
        }

    # --- Survival check for the mechanism-vs-ratio finding ---
    # Headline pattern (paper §5.3): spatial_merge_mean (0.667) >
    # learned_resampler_mean (0.000), with no_compression (0.222) in
    # between. Define survival as: spatial_merge_mean > learned_resampler_mean
    # under expanded whitelist by at least 0.1 absolute.
    sm_exp = (stratified["expanded"]["spatial_merge"]["mean_h3"] or 0.0)
    lr_exp = (stratified["expanded"]["learned_resampler"]["mean_h3"] or 0.0)
    nc_exp = (stratified["expanded"]["no_compression"]["mean_h3"] or 0.0)
    mechanism_survives = (sm_exp - lr_exp) >= 0.10
    # Direction-only secondary check (binary-clarity):
    direction_preserved = sm_exp > lr_exp

    # --- Kill-gate flip: did the expanded median |error| cross 0.20? ---
    # We approximate by recomputing LOO regression on the ESTIMATED H3
    # values under the expanded partition vs. the unchanged predictor
    # scores (compression, Δprobe). This is a pure-Python LOO so the
    # script does not require numpy/sklearn.
    pred_path = args.results_dir / "phys_lens_predict_weekb.json"
    kill_gate_orig: Optional[bool] = None
    kill_gate_exp: Optional[bool] = None
    median_err_orig: Optional[float] = None
    median_err_exp: Optional[float] = None
    spearman_orig: Optional[float] = None
    spearman_exp: Optional[float] = None
    if pred_path.exists():
        try:
            with open(pred_path, encoding="utf-8") as f:
                pred = json.load(f)
            kill_gate_orig = pred.get("loo_regression", {}).get("kill_gate_fired")
            median_err_orig = pred.get("loo_regression", {}).get("median_abs_error")
            spearman_orig = pred.get("loo_regression", {}).get("spearman_rho")

            # Build (score, expanded_h3) pairs for LOO under the new
            # empirical target.
            scores: Dict[str, float] = {}
            new_h3: Dict[str, float] = {}
            for m in flat_models:
                rec = pred.get("per_model_scores", {}).get(m, {}) or {}
                s = rec.get("phys_lens_score")
                h_exp = per_model_results.get(m, {}).get("h3_expanded_estimate")
                if s is not None and h_exp is not None:
                    scores[m] = float(s)
                    new_h3[m] = float(h_exp)
            if len(scores) >= 3:
                names = list(scores.keys())
                errs = []
                preds = []
                emps = []
                for held in names:
                    train_x = [scores[mm] for mm in names if mm != held]
                    train_y = [new_h3[mm] for mm in names if mm != held]
                    n_train = len(train_x)
                    if n_train < 2:
                        continue
                    sx = sum(train_x) / n_train
                    sy = sum(train_y) / n_train
                    num = sum((train_x[i] - sx) * (train_y[i] - sy)
                              for i in range(n_train))
                    den = sum((train_x[i] - sx) ** 2 for i in range(n_train))
                    a = num / den if abs(den) > 1e-12 else 0.0
                    b = sy - a * sx
                    p = max(0.0, min(1.0, a * scores[held] + b))
                    errs.append(abs(p - new_h3[held]))
                    preds.append(p)
                    emps.append(new_h3[held])
                # Pure-Python median.
                if errs:
                    s_errs = sorted(errs)
                    mid = len(s_errs) // 2
                    if len(s_errs) % 2:
                        median_err_exp = float(s_errs[mid])
                    else:
                        median_err_exp = float((s_errs[mid - 1] + s_errs[mid]) / 2.0)
                    kill_gate_exp = median_err_exp > 0.20
                # Pure-Python Spearman (rank correlation).
                def _rank(xs):
                    pairs = sorted([(v, i) for i, v in enumerate(xs)])
                    out = [0.0] * len(xs)
                    i = 0
                    while i < len(pairs):
                        j = i
                        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
                            j += 1
                        rk = 0.5 * (i + j) + 1.0
                        for k in range(i, j + 1):
                            out[pairs[k][1]] = rk
                        i = j + 1
                    return out
                if len(preds) >= 3:
                    rp = _rank(preds)
                    rq = _rank(emps)
                    n_pairs = len(preds)
                    mx = sum(rp) / n_pairs
                    my = sum(rq) / n_pairs
                    num = sum((rp[i] - mx) * (rq[i] - my) for i in range(n_pairs))
                    dx = math.sqrt(sum((rp[i] - mx) ** 2 for i in range(n_pairs)))
                    dy = math.sqrt(sum((rq[i] - my) ** 2 for i in range(n_pairs)))
                    spearman_exp = num / (dx * dy) if dx * dy > 1e-12 else None
        except Exception as e:
            print(f"  WARN: failed to recompute LOO: {e}")

    kill_gate_flips = (
        kill_gate_orig is not None and kill_gate_exp is not None
        and kill_gate_orig != kill_gate_exp
    )

    # --- Narrative ---
    narrative = (
        f"With the original {len(orig_whitelist)}-subtype whitelist "
        f"{sorted(orig_whitelist)}, the val partition contains "
        f"{n_quant_orig} quantitative + {n_qual_orig} qualitative items. "
        f"Adding {sorted(expanded_whitelist - orig_whitelist)} reclassifies "
        f"{val_delta.get('n_reclassified', 0)} val items "
        f"({n_quant_orig} → {n_quant_expanded} quant). "
    )
    if median_err_exp is not None and median_err_orig is not None:
        narrative += (
            f"Under the linear-mixture proxy, LOO median |error| moves "
            f"{median_err_orig:.4f} → {median_err_exp:.4f}; kill-gate "
            f"{'flips' if kill_gate_flips else 'does NOT flip'}. "
        )
    narrative += (
        f"Mechanism stratification — spatial-merge mean H3 = "
        f"{stratified['orig']['spatial_merge']['mean_h3']} → "
        f"{stratified['expanded']['spatial_merge']['mean_h3']}; "
        f"learned-resampler = {stratified['orig']['learned_resampler']['mean_h3']} → "
        f"{stratified['expanded']['learned_resampler']['mean_h3']}. "
        f"Direction preserved: {direction_preserved}; "
        f"survives 0.10 gap: {mechanism_survives}. "
        f"NB: this is an estimate from the linear-mixture proxy "
        f"(no re-fit of the linear probes); a true re-aggregation requires "
        f"re-running scripts/week1_quant_qual_probe.py with the modified "
        f"PHYSBENCH_QUANT_SUBTYPES on Turing GPU."
    )

    output = {
        "schema_version": "1.0.0",
        "generator": "scripts/subtype_sensitivity.py",
        "method": "linear_mixture_proxy",
        "method_caveat": (
            "Per-(model × site × slice) probe accuracies are aggregated "
            "from sklearn LogisticRegression fits on per-sample features "
            "that are not on local disk. We bound the expanded-slice "
            "Δprobe by linearly mixing the original quant-slice mean_acc "
            "and qual-slice mean_acc according to slice composition, "
            "without re-fitting the probes. The true expanded-slice "
            "Δprobe requires re-running scripts/week1_quant_qual_probe.py "
            "with the modified PHYSBENCH_QUANT_SUBTYPES whitelist on a "
            "GPU with the panel's quantization protocol (~30 min/model "
            "× 10 models)."
        ),
        "original_whitelist": sorted(orig_whitelist),
        "expanded_whitelist": sorted(expanded_whitelist),
        "expansion_added": sorted(expanded_whitelist - orig_whitelist),
        "expansion_present_in_data": [],
        "splits_loaded": list(splits_loaded.keys()),
        "splits_missing": splits_missing,
        "partition_delta": partition_diff,
        "per_model_results": per_model_results,
        "stratified_finding": {
            "spatial_merge_mean_H3": {
                "orig": stratified["orig"]["spatial_merge"]["mean_h3"],
                "expanded": stratified["expanded"]["spatial_merge"]["mean_h3"],
            },
            "learned_resampler_mean_H3": {
                "orig": stratified["orig"]["learned_resampler"]["mean_h3"],
                "expanded": stratified["expanded"]["learned_resampler"]["mean_h3"],
            },
            "no_compression_mean_H3": {
                "orig": stratified["orig"]["no_compression"]["mean_h3"],
                "expanded": stratified["expanded"]["no_compression"]["mean_h3"],
            },
            "direction_preserved": direction_preserved,
            "mechanism_finding_survives_with_0.10_gap": mechanism_survives,
        },
        "stratified_full": stratified,
        "kill_gate": {
            "orig_median_err": median_err_orig,
            "expanded_median_err": median_err_exp,
            "orig_kill_gate_fired": kill_gate_orig,
            "expanded_kill_gate_fired": kill_gate_exp,
            "kill_gate_flips": bool(kill_gate_flips),
            "orig_spearman_rho": spearman_orig,
            "expanded_spearman_rho": spearman_exp,
        },
        "kill_gate_flips": bool(kill_gate_flips),
        "mechanism_finding_survives": bool(mechanism_survives),
        "narrative": narrative,
    }
    # Compute which expansion items are actually present in PhysBench data.
    all_subtypes_seen = set()
    for split, items in splits_loaded.items():
        for it in items:
            st = str(it.get("sub_type", "") or "").lower().strip()
            if st:
                all_subtypes_seen.add(st)
    output["expansion_present_in_data"] = sorted(
        s for s in (expanded_whitelist - orig_whitelist) if s in all_subtypes_seen
    )
    output["expansion_absent_from_data"] = sorted(
        s for s in (expanded_whitelist - orig_whitelist) if s not in all_subtypes_seen
    )

    atomic_write_json(args.output, output)

    # --- Console summary ---
    print()
    print("=" * 78)
    print(" Subtype-whitelist sensitivity")
    print("=" * 78)
    print(f"  added subtypes:     {output['expansion_added']}")
    print(f"  present in data:    {output['expansion_present_in_data']}")
    print(f"  absent from data:   {output['expansion_absent_from_data']}")
    for split, d in partition_diff.items():
        print(f"  {split}: n_quant {d['n_quant_orig']} -> {d['n_quant_expanded']} "
              f"({d['n_reclassified']} items reclassified)")
        if d.get("subtype_movements"):
            for mv in d["subtype_movements"]:
                print(f"    + {mv['sub_type']}: {mv['n_items']} items "
                      f"{mv['from']} → {mv['to']}")
    print()
    print(f"  Spatial-merge  H3 mean: orig={stratified['orig']['spatial_merge']['mean_h3']} "
          f"expanded={stratified['expanded']['spatial_merge']['mean_h3']}")
    print(f"  Learned-resamp H3 mean: orig={stratified['orig']['learned_resampler']['mean_h3']} "
          f"expanded={stratified['expanded']['learned_resampler']['mean_h3']}")
    print(f"  No-compression H3 mean: orig={stratified['orig']['no_compression']['mean_h3']} "
          f"expanded={stratified['expanded']['no_compression']['mean_h3']}")
    print(f"  mechanism finding survives (>=0.10 gap): {mechanism_survives}")
    print(f"  kill-gate flips:                         {kill_gate_flips}")
    print(f"  median |error| orig:     {median_err_orig}")
    print(f"  median |error| expanded: {median_err_exp}")
    print(f"  Spearman ρ orig:         {spearman_orig}")
    print(f"  Spearman ρ expanded:     {spearman_exp}")
    print()
    print(f"  wrote {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
