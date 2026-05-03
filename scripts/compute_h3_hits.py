#!/usr/bin/env python3
"""
Compute the empirical H3 hit-rate from per-model permutation_check.json files.

H3 hypothesis (per docs/PRE_REGISTRATION.md):
  Vision-token compression at the multimodal projector causes
  *quantitative-physics features* to be lost, while *qualitative-physics
  features* survive. Operationally, for each prediction target T in
  {answer, task_type, sub_type}, an "H3 hit" requires:
    (a) the quantitative-slice probe at enc_out is reliable (real accuracy
        significantly above the permutation null, p < 0.05), AND
    (b) accuracy DEGRADES at post_proj (enc_out_acc > post_proj_acc on the
        quantitative slice, indicating compression destroyed quant info).

The H3 hit-rate is hits / 3 (denominator = number of targets).

This matches the published values in scripts/phys_lens_predict.py
MODEL_H3_HITS:
  internvl3-8b:    0/3 = 0.000
  gemma4-e4b:      3/3 = 1.000
  qwen2.5-vl-7b:   3/3 = 1.000
  qwen3-vl-8b:     2/3 = 0.667

Usage:
    python scripts/compute_h3_hits.py \\
        --results-dir results/week1_turing \\
        --models llava-onevision-7b phi3.5-vision granite-vision-3.2-2b

Output: prints MODEL_H3_HITS-ready Python lines for each model. Paste them
into scripts/phys_lens_predict.py and re-run the aggregator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple


def _safe_acc(record: dict) -> Optional[float]:
    """Extract real PCA-128 accuracy from a permutation record. Falls back to raw."""
    if record is None:
        return None
    v = record.get("real_acc_pca128_split")
    if v is None:
        v = record.get("real_acc_raw_kfold")
    return float(v) if v is not None else None


def compute_h3_hits(perm_data: dict, verbose: bool = False) -> Tuple[float, dict]:
    """Return (hit_rate, per_target_breakdown) from a permutation_check.json dict.

    Hit definition per target:
        - enc_out quant: significant (p<0.05) above null
        - AND enc_out_acc(quant) > post_proj_acc(quant)   [degradation]
    """
    targets = ["answer", "task_type", "sub_type"]
    breakdown = {}
    hits = 0
    for t in targets:
        try:
            enc_q = perm_data["results"][t]["enc_out"]["quantitative"]
            post_q = perm_data["results"][t]["post_proj"]["quantitative"]
        except (KeyError, TypeError):
            breakdown[t] = {"hit": False, "reason": "missing target/site/slice"}
            continue

        enc_acc = _safe_acc(enc_q)
        post_acc = _safe_acc(post_q)
        enc_sig = bool(enc_q.get("significant"))
        enc_p = enc_q.get("p_value")

        if enc_acc is None or post_acc is None:
            breakdown[t] = {"hit": False, "reason": "no acc",
                            "enc_acc": enc_acc, "post_acc": post_acc}
            continue

        degraded = enc_acc > post_acc
        hit = enc_sig and degraded
        if hit:
            hits += 1

        breakdown[t] = {
            "hit": hit,
            "enc_acc": round(enc_acc, 4),
            "post_acc": round(post_acc, 4),
            "delta": round(enc_acc - post_acc, 4),
            "enc_p": round(enc_p, 4) if enc_p is not None else None,
            "enc_significant": enc_sig,
            "degraded": degraded,
        }
    return hits / max(len(targets), 1), breakdown


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path,
                    default=Path("results/week1_turing"),
                    help="Directory holding <model>_permutation_check.json")
    ap.add_argument("--models", nargs="+", required=True,
                    help="Model keys to compute H3 hit-rate for")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-target breakdown")
    args = ap.parse_args()

    print()
    print("=" * 72)
    print("H3 hit-rate extraction (from permutation_check.json files)")
    print("=" * 72)
    print(f"  {'model':<28} {'hit-rate':>10}  {'breakdown'}")
    print("  " + "-" * 70)

    update_lines = []
    for m in args.models:
        path = args.results_dir / f"{m}_permutation_check.json"
        if not path.exists():
            print(f"  {m:<28} {'MISSING':>10}  ({path} not found)")
            update_lines.append(f'    "{m}": None,  # FIXME: permutation JSON missing')
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as e:
            print(f"  {m:<28} {'ERROR':>10}  parse: {e}")
            update_lines.append(f'    "{m}": None,  # FIXME: permutation JSON parse error')
            continue

        rate, br = compute_h3_hits(data)
        n_hits = sum(1 for v in br.values() if v.get("hit"))
        n_total = len(br)
        summary = f"{n_hits}/{n_total} hits"
        if args.verbose:
            details = "; ".join(
                f"{t}: hit={v.get('hit')} delta={v.get('delta')} p={v.get('enc_p')}"
                for t, v in br.items()
            )
            summary = f"{summary}  [{details}]"
        print(f"  {m:<28} {rate:>10.4f}  {summary}")
        update_lines.append(f'    "{m}":{rate!s:>26},  # {n_hits}/{n_total}')

    print()
    print("=" * 72)
    print("Paste these lines into scripts/phys_lens_predict.py MODEL_H3_HITS:")
    print("=" * 72)
    for line in update_lines:
        print(line)
    print()
    print("Then commit + push, and on Turing:")
    print("  git pull origin physics-steering")
    print("  sbatch turing/09_weekb_aggregate.sh")
    print("Read the final Gate 5 verdict from the aggregator output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
