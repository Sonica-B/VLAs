#!/usr/bin/env python3
"""
Run permutation baselines on the Week 1 cached features.

This answers the reviewer question: "Is the Qwen3-VL-8B task_type quant-slice
probe accuracy of 0.818 actually above chance, or is it overfitting noise on
55 samples?"

For every (model, target, site, slice) that Week 1 reported, we build a null
distribution of 200 shuffled-label probe accuracies and compute a one-sided
empirical p-value for the real accuracy.

A probe is "real" if its real accuracy exceeds the 95th percentile of the
null distribution (p < 0.05).

## Output

Writes `results/week1/{model}_permutation_check.json` for each model with
cached features. Also prints a summary table to stdout.

## Runtime

~1-2 minutes per model (CPU only, sklearn lbfgs on PCA-128 features).
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optim.features import FeatureCache  # noqa: E402
from src.optim.physbench_split import classify_quantitative  # noqa: E402
from src.optim.permutation_baseline import (  # noqa: E402
    permutation_probe,
    p_value_of,
    real_acc_on_reduced,
)

from scripts.run_physbench_eval import load_physbench_data  # noqa: E402


def run_for_model(model_key: str, cache_dir: Path, data_dir: Path,
                  out_dir: Path, n_permutations: int) -> Dict:
    cache = FeatureCache(cache_dir / "features", model_key, "val")
    cached = list(cache.completed_ids())
    if not cached:
        print(f"[{model_key}] no cache, skipping")
        return {}

    # Load the real probe accuracies from the Week 1 results file for comparison.
    results_path = out_dir / f"{model_key}_quant_qual_probe.json"
    real = {}
    if results_path.exists():
        real = json.loads(results_path.read_text())

    # Load samples + build label maps.
    samples = load_physbench_data(str(data_dir), split="val")
    for s in samples:
        s.setdefault("sample_id", f"val_{s.get('idx', '?')}")
    id_to_task = {s["sample_id"]: s.get("task_type", "") for s in samples}
    id_to_sub = {s["sample_id"]: s.get("sub_type", "") for s in samples}
    id_to_answer = {s["sample_id"]: s.get("answer", "") for s in samples}
    id_to_slice = {s["sample_id"]: classify_quantitative(s) for s in samples}

    targets = {"answer": id_to_answer, "task_type": id_to_task, "sub_type": id_to_sub}
    sites = ["enc_out", "post_proj", "llm_8", "llm_16"]
    index_order = list(cache._index["sample_ids"])

    feats_per_site = {site: cache.load_site(site) for site in sites}
    slices = np.array([id_to_slice.get(sid, "") for sid in index_order])

    print(f"\n[{model_key}] Running permutation baselines "
          f"(n={n_permutations}, pca_dim=128)")
    t0 = time.time()
    out = {
        "model": model_key, "n_permutations": n_permutations,
        "pca_dim": 128, "results": {},
    }
    for target, id_map in targets.items():
        t_target = time.time()
        labels = np.array([id_map.get(sid, "") for sid in index_order])
        out["results"][target] = {}
        for site in sites:
            feats = feats_per_site[site]
            for slice_name, mask_fn in [
                ("all", lambda: np.ones(len(labels), dtype=bool)),
                ("quantitative", lambda: slices == "quantitative"),
                ("qualitative", lambda: slices == "qualitative"),
            ]:
                mask = mask_fn()
                # Compute real accuracy on PCA-128 reduced features
                # (apples-to-apples with the null distribution).
                real_acc_pca = real_acc_on_reduced(
                    feats[mask], labels[mask], C=0.1, pca_dim=128,
                )
                # Also record the Week 1 raw-feature k-fold CV accuracy
                # (the headline effect-size number).
                real_acc_raw = None
                if real:
                    try:
                        slot = real["probe_results"][target][site][slice_name]
                        if slot is not None:
                            real_acc_raw = slot.get("mean_acc")
                    except (KeyError, TypeError):
                        pass
                null = permutation_probe(
                    feats[mask], labels[mask],
                    n_permutations=n_permutations, C=0.1, pca_dim=128,
                )
                p_val = None
                if real_acc_pca is not None and null.get("accs"):
                    p_val = p_value_of(real_acc_pca, null["accs"])
                out["results"][target].setdefault(site, {})[slice_name] = {
                    "real_acc_raw_kfold": real_acc_raw,  # from Week 1 tables
                    "real_acc_pca128_split": real_acc_pca,  # for p-value
                    "null_mean": null.get("mean"),
                    "null_p95": null.get("p95"),
                    "null_p99": null.get("p99"),
                    "n_valid_permutations": null.get("n_valid"),
                    "p_value": p_val,
                    "reason": null.get("reason"),
                    "significant": (
                        p_val is not None and p_val < 0.05
                        if p_val is not None else None
                    ),
                }
        print(f"  target={target} done in {time.time() - t_target:.1f}s",
              flush=True)
    print(f"[{model_key}] total {time.time() - t0:.1f}s")

    out_path = out_dir / f"{model_key}_permutation_check.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"[{model_key}] written: {out_path}")

    # Print a compact table of significant-or-not for the H3 targets.
    print(f"\n  {'target':<12} {'site':<12} {'slice':<14} {'real_pca':>9} "
          f"{'null_p95':>10} {'p':>8} {'sig?':>5}")
    print("  " + "-" * 70)
    for target in ["task_type", "sub_type"]:
        for site in sites:
            for slice_name in ["quantitative", "qualitative"]:
                rec = out["results"][target][site][slice_name]
                r = rec["real_acc_pca128_split"]
                p95 = rec["null_p95"]
                p = rec["p_value"]
                sig = rec["significant"]
                r_s = f"{r:.3f}" if r is not None else "  n/a"
                p95_s = f"{p95:.3f}" if p95 is not None else "  n/a"
                p_s = f"{p:.4f}" if p is not None else "  n/a"
                sig_s = "YES" if sig else ("no " if sig is False else "  ?")
                print(f"  {target:<12} {site:<12} {slice_name:<14} {r_s:>9} {p95_s:>10} {p_s:>8} {sig_s:>5}")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="cache/week1", type=Path)
    ap.add_argument("--data-dir", default="data/physbench", type=Path)
    ap.add_argument("--output-dir", default="results/week1", type=Path)
    ap.add_argument("--n-permutations", type=int, default=200)
    ap.add_argument("--models", nargs="+",
                    default=["qwen3-vl-8b", "qwen2.5-vl-7b", "internvl3-8b"])
    args = ap.parse_args()

    print("=" * 80)
    print(" PERMUTATION BASELINE CHECK -- Week 1 probe credibility")
    print("=" * 80)

    for model in args.models:
        try:
            run_for_model(model, args.cache_dir, args.data_dir,
                          args.output_dir, args.n_permutations)
        except Exception as e:
            print(f"[{model}] ERROR: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    return 0


if __name__ == "__main__":
    sys.exit(main())
