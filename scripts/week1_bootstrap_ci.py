#!/usr/bin/env python3
"""
Bootstrap confidence intervals for Week 1 probing results.

Reads the cached features from cache/week1/features/{model}_val/ and
re-fits LogReg probes with bootstrap resampling (default 200 resamples).
Produces tighter CIs than 5-fold CV alone, which was noisy on the 55-sample
quantitative slice (+/- 0.12 CI with k-fold).

Method:
  For each (model, target, site, slice):
    1. Resample N samples with replacement from the slice (bootstrap)
    2. Fit LogReg on 70%, score on 30% (single stratified split)
    3. Repeat B times
    4. Report mean, 95% percentile CI, std

This is CPU-only and runs on already-cached features — no GPU required,
no model load. Runs concurrently with GPU extraction of other models.

Usage:
    # Default: bootstrap the qwen3-vl-8b cache that exists today
    python scripts/week1_bootstrap_ci.py

    # Specific model
    python scripts/week1_bootstrap_ci.py --model qwen2.5-vl-7b

    # More/fewer bootstrap resamples
    python scripts/week1_bootstrap_ci.py --n-bootstrap 500

    # All cached models
    python scripts/week1_bootstrap_ci.py --all
"""

import argparse
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Suppress sklearn convergence warnings — with PCA-128 and n<200 the probe
# score stabilizes long before full lbfgs convergence, and the warning flood
# blocks Windows stdout on slow terminals.
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optim.features import FeatureCache  # noqa: E402
from src.optim.physbench_split import classify_quantitative  # noqa: E402

from scripts.run_physbench_eval import load_physbench_data  # noqa: E402


# ---------------------------------------------------------------------------
# Bootstrap probe.
# ---------------------------------------------------------------------------

def bootstrap_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 100,
    test_frac: float = 0.3,
    C: float = 0.5,
    min_n: int = 10,
    pca_dim: int = 128,
    rng: Optional[np.random.Generator] = None,
) -> Dict:
    """Bootstrap-resample a slice and fit LogReg on each resample.

    Performance note: features are dim 1152-4096. Fitting LogReg on raw
    features at 100 resamples * 4 sites * 3 targets * 3 slices = 3600 fits
    would take ~1 hour. We apply PCA-to-128 ONCE on the full slice before
    bootstrapping, which reduces each fit from ~1s to ~30ms for a total of
    a few minutes. PCA is fit on the full slice (not per-bootstrap) so it
    doesn't leak bootstrap-specific information into the probe.

    Returns:
        dict with mean_acc, ci_low (2.5%), ci_high (97.5%), std_acc,
        n, n_bootstrap, reason (None on success).
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if rng is None:
        rng = np.random.default_rng(42)

    # Filter empty labels + singleton classes (same as fit_probe).
    valid = np.array([bool(str(l).strip()) for l in labels])
    features = features[valid]
    labels = labels[valid]
    counts = Counter(labels.tolist())
    keep = np.array([counts[l] >= 2 for l in labels])
    features = features[keep]
    labels = labels[keep]

    n = len(labels)
    if n < min_n:
        return {
            "mean_acc": None, "ci_low": None, "ci_high": None, "std_acc": None,
            "n": int(n), "n_bootstrap": 0,
            "reason": f"n={n} < min_n={min_n} after filtering",
        }

    n_classes = len(np.unique(labels))
    if n_classes < 2:
        return {
            "mean_acc": None, "ci_low": None, "ci_high": None, "std_acc": None,
            "n": int(n), "n_bootstrap": 0,
            "reason": f"single class '{labels[0]}' after filtering",
        }

    # Step 1: scale + PCA on the full slice (no leakage — PCA is unsupervised).
    scaler_full = StandardScaler()
    features_scaled = scaler_full.fit_transform(features)
    target_dim = min(pca_dim, n - 1, features_scaled.shape[1])
    if target_dim < features_scaled.shape[1]:
        pca = PCA(n_components=target_dim, random_state=42)
        features_reduced = pca.fit_transform(features_scaled)
    else:
        features_reduced = features_scaled

    # Step 2: bootstrap resample + single-split LogReg on reduced features.
    # liblinear is faster than lbfgs for small-sample high-dim and handles
    # multinomial via one-vs-rest, which for 4-way probing is totally fine.
    accs: List[float] = []
    skipped = 0
    first_err: Optional[str] = None
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        Xb = features_reduced[idx]
        yb = labels[idx]
        if len(np.unique(yb)) < 2:
            skipped += 1
            continue
        try:
            # Bootstrap resampling yields duplicates — stratify can fail on
            # small classes after resampling. Use stratified split only when
            # every class has >= 2 members in the resample, else random split.
            yb_counts = Counter(yb.tolist())
            use_stratify = min(yb_counts.values()) >= 2
            Xtr, Xte, ytr, yte = train_test_split(
                Xb, yb, test_size=test_frac, random_state=b,
                stratify=yb if use_stratify else None,
            )
            # After the split, the train set might still have a single-class
            # anomaly — LogReg requires >= 2 classes in y_train. Fall through
            # to skip in that case.
            if len(np.unique(ytr)) < 2:
                skipped += 1
                continue
            # lbfgs for multiclass (liblinear only supports OVR, which sklearn
            # 1.8 no longer auto-wraps). With PCA-128 features the fit is fast.
            # max_iter=200: probe accuracy stabilizes long before full
            # convergence, so capping iterations at 200 gives near-identical
            # numbers at 2-3x speedup vs 500.
            clf = LogisticRegression(
                max_iter=200, C=C, solver="lbfgs",
            )
            clf.fit(Xtr, ytr)
            accs.append(float(clf.score(Xte, yte)))
        except Exception as e:
            skipped += 1
            if first_err is None:
                first_err = f"{type(e).__name__}: {e}"
            continue

    if len(accs) < 10:
        return {
            "mean_acc": None, "ci_low": None, "ci_high": None, "std_acc": None,
            "n": int(n), "n_bootstrap": len(accs),
            "reason": (
                f"only {len(accs)} valid resamples out of {n_bootstrap}"
                + (f"; first error: {first_err}" if first_err else "")
            ),
        }

    accs_arr = np.array(accs)
    return {
        "mean_acc":    float(accs_arr.mean()),
        "ci_low":      float(np.percentile(accs_arr, 2.5)),
        "ci_high":     float(np.percentile(accs_arr, 97.5)),
        "std_acc":     float(accs_arr.std()),
        "n":           int(n),
        "n_bootstrap": int(len(accs)),
        "skipped":     int(skipped),
        "chance":      float(max(Counter(labels.tolist()).values()) / n),
        "pca_dim":     int(target_dim),
        "reason":      None,
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def run_for_model(model_name: str, args: argparse.Namespace) -> Dict:
    print("\n" + "=" * 80)
    print(f" BOOTSTRAP CIs: model='{model_name}' n_bootstrap={args.n_bootstrap}")
    print("=" * 80)

    cache = FeatureCache(args.cache_dir / "features", model_name, "val")
    cached_ids = list(cache.completed_ids())
    if not cached_ids:
        print(f"  no cached features for {model_name}; skipping")
        return {}

    samples = load_physbench_data(str(args.data_dir), split="val")
    for s in samples:
        s.setdefault("sample_id", f"val_{s.get('idx', '?')}")

    id_to_answer = {s["sample_id"]: s.get("answer", "") for s in samples}
    id_to_task = {s["sample_id"]: s.get("task_type", "") for s in samples}
    id_to_sub = {s["sample_id"]: s.get("sub_type", "") for s in samples}
    id_to_slice = {s["sample_id"]: classify_quantitative(s) for s in samples}

    sites = list(cache._index.get("dims", {}).keys())
    print(f"  sites: {sites}")
    print(f"  cached samples: {len(cached_ids)}")

    # Load all features into memory (cheap at this scale).
    feats_per_site: Dict[str, np.ndarray] = {}
    for site in sites:
        feats_per_site[site] = cache.load_site(site)
    labels_order = list(cache._index["sample_ids"])

    results: Dict[str, Dict[str, Dict]] = {}
    targets = {
        "answer":    id_to_answer,
        "task_type": id_to_task,
        "sub_type":  id_to_sub,
    }
    rng = np.random.default_rng(args.seed)

    for target_name, id_map in targets.items():
        results[target_name] = {}
        print(f"  target={target_name}")
        for site in sites:
            t_site = time.time()
            feats = feats_per_site[site]
            labels = np.array([id_map.get(sid, "") for sid in labels_order])
            slices = np.array([id_to_slice.get(sid, "") for sid in labels_order])
            quant_mask = slices == "quantitative"
            qual_mask = slices == "qualitative"

            all_p = bootstrap_probe(feats, labels,
                                    n_bootstrap=args.n_bootstrap, rng=rng)
            quant_p = bootstrap_probe(feats[quant_mask], labels[quant_mask],
                                      n_bootstrap=args.n_bootstrap, rng=rng)
            qual_p = bootstrap_probe(feats[qual_mask], labels[qual_mask],
                                     n_bootstrap=args.n_bootstrap, rng=rng)
            results[target_name][site] = {
                "all": all_p, "quantitative": quant_p, "qualitative": qual_p,
                "n_total": int(len(feats)),
                "n_quant": int(quant_mask.sum()),
                "n_qual": int(qual_mask.sum()),
                "feature_dim": int(feats.shape[1]),
            }
            dt = time.time() - t_site
            print(f"    {site}: {dt:.1f}s  "
                  f"all={all_p.get('mean_acc') or 'n/a'}  "
                  f"q={quant_p.get('mean_acc') or 'n/a'}  "
                  f"ql={qual_p.get('mean_acc') or 'n/a'}",
                  flush=True)

    # Save results.
    out_path = args.output_dir / f"{model_name}_bootstrap_ci.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": model_name,
            "n_bootstrap": args.n_bootstrap,
            "seed": args.seed,
            "targets": list(targets.keys()),
            "results": results,
        }, f, indent=2)
    print(f"  written to {out_path}")

    # Print the task_type target (the real one).
    site_order = ["enc_out", "post_proj", "llm_8", "llm_16"]
    for target in ("answer", "task_type", "sub_type"):
        print(f"\n  target={target}")
        print(f"    {'site':<12} {'all':<26} {'quant (95% CI)':<28} {'qual (95% CI)':<28}")
        print("    " + "-" * 94)
        for site in site_order:
            if site not in results[target]:
                continue
            pr = results[target][site]

            def _fmt(p):
                if p is None or p.get("mean_acc") is None:
                    reason = (p.get("reason") or "n/a") if p else "n/a"
                    return f"skip ({reason[:20]})"
                m = p["mean_acc"]
                lo = p["ci_low"]
                hi = p["ci_high"]
                return f"{m:.3f} [{lo:.3f},{hi:.3f}]"

            print(f"    {site:<12} {_fmt(pr['all']):<26} "
                  f"{_fmt(pr['quantitative']):<28} "
                  f"{_fmt(pr['qualitative']):<28}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b",
                    help="Model name (matches cache subdir)")
    ap.add_argument("--all", action="store_true",
                    help="Run for every model with cached features")
    ap.add_argument("--data-dir", default="data/physbench", type=Path)
    ap.add_argument("--cache-dir", default="cache/week1", type=Path)
    ap.add_argument("--output-dir", default="results/week1", type=Path)
    ap.add_argument("--n-bootstrap", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t_start = time.time()

    if args.all:
        features_root = args.cache_dir / "features"
        if not features_root.exists():
            print(f"No feature caches under {features_root}")
            return 1
        model_names = []
        for sub in sorted(features_root.iterdir()):
            if sub.is_dir() and sub.name.endswith("_val"):
                model_names.append(sub.name[: -len("_val")])
        print(f"Running bootstrap on all cached models: {model_names}")
        for model_name in model_names:
            run_for_model(model_name, args)
    else:
        run_for_model(args.model, args)

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
