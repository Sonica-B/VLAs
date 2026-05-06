"""
Permutation baseline for probe validation.

The probing accuracies reported in Week 1 (e.g. Qwen3-VL-8B task_type
quant slice at 0.818 at enc_out) only matter if they are above chance in
a meaningful way. The standard test is a permutation baseline:

    1. Shuffle the labels randomly (break any real signal).
    2. Re-fit the same probe on the shuffled labels.
    3. Repeat N times.
    4. Compare the real accuracy against the null distribution of
       shuffled-label accuracies. If the real accuracy is above the 95th
       percentile of the null distribution, the probe signal is real.

This module provides `permutation_probe` which runs the full null-
distribution construction, and `p_value_of` which gives a one-sided p-value
for a given measured accuracy relative to that null distribution.

## Why this matters for Week 1 credibility

Reviewers (Codex / Antigravity / NeurIPS) will reasonably ask: "Is your
0.818 task_type probing accuracy actually above chance on n=55 samples
with 4 classes?" Chance is nominally 0.25 (uniform) or ~0.28 (majority
class). But on a small high-dim sample, a well-regularized LogReg can
hit 0.30-0.40 on random labels just by memorizing. The permutation
baseline quantifies exactly this risk and produces a p-value we can
cite in the paper.

## Usage

    from src.optim.permutation_baseline import permutation_probe, p_value_of

    # features: [n, d]   labels: [n] (strings or ints)
    null = permutation_probe(
        features, labels, n_permutations=200, C=0.1,
    )
    # null["accs"] is a list of N shuffled-label accuracies.
    # null["mean"] / null["std"] / null["p95"] describe the null distribution.

    real_acc = 0.818
    p = p_value_of(real_acc, null["accs"])
    print(f"real={real_acc:.3f}  null_mean={null['mean']:.3f}  p={p:.4g}")
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence

import numpy as np

# Suppress sklearn convergence warnings -- probe fits on permuted labels
# often converge slowly because the signal is noise by construction, but
# the score is what we care about, not the loss.
warnings.filterwarnings("ignore")


def _fit_score_reduced(
    X_reduced: np.ndarray,
    y: np.ndarray,
    C: float,
    test_size: float,
    seed: int,
) -> Optional[float]:
    """Fit LogReg on PCA-reduced features with a stratified train/test split.

    Features are assumed ALREADY dimensionality-reduced upstream (PCA happens
    once per slice, not per permutation). This is ~30x faster than fitting
    on raw 4096-dim features and is appropriate for null-distribution
    construction because the null is a stability measure, not an absolute
    accuracy estimate.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    counts = Counter(y.tolist())
    min_class = min(counts.values())
    if min_class < 2:
        return None
    try:
        Xtr, Xte, ytr, yte = train_test_split(
            X_reduced, y, test_size=test_size, random_state=seed, stratify=y,
        )
    except ValueError:
        return None
    if len(np.unique(ytr)) < 2:
        return None
    clf = LogisticRegression(max_iter=200, C=C, solver="lbfgs")
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def permutation_probe(
    features: np.ndarray,
    labels: np.ndarray,
    n_permutations: int = 100,
    C: float = 0.1,
    test_size: float = 0.3,
    seed: int = 42,
    pca_dim: int = 128,
) -> Dict:
    """Build a null distribution of probe accuracies under label shuffling.

    Performance note:
        Raw 4096-dim features make each LogReg fit take ~1s, so 100
        permutations x 36 slots x 3 models x 1s = ~3 hours. With PCA-128
        preprocessing each fit takes ~30ms, bringing total runtime to ~3
        minutes.

    Apples-to-apples caveat:
        The Week 1 reported accuracies use RAW features (no PCA). To make
        the p-value computation valid, the caller must ALSO compute the
        "real" (unshuffled) accuracy on the same PCA-128 features, using
        this module's `real_acc_on_reduced` helper. The p-value is then
        valid relative to a PCA-128-reference probe, not the raw-feature
        probe reported in Week 1 tables.

        Note: the two probe-accuracy numbers in the paper have different
        meanings -- the raw-feature number is the headline effect size,
        the PCA-128 number is the significance test.

    Args:
        features: [n, d] raw feature matrix (pre-PCA).
        labels: [n] label array (strings or ints).
        n_permutations: number of shuffle-refit iterations. Default 100
            gives a minimum p-value of ~0.01.
        C: LogReg regularization. Should match the C used for the real
            probe.
        test_size: train/test split ratio.
        seed: base RNG seed for reproducibility.
        pca_dim: PCA reduction dimension. Default 128. Set to 0 or None
            to disable PCA and use raw features (slow).

    Returns:
        Dict with keys:
            accs: list of successful shuffle accuracies
            n_valid: number of successful shuffles
            mean, std, median: summary stats of the null distribution
            p95, p99: significance cutoffs
            chance: majority-class baseline
            pca_dim: effective PCA dim (for logging)
            reason: None on success, else a short skip explanation
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    valid = np.array([bool(str(l).strip()) for l in labels])
    features = np.asarray(features)[valid]
    labels = np.asarray(labels)[valid]

    counts = Counter(labels.tolist())
    keep = np.array([counts[l] >= 2 for l in labels])
    features = features[keep]
    labels = labels[keep]

    n = len(labels)
    if n < 10:
        return _empty_null(n, reason=f"n={n} < 10 after filtering")
    n_classes = len(np.unique(labels))
    if n_classes < 2:
        return _empty_null(n, reason="single class after filtering")

    # One-shot dimensionality reduction (unsupervised, no label leakage).
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    if pca_dim and pca_dim > 0:
        target_dim = min(pca_dim, n - 1, features_scaled.shape[1])
        if target_dim < features_scaled.shape[1]:
            pca = PCA(n_components=target_dim, random_state=42)
            features_reduced = pca.fit_transform(features_scaled)
        else:
            features_reduced = features_scaled
            target_dim = features_scaled.shape[1]
    else:
        features_reduced = features_scaled
        target_dim = features_scaled.shape[1]

    chance = max(Counter(labels.tolist()).values()) / n
    rng = np.random.default_rng(seed)
    accs: List[float] = []
    for k in range(n_permutations):
        y_perm = labels.copy()
        rng.shuffle(y_perm)
        if min(Counter(y_perm.tolist()).values()) < 2:
            continue
        acc = _fit_score_reduced(
            features_reduced, y_perm, C=C, test_size=test_size, seed=seed + k,
        )
        if acc is not None:
            accs.append(acc)

    if len(accs) < 10:
        return _empty_null(n, reason=f"only {len(accs)} valid permutations out of {n_permutations}")

    arr = np.asarray(accs)
    return {
        "accs": accs,
        "n": int(n),
        "n_valid": int(len(accs)),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "chance": float(chance),
        "pca_dim": int(target_dim),
        "reason": None,
    }


def real_acc_on_reduced(
    features: np.ndarray,
    labels: np.ndarray,
    C: float = 0.1,
    test_size: float = 0.3,
    seed: int = 42,
    pca_dim: int = 128,
) -> Optional[float]:
    """Compute the real (unshuffled) probe accuracy on PCA-reduced features.

    This is the number that should be compared against the null distribution
    from `permutation_probe(pca_dim=128)` for a valid p-value. The caller
    can then also report the raw-feature k-fold CV accuracy separately as
    the effect-size number.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    valid = np.array([bool(str(l).strip()) for l in labels])
    features = np.asarray(features)[valid]
    labels = np.asarray(labels)[valid]
    counts = Counter(labels.tolist())
    keep = np.array([counts[l] >= 2 for l in labels])
    features = features[keep]
    labels = labels[keep]
    n = len(labels)
    if n < 10 or len(np.unique(labels)) < 2:
        return None

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    if pca_dim and pca_dim > 0:
        target_dim = min(pca_dim, n - 1, features_scaled.shape[1])
        if target_dim < features_scaled.shape[1]:
            features_reduced = PCA(n_components=target_dim, random_state=42).fit_transform(features_scaled)
        else:
            features_reduced = features_scaled
    else:
        features_reduced = features_scaled

    try:
        Xtr, Xte, ytr, yte = train_test_split(
            features_reduced, labels, test_size=test_size,
            random_state=seed, stratify=labels,
        )
    except ValueError:
        return None
    if len(np.unique(ytr)) < 2:
        return None
    clf = LogisticRegression(max_iter=200, C=C, solver="lbfgs")
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def _empty_null(n: int, reason: str) -> Dict:
    return {
        "accs": [], "n": int(n), "n_valid": 0,
        "mean": None, "std": None, "median": None,
        "p95": None, "p99": None, "chance": None,
        "reason": reason,
    }


def p_value_of(real_acc: float, null_accs: Sequence[float]) -> float:
    """One-sided empirical p-value: fraction of null accuracies >= real_acc.

    Adds the pseudo-count +1 in numerator and denominator to avoid p=0
    when the real value exceeds every permutation (standard Davison-Hinkley
    convention).
    """
    if not null_accs:
        return float("nan")
    n = len(null_accs)
    n_gte = sum(1 for a in null_accs if a >= real_acc)
    return (n_gte + 1) / (n + 1)
