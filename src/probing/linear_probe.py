"""
Linear ridge regression probe for predicting physics properties from VLM activations.

This is the primary probe used in Phase 1. Ridge regression is preferred over
a vanilla linear regression because:
  1. Activation dimensions D are high (1024–3584) but samples per patch are moderate
  2. Ridge regularization prevents overfitting on the flat activation manifold
  3. Closed-form solution is fast and reproducible

Per-patch R² scores are computed by training one probe per patch position and
averaging across the spatial dimension — these form the physics saliency maps.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


class LinearProbe:
    """Linear ridge regression probe for physics property prediction.

    Fits a scikit-learn Ridge regressor on flattened per-patch activations
    to predict a scalar physics property (e.g., mass, friction).

    Args:
        input_dim: Feature dimension D (activation hidden dim at the target stage).
        alpha: Ridge L2 regularization strength. Larger = more regularization.
        fit_intercept: Whether to fit a bias term. Default True.
        normalize_features: Apply StandardScaler before fitting. Default True.

    Example:
        >>> probe = LinearProbe(input_dim=1280, alpha=1.0)
        >>> probe.fit(X_train, y_train)
        >>> r2 = probe.score(X_test, y_test)
    """

    def __init__(
        self,
        input_dim: int,
        alpha: float = 1.0,
        fit_intercept: bool = True,
        normalize_features: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize_features = normalize_features

        self._ridge = Ridge(alpha=alpha, fit_intercept=fit_intercept, max_iter=1000, solver="lsqr")
        self._scaler: Optional[StandardScaler] = StandardScaler() if normalize_features else None
        self._is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearProbe":
        """Fit the probe on activation features X and physics labels y.

        Args:
            X: float32 array [N_samples, D] — patch activations.
            y: float32 array [N_samples] — physics property values.
               NaN values are automatically excluded.

        Returns:
            self (for chaining).
        """
        # Exclude NaN labels (background patches)
        valid = ~np.isnan(y)
        X_valid, y_valid = X[valid], y[valid]

        if len(y_valid) < 10:
            raise ValueError(
                f"Too few valid (non-NaN) samples: {len(y_valid)}. "
                "Check that physics labels are assigned correctly."
            )

        if self._scaler is not None:
            X_valid = self._scaler.fit_transform(X_valid)

        self._ridge.fit(X_valid, y_valid)
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict physics property values for a set of activations.

        Args:
            X: float32 array [N_samples, D].

        Returns:
            Predicted values, float32 array [N_samples].
        """
        self._check_fitted()
        if self._scaler is not None:
            X = self._scaler.transform(X)
        return self._ridge.predict(X).astype(np.float32)

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics on test data.

        Args:
            X: float32 array [N_test, D].
            y: float32 array [N_test] — ground truth labels (may contain NaN).

        Returns:
            Dict with keys: r2, pearson_r, mse, mae.
        """
        self._check_fitted()
        valid = ~np.isnan(y)
        X_valid, y_valid = X[valid], y[valid]

        y_pred = self.predict(X_valid)
        r2 = r2_score(y_valid, y_pred)
        pearson_r, _ = pearsonr(y_valid, y_pred) if len(y_valid) > 2 else (0.0, 1.0)
        mse = float(np.mean((y_valid - y_pred) ** 2))
        mae = float(np.mean(np.abs(y_valid - y_pred)))

        return {"r2": float(r2), "pearson_r": float(pearson_r), "mse": mse, "mae": mae}

    def get_coefficients(self) -> np.ndarray:
        """Return the fitted weight vector [D]."""
        self._check_fitted()
        return self._ridge.coef_.astype(np.float32)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("LinearProbe has not been fitted. Call fit() first.")


def compute_per_patch_r2(
    X_all: np.ndarray,
    y_all: np.ndarray,
    patch_grid_size: int = 14,
    alpha: float = 1.0,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> np.ndarray:
    """Compute per-patch R² scores for generating physics saliency maps.

    For each patch position p ∈ [0, N_patches), fits a separate linear probe
    using activations from patch p across all samples. Returns a 1D array
    of R² values that can be reshaped to [patch_grid_size, patch_grid_size].

    Args:
        X_all: float32 array [N_samples, N_patches, D] — all patch activations.
        y_all: float32 array [N_samples, N_patches] — per-patch physics labels.
        patch_grid_size: Spatial grid size (e.g., 14 for a 14×14 grid).
        alpha: Ridge regularization strength.
        train_fraction: Fraction of samples used for training per patch probe.
        seed: Random seed for train/test split.

    Returns:
        r2_per_patch: float32 array [N_patches] — R² at each patch position.
    """
    N_samples, N_patches, D = X_all.shape
    rng = np.random.default_rng(seed)
    n_train = int(N_samples * train_fraction)
    indices = rng.permutation(N_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]

    r2_per_patch = np.zeros(N_patches, dtype=np.float32)

    for p in range(N_patches):
        X_p = X_all[:, p, :]       # [N_samples, D]
        y_p = y_all[:, p]          # [N_samples]

        # Skip patches that are all background in training set
        y_train = y_p[train_idx]
        if np.sum(~np.isnan(y_train)) < 5:
            r2_per_patch[p] = 0.0
            continue

        probe = LinearProbe(input_dim=D, alpha=alpha)
        try:
            probe.fit(X_p[train_idx], y_train)
            metrics = probe.score(X_p[test_idx], y_p[test_idx])
            r2_per_patch[p] = max(0.0, metrics["r2"])  # Clip negative R² to 0
        except Exception:
            r2_per_patch[p] = 0.0

    return r2_per_patch


def bootstrap_r2_ci(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    n_resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for R².

    Args:
        X: [N, D] activation features.
        y: [N] physics labels.
        alpha: Ridge regularization.
        n_resamples: Number of bootstrap resamples.
        confidence: Confidence level (e.g., 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        (r2_mean, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    r2_samples = []

    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        X_b, y_b = X[idx], y[idx]
        probe = LinearProbe(input_dim=X.shape[1], alpha=alpha)
        try:
            probe.fit(X_b, y_b)
            metrics = probe.score(X, y)   # Evaluate on full data
            r2_samples.append(metrics["r2"])
        except Exception:
            continue

    r2_arr = np.array(r2_samples)
    alpha_tail = (1 - confidence) / 2
    return (
        float(np.mean(r2_arr)),
        float(np.quantile(r2_arr, alpha_tail)),
        float(np.quantile(r2_arr, 1 - alpha_tail)),
    )
