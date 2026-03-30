"""
Unit tests for LinearProbe and MLPProbe.

Tests fitting, prediction, scoring, edge cases (NaN labels, small datasets),
and the per-patch R² computation used for saliency maps.
"""

import numpy as np
import pytest
import torch

from src.probing.linear_probe import LinearProbe, compute_per_patch_r2, bootstrap_r2_ci
from src.probing.mlp_probe import MLPProbe, MLPProbeNetwork


def make_synthetic_data(
    n_samples: int = 200,
    n_features: int = 64,
    noise: float = 0.1,
    seed: int = 0,
) -> tuple:
    """Generate a simple linear regression problem: y = w·x + noise.

    Returns (X, y, true_r2) where true_r2 is the expected R² (approximate).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    w = rng.standard_normal(n_features).astype(np.float32)
    w /= np.linalg.norm(w)
    y = X @ w + noise * rng.standard_normal(n_samples).astype(np.float32)
    return X, y


def make_data_with_nan(n_samples: int = 200, n_features: int = 64, nan_fraction: float = 0.3):
    """Generate data where some labels are NaN (background patches)."""
    X, y = make_synthetic_data(n_samples, n_features)
    rng = np.random.default_rng(42)
    nan_mask = rng.random(n_samples) < nan_fraction
    y[nan_mask] = float("nan")
    return X, y


class TestLinearProbe:
    """Tests for LinearProbe."""

    def test_fit_and_score_basic(self):
        """Probe should achieve R² > 0.5 on a linearly separable problem."""
        X, y = make_synthetic_data(n_samples=500, n_features=32, noise=0.05)
        n_train = 400
        probe = LinearProbe(input_dim=32, alpha=0.1)
        probe.fit(X[:n_train], y[:n_train])
        metrics = probe.score(X[n_train:], y[n_train:])
        assert metrics["r2"] > 0.5, f"Expected R² > 0.5, got {metrics['r2']:.3f}"
        assert "pearson_r" in metrics
        assert "mse" in metrics
        assert "mae" in metrics

    def test_fit_excludes_nan_labels(self):
        """Probe.fit() should not crash when y contains NaN values."""
        X, y = make_data_with_nan(n_samples=200, nan_fraction=0.3)
        probe = LinearProbe(input_dim=64)
        probe.fit(X[:150], y[:150])  # Should not raise
        assert probe._is_fitted

    def test_predict_shape(self):
        """predict() should return [N_samples] array."""
        X, y = make_synthetic_data()
        probe = LinearProbe(input_dim=64)
        probe.fit(X[:150], y[:150])
        preds = probe.predict(X[150:])
        assert preds.shape == (50,)
        assert preds.dtype == np.float32

    def test_score_excludes_nan(self):
        """score() should handle NaN in y_test gracefully."""
        X, y = make_data_with_nan(n_samples=200, nan_fraction=0.2)
        probe = LinearProbe(input_dim=64)
        probe.fit(X[:150], y[:150])
        # Add some NaN to test set
        y_test = y[150:].copy()
        metrics = probe.score(X[150:], y_test)
        assert not np.isnan(metrics["r2"]), "R² should not be NaN"

    def test_not_fitted_raises(self):
        """predict() and score() should raise if called before fit()."""
        probe = LinearProbe(input_dim=64)
        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.predict(np.zeros((10, 64)))

    def test_coefficients_shape(self):
        """get_coefficients() should return [D] float32 array."""
        X, y = make_synthetic_data(n_features=32)
        probe = LinearProbe(input_dim=32)
        probe.fit(X[:150], y[:150])
        coef = probe.get_coefficients()
        assert coef.shape == (32,)
        assert coef.dtype == np.float32

    def test_too_few_samples_raises(self):
        """fit() should raise if fewer than 10 valid samples."""
        X = np.zeros((5, 64), dtype=np.float32)
        y = np.ones(5, dtype=np.float32)
        probe = LinearProbe(input_dim=64)
        with pytest.raises(ValueError, match="Too few valid"):
            probe.fit(X, y)

    def test_normalization_enabled(self):
        """Probe with normalize_features=True should fit on scaled data."""
        X, y = make_synthetic_data(n_samples=300, n_features=32, noise=0.1)
        # Scale X to have very different ranges per feature
        X[:, 0] *= 1000
        probe = LinearProbe(input_dim=32, normalize_features=True)
        probe.fit(X[:200], y[:200])
        metrics = probe.score(X[200:], y[200:])
        assert metrics["r2"] > 0.3, "Normalized probe should achieve reasonable R²"


class TestComputePerPatchR2:
    """Tests for per-patch R² computation."""

    def test_output_shape(self):
        """Should return [N_patches] array."""
        N_samples, N_patches, D = 50, 9, 32
        X_all = np.random.randn(N_samples, N_patches, D).astype(np.float32)
        y_all = np.random.randn(N_samples, N_patches).astype(np.float32)
        r2 = compute_per_patch_r2(X_all, y_all, patch_grid_size=3)
        assert r2.shape == (N_patches,)
        assert r2.dtype == np.float32

    def test_values_non_negative(self):
        """R² values should be clipped to [0, ∞) (negatives → 0)."""
        N_samples, N_patches, D = 50, 9, 32
        X_all = np.random.randn(N_samples, N_patches, D).astype(np.float32)
        y_all = np.random.randn(N_samples, N_patches).astype(np.float32)
        r2 = compute_per_patch_r2(X_all, y_all, patch_grid_size=3)
        assert np.all(r2 >= 0.0), "Per-patch R² should be non-negative"

    def test_with_all_nan_patches(self):
        """Patches with all NaN labels should get R² = 0."""
        N_samples, N_patches, D = 30, 4, 16
        X_all = np.random.randn(N_samples, N_patches, D).astype(np.float32)
        y_all = np.full((N_samples, N_patches), float("nan"), dtype=np.float32)
        r2 = compute_per_patch_r2(X_all, y_all, patch_grid_size=2)
        assert np.all(r2 == 0.0), "All-NaN patches should have R² = 0"


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_returns_three_values(self):
        """Should return (mean, lower, upper)."""
        X, y = make_synthetic_data(n_samples=100, n_features=16)
        result = bootstrap_r2_ci(X[:80], y[:80], n_resamples=50, seed=42)
        assert len(result) == 3

    def test_ci_ordering(self):
        """CI lower bound should be ≤ mean ≤ upper bound."""
        X, y = make_synthetic_data(n_samples=100, n_features=16)
        mean, lower, upper = bootstrap_r2_ci(X[:80], y[:80], n_resamples=100, seed=42)
        assert lower <= mean <= upper, f"Expected lower ≤ mean ≤ upper, got {lower:.3f} ≤ {mean:.3f} ≤ {upper:.3f}"


class TestMLPProbeNetwork:
    """Tests for MLPProbeNetwork architecture."""

    def test_forward_output_shape(self):
        """Forward pass should return [B, 1] tensor."""
        net = MLPProbeNetwork(input_dim=64, hidden_dim=128)
        x = torch.randn(32, 64)
        out = net(x)
        assert out.shape == (32, 1), f"Expected (32, 1), got {out.shape}"

    def test_parameter_count(self):
        """Should have approximately (D*H + H) + (H*1 + 1) parameters."""
        net = MLPProbeNetwork(input_dim=64, hidden_dim=128)
        n_params = sum(p.numel() for p in net.parameters())
        expected = (64 * 128 + 128) + (128 * 1 + 1)
        assert n_params == expected, f"Expected {expected} params, got {n_params}"


class TestMLPProbe:
    """Tests for MLPProbe training and prediction."""

    def test_fit_and_score_basic(self):
        """MLPProbe should achieve R² > 0.4 on a simple linear problem."""
        X, y = make_synthetic_data(n_samples=400, n_features=32, noise=0.1)
        probe = MLPProbe(
            input_dim=32, hidden_dim=64, epochs=20, batch_size=64, device="cpu"
        )
        probe.fit(X[:300], y[:300], X[300:350], y[300:350])
        metrics = probe.score(X[350:], y[350:])
        assert metrics["r2"] > 0.4, f"Expected R² > 0.4, got {metrics['r2']:.3f}"

    def test_fit_with_nan_labels(self):
        """MLPProbe.fit() should not crash with NaN labels."""
        X, y = make_data_with_nan(n_samples=200, nan_fraction=0.25)
        probe = MLPProbe(input_dim=64, epochs=5, device="cpu")
        probe.fit(X[:150], y[:150])
        assert probe._is_fitted

    def test_predict_not_fitted_raises(self):
        """predict() should raise before fit()."""
        probe = MLPProbe(input_dim=64, device="cpu")
        with pytest.raises(RuntimeError, match="not been fitted"):
            probe.predict(np.zeros((10, 64)))

    def test_training_history_populated(self):
        """training_history should have train_loss entries after fitting."""
        X, y = make_synthetic_data(n_samples=100, n_features=16)
        probe = MLPProbe(input_dim=16, epochs=5, device="cpu", batch_size=32)
        probe.fit(X[:80], y[:80])
        assert len(probe.training_history["train_loss"]) > 0

    def test_early_stopping(self):
        """With patience=2, training should stop before max_epochs on plateaued val loss."""
        X, y = make_synthetic_data(n_samples=200, n_features=16, noise=5.0)  # High noise
        probe = MLPProbe(
            input_dim=16, epochs=100, patience=2, device="cpu",
            early_stopping_patience=2, batch_size=32
        )
        probe.fit(X[:120], y[:120], X[120:160], y[120:160])
        # Should stop well before 100 epochs due to early stopping
        assert len(probe.training_history["train_loss"]) < 90, (
            "Expected early stopping to trigger before 90 epochs"
        )
