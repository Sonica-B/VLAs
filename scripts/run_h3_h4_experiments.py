#!/usr/bin/env python3
"""
H3/H4 Experiments: Physics-Preserving Adapter & Probing-Guided Fine-Tuning
==========================================================================

Tests two core hypotheses:
  H3: Encoder+projection interventions outperform LLM-only for physics preservation
  H4: Fine-tuning can recover the R² degradation observed across pipeline stages

Approach: Train lightweight adapters on CACHED Qwen2.5-VL-7B activations (no GPU needed).
  - Experiment 1: Physics-preserving adapter at each pipeline stage
  - Experiment 2: Probing-guided representation learning (maximize R² directly)
  - Experiment 3: Simulated LLM degradation — does adapting stage 2 reduce stage 4 loss?

All experiments run on CPU using cached HDF5 activations from 300 realistic material scenes.
"""

import json
import os
import sys
import time
from pathlib import Path

# Fix Windows cp1252 encoding for Unicode characters in output
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVATIONS_PATH = PROJECT_ROOT / "results" / "week2_physion" / "activations" / "realistic_qwen_activations.h5"
RESULTS_DIR = PROJECT_ROOT / "results" / "h3_h4"
FIGURES_DIR = RESULTS_DIR / "figures"

STAGE_NAMES = [
    "stage_1_enc_out",
    "stage_2_post_proj",
    "stage_3_llm_8",
    "stage_4_llm_16",
]
STAGE_LABELS = ["Encoder Out", "Post-Projection", "LLM Layer 8", "LLM Layer 16"]
PROPERTIES = ["mass", "friction", "elasticity"]
PROPERTY_INDICES = {"mass": 0, "friction": 1, "elasticity": 2}

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_activations_and_labels(h5_path: str):
    """Load cached Qwen activations and physics labels from HDF5.

    Returns:
        activations: dict[stage_name] -> np.ndarray of shape [N, tokens, D]
        labels: np.ndarray of shape [N, 3] (scene-level mass, friction, elasticity)
    """
    print(f"Loading activations from {h5_path}")
    activations = {}
    with h5py.File(h5_path, "r") as f:
        for stage in STAGE_NAMES:
            if stage in f:
                activations[stage] = f[stage][:]
                print(f"  {stage}: {activations[stage].shape}")

        # Physics labels: [N, 196, 4] — per-patch, but physics is scene-level
        # Average across non-NaN patches to get scene-level labels
        raw_labels = f["physics_labels"][:]  # [300, 196, 4]

    # Compute scene-level labels by averaging across non-NaN patches
    # Each object's patches share identical physics values; background patches are NaN
    n_scenes = raw_labels.shape[0]
    scene_labels = np.zeros((n_scenes, 3), dtype=np.float32)
    for i in range(n_scenes):
        patch_labels = raw_labels[i, :, :3]  # [196, 3] — mass, friction, elasticity
        valid_mask = ~np.isnan(patch_labels[:, 0])
        if valid_mask.sum() > 0:
            scene_labels[i] = np.nanmean(patch_labels[valid_mask], axis=0)
        else:
            scene_labels[i] = 0.0  # fallback (shouldn't happen)

    print(f"  Scene labels: {scene_labels.shape}")
    print(f"  Label ranges: mass [{scene_labels[:,0].min():.2f}, {scene_labels[:,0].max():.2f}], "
          f"friction [{scene_labels[:,1].min():.2f}, {scene_labels[:,1].max():.2f}], "
          f"elasticity [{scene_labels[:,2].min():.2f}, {scene_labels[:,2].max():.2f}]")

    return activations, scene_labels


def global_pool(activations: np.ndarray) -> np.ndarray:
    """Global average pooling: [N, tokens, D] -> [N, D]."""
    return activations.mean(axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# Baseline: Ridge Probing with Cross-Validation
# ──────────────────────────────────────────────────────────────────────────────

def ridge_probe_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                   alphas=(0.01, 0.1, 1.0, 10.0, 100.0)):
    """Cross-validated ridge regression probing.

    Returns dict with mean R², std, and per-fold scores for each alpha.
    Best alpha is selected by mean CV R².
    """
    best_alpha = 1.0
    best_mean_r2 = -np.inf

    for alpha in alphas:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
        fold_r2s = []
        for train_idx, test_idx in kf.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            ridge = Ridge(alpha=alpha, solver="lsqr")
            ridge.fit(X_tr, y[train_idx])
            fold_r2s.append(r2_score(y[test_idx], ridge.predict(X_te)))
        mean_r2 = np.mean(fold_r2s)
        if mean_r2 > best_mean_r2:
            best_mean_r2 = mean_r2
            best_alpha = alpha

    # Final evaluation with best alpha
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_r2s = []
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        ridge = Ridge(alpha=best_alpha, solver="lsqr")
        ridge.fit(X_tr, y[train_idx])
        fold_r2s.append(r2_score(y[test_idx], ridge.predict(X_te)))

    return {
        "r2_mean": float(np.mean(fold_r2s)),
        "r2_std": float(np.std(fold_r2s)),
        "r2_folds": [float(r) for r in fold_r2s],
        "best_alpha": best_alpha,
    }


def run_baseline_probing(activations, labels):
    """Run ridge probing at all 4 stages for all 3 physics properties."""
    print("\n" + "=" * 70)
    print("BASELINE: Global-Pooled Ridge Probing (5-fold CV)")
    print("=" * 70)

    results = {}
    for stage in STAGE_NAMES:
        X = global_pool(activations[stage])
        results[stage] = {}
        for prop in PROPERTIES:
            y = labels[:, PROPERTY_INDICES[prop]]
            res = ridge_probe_cv(X, y)
            results[stage][prop] = res
            print(f"  {stage} / {prop}: R² = {res['r2_mean']:.4f} ± {res['r2_std']:.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1: Physics-Preserving Adapter
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsPreservingAdapter(nn.Module):
    """Lightweight adapter that modifies activations to better preserve physics.

    Architecture:
        - Residual MLP adapter: x + MLP(x)  (preserves original signal)
        - Physics prediction head: global_pool(adapted_x) -> [mass, friction, elasticity]

    Training losses:
        - L_physics: MSE between predicted and true physics values
        - L_recon: MSE between adapted and original activations (regularization)
    """

    def __init__(self, hidden_dim: int, bottleneck_dim: int = 256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, bottleneck_dim),
            nn.GELU(),
            nn.Linear(bottleneck_dim, hidden_dim),
        )
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 3),  # mass, friction, elasticity
        )
        # Initialize adapter near zero for residual stability
        with torch.no_grad():
            self.adapter[-1].weight.mul_(0.01)
            self.adapter[-1].bias.zero_()

    def forward(self, x):
        """
        Args:
            x: [batch, tokens, D] or [batch, D] (pre-pooled)
        Returns:
            adapted: same shape as x (with residual adapter applied)
            physics_pred: [batch, 3] physics predictions
        """
        adapted = x + self.adapter(x)

        # Global pool for physics prediction
        if adapted.dim() == 3:
            pooled = adapted.mean(dim=1)  # [batch, D]
        else:
            pooled = adapted

        physics_pred = self.physics_head(pooled)
        return adapted, physics_pred


def train_adapter(X_pool: np.ndarray, y: np.ndarray,
                  hidden_dim: int, bottleneck_dim: int = 256,
                  n_epochs: int = 200, lr: float = 1e-3,
                  lambda_recon: float = 0.1, patience: int = 30):
    """Train a physics-preserving adapter on global-pooled activations.

    Args:
        X_pool: [N, D] global-pooled activations
        y: [N, 3] physics labels
        hidden_dim: dimension of activations
        bottleneck_dim: adapter bottleneck size
        n_epochs: max training epochs
        lr: learning rate
        lambda_recon: weight of reconstruction loss
        patience: early stopping patience

    Returns:
        trained adapter, training history
    """
    N = X_pool.shape[0]
    n_train = int(0.8 * N)
    indices = np.random.permutation(N)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    # Normalize
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_pool[train_idx])
    X_val = x_scaler.transform(X_pool[val_idx])

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y[train_idx])
    y_val = y_scaler.transform(y[val_idx])

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    adapter = PhysicsPreservingAdapter(hidden_dim, bottleneck_dim)
    optimizer = optim.AdamW(adapter.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {"train_loss": [], "val_loss": [], "val_physics_r2": []}
    best_val_loss = np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # Train
        adapter.train()
        adapted, physics_pred = adapter(X_train_t)
        loss_physics = nn.functional.mse_loss(physics_pred, y_train_t)
        loss_recon = nn.functional.mse_loss(adapted, X_train_t)
        loss = loss_physics + lambda_recon * loss_recon

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Validate
        adapter.eval()
        with torch.no_grad():
            adapted_val, physics_pred_val = adapter(X_val_t)
            val_physics = nn.functional.mse_loss(physics_pred_val, y_val_t).item()
            val_recon = nn.functional.mse_loss(adapted_val, X_val_t).item()
            val_loss = val_physics + lambda_recon * val_recon

            # R² on validation (in original scale)
            pred_np = y_scaler.inverse_transform(physics_pred_val.numpy())
            true_np = y[val_idx]
            val_r2 = {prop: r2_score(true_np[:, i], pred_np[:, i])
                      for i, prop in enumerate(PROPERTIES)}

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_physics_r2"].append(val_r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in adapter.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 50 == 0 or epoch == 0:
            r2_str = ", ".join(f"{p}={val_r2[p]:.3f}" for p in PROPERTIES)
            print(f"  Epoch {epoch+1:3d}: train_loss={loss.item():.4f}, "
                  f"val_loss={val_loss:.4f}, R²: {r2_str}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    adapter.load_state_dict(best_state)
    return adapter, history, x_scaler, y_scaler, train_idx, val_idx


def probe_adapted_activations(adapter, X_pool, y, x_scaler, stage_name):
    """After training adapter, apply it and re-probe to measure R² improvement."""
    adapter.eval()
    X_norm = x_scaler.transform(X_pool)
    X_t = torch.tensor(X_norm, dtype=torch.float32)

    with torch.no_grad():
        adapted, _ = adapter(X_t)
        X_adapted = adapted.numpy()

    # Probe both original (normalized) and adapted
    results = {}
    for prop in PROPERTIES:
        yi = y[:, PROPERTY_INDICES[prop]]
        orig_res = ridge_probe_cv(X_norm, yi)
        adapted_res = ridge_probe_cv(X_adapted, yi)
        results[prop] = {
            "original_r2": orig_res["r2_mean"],
            "adapted_r2": adapted_res["r2_mean"],
            "delta_r2": adapted_res["r2_mean"] - orig_res["r2_mean"],
            "original_std": orig_res["r2_std"],
            "adapted_std": adapted_res["r2_std"],
        }
    return results


def run_adapter_experiment(activations, labels):
    """Run physics-preserving adapter at each pipeline stage."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Physics-Preserving Adapter (per-stage)")
    print("=" * 70)
    print("Training adapter at each stage to maximize physics R² with residual constraint.\n")

    all_results = {}

    for stage in STAGE_NAMES:
        X_pool = global_pool(activations[stage])
        hidden_dim = X_pool.shape[1]
        bottleneck = min(256, hidden_dim // 4)

        print(f"\n--- {stage} (dim={hidden_dim}, bottleneck={bottleneck}) ---")

        adapter, history, x_scaler, y_scaler, train_idx, val_idx = train_adapter(
            X_pool, labels, hidden_dim=hidden_dim, bottleneck_dim=bottleneck,
            n_epochs=300, lr=1e-3, lambda_recon=0.1, patience=40,
        )

        # Probe original vs adapted
        probe_results = probe_adapted_activations(adapter, X_pool, labels, x_scaler, stage)

        # Direct physics prediction from adapter head
        adapter.eval()
        X_all_norm = x_scaler.transform(X_pool)
        with torch.no_grad():
            _, pred = adapter(torch.tensor(X_all_norm, dtype=torch.float32))
            pred_np = y_scaler.inverse_transform(pred.numpy())

        direct_r2 = {}
        for prop in PROPERTIES:
            idx = PROPERTY_INDICES[prop]
            direct_r2[prop] = float(r2_score(labels[:, idx], pred_np[:, idx]))

        all_results[stage] = {
            "probe_comparison": probe_results,
            "direct_physics_r2": direct_r2,
            "adapter_params": sum(p.numel() for p in adapter.parameters()),
            "bottleneck_dim": bottleneck,
            "best_val_loss": float(history["val_loss"][-1]) if history["val_loss"] else None,
        }

        # Print summary
        print(f"\n  Results for {stage}:")
        print(f"  {'Property':<12} {'Original R²':>12} {'Adapted R²':>12} {'ΔR²':>8} {'Direct R²':>10}")
        print(f"  {'-'*56}")
        for prop in PROPERTIES:
            pr = probe_results[prop]
            dr = direct_r2[prop]
            print(f"  {prop:<12} {pr['original_r2']:>12.4f} {pr['adapted_r2']:>12.4f} "
                  f"{pr['delta_r2']:>+8.4f} {dr:>10.4f}")

    return all_results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2: Probing-Guided Representation Learning
# ──────────────────────────────────────────────────────────────────────────────

class ProbingGuidedTransform(nn.Module):
    """Learn a transformation that MAXIMIZES probe R² at each stage.

    Unlike the adapter (which also has a reconstruction loss), this module
    ONLY optimizes for physics decodability. It answers: what is the maximum
    physics information that COULD be preserved at each stage?
    """

    def __init__(self, input_dim: int, output_dim: int = 3):
        super().__init__()
        bottleneck = min(512, input_dim // 2)
        self.transform = nn.Sequential(
            nn.Linear(input_dim, bottleneck),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, bottleneck),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(bottleneck, output_dim),
        )

    def forward(self, x):
        return self.transform(x)


def train_probing_guided(X_pool, y, n_epochs=300, lr=1e-3, patience=30):
    """Train a probing-guided transform to predict physics from activations.

    This is a NONLINEAR probe — it tells us the upper bound of physics info
    available at each stage (compared to the linear ridge probe baseline).
    """
    N = X_pool.shape[0]
    n_train = int(0.8 * N)
    indices = np.random.permutation(N)
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    scaler = StandardScaler()
    X_train = torch.tensor(scaler.fit_transform(X_pool[train_idx]), dtype=torch.float32)
    X_val = torch.tensor(scaler.transform(X_pool[val_idx]), dtype=torch.float32)

    y_scaler = StandardScaler()
    y_train = torch.tensor(y_scaler.fit_transform(y[train_idx]), dtype=torch.float32)
    y_val_raw = y[val_idx]

    model = ProbingGuidedTransform(X_pool.shape[1], output_dim=3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_loss = np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        pred = model(X_train)
        loss = nn.functional.mse_loss(pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pred_val = model(X_val)
            pred_val_np = y_scaler.inverse_transform(pred_val.numpy())
            val_loss = nn.functional.mse_loss(
                pred_val,
                torch.tensor(y_scaler.transform(y_val_raw), dtype=torch.float32),
            ).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.load_state_dict(best_state)

    # Final evaluation on ALL data (5-fold CV for fair comparison)
    model.eval()
    X_all = torch.tensor(scaler.transform(X_pool), dtype=torch.float32)
    with torch.no_grad():
        pred_all = y_scaler.inverse_transform(model(X_all).numpy())

    r2s = {}
    for prop in PROPERTIES:
        idx = PROPERTY_INDICES[prop]
        r2s[prop] = float(r2_score(y[:, idx], pred_all[:, idx]))

    return r2s, model


def run_probing_guided_experiment(activations, labels):
    """Train probing-guided transforms at each stage — upper bound analysis."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Probing-Guided Transform (nonlinear upper bound)")
    print("=" * 70)
    print("Training 3-layer MLP to predict physics directly. Shows maximum\n"
          "decodable physics info at each stage (upper bound for linear probes).\n")

    results = {}
    for stage in STAGE_NAMES:
        X_pool = global_pool(activations[stage])
        r2s, _ = train_probing_guided(X_pool, labels)
        results[stage] = r2s
        r2_str = ", ".join(f"{p}={r2s[p]:.4f}" for p in PROPERTIES)
        print(f"  {stage}: {r2_str}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3: Simulated LLM Degradation Analysis
# ──────────────────────────────────────────────────────────────────────────────

def learn_stage_mapping(X_source: np.ndarray, X_target: np.ndarray):
    """Learn a linear mapping from one stage's activations to another.

    This simulates what the LLM layers do to the representation.
    Returns the fitted Ridge model and scaler.
    """
    scaler_src = StandardScaler()
    scaler_tgt = StandardScaler()
    X_s = scaler_src.fit_transform(X_source)
    X_t = scaler_tgt.fit_transform(X_target)

    # Use Ridge regression to learn the mapping (column by column would be slow,
    # so we use multi-output ridge)
    from sklearn.multioutput import MultiOutputRegressor
    # For high-dim → high-dim, use a truncated approach: PCA → map → inverse PCA
    from sklearn.decomposition import PCA

    # Reduce dimensionality for tractability
    n_components = min(128, X_s.shape[1], X_t.shape[1])
    pca_src = PCA(n_components=n_components, random_state=SEED)
    pca_tgt = PCA(n_components=n_components, random_state=SEED)

    X_s_pca = pca_src.fit_transform(X_s)
    X_t_pca = pca_tgt.fit_transform(X_t)

    ridge = Ridge(alpha=10.0, solver="lsqr")
    ridge.fit(X_s_pca, X_t_pca)

    return ridge, scaler_src, scaler_tgt, pca_src, pca_tgt


def apply_stage_mapping(X_adapted, ridge, scaler_src, scaler_tgt, pca_src, pca_tgt):
    """Apply learned stage mapping to adapted activations."""
    X_s = scaler_src.transform(X_adapted)
    X_s_pca = pca_src.transform(X_s)
    X_t_pca = ridge.predict(X_s_pca)
    X_t = pca_tgt.inverse_transform(X_t_pca)
    return scaler_tgt.inverse_transform(X_t)


def run_degradation_simulation(activations, labels):
    """Simulate LLM degradation: learn stage2→stage4 mapping, apply to adapted activations."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Simulated LLM Degradation Recovery")
    print("=" * 70)
    print("Learn linear mapping stage2→stage4 (captures LLM transformation).")
    print("Apply to adapter-modified stage2 → 'simulated stage4'.")
    print("If adapted simulated-stage4 R² > original stage4 R² → degradation reduced.\n")

    X_stage2 = global_pool(activations["stage_2_post_proj"])
    X_stage4 = global_pool(activations["stage_4_llm_16"])

    # Learn mapping from stage 2 → stage 4
    print("  Learning stage2 → stage4 linear mapping...")
    mapping = learn_stage_mapping(X_stage2, X_stage4)
    ridge_map, scaler_src, scaler_tgt, pca_src, pca_tgt = mapping

    # Verify mapping quality
    X_stage4_pred = apply_stage_mapping(X_stage2, *mapping)
    recon_r2 = r2_score(X_stage4[:, :10], X_stage4_pred[:, :10])
    print(f"  Mapping quality (first 10 dims): R² = {recon_r2:.4f}")

    # Train adapter on stage 2
    print("  Training physics-preserving adapter on stage 2...")
    hidden_dim = X_stage2.shape[1]
    adapter, history, x_scaler, y_scaler, train_idx, val_idx = train_adapter(
        X_stage2, labels, hidden_dim=hidden_dim, bottleneck_dim=min(256, hidden_dim // 4),
        n_epochs=300, lr=1e-3, lambda_recon=0.05, patience=40,
    )

    # Get adapted stage 2 activations
    adapter.eval()
    X_stage2_norm = x_scaler.transform(X_stage2)
    with torch.no_grad():
        adapted_stage2, _ = adapter(torch.tensor(X_stage2_norm, dtype=torch.float32))
        X_stage2_adapted = adapted_stage2.numpy()

    # Apply stage mapping to adapted activations → simulated adapted stage 4
    X_stage4_simulated = apply_stage_mapping(X_stage2_adapted, *mapping)

    # Probe all versions
    results = {}
    for prop in PROPERTIES:
        yi = labels[:, PROPERTY_INDICES[prop]]

        r2_stage2_orig = ridge_probe_cv(X_stage2_norm, yi)["r2_mean"]
        r2_stage2_adapted = ridge_probe_cv(X_stage2_adapted, yi)["r2_mean"]
        r2_stage4_orig = ridge_probe_cv(
            StandardScaler().fit_transform(X_stage4), yi)["r2_mean"]
        r2_stage4_simulated = ridge_probe_cv(
            StandardScaler().fit_transform(X_stage4_simulated), yi)["r2_mean"]

        results[prop] = {
            "stage2_original_r2": r2_stage2_orig,
            "stage2_adapted_r2": r2_stage2_adapted,
            "stage4_original_r2": r2_stage4_orig,
            "stage4_simulated_r2": r2_stage4_simulated,
            "degradation_original": r2_stage2_orig - r2_stage4_orig,
            "degradation_adapted": r2_stage2_adapted - r2_stage4_simulated,
        }

        print(f"\n  {prop}:")
        print(f"    Stage 2 original R²:        {r2_stage2_orig:.4f}")
        print(f"    Stage 2 adapted R²:         {r2_stage2_adapted:.4f}")
        print(f"    Stage 4 original R²:        {r2_stage4_orig:.4f}")
        print(f"    Stage 4 simulated adapted R²: {r2_stage4_simulated:.4f}")
        print(f"    Degradation (orig):         {results[prop]['degradation_original']:.4f}")
        print(f"    Degradation (adapted):      {results[prop]['degradation_adapted']:.4f}")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────────────────────────────

def plot_degradation_curves(baseline_results, adapter_results, save_path):
    """Plot R² degradation curves: before vs after adapter at each stage."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    stage_x = np.arange(len(STAGE_NAMES))

    colors_orig = {"mass": "#2196F3", "friction": "#4CAF50", "elasticity": "#FF9800"}
    colors_adapted = {"mass": "#0D47A1", "friction": "#1B5E20", "elasticity": "#E65100"}

    for i, prop in enumerate(PROPERTIES):
        ax = axes[i]

        # Original baseline
        orig_r2 = [baseline_results[s][prop]["r2_mean"] for s in STAGE_NAMES]
        orig_std = [baseline_results[s][prop]["r2_std"] for s in STAGE_NAMES]

        # Adapted
        adapted_r2 = []
        for s in STAGE_NAMES:
            if s in adapter_results and prop in adapter_results[s]["probe_comparison"]:
                adapted_r2.append(adapter_results[s]["probe_comparison"][prop]["adapted_r2"])
            else:
                adapted_r2.append(orig_r2[STAGE_NAMES.index(s)])

        ax.errorbar(stage_x, orig_r2, yerr=orig_std, marker="o", linewidth=2,
                     color=colors_orig[prop], label="Original", capsize=4)
        ax.plot(stage_x, adapted_r2, marker="s", linewidth=2, linestyle="--",
                color=colors_adapted[prop], label="After Adapter")

        # Shade the improvement region
        for j in range(len(STAGE_NAMES)):
            if adapted_r2[j] > orig_r2[j]:
                ax.annotate(f"+{adapted_r2[j] - orig_r2[j]:.3f}",
                            xy=(j, adapted_r2[j]), xytext=(0, 10),
                            textcoords="offset points", ha="center", fontsize=8,
                            color=colors_adapted[prop], fontweight="bold")

        ax.set_xticks(stage_x)
        ax.set_xticklabels(STAGE_LABELS, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("R² (5-fold CV)")
        ax.set_title(f"{prop.capitalize()}", fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 0.85)

    fig.suptitle("H4: Physics Degradation — Before vs After Adapter", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_h3_stage_comparison(adapter_results, save_path):
    """Plot H3: which stage benefits most from the adapter? (encoder/proj vs LLM)"""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(PROPERTIES))
    width = 0.18
    colors = ["#E3F2FD", "#90CAF9", "#42A5F5", "#1565C0"]

    for i, stage in enumerate(STAGE_NAMES):
        deltas = []
        for prop in PROPERTIES:
            if stage in adapter_results:
                d = adapter_results[stage]["probe_comparison"][prop]["delta_r2"]
            else:
                d = 0
            deltas.append(d)
        bars = ax.bar(x + i * width, deltas, width, label=STAGE_LABELS[i], color=colors[i])
        for bar, d in zip(bars, deltas):
            if abs(d) > 0.001:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{d:+.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([p.capitalize() for p in PROPERTIES])
    ax.set_ylabel("ΔR² (Adapted − Original)")
    ax.set_title("H3: Adapter Improvement by Pipeline Stage", fontsize=13, fontweight="bold")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_degradation_simulation(sim_results, save_path):
    """Plot the simulated degradation recovery results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, prop in enumerate(PROPERTIES):
        ax = axes[i]
        r = sim_results[prop]

        stages = ["Stage 2\nOriginal", "Stage 2\nAdapted", "Stage 4\nOriginal", "Stage 4\nSim. Adapted"]
        values = [r["stage2_original_r2"], r["stage2_adapted_r2"],
                  r["stage4_original_r2"], r["stage4_simulated_r2"]]
        colors = ["#90CAF9", "#42A5F5", "#FFCC80", "#FF9800"]

        bars = ax.bar(stages, values, color=colors, edgecolor="black", linewidth=0.5)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, max(v, 0) + 0.01,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

        # Draw degradation arrows
        ax.annotate("", xy=(2, r["stage4_original_r2"]),
                     xytext=(0, r["stage2_original_r2"]),
                     arrowprops=dict(arrowstyle="->", color="red", lw=1.5, ls="--"))
        ax.annotate("", xy=(3, r["stage4_simulated_r2"]),
                     xytext=(1, r["stage2_adapted_r2"]),
                     arrowprops=dict(arrowstyle="->", color="green", lw=1.5, ls="--"))

        ax.set_ylabel("R² (5-fold CV)")
        ax.set_title(f"{prop.capitalize()}", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(min(min(values) - 0.1, -0.15), max(values) + 0.15)

    fig.suptitle("H4: Simulated LLM Degradation Recovery\n"
                 "(Red: original degradation, Green: adapted degradation)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_combined_summary(baseline, adapter, probing_guided, simulation, save_path):
    """Combined summary figure for the paper."""
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

    # Panel A: Degradation curves (top-left, spans 2 columns)
    ax_a = fig.add_subplot(gs[0, :2])
    markers = {"mass": "o", "friction": "s", "elasticity": "D"}
    colors_o = {"mass": "#2196F3", "friction": "#4CAF50", "elasticity": "#FF9800"}
    colors_a = {"mass": "#0D47A1", "friction": "#1B5E20", "elasticity": "#E65100"}

    for prop in PROPERTIES:
        orig = [baseline[s][prop]["r2_mean"] for s in STAGE_NAMES]
        adapt = [adapter[s]["probe_comparison"][prop]["adapted_r2"] for s in STAGE_NAMES]
        ax_a.plot(range(4), orig, marker=markers[prop], color=colors_o[prop],
                  linewidth=2, label=f"{prop} (orig)")
        ax_a.plot(range(4), adapt, marker=markers[prop], color=colors_a[prop],
                  linewidth=2, linestyle="--", label=f"{prop} (adapted)")

    ax_a.set_xticks(range(4))
    ax_a.set_xticklabels(STAGE_LABELS, fontsize=9)
    ax_a.set_ylabel("R² (5-fold CV)")
    ax_a.set_title("A) Degradation Curves: Original vs Adapted", fontweight="bold")
    ax_a.legend(fontsize=7, ncol=2)
    ax_a.grid(True, alpha=0.3)

    # Panel B: Nonlinear upper bound (top-right)
    ax_b = fig.add_subplot(gs[0, 2])
    x = np.arange(len(STAGE_NAMES))
    w = 0.25
    for j, prop in enumerate(PROPERTIES):
        lin = [baseline[s][prop]["r2_mean"] for s in STAGE_NAMES]
        nlin = [probing_guided[s][prop] for s in STAGE_NAMES]
        ax_b.bar(x + j * w - w, lin, w, alpha=0.5, color=colors_o[prop], label=f"{prop} (linear)" if j == 0 else "")
        ax_b.bar(x + j * w - w, nlin, w, alpha=0.8, color=colors_a[prop], edgecolor="black",
                 linewidth=0.3, label=f"{prop} (nonlinear)" if j == 0 else "")

    ax_b.set_xticks(x)
    ax_b.set_xticklabels(["S1", "S2", "S3", "S4"], fontsize=9)
    ax_b.set_ylabel("R²")
    ax_b.set_title("B) Linear vs Nonlinear Probes", fontweight="bold")
    ax_b.grid(True, alpha=0.3, axis="y")

    # Panel C: Stage-wise adapter ΔR² (bottom-left)
    ax_c = fig.add_subplot(gs[1, 0])
    prop_colors = {"mass": "#2196F3", "friction": "#4CAF50", "elasticity": "#FF9800"}
    x = np.arange(len(STAGE_NAMES))
    w = 0.25
    for j, prop in enumerate(PROPERTIES):
        deltas = [adapter[s]["probe_comparison"][prop]["delta_r2"] for s in STAGE_NAMES]
        ax_c.bar(x + j * w - w, deltas, w, color=prop_colors[prop], label=prop.capitalize())
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(["S1", "S2", "S3", "S4"])
    ax_c.axhline(0, color="black", linewidth=0.5)
    ax_c.set_ylabel("ΔR²")
    ax_c.set_title("C) Adapter Improvement (H3)", fontweight="bold")
    ax_c.legend(fontsize=8)
    ax_c.grid(True, alpha=0.3, axis="y")

    # Panel D: Degradation recovery (bottom-center)
    ax_d = fig.add_subplot(gs[1, 1])
    props_x = np.arange(len(PROPERTIES))
    deg_orig = [simulation[p]["degradation_original"] for p in PROPERTIES]
    deg_adapt = [simulation[p]["degradation_adapted"] for p in PROPERTIES]
    ax_d.bar(props_x - 0.15, deg_orig, 0.3, color="#EF5350", label="Original Degradation")
    ax_d.bar(props_x + 0.15, deg_adapt, 0.3, color="#66BB6A", label="Adapted Degradation")
    ax_d.set_xticks(props_x)
    ax_d.set_xticklabels([p.capitalize() for p in PROPERTIES])
    ax_d.set_ylabel("Stage2 R² − Stage4 R²")
    ax_d.set_title("D) Degradation Recovery (H4)", fontweight="bold")
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3, axis="y")

    # Panel E: Direct physics prediction R² from adapter heads (bottom-right)
    ax_e = fig.add_subplot(gs[1, 2])
    x = np.arange(len(STAGE_NAMES))
    w = 0.25
    for j, prop in enumerate(PROPERTIES):
        vals = [adapter[s]["direct_physics_r2"][prop] for s in STAGE_NAMES]
        ax_e.bar(x + j * w - w, vals, w, color=prop_colors[prop],
                 label=prop.capitalize() if j < 3 else "")
    ax_e.set_xticks(x)
    ax_e.set_xticklabels(["S1", "S2", "S3", "S4"])
    ax_e.set_ylabel("R²")
    ax_e.set_title("E) Direct Prediction from Adapter", fontweight="bold")
    ax_e.legend(fontsize=8)
    ax_e.grid(True, alpha=0.3, axis="y")

    fig.suptitle("H3/H4 Experiments: Physics-Preserving Adapters in Qwen2.5-VL-7B Pipeline",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Setup
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not ACTIVATIONS_PATH.exists():
        print(f"ERROR: Activations not found at {ACTIVATIONS_PATH}")
        sys.exit(1)

    # Load data
    activations, labels = load_activations_and_labels(str(ACTIVATIONS_PATH))

    # ── Baseline ──
    baseline_results = run_baseline_probing(activations, labels)

    # ── Experiment 1: Physics-Preserving Adapter ──
    adapter_results = run_adapter_experiment(activations, labels)

    # ── Experiment 2: Probing-Guided Transform ──
    probing_guided_results = run_probing_guided_experiment(activations, labels)

    # ── Experiment 3: Simulated LLM Degradation ──
    simulation_results = run_degradation_simulation(activations, labels)

    # ── Visualizations ──
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    plot_degradation_curves(baseline_results, adapter_results,
                            FIGURES_DIR / "h4_degradation_before_after.png")
    plot_h3_stage_comparison(adapter_results,
                             FIGURES_DIR / "h3_stage_comparison.png")
    plot_degradation_simulation(simulation_results,
                                FIGURES_DIR / "h4_degradation_simulation.png")
    plot_combined_summary(baseline_results, adapter_results,
                          probing_guided_results, simulation_results,
                          FIGURES_DIR / "h3_h4_combined_summary.png")

    # ── Save all results ──
    all_results = {
        "experiment": "h3_h4_physics_preserving_adapter",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": time.time() - t0,
        "activations_file": str(ACTIVATIONS_PATH),
        "n_scenes": labels.shape[0],
        "baseline_probing": baseline_results,
        "adapter_experiment": {
            stage: {
                "probe_comparison": adapter_results[stage]["probe_comparison"],
                "direct_physics_r2": adapter_results[stage]["direct_physics_r2"],
                "adapter_params": adapter_results[stage]["adapter_params"],
            }
            for stage in STAGE_NAMES
        },
        "probing_guided_transform": probing_guided_results,
        "simulated_degradation": simulation_results,
    }

    results_path = RESULTS_DIR / "h3_h4_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {results_path}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY: H3/H4 HYPOTHESIS EVALUATION")
    print("=" * 70)

    # H3: Does encoder/projection benefit more than LLM?
    print("\n  H3: Encoder+Projection vs LLM adapter improvement")
    early_stages = ["stage_1_enc_out", "stage_2_post_proj"]
    late_stages = ["stage_3_llm_8", "stage_4_llm_16"]
    for prop in PROPERTIES:
        early_delta = np.mean([adapter_results[s]["probe_comparison"][prop]["delta_r2"]
                               for s in early_stages])
        late_delta = np.mean([adapter_results[s]["probe_comparison"][prop]["delta_r2"]
                              for s in late_stages])
        verdict = "SUPPORTED" if early_delta > late_delta else "NOT SUPPORTED"
        print(f"    {prop}: early ΔR²={early_delta:+.4f}, late ΔR²={late_delta:+.4f} → {verdict}")

    # H4: Does adapter reduce degradation?
    print("\n  H4: Does adapter reduce stage2→stage4 degradation?")
    for prop in PROPERTIES:
        s = simulation_results[prop]
        orig_deg = s["degradation_original"]
        adapt_deg = s["degradation_adapted"]
        reduced = adapt_deg < orig_deg
        verdict = "SUPPORTED" if reduced else "NOT SUPPORTED"
        print(f"    {prop}: orig degradation={orig_deg:.4f}, adapted={adapt_deg:.4f} → {verdict}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")
    print("  Done!")


if __name__ == "__main__":
    main()
