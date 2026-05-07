"""
H3/H4 Experiment: Physics-Preserving Adapter
Tests whether a lightweight adapter trained on cached activations can recover
physics information lost through the VLM pipeline.

Uses cached Qwen2.5-VL-7B activations from qwen_activations.h5.
"""
import h5py
import numpy as np
import torch
import torch.nn as nn
import json
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

np.random.seed(42)
torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================================================
# STEP 1: Load cached activations
# ============================================================
print("\n" + "=" * 60)
print("STEP 1: Loading cached activations")
print("=" * 60)

h5_path = Path("results/week2_qwen/activations/qwen_activations.h5")
assert h5_path.exists(), f"HDF5 not found at {h5_path}"

with h5py.File(h5_path, "r") as f:
    raw = {}
    for k in f.keys():
        raw[k] = f[k][:]
        print(f"  {k}: {raw[k].shape} {raw[k].dtype}")

# Dimensions per stage
STAGE_NAMES = ["stage_1_enc_out", "stage_2_post_proj", "stage_3_llm_8", "stage_4_llm_16"]
STAGE_SHORT = ["Encoder", "Merger", "LLM-8", "LLM-16"]
PHYSICS_PROPS = ["mass", "friction", "elasticity"]  # stability is all-NaN in cache
N_PHYSICS = len(PHYSICS_PROPS)

# 200 scenes, 196 patches per scene for physics labels
N_SCENES = 200
PATCHES_PER_SCENE = 196
assert raw["physics_labels"].shape[0] == N_SCENES * PATCHES_PER_SCENE

# Tokens per scene per stage
tokens_per_scene = {}
for s in STAGE_NAMES:
    tps = raw[s].shape[0] // N_SCENES
    tokens_per_scene[s] = tps
    print(f"  {s}: {tps} tokens/scene")

# ============================================================
# STEP 2: Build per-scene mean-pooled representations
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Building per-scene representations (mean-pool)")
print("=" * 60)

# Mean-pool activations per scene
scene_acts = {}
for s in STAGE_NAMES:
    tps = tokens_per_scene[s]
    dim = raw[s].shape[1]
    reshaped = raw[s].reshape(N_SCENES, tps, dim)
    scene_acts[s] = reshaped.mean(axis=1)  # (200, dim)
    print(f"  {s}: {scene_acts[s].shape}")

# Mean-pool physics labels per scene (nanmean to skip background patches)
# Only use first 3 columns (mass, friction, elasticity); col 3 stability is all-NaN
physics_raw = raw["physics_labels"].reshape(N_SCENES, PATCHES_PER_SCENE, 4)
physics_raw = physics_raw[:, :, :N_PHYSICS]  # drop stability column
scene_physics = np.nanmean(physics_raw, axis=1)  # (200, 3)
print(f"  physics_labels: {scene_physics.shape}")

# Check for any remaining NaN scenes and drop them
valid_mask = ~np.any(np.isnan(scene_physics), axis=1)
print(f"  Valid scenes: {valid_mask.sum()} / {N_SCENES}")

# Filter to valid scenes
for s in STAGE_NAMES:
    scene_acts[s] = scene_acts[s][valid_mask]
scene_physics = scene_physics[valid_mask]
n_valid = len(scene_physics)

# Train/test split indices
train_idx, test_idx = train_test_split(
    np.arange(n_valid), test_size=0.2, random_state=42
)
print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

# ============================================================
# STEP 3: Baseline global probing (Ridge regression)
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Baseline global probing (Ridge regression)")
print("=" * 60)

baseline_r2 = {}  # stage -> property -> R²
for s, sname in zip(STAGE_NAMES, STAGE_SHORT):
    X = scene_acts[s]
    X_train, X_test = X[train_idx], X[test_idx]
    baseline_r2[s] = {}
    for pi, pname in enumerate(PHYSICS_PROPS):
        y = scene_physics[:, pi]
        y_train, y_test = y[train_idx], y[test_idx]
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        r2 = r2_score(y_test, pred)
        baseline_r2[s][pname] = r2
    mean_r2 = np.mean(list(baseline_r2[s].values()))
    print(f"  {sname:10s} | " +
          " | ".join(f"{p}: {baseline_r2[s][p]:.3f}" for p in PHYSICS_PROPS) +
          f" | mean: {mean_r2:.3f}")

# ============================================================
# STEP 4: Physics-Preserving Adapter
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: Training Physics-Preserving Adapter")
print("=" * 60)


class PhysicsAdapter(nn.Module):
    """Lightweight adapter that modifies activations to preserve physics."""
    def __init__(self, hidden_dim, physics_dim=4, bottleneck=256):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_dim),
        )
        self.physics_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Linear(128, physics_dim),
        )

    def forward(self, x):
        adapted = x + 0.1 * self.adapter(x)  # small residual
        physics_pred = self.physics_head(adapted)
        return adapted, physics_pred


def train_adapter(X_train, y_train, hidden_dim, epochs=300, lr=1e-3, reg=0.01):
    """Train a PhysicsAdapter and return adapted activations."""
    adapter = PhysicsAdapter(hidden_dim, physics_dim=N_PHYSICS).to(DEVICE)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)

    adapter.train()
    for epoch in range(epochs):
        adapted, pred = adapter(X_t)
        loss_physics = nn.functional.mse_loss(pred, y_t)
        loss_reg = reg * torch.mean(adapter.adapter[0](X_t) ** 2)
        loss = loss_physics + loss_reg
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch+1}/{epochs}: loss={loss.item():.4f} "
                  f"(physics={loss_physics.item():.4f}, reg={loss_reg.item():.4f})")

    adapter.eval()
    return adapter


def apply_adapter(adapter, X):
    """Apply trained adapter to activations."""
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        adapted, _ = adapter(X_t)
        return adapted.cpu().numpy()


# Train adapter on post-merger (stage_2) activations — where physics info drops
target_stage = "stage_2_post_proj"
print(f"\nTraining adapter on {target_stage}...")
hidden_dim = scene_acts[target_stage].shape[1]
adapter_main = train_adapter(
    scene_acts[target_stage][train_idx],
    scene_physics[train_idx],
    hidden_dim,
    epochs=300,
)

# ============================================================
# STEP 5: Re-probe adapted activations
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Re-probing adapted activations (post-merger adapter)")
print("=" * 60)

# Apply adapter to post-merger activations and re-probe
adapted_acts = apply_adapter(adapter_main, scene_acts[target_stage])
X_train_a, X_test_a = adapted_acts[train_idx], adapted_acts[test_idx]

adapted_r2 = {}
for pi, pname in enumerate(PHYSICS_PROPS):
    y = scene_physics[:, pi]
    y_train, y_test = y[train_idx], y[test_idx]
    model = Ridge(alpha=1.0)
    model.fit(X_train_a, y_train)
    pred = model.predict(X_test_a)
    r2 = r2_score(y_test, pred)
    adapted_r2[pname] = r2

print(f"\n  {'Property':12s} | {'Baseline':>10s} | {'Adapted':>10s} | {'Delta':>10s}")
print(f"  {'-'*12}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
for p in PHYSICS_PROPS:
    b = baseline_r2[target_stage][p]
    a = adapted_r2[p]
    d = a - b
    print(f"  {p:12s} | {b:10.4f} | {a:10.4f} | {d:+10.4f}")
mean_b = np.mean([baseline_r2[target_stage][p] for p in PHYSICS_PROPS])
mean_a = np.mean(list(adapted_r2.values()))
print(f"  {'MEAN':12s} | {mean_b:10.4f} | {mean_a:10.4f} | {mean_a - mean_b:+10.4f}")

# ============================================================
# STEP 6: Component analysis (H3) — adapter at each stage
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: Component analysis (H3) — adapter at each pipeline stage")
print("=" * 60)

component_results = {}
for s, sname in zip(STAGE_NAMES, STAGE_SHORT):
    print(f"\n  --- Adapter at {sname} ({s}) ---")
    hdim = scene_acts[s].shape[1]
    adapter_i = train_adapter(
        scene_acts[s][train_idx],
        scene_physics[train_idx],
        hdim,
        epochs=300,
    )
    adapted_i = apply_adapter(adapter_i, scene_acts[s])
    X_tr, X_te = adapted_i[train_idx], adapted_i[test_idx]

    stage_r2 = {}
    for pi, pname in enumerate(PHYSICS_PROPS):
        y = scene_physics[:, pi]
        y_train, y_test = y[train_idx], y[test_idx]
        model = Ridge(alpha=1.0)
        model.fit(X_tr, y_train)
        pred = model.predict(X_te)
        r2 = r2_score(y_test, pred)
        stage_r2[pname] = r2

    mean_r2 = np.mean(list(stage_r2.values()))
    component_results[sname] = {
        "per_property": stage_r2,
        "mean_r2": float(mean_r2),
        "baseline_mean_r2": float(np.mean(list(baseline_r2[s].values()))),
        "delta": float(mean_r2 - np.mean(list(baseline_r2[s].values()))),
    }
    print(f"    Baseline mean R²: {component_results[sname]['baseline_mean_r2']:.4f}")
    print(f"    Adapted  mean R²: {mean_r2:.4f}")
    print(f"    Delta:             {component_results[sname]['delta']:+.4f}")

# Summary table
print("\n  === H3 Component Analysis Summary ===")
print(f"  {'Stage':10s} | {'Baseline':>10s} | {'Adapted':>10s} | {'Delta':>10s}")
print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
for sname in STAGE_SHORT:
    r = component_results[sname]
    print(f"  {sname:10s} | {r['baseline_mean_r2']:10.4f} | {r['mean_r2']:10.4f} | {r['delta']:+10.4f}")

best_stage = max(component_results, key=lambda k: component_results[k]["delta"])
print(f"\n  Best stage for physics recovery: {best_stage} "
      f"(delta = {component_results[best_stage]['delta']:+.4f})")

# ============================================================
# STEP 7: Save results
# ============================================================
print("\n" + "=" * 60)
print("STEP 7: Saving results")
print("=" * 60)

out_dir = Path("results/h3_h4")
out_dir.mkdir(parents=True, exist_ok=True)

results = {
    "experiment": "H3/H4 Physics-Preserving Adapter",
    "model": "Qwen2.5-VL-7B",
    "n_scenes": int(n_valid),
    "train_size": int(len(train_idx)),
    "test_size": int(len(test_idx)),
    "adapter_config": {
        "bottleneck": 256,
        "residual_scale": 0.1,
        "epochs": 300,
        "lr": 1e-3,
        "reg_weight": 0.01,
    },
    "baseline_probing": {
        sname: {
            "per_property": {p: float(baseline_r2[s][p]) for p in PHYSICS_PROPS},
            "mean_r2": float(np.mean(list(baseline_r2[s].values()))),
        }
        for s, sname in zip(STAGE_NAMES, STAGE_SHORT)
    },
    "h4_adapter_on_merger": {
        "stage": "stage_2_post_proj",
        "baseline": {p: float(baseline_r2[target_stage][p]) for p in PHYSICS_PROPS},
        "adapted": {p: float(v) for p, v in adapted_r2.items()},
        "mean_baseline": float(mean_b),
        "mean_adapted": float(mean_a),
        "mean_delta": float(mean_a - mean_b),
    },
    "h3_component_analysis": component_results,
    "h3_best_stage": best_stage,
}

out_path = out_dir / "adapter_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to {out_path}")

print("\n" + "=" * 60)
print("DONE — H3/H4 Adapter Experiment Complete")
print("=" * 60)
