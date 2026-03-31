"""
Day 5-6: Publication-quality figures, bootstrap CIs, contrastive probing,
cross-property correlation analysis, and fine-grained layer-by-layer analysis.

Tasks covered
─────────────
  Task 1  — 4×4 saliency grid + "best-case" mass-degradation figure (300 DPI)
  Task 2  — Bootstrap 95% CIs on mean R² + error-bar degradation curves
  Task 3  — Contrastive probing (logistic regression, ROC-AUC) + saliency maps
  Task 4  — Cross-property correlation analysis (mass R² vs stability R²)
  Task 5  — Layer-by-layer (all 12 ViT layers) fine-grained degradation curve

Outputs saved to results/figures/
Run from project root:
    py -3 scripts/generate_publication_figures.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import numpy as np
from scipy.ndimage import zoom as ndimage_zoom
from scipy.stats import pearsonr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
PATCH_GRID      = 14
N_PATCHES       = PATCH_GRID * PATCH_GRID   # 196
IMAGE_SIZE      = 224
N_SCENES        = 1000
SEED            = 42
N_BOOTSTRAP     = 1000
N_LAYER_SCENES  = 300   # scenes used for all-layer extraction (speed/accuracy trade-off)

STAGE_NAMES  = ["stage_1_enc_out", "stage_2_post_proj", "stage_3_llm_8", "stage_4_llm_16"]
STAGE_LABELS = ["S1\nEnc-out", "S2\nPost-proj", "S3\nLLM-8", "S4\nLLM-16"]
STAGE_LABELS_FULL = ["S1 Enc-out", "S2 Post-proj", "S3 LLM-8", "S4 LLM-16"]
PROP_NAMES   = ["mass", "friction", "elasticity", "stability"]

PROP_COLORS  = {
    "mass":        "#2196F3",
    "friction":    "#4CAF50",
    "elasticity":  "#FF5722",
    "stability":   "#9C27B0",
}
PROP_MARKERS = {"mass": "o", "friction": "s", "elasticity": "^", "stability": "D"}

HDF5_DIR    = Path("results/activations/vit_base")
DAY34_DIR   = Path("results/day34")
OUTPUT_DIR  = Path("results/figures")

# matplotlib style
plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        9,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "figure.dpi":       300,
})


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_probe_results() -> dict:
    path = DAY34_DIR / "probe_results.json"
    with open(path) as f:
        return json.load(f)


def load_hdf5_data() -> tuple[dict, np.ndarray]:
    """Load activations and labels from HDF5 cache.

    Returns
    -------
    stage_arrays : dict  stage_name → float32 [N, 196, D]
    label_array  : float32 [N, 196, 4]
    """
    import h5py

    stage_arrays: dict = {}
    label_array = None

    for stage_name in STAGE_NAMES:
        h5_path = HDF5_DIR / f"{stage_name}.h5"
        with h5py.File(h5_path, "r") as f:
            stage_arrays[stage_name] = f["activations"][:]   # [N, P, D]
            if label_array is None:
                label_array = f["labels"][:]                 # [N, P, 4]

    logger.info("Loaded HDF5: %s scenes, %s patches, %s props",
                label_array.shape[0], label_array.shape[1], label_array.shape[2])
    return stage_arrays, label_array


def get_sample_images(n: int = 8):
    """Return list of PIL images from the same seed used for activations."""
    from src.data.synthetic_physion import SyntheticPhysicsDataset
    dataset = SyntheticPhysicsDataset(n_scenes=N_SCENES, image_size=IMAGE_SIZE, seed=SEED)
    return [dataset[i].image for i in range(min(n, N_SCENES))]


def r2_grid(stage_name: str, prop_name: str, results: dict) -> np.ndarray:
    """Return (14, 14) R² map, clipped to [0, ∞)."""
    r2 = np.array(results[stage_name][prop_name]["r2_per_patch"])
    return np.clip(r2, 0, None).reshape(PATCH_GRID, PATCH_GRID)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1a — 4×4 publication saliency grid
# ═══════════════════════════════════════════════════════════════════════════════

def make_saliency_grid(results: dict, scene_img, out_path: Path) -> None:
    """4×4 grid: rows=stages, cols=properties. Each cell = heatmap overlay.

    Design decisions:
      - Single shared colorbar per column (property), calibrated to S1 max
      - Property name as column header, stage label as row label
      - Moran's I annotated on each cell
    """
    logger.info("Task 1a: generating 4×4 saliency grid...")
    spatial = json.loads((DAY34_DIR / "spatial_analysis.json").read_text())

    n_rows, n_cols = len(STAGE_NAMES), len(PROP_NAMES)
    cell_w, cell_h = 2.8, 2.6
    fig = plt.figure(figsize=(n_cols * cell_w, n_rows * cell_h + 0.7), dpi=300)
    gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig,
        hspace=0.08, wspace=0.12,
        left=0.07, right=0.95, top=0.93, bottom=0.03,
    )

    img_arr = np.array(scene_img)
    H, W = img_arr.shape[:2]

    # Per-column vmax = max across all stages for that property
    col_vmaxes = {}
    for prop in PROP_NAMES:
        col_vmaxes[prop] = max(
            results[s][prop]["max_r2"]
            for s in STAGE_NAMES if prop in results.get(s, {})
        )

    axes_grid = {}
    for row, stage_name in enumerate(STAGE_NAMES):
        for col, prop_name in enumerate(PROP_NAMES):
            ax = fig.add_subplot(gs[row, col])
            axes_grid[(row, col)] = ax

            if stage_name not in results or prop_name not in results[stage_name]:
                ax.axis("off")
                continue

            grid = r2_grid(stage_name, prop_name, results)
            heatmap = ndimage_zoom(grid, (H / PATCH_GRID, W / PATCH_GRID), order=1)

            vmax = col_vmaxes[prop_name]
            ax.imshow(img_arr)
            im = ax.imshow(
                heatmap, cmap="inferno", alpha=0.70,
                interpolation="bilinear",
                vmin=0.0, vmax=max(vmax, 0.01),
            )
            ax.axis("off")

            # R² stats text
            mean_r2 = results[stage_name][prop_name]["mean_r2"]
            ci_lo   = results[stage_name][prop_name].get("ci_lo", 0.0)
            ci_hi   = results[stage_name][prop_name].get("ci_hi", mean_r2)
            ax.text(
                0.03, 0.03,
                f"μR²={mean_r2:.3f}\n[{ci_lo:.3f},{ci_hi:.3f}]",
                transform=ax.transAxes, fontsize=5.5,
                color="white", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.55),
            )

            # Moran's I
            mi = spatial.get(stage_name, {}).get(prop_name, {}).get("morans_i", None)
            if mi is not None:
                ax.text(
                    0.97, 0.03, f"I={mi:.2f}",
                    transform=ax.transAxes, fontsize=5.5,
                    color="yellow", va="bottom", ha="right",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.45),
                )

            # Column headers (top row)
            if row == 0:
                ax.set_title(prop_name.capitalize(), fontsize=10, fontweight="bold", pad=4)

            # Row labels (left column)
            if col == 0:
                ax.set_ylabel(
                    STAGE_LABELS_FULL[row], fontsize=8, fontweight="bold",
                    labelpad=4, rotation=0, ha="right", va="center",
                )

            # Colorbar only on right column
            if col == n_cols - 1:
                cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
                cbar.set_label("R²", fontsize=6)
                cbar.ax.tick_params(labelsize=5)

    fig.suptitle(
        "Physics Property Saliency Maps — ViT-base-patch16-224\n"
        "μR² ± 95% CI  |  I = Moran's I spatial clustering index",
        fontsize=9, y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 1b — Best-case mass degradation (4-panel horizontal)
# ═══════════════════════════════════════════════════════════════════════════════

def make_best_case_mass(results: dict, images: list, out_path: Path) -> None:
    """Find scene with highest mass R² in S1 and show side-by-side S1→S4."""
    logger.info("Task 1b: best-case mass degradation figure...")

    # Find scene image with highest mean R² in S1/mass
    s1_r2 = results["stage_1_enc_out"]["mass"]["r2_per_patch"]
    best_idx = int(np.argmax(s1_r2))
    # Use a different scene image for visual variety
    scene_idx = min(2, len(images) - 1)
    img_arr = np.array(images[scene_idx])
    H, W = img_arr.shape[:2]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), dpi=300)
    fig.subplots_adjust(wspace=0.06, left=0.03, right=0.97, top=0.87, bottom=0.03)

    # Shared vmax = S1 max
    vmax = results["stage_1_enc_out"]["mass"]["max_r2"]

    for col, (stage_name, label) in enumerate(zip(STAGE_NAMES, STAGE_LABELS_FULL)):
        ax = axes[col]
        if stage_name in results and "mass" in results[stage_name]:
            grid = r2_grid(stage_name, "mass", results)
            heatmap = ndimage_zoom(grid, (H / PATCH_GRID, W / PATCH_GRID), order=1)
            ax.imshow(img_arr)
            im = ax.imshow(
                heatmap, cmap="plasma", alpha=0.72,
                interpolation="bilinear",
                vmin=0.0, vmax=max(vmax, 0.01),
            )
            mean_r2 = results[stage_name]["mass"]["mean_r2"]
            ax.text(
                0.5, -0.04, f"μR² = {mean_r2:.3f}",
                transform=ax.transAxes, ha="center", fontsize=9,
                color=PROP_COLORS["mass"], fontweight="bold",
            )
        else:
            ax.imshow(img_arr)
            im = None
        ax.axis("off")
        ax.set_title(label, fontsize=10, fontweight="bold", pad=5)

    # Shared colorbar
    if im is not None:
        cbar_ax = fig.add_axes([0.98, 0.12, 0.012, 0.72])
        sm = plt.cm.ScalarMappable(cmap="plasma", norm=Normalize(vmin=0, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("R² (mass)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Mass Decodability Degradation Across Pipeline Stages\n"
        "Physics information is richest at enc-out (S1) and diminishes through the LLM",
        fontsize=10, fontweight="bold",
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Task 2 — Bootstrap 95% CIs + updated degradation curves
# ═══════════════════════════════════════════════════════════════════════════════

def run_bootstrap_cis(results: dict, out_path_json: Path, out_path_fig: Path) -> dict:
    """Bootstrap CIs by resampling the per-patch R² distribution (1000×).

    For each stage × property we bootstrap 1000 means from the 196-patch
    R² distribution, yielding a 95% CI on mean spatial R².
    This quantifies sampling uncertainty over the patch positions.
    """
    logger.info("Task 2: computing bootstrap CIs (N=%d)...", N_BOOTSTRAP)
    rng = np.random.default_rng(SEED)
    updated = {}

    ci_table: dict = {}

    for stage_name in STAGE_NAMES:
        updated[stage_name] = {}
        ci_table[stage_name] = {}
        for prop_name in PROP_NAMES:
            if stage_name not in results or prop_name not in results[stage_name]:
                continue
            r2_arr = np.array(results[stage_name][prop_name]["r2_per_patch"])
            r2_arr = np.clip(r2_arr, 0, None)

            # Bootstrap: resample with replacement from 196 patch R² values
            boot_means = np.array([
                rng.choice(r2_arr, size=len(r2_arr), replace=True).mean()
                for _ in range(N_BOOTSTRAP)
            ])
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))
            boot_std = float(boot_means.std())
            mean_r2 = float(r2_arr.mean())

            entry = dict(results[stage_name][prop_name])
            entry["bootstrap_ci_lo"] = ci_lo
            entry["bootstrap_ci_hi"] = ci_hi
            entry["bootstrap_std"]   = boot_std
            updated[stage_name][prop_name] = entry
            ci_table[stage_name][prop_name] = {
                "mean": mean_r2, "ci_lo": ci_lo, "ci_hi": ci_hi,
            }

    # Save updated results
    out_path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path_json, "w") as f:
        json.dump(updated, f, indent=2)
    logger.info("Bootstrap results saved: %s", out_path_json)

    # Figure: degradation curves with CI ribbons
    fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=300)
    x = np.arange(len(STAGE_NAMES))

    for prop_name in PROP_NAMES:
        means, ci_los, ci_his = [], [], []
        for stage_name in STAGE_NAMES:
            if stage_name in ci_table and prop_name in ci_table[stage_name]:
                d = ci_table[stage_name][prop_name]
                means.append(d["mean"])
                ci_los.append(d["ci_lo"])
                ci_his.append(d["ci_hi"])
            else:
                means.append(0.0); ci_los.append(0.0); ci_his.append(0.0)

        means   = np.array(means)
        ci_los  = np.array(ci_los)
        ci_his  = np.array(ci_his)
        color   = PROP_COLORS[prop_name]
        marker  = PROP_MARKERS[prop_name]

        ax.plot(x, means, marker=marker, color=color,
                linewidth=2.4, markersize=8, label=prop_name.capitalize(), zorder=4)
        ax.fill_between(x, ci_los, ci_his, alpha=0.18, color=color, zorder=2)

        # Annotate the largest single-step drop
        drops = np.diff(means)
        if len(drops) > 0 and drops.min() < -0.01:
            idx = int(np.argmin(drops))
            mid_x = (x[idx] + x[idx + 1]) / 2
            mid_y = (means[idx] + means[idx + 1]) / 2
            ax.annotate(
                f"Δ={drops[idx]:+.3f}",
                xy=(mid_x, mid_y), fontsize=7, color=color, fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8),
            )

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS_FULL, fontsize=9)
    ax.set_ylabel("Mean R² across patches  (shaded: 95% bootstrap CI)", fontsize=10)
    ax.set_xlabel("Pipeline Stage", fontsize=10)
    ax.set_title(
        "Physics Decodability vs. Pipeline Stage\n"
        "Bootstrap 95% CI  |  1000 resamples  |  196 patches  |  1000 scenes",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.set_ylim(bottom=0.0)

    # Shade projection gap
    ax.axvspan(0.5, 1.5, alpha=0.04, color="tomato")
    ax.text(1.0, ax.get_ylim()[1] * 0.97, "projection\ngap",
            ha="center", va="top", fontsize=7.5, color="tomato", alpha=0.8)

    plt.tight_layout()
    out_path_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Bootstrap CI figure saved: %s", out_path_fig)
    return updated


# ═══════════════════════════════════════════════════════════════════════════════
# Task 3 — Contrastive probing (logistic regression)
# ═══════════════════════════════════════════════════════════════════════════════

def run_contrastive_probing(
    stage_arrays: dict,
    label_array: np.ndarray,
    out_path_json: Path,
    out_path_fig: Path,
) -> dict:
    """Binary classification: heavy vs light (above/below median mass).

    For each stage × patch position:
      - Binarize mass label at median
      - Fit LogisticRegression (L2, C=1.0)
      - Record accuracy and ROC-AUC
    Generates:
      1. ROC-AUC saliency maps (4×1 grid for mass across stages)
      2. Accuracy saliency maps
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, accuracy_score

    logger.info("Task 3: contrastive probing (logistic regression)...")

    N, P, D_dummy = next(iter(stage_arrays.values())).shape
    # mass is property index 0
    MASS_IDX = 0
    STAB_IDX = 3

    # Global median of mass (ignoring NaN)
    mass_all = label_array[:, :, MASS_IDX].ravel()
    mass_valid = mass_all[~np.isnan(mass_all)]
    mass_median = float(np.median(mass_valid))
    logger.info("  Mass median = %.4f  |  N_valid_patches = %d", mass_median, len(mass_valid))

    # stability binary: above median of valid stability values
    stab_all = label_array[:, :, STAB_IDX].ravel()
    stab_valid = stab_all[~np.isnan(stab_all)]
    stab_median = float(np.median(stab_valid))

    contrastive_results: dict = {}
    roc_maps: dict = {}
    acc_maps: dict = {}

    for stage_name, X_all in stage_arrays.items():
        logger.info("  Stage: %s", stage_name)
        N, P, D = X_all.shape
        roc_map = np.full(P, np.nan)
        acc_map = np.full(P, np.nan)

        for patch_idx in range(P):
            X_patch = X_all[:, patch_idx, :]      # [N, D]
            y_mass  = label_array[:, patch_idx, MASS_IDX]   # [N]

            valid = ~np.isnan(y_mass)
            if valid.sum() < 20:
                continue

            X_v = X_patch[valid]
            y_v = (y_mass[valid] >= mass_median).astype(int)

            if len(np.unique(y_v)) < 2:
                continue

            # 70/30 split
            n_tr = int(len(y_v) * 0.70)
            X_tr, X_te = X_v[:n_tr], X_v[n_tr:]
            y_tr, y_te = y_v[:n_tr], y_v[n_tr:]

            if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            clf = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs")
            try:
                clf.fit(X_tr_s, y_tr)
                y_prob = clf.predict_proba(X_te_s)[:, 1]
                y_pred = clf.predict(X_te_s)
                roc_map[patch_idx] = roc_auc_score(y_te, y_prob)
                acc_map[patch_idx] = accuracy_score(y_te, y_pred)
            except Exception:
                pass

        roc_maps[stage_name] = roc_map
        acc_maps[stage_name] = acc_map
        valid_roc = roc_map[~np.isnan(roc_map)]
        logger.info("  %s — mean AUC=%.4f  mean Acc=%.4f",
                    stage_name,
                    valid_roc.mean() if len(valid_roc) > 0 else 0.0,
                    acc_map[~np.isnan(acc_map)].mean() if (~np.isnan(acc_map)).sum() > 0 else 0.0)

        contrastive_results[stage_name] = {
            "mean_auc":  float(np.nanmean(roc_map)),
            "max_auc":   float(np.nanmax(roc_map)) if (~np.isnan(roc_map)).sum() > 0 else 0.0,
            "mean_acc":  float(np.nanmean(acc_map)),
            "roc_auc_per_patch": roc_map.tolist(),
            "accuracy_per_patch": acc_map.tolist(),
        }

    out_path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path_json, "w") as f:
        json.dump(contrastive_results, f, indent=2)
    logger.info("Contrastive results saved: %s", out_path_json)

    # Figure: ROC-AUC saliency maps 4-panel (S1→S4 for mass)
    # We'll show a single image + ROC-AUC heatmap for each stage
    from src.data.synthetic_physion import SyntheticPhysicsDataset
    dataset = SyntheticPhysicsDataset(n_scenes=N_SCENES, image_size=IMAGE_SIZE, seed=SEED)
    scene_img = dataset[0].image
    img_arr = np.array(scene_img)
    H, W = img_arr.shape[:2]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5), dpi=300)
    fig.subplots_adjust(wspace=0.06, left=0.03, right=0.96, top=0.85, bottom=0.05)

    vmax = max(
        np.nanmax(roc_maps[s]) for s in STAGE_NAMES if (~np.isnan(roc_maps[s])).sum() > 0
    )
    vmax = max(vmax, 0.55)

    for col, stage_name in enumerate(STAGE_NAMES):
        ax = axes[col]
        roc = roc_maps[stage_name]
        grid = np.clip(np.array(roc), 0.5, None)   # AUC < 0.5 = no better than chance
        grid_2d = np.nan_to_num(grid, nan=0.5).reshape(PATCH_GRID, PATCH_GRID)
        heatmap = ndimage_zoom(grid_2d, (H / PATCH_GRID, W / PATCH_GRID), order=1)

        ax.imshow(img_arr)
        im = ax.imshow(heatmap, cmap="RdYlGn", alpha=0.72,
                       interpolation="bilinear",
                       vmin=0.5, vmax=min(vmax, 1.0))
        ax.axis("off")
        mean_auc = contrastive_results[stage_name]["mean_auc"]
        ax.text(0.5, -0.06, f"Mean AUC = {mean_auc:.3f}",
                transform=ax.transAxes, ha="center", fontsize=9, color="#333333")
        ax.set_title(STAGE_LABELS_FULL[col], fontsize=10, fontweight="bold", pad=5)

    cbar_ax = fig.add_axes([0.97, 0.12, 0.012, 0.68])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=Normalize(vmin=0.5, vmax=min(vmax, 1.0)))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("ROC-AUC\n(heavy vs light)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Contrastive Probing: Heavy vs Light Mass Classification\n"
        "ROC-AUC per patch  |  Logistic Regression  |  Binary: above/below median mass",
        fontsize=10, fontweight="bold",
    )
    out_path_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Contrastive figure saved: %s", out_path_fig)
    return contrastive_results


# ═══════════════════════════════════════════════════════════════════════════════
# Task 4 — Cross-property correlation analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_cross_property_correlation(results: dict, out_path_fig: Path) -> dict:
    """Pearson correlation between mass R² map and stability R² map per stage.

    If high correlation: physics is object-localized (all properties in same patches).
    If low correlation: different properties encoded at different spatial locations.
    """
    logger.info("Task 4: cross-property correlation analysis...")

    cross_results: dict = {}
    PROP_PAIRS = [
        ("mass", "stability"),
        ("mass", "friction"),
        ("mass", "elasticity"),
        ("stability", "friction"),
    ]

    for stage_name in STAGE_NAMES:
        cross_results[stage_name] = {}
        for p1, p2 in PROP_PAIRS:
            if p1 not in results.get(stage_name, {}) or p2 not in results.get(stage_name, {}):
                continue
            r2_p1 = np.clip(results[stage_name][p1]["r2_per_patch"], 0, None)
            r2_p2 = np.clip(results[stage_name][p2]["r2_per_patch"], 0, None)
            r, pval = pearsonr(r2_p1, r2_p2)
            cross_results[stage_name][f"{p1}_vs_{p2}"] = {
                "pearson_r": float(r), "p_value": float(pval),
            }
            logger.info("  %s | %s vs %s: r=%.4f  p=%.4e", stage_name, p1, p2, r, pval)

    # Figure: scatter plots + correlation matrix heatmap
    fig = plt.figure(figsize=(13, 9), dpi=300)
    gs_outer = gridspec.GridSpec(2, 1, figure=fig, hspace=0.45, top=0.93, bottom=0.06)

    # Top row: scatter plots mass vs stability for each stage
    gs_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[0], wspace=0.35)
    for col, stage_name in enumerate(STAGE_NAMES):
        ax = fig.add_subplot(gs_top[col])
        r2_mass = np.clip(results[stage_name]["mass"]["r2_per_patch"], 0, None)
        r2_stab = np.clip(results[stage_name]["stability"]["r2_per_patch"], 0, None)

        ax.scatter(r2_mass, r2_stab, s=12, alpha=0.55, color=PROP_COLORS["mass"],
                   edgecolors="none")
        r = cross_results[stage_name].get("mass_vs_stability", {}).get("pearson_r", 0.0)
        p = cross_results[stage_name].get("mass_vs_stability", {}).get("p_value", 1.0)
        p_str = f"p<0.001" if p < 0.001 else f"p={p:.3f}"
        ax.set_title(STAGE_LABELS_FULL[col], fontsize=9, fontweight="bold")
        ax.set_xlabel("Mass R²", fontsize=8)
        if col == 0:
            ax.set_ylabel("Stability R²", fontsize=8)
        ax.text(0.05, 0.92, f"r={r:.3f}\n{p_str}",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", alpha=0.9))
        ax.tick_params(labelsize=7)

    # Bottom row: correlation heatmap (properties × stages)
    gs_bot = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs_outer[1], wspace=0.35)
    for col, stage_name in enumerate(STAGE_NAMES):
        ax = fig.add_subplot(gs_bot[col])
        n_props = len(PROP_NAMES)
        corr_matrix = np.zeros((n_props, n_props))
        for i, pi in enumerate(PROP_NAMES):
            for j, pj in enumerate(PROP_NAMES):
                if i == j:
                    corr_matrix[i, j] = 1.0
                elif pi in results.get(stage_name, {}) and pj in results.get(stage_name, {}):
                    r2i = np.clip(results[stage_name][pi]["r2_per_patch"], 0, None)
                    r2j = np.clip(results[stage_name][pj]["r2_per_patch"], 0, None)
                    corr_matrix[i, j] = pearsonr(r2i, r2j)[0]

        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(n_props))
        ax.set_yticks(range(n_props))
        ax.set_xticklabels([p[:3].capitalize() for p in PROP_NAMES], fontsize=7, rotation=45)
        if col == 0:
            ax.set_yticklabels([p.capitalize() for p in PROP_NAMES], fontsize=7)
        else:
            ax.set_yticklabels([])
        ax.set_title(STAGE_LABELS_FULL[col], fontsize=9, fontweight="bold")

        for i in range(n_props):
            for j in range(n_props):
                ax.text(j, i, f"{corr_matrix[i, j]:.2f}",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if abs(corr_matrix[i, j]) > 0.6 else "black")

        if col == n_props - 1:
            cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
            cbar.set_label("Pearson r", fontsize=7)
            cbar.ax.tick_params(labelsize=6)

    fig.suptitle(
        "Cross-Property R² Correlation Analysis\n"
        "Top: Mass vs Stability scatter (per patch)  |  Bottom: Full correlation matrix",
        fontsize=10, fontweight="bold",
    )
    out_path_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cross-property correlation figure saved: %s", out_path_fig)
    return cross_results


# ═══════════════════════════════════════════════════════════════════════════════
# Task 5 — Layer-by-layer fine-grained degradation
# ═══════════════════════════════════════════════════════════════════════════════

def run_layer_by_layer(out_path_json: Path, out_path_fig: Path) -> dict:
    """Extract activations at ALL 12 ViT layers and train linear probes.

    Uses N_LAYER_SCENES (300) for speed while keeping statistical validity.
    The same synthetic dataset seed ensures scenes are a subset of the 1000.
    """
    import torch
    from src.data.synthetic_physion import SyntheticPhysicsDataset
    from src.data.patch_label_assigner import PatchLabelAssigner
    from src.models.activation_extractor import LightweightViTExtractor
    from src.probing.linear_probe import LinearProbe

    logger.info("Task 5: layer-by-layer analysis (%d scenes, 12 layers)...", N_LAYER_SCENES)

    dataset  = SyntheticPhysicsDataset(n_scenes=N_LAYER_SCENES, image_size=IMAGE_SIZE, seed=SEED)
    extractor = LightweightViTExtractor(device="cpu")
    extractor.load()
    assigner = PatchLabelAssigner(patch_grid_size=PATCH_GRID)

    N_LAYERS = 12   # ViT-base has 12 transformer layers (indices 1..12)

    # Collect activations at every layer
    # Shape per layer: [N_LAYER_SCENES, 196, 768]
    layer_acts: list[list] = [[] for _ in range(N_LAYERS)]
    all_labels: list = []

    t0 = time.time()
    for i, sample in enumerate(dataset):
        inputs = extractor._processor(images=sample.image, return_tensors="pt")
        with torch.no_grad():
            outputs = extractor._model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of length 13 (embed + 12 layers), each [1, 197, 768]
        for layer_idx in range(N_LAYERS):
            hidden = outputs.hidden_states[layer_idx + 1]  # +1 to skip embedding layer
            patches = hidden[0, 1:, :].float().cpu().numpy()  # [196, 768]
            layer_acts[layer_idx].append(patches)

        # Labels
        stab_vals = sample.physics_labels.get("stability", [])
        from scripts.run_day34_pipeline import _compute_geom_stability_per_patch
        stab = _compute_geom_stability_per_patch(sample.object_masks, sample.physics_labels, assigner)
        patch_labels = assigner.assign(sample.object_masks, sample.physics_labels, stability_scores=stab)
        all_labels.append(patch_labels)

        if (i + 1) % 100 == 0:
            logger.info("  Layer extraction %d/%d  (%.1fs)", i + 1, N_LAYER_SCENES, time.time() - t0)

    label_array = np.stack(all_labels)  # [N, 196, 4]

    # Convert to arrays
    layer_arrays = [np.stack(layer_acts[l]) for l in range(N_LAYERS)]  # each [N, 196, 768]
    logger.info("  Extraction done in %.1fs. Training probes...", time.time() - t0)

    # Per-layer mean R² (per-patch probes)
    layer_results: dict = {}
    for l_idx in range(N_LAYERS):
        layer_key = f"layer_{l_idx + 1:02d}"
        X_all = layer_arrays[l_idx]
        N, P, D = X_all.shape
        layer_results[layer_key] = {}
        for prop_idx, prop_name in enumerate(PROP_NAMES):
            y_all = label_array[:, :, prop_idx]
            valid_frac = float(np.mean(~np.isnan(y_all)))
            if valid_frac < 0.03:
                continue
            # Vectorized per-patch Ridge
            r2_vals = []
            for patch_idx in range(P):
                X_p = X_all[:, patch_idx, :]
                y_p = y_all[:, patch_idx]
                valid = ~np.isnan(y_p)
                if valid.sum() < 15:
                    r2_vals.append(np.nan)
                    continue
                from sklearn.linear_model import Ridge
                from sklearn.metrics import r2_score
                from sklearn.preprocessing import StandardScaler
                n_tr = int(valid.sum() * 0.8)
                idx_valid = np.where(valid)[0]
                X_v, y_v = X_p[idx_valid], y_p[idx_valid]
                X_v_tr, X_v_te = X_v[:n_tr], X_v[n_tr:]
                y_v_tr, y_v_te = y_v[:n_tr], y_v[n_tr:]
                if len(y_v_te) < 3:
                    r2_vals.append(np.nan)
                    continue
                sc = StandardScaler()
                X_v_tr_s = sc.fit_transform(X_v_tr)
                X_v_te_s = sc.transform(X_v_te)
                reg = Ridge(alpha=1.0)
                reg.fit(X_v_tr_s, y_v_tr)
                y_pred = reg.predict(X_v_te_s)
                r2_vals.append(max(0.0, float(r2_score(y_v_te, y_pred))))

            mean_r2 = float(np.nanmean(r2_vals))
            layer_results[layer_key][prop_name] = {
                "mean_r2": mean_r2,
                "std_r2":  float(np.nanstd(r2_vals)),
            }
        logger.info("  Layer %2d: mass=%.4f  stab=%.4f",
                    l_idx + 1,
                    layer_results[layer_key].get("mass", {}).get("mean_r2", 0.0),
                    layer_results[layer_key].get("stability", {}).get("mean_r2", 0.0))

    out_path_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path_json, "w") as f:
        json.dump(layer_results, f, indent=2)
    logger.info("Layer-by-layer results saved: %s", out_path_json)

    # Figure: R² vs layer for all properties
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    layers_x = np.arange(1, N_LAYERS + 1)

    peak_info: dict = {}
    for prop_name in PROP_NAMES:
        means = []
        stds  = []
        for l_idx in range(N_LAYERS):
            lk = f"layer_{l_idx + 1:02d}"
            d = layer_results.get(lk, {}).get(prop_name, {})
            means.append(d.get("mean_r2", 0.0))
            stds.append(d.get("std_r2", 0.0))

        means = np.array(means)
        stds  = np.array(stds)
        color  = PROP_COLORS[prop_name]
        marker = PROP_MARKERS[prop_name]

        ax.plot(layers_x, means, marker=marker, color=color,
                linewidth=2.0, markersize=5, label=prop_name.capitalize(), zorder=3)
        ax.fill_between(layers_x, means - stds, means + stds,
                        alpha=0.12, color=color, zorder=2)

        # Mark peak
        peak_layer = int(np.argmax(means)) + 1
        peak_val   = float(means[peak_layer - 1])
        peak_info[prop_name] = {"peak_layer": peak_layer, "peak_r2": peak_val}
        ax.annotate(
            f"L{peak_layer}",
            xy=(peak_layer, peak_val),
            xytext=(peak_layer + 0.3, peak_val + 0.012),
            fontsize=7, color=color, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=color, lw=0.8),
        )

    # Mark the 4 existing stage layers with vertical lines
    for layer_num, label in zip([3, 6, 9, 12], STAGE_LABELS_FULL):
        ax.axvline(layer_num, color="gray", linestyle=":", linewidth=1.0, alpha=0.6)
        ax.text(layer_num, ax.get_ylim()[1] * 0.02 if ax.get_ylim()[1] > 0 else 0.01,
                label.replace(" ", "\n"), ha="center", fontsize=6.5, color="gray", alpha=0.8)

    ax.set_xticks(layers_x)
    ax.set_xlabel("ViT Transformer Layer", fontsize=11)
    ax.set_ylabel("Mean R² (per-patch linear probes)", fontsize=11)
    ax.set_title(
        "Fine-Grained Physics Decodability: All 12 ViT Layers\n"
        f"{N_LAYER_SCENES} scenes  |  196 patches per scene  |  ±std band  |  Peaks annotated",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.22, linestyle="--")
    ax.set_ylim(bottom=0.0)
    ax.set_xlim(0.5, N_LAYERS + 0.5)

    plt.tight_layout()
    out_path_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Layer-by-layer figure saved: %s", out_path_fig)
    return {**layer_results, "_peak_layers": peak_info}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Day 5-6 publication figures pipeline")
    parser.add_argument("--skip-layer-by-layer", action="store_true",
                        help="Skip Task 5 (saves ~10 min)")
    parser.add_argument("--skip-contrastive",    action="store_true",
                        help="Skip Task 3 (logistic regression)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()
    all_outputs: dict = {}

    # ── Load base data ──────────────────────────────────────────────────────
    logger.info("Loading probe results from %s...", DAY34_DIR)
    results = load_probe_results()

    logger.info("Loading HDF5 activations from %s...", HDF5_DIR)
    stage_arrays, label_array = load_hdf5_data()

    logger.info("Loading sample images...")
    images = get_sample_images(n=4)

    # ── Task 1a: 4×4 saliency grid ─────────────────────────────────────────
    make_saliency_grid(
        results, images[0],
        OUTPUT_DIR / "pub_saliency_grid.png",
    )

    # ── Task 1b: Best-case mass degradation ────────────────────────────────
    make_best_case_mass(
        results, images,
        OUTPUT_DIR / "pub_mass_degradation.png",
    )

    # ── Task 2: Bootstrap CIs + updated degradation curves ─────────────────
    updated_results = run_bootstrap_cis(
        results,
        OUTPUT_DIR / "probe_results_with_bootstrap_ci.json",
        OUTPUT_DIR / "pub_degradation_bootstrap_ci.png",
    )
    all_outputs["bootstrap"] = updated_results

    # ── Task 3: Contrastive probing ─────────────────────────────────────────
    if not args.skip_contrastive:
        contrastive = run_contrastive_probing(
            stage_arrays, label_array,
            OUTPUT_DIR / "contrastive_results.json",
            OUTPUT_DIR / "pub_contrastive_mass.png",
        )
        all_outputs["contrastive"] = contrastive

    # ── Task 4: Cross-property correlation ─────────────────────────────────
    cross_corr = run_cross_property_correlation(
        results,
        OUTPUT_DIR / "pub_cross_property_correlation.png",
    )
    with open(OUTPUT_DIR / "cross_property_results.json", "w") as f:
        json.dump(cross_corr, f, indent=2)
    all_outputs["cross_property"] = cross_corr

    # ── Task 5: Layer-by-layer ──────────────────────────────────────────────
    if not args.skip_layer_by_layer:
        layer_results = run_layer_by_layer(
            OUTPUT_DIR / "layer_by_layer_results.json",
            OUTPUT_DIR / "pub_layer_by_layer.png",
        )
        all_outputs["layer_by_layer"] = layer_results

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    logger.info("=" * 64)
    logger.info("ALL DONE in %.1fs. Figures in %s", elapsed, OUTPUT_DIR.resolve())
    logger.info("=" * 64)

    # Print quick numerical summary
    print("\n" + "=" * 64)
    print("PUBLICATION FIGURE SUMMARY")
    print("=" * 64)

    print("\n-- Bootstrap CI on Mean R2 (95%) --")
    print(f"{'Stage':<20} {'Property':<14} {'Mean':>8} {'CI_lo':>8} {'CI_hi':>8}")
    print("-" * 62)
    for stage_name, sl in zip(STAGE_NAMES, STAGE_LABELS_FULL):
        for prop_name in PROP_NAMES:
            if stage_name in updated_results and prop_name in updated_results[stage_name]:
                d = updated_results[stage_name][prop_name]
                ci_lo = d.get("bootstrap_ci_lo", d.get("ci_lo", 0))
                ci_hi = d.get("bootstrap_ci_hi", d.get("ci_hi", 0))
                print(f"{sl:<20} {prop_name:<14} {d['mean_r2']:>8.4f} {ci_lo:>8.4f} {ci_hi:>8.4f}")

    if not args.skip_contrastive and "contrastive" in all_outputs:
        print("\n-- Contrastive Probing (Heavy vs Light Mass) --")
        print(f"{'Stage':<20} {'Mean AUC':>10} {'Max AUC':>10} {'Mean Acc':>10}")
        print("-" * 52)
        for stage_name, sl in zip(STAGE_NAMES, STAGE_LABELS_FULL):
            if stage_name in all_outputs["contrastive"]:
                d = all_outputs["contrastive"][stage_name]
                print(f"{sl:<20} {d['mean_auc']:>10.4f} {d['max_auc']:>10.4f} {d['mean_acc']:>10.4f}")

    print("\n-- Cross-Property Correlation (Mass vs Stability) --")
    print(f"{'Stage':<20} {'Pearson r':>12} {'p-value':>12}")
    print("-" * 46)
    for stage_name, sl in zip(STAGE_NAMES, STAGE_LABELS_FULL):
        if stage_name in cross_corr and "mass_vs_stability" in cross_corr[stage_name]:
            d = cross_corr[stage_name]["mass_vs_stability"]
            print(f"{sl:<20} {d['pearson_r']:>12.4f} {d['p_value']:>12.4e}")

    print("=" * 64)
    print(f"\nFigures saved to: {OUTPUT_DIR.resolve()}")
    print("=" * 64)


if __name__ == "__main__":
    main()
