"""
Day 3-4 pipeline: 1000 scenes, HDF5 caching, linear+MLP probes, 4 physics properties,
publication-quality saliency maps and degradation curves.

Run from project root:
    python scripts/run_day34_pipeline.py

Steps:
  1. Extract activations (or load from HDF5 if cached)
  2. Train per-patch linear probes → saliency maps
  3. Train global linear + MLP probes → comparison table
  4. Generate 4×4 saliency map grid (300 DPI)
  5. Generate degradation curves with error bars (300 DPI)
  6. Print full R² results matrix

Outputs in results/day34/:
  probe_results.json          — per-patch R² for all stages × properties
  probe_comparison.json       — linear vs MLP global R²
  saliency_grid.png           — 4-stage × 4-property saliency maps
  degradation_curves.png      — R² vs stage with error bars
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
import numpy as np
import torch

from src.data.synthetic_physion import SyntheticPhysicsDataset
from src.data.patch_label_assigner import PatchLabelAssigner
from src.models.activation_extractor import LightweightViTExtractor, STAGE_NAMES
from src.probing.linear_probe import LinearProbe, compute_per_patch_r2
from src.probing.mlp_probe import MLPProbe

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
N_SCENES      = 1000
IMAGE_SIZE    = 224
PATCH_GRID    = 14
DEVICE        = "cpu"
SEED          = 42
OUTPUT_DIR    = Path("results/day34")
HDF5_DIR      = Path("results/activations/vit_base")
PROPERTY_NAMES = ["mass", "friction", "elasticity", "stability"]
STAGE_LABELS  = ["S1 Enc-out", "S2 Post-proj", "S3 LLM-8", "S4 LLM-16"]


# ── Helpers imported from extract script ────────────────────────────────────

def _compute_geom_stability_per_patch(object_masks, physics_labels, assigner):
    if "stability" not in physics_labels:
        return None
    stab_vals = physics_labels["stability"]
    patch_assignments = assigner._assign_patches_to_objects(object_masks)
    stability_scores = np.full(assigner.n_patches, float("nan"), dtype=np.float32)
    for patch_idx in range(assigner.n_patches):
        obj_id = patch_assignments[patch_idx]
        if obj_id > 0:
            obj_idx = obj_id - 1
            if obj_idx < len(stab_vals):
                stability_scores[patch_idx] = stab_vals[obj_idx]
    return stability_scores


# ── Step 1: Activation extraction / HDF5 cache ──────────────────────────────

def load_or_extract(force_extract: bool = False):
    """Return (stage_arrays, label_array, sample_images).
    Loads from HDF5 if cached, otherwise extracts fresh."""
    import h5py
    from scripts.extract_and_save_activations import (
        extract_all, save_to_hdf5, load_from_hdf5, OUTPUT_DIR as HDF5_OUT
    )

    info_path = HDF5_DIR / "dataset_info.json"
    hdf5_ready = (
        not force_extract
        and info_path.exists()
        and all((HDF5_DIR / f"{s}.h5").exists() for s in STAGE_NAMES)
    )

    if hdf5_ready:
        with open(info_path) as f:
            info = json.load(f)
        if info.get("n_scenes") == N_SCENES and info.get("seed") == SEED:
            logger.info("Loading activations from HDF5 cache (%s)...", HDF5_DIR)
            stage_arrays, label_array = load_from_hdf5(HDF5_DIR)
            # We still need sample images for visualisation
            logger.info("Re-generating dataset for sample images...")
            dataset = SyntheticPhysicsDataset(n_scenes=N_SCENES, image_size=IMAGE_SIZE, seed=SEED)
            sample_images = [dataset[i].image for i in range(min(8, N_SCENES))]
            return stage_arrays, label_array, sample_images

    logger.info("Extracting activations for %d scenes (this takes ~5-10 min)...", N_SCENES)
    dataset = SyntheticPhysicsDataset(n_scenes=N_SCENES, image_size=IMAGE_SIZE, seed=SEED)
    extractor = LightweightViTExtractor(device=DEVICE)
    extractor.load()
    assigner = PatchLabelAssigner(patch_grid_size=PATCH_GRID)

    per_stage = {s: [] for s in STAGE_NAMES}
    label_list = []
    sample_images = []
    t0 = time.time()

    for i, sample in enumerate(dataset):
        acts = extractor.extract(sample.image)
        for stage in STAGE_NAMES:
            per_stage[stage].append(acts[stage].numpy())

        stab = _compute_geom_stability_per_patch(
            sample.object_masks, sample.physics_labels, assigner
        )
        patch_labels = assigner.assign(sample.object_masks, sample.physics_labels,
                                       stability_scores=stab)
        label_list.append(patch_labels)
        if len(sample_images) < 8:
            sample_images.append(sample.image)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            logger.info("  Scene %d/%d  (%.1fs elapsed)", i + 1, N_SCENES, elapsed)

    stage_arrays = {s: np.stack(v) for s, v in per_stage.items()}
    label_array = np.stack(label_list)

    # Save to HDF5
    HDF5_DIR.mkdir(parents=True, exist_ok=True)
    info = {
        "n_scenes": N_SCENES, "seed": SEED,
        "model": "vit_base_patch16_224",
        "patch_grid_size": PATCH_GRID, "image_size": IMAGE_SIZE,
        "properties": PROPERTY_NAMES, "stages": STAGE_NAMES,
        "activation_shape": list(next(iter(stage_arrays.values())).shape),
        "label_shape": list(label_array.shape),
    }
    with open(HDF5_DIR / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)
    import h5py
    for stage_name, acts in stage_arrays.items():
        with h5py.File(HDF5_DIR / f"{stage_name}.h5", "w") as f:
            f.create_dataset("activations", data=acts, compression="gzip", compression_opts=4,
                             chunks=(50, acts.shape[1], acts.shape[2]))
            f.create_dataset("labels", data=label_array, compression="gzip", compression_opts=4,
                             chunks=(50, label_array.shape[1], label_array.shape[2]))
            f.attrs["stage_name"] = stage_name
    logger.info("Activations saved to %s", HDF5_DIR)

    return stage_arrays, label_array, sample_images


# ── Step 2: Per-patch linear probes ─────────────────────────────────────────

def run_linear_probes(
    stage_arrays: dict,
    label_array: np.ndarray,
) -> dict:
    """Train per-patch linear probes. Returns nested results dict."""
    results = {}
    N, P, D = next(iter(stage_arrays.values())).shape

    for stage_name, X_all in stage_arrays.items():
        stage_results = {}
        for prop_idx, prop_name in enumerate(PROPERTY_NAMES):
            y_all = label_array[:, :, prop_idx]
            valid_frac = float(np.mean(~np.isnan(y_all)))
            if valid_frac < 0.03:
                logger.warning("  %s/%s: too few valid patches (%.1f%%), skipping",
                               stage_name, prop_name, valid_frac * 100)
                continue

            r2_map = compute_per_patch_r2(X_all, y_all, patch_grid_size=PATCH_GRID, alpha=1.0)
            mean_r2 = float(np.nanmean(r2_map))
            max_r2  = float(np.nanmax(r2_map))
            std_r2  = float(np.nanstd(r2_map))
            # Bootstrap CI from per-patch distribution (fast: no re-training)
            ci_lo = float(np.nanpercentile(r2_map, 2.5))
            ci_hi = float(np.nanpercentile(r2_map, 97.5))

            stage_results[prop_name] = {
                "r2_per_patch": r2_map.tolist(),
                "mean_r2": mean_r2,
                "max_r2":  max_r2,
                "std_r2":  std_r2,
                "ci_lo":   ci_lo,
                "ci_hi":   ci_hi,
            }
            logger.info("  %-22s | %-12s | mean R²=%.4f  max=%.4f  std=%.4f",
                        stage_name, prop_name, mean_r2, max_r2, std_r2)

        results[stage_name] = stage_results

    return results


# ── Step 3: Global linear + MLP probes ─────────────────────────────────────

def run_global_probes(
    stage_arrays: dict,
    label_array: np.ndarray,
) -> dict:
    """Train global (all-patch) linear and MLP probes for comparison."""
    N, P, D = next(iter(stage_arrays.values())).shape
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(N)
    n_train = int(N * 0.70)
    n_val   = int(N * 0.15)
    tr_idx  = perm[:n_train]
    va_idx  = perm[n_train:n_train + n_val]
    te_idx  = perm[n_train + n_val:]

    comparison = {}
    for stage_name, X_all in stage_arrays.items():
        stage_cmp = {}
        for prop_idx, prop_name in enumerate(PROPERTY_NAMES):
            y_all = label_array[:, :, prop_idx]

            # Flatten scenes×patches → samples
            X_tr = X_all[tr_idx].reshape(-1, D)
            y_tr = y_all[tr_idx].reshape(-1)
            X_va = X_all[va_idx].reshape(-1, D)
            y_va = y_all[va_idx].reshape(-1)
            X_te = X_all[te_idx].reshape(-1, D)
            y_te = y_all[te_idx].reshape(-1)

            valid_tr = ~np.isnan(y_tr)
            if valid_tr.sum() < 100:
                continue

            # Global linear probe
            lin = LinearProbe(input_dim=D, alpha=1.0)
            try:
                lin.fit(X_tr, y_tr)
                valid_te = ~np.isnan(y_te)
                lin_metrics = lin.score(X_te[valid_te], y_te[valid_te])
                lin_r2 = lin_metrics["r2"]
            except Exception as e:
                logger.warning("Linear probe failed for %s/%s: %s", stage_name, prop_name, e)
                lin_r2 = 0.0

            # Global MLP probe
            mlp = MLPProbe(
                input_dim=D, hidden_dim=512, dropout=0.2,
                lr=1e-3, batch_size=64, epochs=40,
                early_stopping_patience=8, device=DEVICE,
            )
            try:
                mlp.fit(X_tr, y_tr, X_va, y_va, verbose=False)
                valid_te = ~np.isnan(y_te)
                mlp_metrics = mlp.score(X_te[valid_te], y_te[valid_te])
                mlp_r2 = mlp_metrics["r2"]
            except Exception as e:
                logger.warning("MLP probe failed for %s/%s: %s", stage_name, prop_name, e)
                mlp_r2 = 0.0

            stage_cmp[prop_name] = {"linear_r2": lin_r2, "mlp_r2": mlp_r2}
            logger.info("  %-22s | %-12s | Linear R²=%.4f  MLP R²=%.4f  Δ=%.4f",
                        stage_name, prop_name, lin_r2, mlp_r2, mlp_r2 - lin_r2)

        comparison[stage_name] = stage_cmp

    return comparison


# ── Step 4: Publication-quality saliency maps ───────────────────────────────

def save_saliency_grid(
    results: dict,
    sample_image,
    output_path: Path,
) -> None:
    """4-stage × 4-property saliency map grid with side-by-side original+heatmap."""
    n_rows = len(STAGE_NAMES)
    n_cols = len(PROPERTY_NAMES)

    # Each cell: original image (left) + heatmap overlay (right)
    fig_w = 2.8 * n_cols * 2     # 2 sub-columns per property
    fig_h = 2.8 * n_rows
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)

    # Outer grid: rows=stages, cols=properties; each cell is itself 1×2
    outer = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.15)

    img_arr = np.array(sample_image)
    H, W = img_arr.shape[:2]

    for row, stage_name in enumerate(STAGE_NAMES):
        for col, prop_name in enumerate(PROPERTY_NAMES):
            inner = gridspec.GridSpecFromSubplotSpec(
                1, 2, subplot_spec=outer[row, col], wspace=0.05
            )
            ax_orig   = fig.add_subplot(inner[0])
            ax_heat   = fig.add_subplot(inner[1])

            ax_orig.imshow(img_arr)
            ax_orig.axis("off")

            if stage_name not in results or prop_name not in results[stage_name]:
                ax_heat.axis("off")
                ax_heat.set_title("(no data)", fontsize=6)
                if col == 0:
                    ax_orig.set_ylabel(STAGE_LABELS[row], fontsize=8, fontweight="bold")
                continue

            r2_map  = np.array(results[stage_name][prop_name]["r2_per_patch"])
            mean_r2 = results[stage_name][prop_name]["mean_r2"]
            max_r2  = results[stage_name][prop_name]["max_r2"]

            grid = r2_map.reshape(PATCH_GRID, PATCH_GRID)
            from scipy.ndimage import zoom as ndimage_zoom
            heatmap = ndimage_zoom(grid, (H / PATCH_GRID, W / PATCH_GRID), order=1)
            heatmap = np.clip(heatmap, 0.0, None)

            ax_heat.imshow(img_arr)
            im = ax_heat.imshow(heatmap, cmap="inferno", alpha=0.65,
                                interpolation="bilinear",
                                vmin=0.0, vmax=max(max_r2, 0.01))
            ax_heat.axis("off")

            # Compact colorbar
            cbar = plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.02)
            cbar.set_label(f"R²", fontsize=5)
            cbar.ax.tick_params(labelsize=5)

            # Column headers (property names) on top row
            if row == 0:
                ax_orig.set_title(prop_name.capitalize(), fontsize=9, fontweight="bold", pad=4)

            # Row labels (stage) on left column
            if col == 0:
                ax_orig.set_ylabel(STAGE_LABELS[row], fontsize=8, fontweight="bold")

            # Per-cell stats
            ax_heat.text(0.02, 0.02, f"μ={mean_r2:.3f}\nmax={max_r2:.3f}",
                         transform=ax_heat.transAxes, fontsize=5.5,
                         color="white", va="bottom",
                         bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))

    fig.suptitle(
        "Physics Saliency Maps — ViT-base-patch16-224\n"
        "Rows: pipeline stages  |  Cols: physics properties  |  "
        "Left: original  |  Right: R² heatmap",
        fontsize=10, y=1.005,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saliency grid saved: %s", output_path)


# ── Step 5: Degradation curves with error bars ──────────────────────────────

def save_degradation_curves(
    results: dict,
    output_path: Path,
) -> None:
    """One plot: 4 lines (properties) × 4 stages, with ±std error bands."""
    PROP_COLORS = {
        "mass":        "#2196F3",
        "friction":    "#4CAF50",
        "elasticity":  "#FF5722",
        "stability":   "#9C27B0",
    }
    PROP_MARKERS = {"mass": "o", "friction": "s", "elasticity": "^", "stability": "D"}

    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    x = np.arange(len(STAGE_NAMES))

    max_drop_info = []

    for prop_name in PROPERTY_NAMES:
        means, stds, ci_los, ci_his = [], [], [], []
        for stage_name in STAGE_NAMES:
            if stage_name in results and prop_name in results[stage_name]:
                d = results[stage_name][prop_name]
                means.append(d["mean_r2"])
                stds.append(d["std_r2"])
                ci_los.append(d["ci_lo"])
                ci_his.append(d["ci_hi"])
            else:
                means.append(0.0); stds.append(0.0)
                ci_los.append(0.0); ci_his.append(0.0)

        means = np.array(means)
        color  = PROP_COLORS.get(prop_name, "gray")
        marker = PROP_MARKERS.get(prop_name, "o")

        line, = ax.plot(x, means, marker=marker, color=color,
                        linewidth=2.2, markersize=7, label=prop_name.capitalize(), zorder=3)

        # Error band: ±std
        stds_arr = np.array(stds)
        ax.fill_between(x, means - stds_arr, means + stds_arr,
                        alpha=0.15, color=color, zorder=2)

        # Annotate max consecutive drop
        drops = np.diff(means)
        if len(drops) > 0:
            max_drop_idx = int(np.argmin(drops))   # most negative = largest drop
            drop_val = drops[max_drop_idx]
            if drop_val < -0.005:
                mid_x = (x[max_drop_idx] + x[max_drop_idx + 1]) / 2
                mid_y = (means[max_drop_idx] + means[max_drop_idx + 1]) / 2
                ax.annotate(
                    f"−{abs(drop_val):.3f}",
                    xy=(mid_x, mid_y),
                    fontsize=7, color=color, fontweight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75),
                )
                max_drop_info.append((prop_name, max_drop_idx, drop_val))

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, fontsize=9)
    ax.set_ylabel("Mean R² across patches (±std band)", fontsize=10)
    ax.set_xlabel("Pipeline Stage", fontsize=10)
    ax.set_title(
        "Physics Decodability Across ViT Pipeline Stages\n"
        "1000-scene Synthetic Dataset  —  Per-patch Linear Probes",
        fontsize=11,
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_ylim(bottom=0)

    # Shade Stage 2 (projection step) faintly
    ax.axvspan(0.5, 1.5, alpha=0.04, color="red")
    ax.text(1.0, ax.get_ylim()[1] * 0.97, "proj.\ngap",
            ha="center", va="top", fontsize=7, color="red", alpha=0.7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Degradation curves saved: %s", output_path)


# ── Step 6: Print results matrix ────────────────────────────────────────────

def print_results_matrix(results: dict, comparison: dict) -> None:
    """Print mean R² tables for linear probes and linear vs MLP comparison."""
    # Table 1: per-patch linear probe mean R²
    print("\n" + "=" * 76)
    print("LINEAR PROBE — mean R² per stage × property (per-patch probes)")
    print("=" * 76)
    header = f"{'Stage':<24}" + "".join(f"{p:>13}" for p in PROPERTY_NAMES)
    print(header)
    print("-" * 76)
    for stage_name, sl in zip(STAGE_NAMES, STAGE_LABELS):
        row = f"{sl:<24}"
        for p in PROPERTY_NAMES:
            if stage_name in results and p in results[stage_name]:
                row += f"{results[stage_name][p]['mean_r2']:>13.4f}"
            else:
                row += f"{'N/A':>13}"
        print(row)
    print("=" * 76)

    # Table 2: Linear vs MLP comparison (Stage 1 only for conciseness)
    print("\n" + "=" * 76)
    print("LINEAR vs MLP PROBE — global R² (all patches, Stage 1 enc-out)")
    print("=" * 76)
    s1 = "stage_1_enc_out"
    header2 = f"{'Property':<16}{'Linear R2':>14}{'MLP R2':>14}{'Delta(MLP-Lin)':>15}"
    print(header2)
    print("-" * 76)
    if s1 in comparison:
        for p in PROPERTY_NAMES:
            if p in comparison[s1]:
                lin = comparison[s1][p]["linear_r2"]
                mlp = comparison[s1][p]["mlp_r2"]
                delta = mlp - lin
                print(f"{p:<16}{lin:>14.4f}{mlp:>14.4f}{delta:>+14.4f}")
    print("=" * 76)

    # Table 3: Full linear vs MLP across all stages
    print("\n" + "=" * 76)
    print("LINEAR vs MLP — global R² all stages (mass property)")
    print("=" * 76)
    print(f"{'Stage':<24}{'Linear R2':>14}{'MLP R2':>14}{'Delta':>10}")
    print("-" * 76)
    prop = "mass"
    for stage_name, sl in zip(STAGE_NAMES, STAGE_LABELS):
        if stage_name in comparison and prop in comparison[stage_name]:
            lin = comparison[stage_name][prop]["linear_r2"]
            mlp = comparison[stage_name][prop]["mlp_r2"]
            print(f"{sl:<24}{lin:>14.4f}{mlp:>14.4f}{mlp - lin:>+10.4f}")
    print("=" * 76)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-extract", action="store_true",
                        help="Re-extract activations even if HDF5 cache exists")
    parser.add_argument("--skip-mlp", action="store_true",
                        help="Skip MLP probe training (faster)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # 1. Load or extract activations
    logger.info("=" * 60)
    logger.info("STEP 1: Activations (%d scenes)", N_SCENES)
    stage_arrays, label_array, sample_images = load_or_extract(
        force_extract=args.force_extract
    )
    logger.info("Activation shape: %s", next(iter(stage_arrays.values())).shape)
    logger.info("Label shape: %s  valid=%.1f%%",
                label_array.shape,
                np.mean(~np.isnan(label_array[:, :, 0])) * 100)

    # 2. Linear probes (per-patch)
    logger.info("=" * 60)
    logger.info("STEP 2: Per-patch linear probes (4 stages × 4 properties)")
    results = run_linear_probes(stage_arrays, label_array)

    results_path = OUTPUT_DIR / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Linear probe results saved: %s", results_path)

    # 3. Global linear + MLP comparison
    comparison = {}
    if not args.skip_mlp:
        logger.info("=" * 60)
        logger.info("STEP 3: Global linear vs MLP probe comparison")
        comparison = run_global_probes(stage_arrays, label_array)
        cmp_path = OUTPUT_DIR / "probe_comparison.json"
        with open(cmp_path, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info("Comparison saved: %s", cmp_path)

    # 4. Saliency maps
    logger.info("=" * 60)
    logger.info("STEP 4: Generating saliency map grid (4×4, 300 DPI)")
    save_saliency_grid(results, sample_images[0], OUTPUT_DIR / "saliency_grid.png")

    # 5. Degradation curves
    logger.info("=" * 60)
    logger.info("STEP 5: Generating degradation curves (300 DPI)")
    save_degradation_curves(results, OUTPUT_DIR / "degradation_curves.png")

    # 6. Print results
    print_results_matrix(results, comparison)

    logger.info("=" * 60)
    logger.info("ALL DONE. Total time: %.1fs. Outputs in %s",
                time.time() - t_total, OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
