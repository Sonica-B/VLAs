"""
Per-patch spatial analysis of physics decodability.

Loads probe results from results/day34/probe_results.json and performs:
  1. Identifies which patch positions carry highest R² for each property
  2. Moran's I — spatial autocorrelation: are high-R² patches clustered?
  3. Spatial precision / recall using object masks as ground truth
  4. Tests Hypothesis 1: physics encoding is spatially structured

Run from project root:
    python scripts/analyze_spatial_patterns.py

Outputs in results/day34/:
    spatial_analysis.json   — all spatial metrics
    spatial_heatmaps.png    — top-patch maps for each property
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom as ndimage_zoom

from src.data.synthetic_physion import SyntheticPhysicsDataset
from src.data.patch_label_assigner import PatchLabelAssigner
from src.models.activation_extractor import STAGE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PATCH_GRID    = 14
N_PATCHES     = PATCH_GRID * PATCH_GRID
IMAGE_SIZE    = 224
N_SCENES      = 1000
SEED          = 42
RESULTS_DIR   = Path("results/day34")
OUTPUT_DIR    = Path("results/day34")
PROPERTY_NAMES = ["mass", "friction", "elasticity", "stability"]
STAGE_LABELS  = ["S1 Enc-out", "S2 Post-proj", "S3 LLM-8", "S4 LLM-16"]


# ── Moran's I ───────────────────────────────────────────────────────────────

def morans_i(values: np.ndarray, grid_size: int = 14) -> float:
    """Compute Moran's I spatial autocorrelation for a patch grid.

    Uses queen contiguity (8-neighbourhood) weights.
    Positive I → spatial clustering; near 0 → random; negative → dispersion.

    Args:
        values: [N_patches] float array (NaN ignored via zero-weight trick).
        grid_size: Spatial grid side length.

    Returns:
        Moran's I value in roughly [-1, 1].
    """
    n = grid_size * grid_size
    vals = values.reshape(grid_size, grid_size).astype(float)

    # Replace NaN with mean to avoid propagation
    nan_mask = np.isnan(vals)
    mean_val = np.nanmean(vals)
    vals[nan_mask] = mean_val

    z = vals - mean_val  # deviations from mean

    # Build 8-neighbour weight matrix (row-normalised)
    W_sum = 0.0
    numerator = 0.0
    denom = float(np.sum(z ** 2))

    for i in range(grid_size):
        for j in range(grid_size):
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_size and 0 <= nj < grid_size:
                        numerator += z[i, j] * z[ni, nj]
                        W_sum += 1.0

    if denom == 0 or W_sum == 0:
        return 0.0
    return float((n / W_sum) * (numerator / denom))


# ── Spatial precision / recall ──────────────────────────────────────────────

def spatial_precision_recall(
    r2_map: np.ndarray,
    object_coverage: np.ndarray,
    threshold_quantile: float = 0.75,
) -> dict:
    """Precision/recall of high-R² patches vs object-covered patches.

    Precision: of patches predicted high-R², what fraction actually
               covers an object?
    Recall: of object patches, what fraction are high-R²?

    Args:
        r2_map: [N_patches] R² per patch.
        object_coverage: [N_patches] bool/float — True if patch covers an object.
        threshold_quantile: Top-Q fraction defines "high R²". Default 0.75.

    Returns:
        Dict with precision, recall, f1, threshold.
    """
    threshold = float(np.nanquantile(r2_map, threshold_quantile))
    predicted_high = r2_map >= threshold          # predicted positive
    actual_positive = object_coverage > 0         # actual positive

    tp = np.sum(predicted_high & actual_positive)
    fp = np.sum(predicted_high & ~actual_positive)
    fn = np.sum(~predicted_high & actual_positive)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "threshold": float(threshold),
        "n_high_r2": int(np.sum(predicted_high)),
        "n_object_patches": int(np.sum(actual_positive)),
        "tp": int(tp), "fp": int(fp), "fn": int(fn),
    }


# ── Compute mean object coverage map ────────────────────────────────────────

def compute_mean_object_coverage(n_scenes: int = 200, seed: int = SEED) -> np.ndarray:
    """Compute average fraction of each patch covered by objects across scenes.

    Returns:
        coverage: [N_patches] float in [0, 1].
    """
    dataset = SyntheticPhysicsDataset(n_scenes=n_scenes, image_size=IMAGE_SIZE, seed=seed)
    assigner = PatchLabelAssigner(patch_grid_size=PATCH_GRID)
    coverage_sum = np.zeros(N_PATCHES, dtype=np.float32)

    for sample in dataset:
        assignments = assigner._assign_patches_to_objects(sample.object_masks)
        coverage_sum += (assignments > 0).astype(np.float32)

    return coverage_sum / n_scenes


# ── Top-patch visualisation ──────────────────────────────────────────────────

def save_spatial_heatmaps(
    probe_results: dict,
    coverage_map: np.ndarray,
    output_path: Path,
) -> None:
    """4-row (properties) × 4-col (stages) grid of R² patch maps."""
    n_rows = len(PROPERTY_NAMES)
    n_cols = len(STAGE_NAMES)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.2 * n_cols, 3.0 * n_rows), dpi=200)

    cov_grid = coverage_map.reshape(PATCH_GRID, PATCH_GRID)

    for row, prop_name in enumerate(PROPERTY_NAMES):
        for col, (stage_name, sl) in enumerate(zip(STAGE_NAMES, STAGE_LABELS)):
            ax = axes[row, col] if n_rows > 1 else axes[col]

            if stage_name in probe_results and prop_name in probe_results[stage_name]:
                r2 = np.array(probe_results[stage_name][prop_name]["r2_per_patch"])
                grid = r2.reshape(PATCH_GRID, PATCH_GRID)
            else:
                grid = np.zeros((PATCH_GRID, PATCH_GRID))

            im = ax.imshow(grid, cmap="plasma", vmin=0,
                           vmax=max(grid.max(), 0.01), interpolation="nearest")
            # Overlay object coverage contour
            ax.contour(cov_grid, levels=[0.3], colors="white", linewidths=0.8, alpha=0.7)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=5)
            ax.set_xticks([]); ax.set_yticks([])

            mean_r2 = probe_results.get(stage_name, {}).get(prop_name, {}).get("mean_r2", 0)
            ax.set_title(f"{sl}\nμR²={mean_r2:.3f}", fontsize=7)

            if col == 0:
                ax.set_ylabel(prop_name.capitalize(), fontsize=8, fontweight="bold")

    fig.suptitle(
        "Per-patch R² Maps  (plasma=R², white contour=object coverage)\n"
        "ViT-base-patch16-224 — 1000-scene Synthetic Dataset",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Spatial heatmaps saved: %s", output_path)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    results_path = RESULTS_DIR / "probe_results.json"
    if not results_path.exists():
        logger.error(
            "probe_results.json not found. Run run_day34_pipeline.py first."
        )
        sys.exit(1)

    with open(results_path) as f:
        probe_results = json.load(f)

    logger.info("Computing mean object coverage map (200 scenes)...")
    coverage_map = compute_mean_object_coverage(n_scenes=200)
    logger.info("Coverage map: %.1f%% of patches covered on average",
                float(np.mean(coverage_map > 0.3)) * 100)

    spatial_metrics = {}

    logger.info("\n%s", "=" * 72)
    logger.info("SPATIAL ANALYSIS — Moran's I and Precision/Recall")
    logger.info("%s", "=" * 72)

    for stage_name, sl in zip(STAGE_NAMES, STAGE_LABELS):
        stage_metrics = {}
        for prop_name in PROPERTY_NAMES:
            if stage_name not in probe_results or prop_name not in probe_results[stage_name]:
                continue

            r2_map = np.array(probe_results[stage_name][prop_name]["r2_per_patch"])

            moran = morans_i(r2_map, grid_size=PATCH_GRID)
            pr = spatial_precision_recall(r2_map, coverage_map, threshold_quantile=0.75)

            # Top-5 patch positions (row, col)
            top_indices = np.argsort(r2_map)[::-1][:5]
            top_positions = [(int(i // PATCH_GRID), int(i % PATCH_GRID))
                             for i in top_indices]

            stage_metrics[prop_name] = {
                "morans_i": moran,
                "precision": pr["precision"],
                "recall":    pr["recall"],
                "f1":        pr["f1"],
                "top_patch_positions": top_positions,
                "n_high_r2_patches": pr["n_high_r2"],
                "n_object_patches":  pr["n_object_patches"],
            }

            logger.info(
                "  %-22s | %-12s | Moran's I=%+.3f  Prec=%.3f  Rec=%.3f  F1=%.3f",
                sl, prop_name, moran, pr["precision"], pr["recall"], pr["f1"],
            )

        spatial_metrics[stage_name] = stage_metrics

    # Summary: is physics spatially structured? (Hypothesis 1)
    all_morans = [
        spatial_metrics[s][p]["morans_i"]
        for s in STAGE_NAMES if s in spatial_metrics
        for p in PROPERTY_NAMES if p in spatial_metrics.get(s, {})
    ]
    mean_moran = float(np.mean(all_morans)) if all_morans else 0.0
    all_f1 = [
        spatial_metrics[s][p]["f1"]
        for s in STAGE_NAMES if s in spatial_metrics
        for p in PROPERTY_NAMES if p in spatial_metrics.get(s, {})
    ]
    mean_f1 = float(np.mean(all_f1)) if all_f1 else 0.0

    hypothesis1_supported = mean_moran > 0.1 and mean_f1 > 0.5

    print("\n" + "=" * 72)
    print("HYPOTHESIS 1: Is physics information spatially structured?")
    print("=" * 72)
    print(f"  Mean Moran's I (all stages/props): {mean_moran:+.4f}")
    print(f"  Mean spatial F1 (high-R² vs objects): {mean_f1:.4f}")
    print(f"  Verdict: {'SUPPORTED' if hypothesis1_supported else 'NOT SUPPORTED / WEAK'}")
    print(f"  Interpretation:")
    if mean_moran > 0.15:
        print("    [+] High-R2 patches cluster spatially (physics-informative regions group together)")
    elif mean_moran > 0.05:
        print("    [~] Mild spatial clustering of physics-informative patches")
    else:
        print("    [-] Physics information is spatially diffuse (distributed across patches)")
    if mean_f1 > 0.6:
        print("    [+] High-R2 patches strongly overlap with object-covered patches")
    elif mean_f1 > 0.4:
        print("    [~] Moderate overlap between high-R2 and object patches")
    else:
        print("    [-] High-R2 patches do not consistently align with object locations")
    print("=" * 72)

    spatial_metrics["_summary"] = {
        "mean_morans_i": mean_moran,
        "mean_f1":       mean_f1,
        "hypothesis1_supported": hypothesis1_supported,
    }

    out_path = OUTPUT_DIR / "spatial_analysis.json"
    with open(out_path, "w") as f:
        json.dump(spatial_metrics, f, indent=2)
    logger.info("Spatial analysis saved: %s", out_path)

    logger.info("Generating spatial heatmap visualisation...")
    save_spatial_heatmaps(probe_results, coverage_map, OUTPUT_DIR / "spatial_heatmaps.png")


if __name__ == "__main__":
    main()
