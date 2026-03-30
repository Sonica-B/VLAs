"""
End-to-end pipeline test using synthetic data and a lightweight ViT model.

Validates the full Phase 1 pipeline on CPU:
  Synthetic physics scenes
    → google/vit-base-patch16-224 activations
    → PatchLabelAssigner (physics labels per patch)
    → LinearProbe (per-patch R² scores)
    → PhysicsSaliencyMap (heatmap visualisation)

Run from the project root:
    python scripts/test_pipeline_e2e.py

Expected output:
  results/e2e_test/probe_results.json  — per-stage R² scores
  results/e2e_test/saliency_maps.png   — 4-stage × 3-property heatmap grid
  results/e2e_test/degradation_curve.png — R² vs pipeline stage line chart
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Add project root to path so src/ imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — no display required

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.synthetic_physion import SyntheticPhysicsDataset
from src.data.patch_label_assigner import PatchLabelAssigner
from src.models.activation_extractor import LightweightViTExtractor, STAGE_NAMES
from src.probing.linear_probe import LinearProbe, compute_per_patch_r2
from src.visualization.saliency_map import PhysicsSaliencyMap

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
N_SCENES = 100
IMAGE_SIZE = 224      # ViT-base-patch16-224 expects 224×224
PATCH_GRID_SIZE = 14  # 14×14 = 196 patches  (224 / 16 patch size)
DEVICE = "cpu"
OUTPUT_DIR = Path("results/e2e_test")
PROPERTY_NAMES = ["mass", "friction", "elasticity"]   # stability needs temporal data


def collect_activations_and_labels(
    dataset: SyntheticPhysicsDataset,
    extractor: LightweightViTExtractor,
    assigner: PatchLabelAssigner,
) -> tuple[dict[str, np.ndarray], np.ndarray, list]:
    """Run forward passes on all scenes and collect stacked arrays.

    Returns:
        stage_arrays:  dict[stage_name → np.ndarray [N, N_patches, D]]
        label_array:   np.ndarray [N, N_patches, 4]  (NaN for background)
        sample_images: list of first 8 PIL images (for visualisation)
    """
    per_stage: dict[str, list[np.ndarray]] = {s: [] for s in STAGE_NAMES}
    label_list: list[np.ndarray] = []
    sample_images = []

    for i, sample in enumerate(dataset):
        if i % 20 == 0:
            logger.info(f"  Extracting scene {i + 1}/{len(dataset)} ...")

        acts = extractor.extract(sample.image)

        for stage in STAGE_NAMES:
            per_stage[stage].append(acts[stage].numpy())  # [196, 768]

        patch_labels = assigner.assign(sample.object_masks, sample.physics_labels)
        label_list.append(patch_labels)  # [196, 4]

        if len(sample_images) < 8:
            sample_images.append(sample.image)

    stage_arrays = {s: np.stack(v) for s, v in per_stage.items()}  # [N, 196, D]
    label_array = np.stack(label_list)   # [N, 196, 4]

    logger.info(f"  Activation shape per stage: {next(iter(stage_arrays.values())).shape}")
    logger.info(f"  Label shape: {label_array.shape}")
    logger.info(
        f"  Valid (non-NaN) patch-label fraction: "
        f"{np.mean(~np.isnan(label_array[:, :, 0])):.1%}"
    )
    return stage_arrays, label_array, sample_images


def train_probes(
    stage_arrays: dict[str, np.ndarray],
    label_array: np.ndarray,
) -> dict[str, dict[str, dict]]:
    """Train per-patch linear probes at each pipeline stage × physics property.

    Returns:
        results[stage_name][prop_name] = {"r2_per_patch": [...], "mean_r2": float}
    """
    PROP_ORDER = ["mass", "friction", "elasticity", "stability"]

    results: dict[str, dict[str, dict]] = {}

    for stage_name, X_all in stage_arrays.items():
        stage_results: dict[str, dict] = {}

        for prop_idx, prop_name in enumerate(PROP_ORDER[:3]):
            y_all = label_array[:, :, prop_idx]  # [N, 196]

            valid_frac = float(np.mean(~np.isnan(y_all)))
            if valid_frac < 0.05:
                logger.info(f"  {stage_name} / {prop_name}: skip (valid={valid_frac:.1%})")
                continue

            r2_map = compute_per_patch_r2(
                X_all, y_all,
                patch_grid_size=PATCH_GRID_SIZE,
                alpha=1.0,
            )
            mean_r2 = float(np.nanmean(r2_map))
            max_r2 = float(np.nanmax(r2_map))

            stage_results[prop_name] = {
                "r2_per_patch": r2_map.tolist(),
                "mean_r2": mean_r2,
                "max_r2": max_r2,
            }
            logger.info(
                f"  {stage_name:20s} | {prop_name:12s} | "
                f"mean R²={mean_r2:.4f}  max R²={max_r2:.4f}"
            )

        results[stage_name] = stage_results

    return results


def save_saliency_grid(
    results: dict,
    sample_image,
    output_path: Path,
) -> None:
    """Save a 4-stage × 3-property grid of saliency heatmaps."""
    viz = PhysicsSaliencyMap(patch_grid_size=PATCH_GRID_SIZE, dpi=120)

    n_rows = len(STAGE_NAMES)
    n_cols = len(PROPERTY_NAMES)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for row, stage_name in enumerate(STAGE_NAMES):
        for col, prop_name in enumerate(PROPERTY_NAMES):
            ax = axes[row, col]
            ax.axis("off")

            if stage_name not in results or prop_name not in results[stage_name]:
                ax.set_title(f"{stage_name[-7:]}\n{prop_name}\n(no data)", fontsize=7)
                continue

            r2_scores = np.array(results[stage_name][prop_name]["r2_per_patch"])
            W, H = sample_image.size
            image_arr = np.array(sample_image)
            heatmap = viz.scores_to_heatmap(r2_scores, (H, W))

            ax.imshow(image_arr)
            im = ax.imshow(heatmap, cmap="inferno", alpha=0.6, interpolation="bilinear",
                           vmin=0.0, vmax=max(r2_scores.max(), 0.01))

            mean_r2 = results[stage_name][prop_name]["mean_r2"]
            stage_short = stage_name.replace("stage_", "S").replace("_enc_out", "").replace(
                "_post_proj", "").replace("_llm_8", "").replace("_llm_16", "")
            ax.set_title(f"{stage_short} | {prop_name}\nR²={mean_r2:.3f}", fontsize=8)

            if row == n_rows - 1:
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("R²", fontsize=7)

    fig.suptitle(
        "Physics Saliency Maps — ViT-base-patch16-224 on Synthetic Data\n"
        "(Row = pipeline stage, Col = physics property)",
        fontsize=10,
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Saliency grid saved: {output_path}")


def save_degradation_curve(results: dict, output_path: Path) -> None:
    """Plot mean R² across the 4 pipeline stages (degradation curve)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    stage_labels = ["S1 enc-out", "S2 post-proj", "S3 LLM-8", "S4 LLM-16"]
    x = np.arange(len(STAGE_NAMES))

    for prop_name in PROPERTY_NAMES:
        y_vals = []
        for stage_name in STAGE_NAMES:
            if stage_name in results and prop_name in results[stage_name]:
                y_vals.append(results[stage_name][prop_name]["mean_r2"])
            else:
                y_vals.append(0.0)
        ax.plot(x, y_vals, marker="o", label=prop_name, linewidth=2)

    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels)
    ax.set_xlabel("Pipeline Stage")
    ax.set_ylabel("Mean R² across patches")
    ax.set_title("Physics Decodability Across Pipeline Stages\n(ViT-base, Synthetic Data)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=-0.01)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"  Degradation curve saved: {output_path}")


def print_summary(results: dict) -> None:
    """Print a compact results table to stdout."""
    print("\n" + "=" * 70)
    print("E2E TEST RESULTS — mean R² per stage × property")
    print("=" * 70)
    header = f"{'Stage':<22}" + "".join(f"{p:>12}" for p in PROPERTY_NAMES)
    print(header)
    print("-" * 70)
    for stage_name in STAGE_NAMES:
        row = f"{stage_name:<22}"
        for prop_name in PROPERTY_NAMES:
            if stage_name in results and prop_name in results[stage_name]:
                val = results[stage_name][prop_name]["mean_r2"]
                row += f"{val:>12.4f}"
            else:
                row += f"{'N/A':>12}"
        print(row)
    print("=" * 70)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Synthetic dataset ----------------------------------------
    logger.info("Step 1: Generating synthetic physics dataset (%d scenes)...", N_SCENES)
    dataset = SyntheticPhysicsDataset(n_scenes=N_SCENES, image_size=IMAGE_SIZE, seed=42)
    logger.info(f"  Generated {len(dataset)} scenes.")

    # Quick sanity check on the dataset
    sample0 = dataset[0]
    assert sample0.image.size == (IMAGE_SIZE, IMAGE_SIZE), "Image size mismatch"
    assert sample0.object_masks.shape == (IMAGE_SIZE, IMAGE_SIZE), "Mask shape mismatch"
    assert "mass" in sample0.physics_labels, "Physics labels missing"
    logger.info(
        f"  Sample 0: {sample0.num_objects} objects, "
        f"mask unique ids={np.unique(sample0.object_masks).tolist()}"
    )

    # ---- Step 2: Load ViT model -------------------------------------------
    logger.info("Step 2: Loading google/vit-base-patch16-224 on CPU...")
    extractor = LightweightViTExtractor(device=DEVICE)
    extractor.load()
    logger.info(
        f"  Hidden dim: {extractor.HIDDEN_DIM}, "
        f"N patches: {extractor.N_PATCHES}, "
        f"Grid: {extractor.PATCH_GRID_SIZE}×{extractor.PATCH_GRID_SIZE}"
    )

    # Quick hook test
    test_acts = extractor.extract(dataset[0].image)
    for stage in STAGE_NAMES:
        assert stage in test_acts, f"Missing stage: {stage}"
        assert test_acts[stage].shape == (196, 768), \
            f"Bad shape for {stage}: {test_acts[stage].shape}"
    logger.info("  Forward pass OK — all 4 stages captured.")

    # ---- Step 3: Extract activations & assign patch labels ----------------
    logger.info("Step 3: Extracting activations and assigning patch labels...")
    assigner = PatchLabelAssigner(patch_grid_size=PATCH_GRID_SIZE)
    stage_arrays, label_array, sample_images = collect_activations_and_labels(
        dataset, extractor, assigner
    )

    # ---- Step 4: Train linear probes ---------------------------------------
    logger.info("Step 4: Training linear probes (ridge regression per patch position)...")
    results = train_probes(stage_arrays, label_array)

    # ---- Step 5: Save results to JSON -------------------------------------
    results_path = OUTPUT_DIR / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Step 5: Probe results saved to {results_path}")

    # ---- Step 6: Generate saliency maps ------------------------------------
    logger.info("Step 6: Generating physics saliency map grid...")
    save_saliency_grid(results, sample_images[0], OUTPUT_DIR / "saliency_maps.png")

    # ---- Step 7: Degradation curve -----------------------------------------
    logger.info("Step 7: Generating pipeline degradation curve...")
    save_degradation_curve(results, OUTPUT_DIR / "degradation_curve.png")

    # ---- Summary -----------------------------------------------------------
    print_summary(results)
    print(f"\nAll outputs saved to: {OUTPUT_DIR.resolve()}")
    print("E2E test PASSED")


if __name__ == "__main__":
    main()
