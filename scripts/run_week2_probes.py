#!/usr/bin/env python3
"""
Week 2 Day 8-9: Run physics probes on real VLM (Qwen2.5-VL-7B) activations.

This script runs the CRITICAL experiments that Week 1 failed to deliver:

1. Linear ridge probes: mass prediction at all 4 pipeline stages
2. Permutation baseline: shuffle mass labels → retrain → should get R² ≈ 0
3. Visual control: probe for HUE at the same patches → if mass R² >> hue R²,
   that's evidence of physics-specific encoding
4. Cross-stage comparison: does projection (Stage 1→2) differentially degrade
   physics vs. appearance features?

Reports:
  - R² for mass at each of the 4 real pipeline stages
  - Permutation baseline R² (should be ~0)
  - Visual control R² (hue probe)
  - Projection layer differential degradation analysis

Usage:
  python scripts/run_week2_probes.py --activation-dir results/activations/qwen2_5_vl_7b/

For testing without GPU (uses ViT-base activations):
  python scripts/run_week2_probes.py --activation-dir results/activations/vit_base/ --test-mode
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.probing.linear_probe import LinearProbe, bootstrap_r2_ci
from src.models.activation_extractor import STAGE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Column indices in the extended patch_labels array
LABEL_COLUMNS = {
    "mass": 0,
    "friction": 1,
    "elasticity": 2,
    "stability": 3,
    "hue": 4,  # visual control variable
}


def load_activations_and_labels(
    activation_dir: Path,
    stage: str,
    target_col: int,
    max_files: Optional[int] = None,
    exclude_background: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load activations and labels from HDF5 files.

    Args:
        activation_dir: Directory containing scene_XXXX.h5 files.
        stage: Pipeline stage key.
        target_col: Column index in patch_labels for the target variable.
        max_files: Limit number of files (for debugging).
        exclude_background: Remove patches with NaN labels.

    Returns:
        (X, y): X is [N_valid_patches, D], y is [N_valid_patches].
    """
    h5_files = sorted(activation_dir.glob("scene_*.h5"))
    if max_files:
        h5_files = h5_files[:max_files]

    X_list, y_list = [], []
    n_skipped = 0

    for h5_path in h5_files:
        try:
            with h5py.File(h5_path, "r") as f:
                if stage not in f:
                    n_skipped += 1
                    continue
                if "patch_labels" not in f:
                    n_skipped += 1
                    continue

                acts = f[stage][:]              # [N_tokens, D]
                labels = f["patch_labels"][:]   # [N_patches, 5]

                # Match dimensions: use min of acts and labels row count
                n = min(acts.shape[0], labels.shape[0])
                acts = acts[:n]
                labels = labels[:n]

                if target_col >= labels.shape[1]:
                    n_skipped += 1
                    continue

                y = labels[:, target_col]
                X_list.append(acts)
                y_list.append(y)
        except Exception as e:
            logger.warning(f"Failed to load {h5_path.name}: {e}")
            n_skipped += 1

    if not X_list:
        raise RuntimeError(f"No valid HDF5 files found with stage '{stage}'")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)

    if exclude_background:
        valid = ~np.isnan(y_all)
        X_all = X_all[valid]
        y_all = y_all[valid]

    logger.info(f"  Loaded {len(h5_files)-n_skipped} files, {X_all.shape[0]} patches, D={X_all.shape[1]}")
    return X_all, y_all


def train_and_evaluate_probe(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    train_frac: float = 0.7,
    seed: int = 42,
) -> Dict[str, float]:
    """Train a linear probe and return test metrics."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_train = int(n * train_frac)

    X_train, y_train = X[idx[:n_train]], y[idx[:n_train]]
    X_test, y_test = X[idx[n_train:]], y[idx[n_train:]]

    probe = LinearProbe(input_dim=X.shape[1], alpha=alpha)
    probe.fit(X_train, y_train)
    metrics = probe.score(X_test, y_test)
    return metrics


def run_permutation_baseline(
    X: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 5,
    alpha: float = 1.0,
    seed: int = 42,
) -> Dict[str, float]:
    """Run permutation baseline: shuffle labels, retrain, average R².

    If the original R² is significantly above the permuted R², the probe
    is detecting a real signal rather than statistical artifact.
    """
    rng = np.random.default_rng(seed)
    perm_r2s = []

    for i in range(n_permutations):
        y_perm = rng.permutation(y)
        metrics = train_and_evaluate_probe(X, y_perm, alpha=alpha, seed=seed + i + 1000)
        perm_r2s.append(metrics["r2"])

    return {
        "permutation_r2_mean": float(np.mean(perm_r2s)),
        "permutation_r2_std": float(np.std(perm_r2s)),
        "permutation_r2_max": float(np.max(perm_r2s)),
        "n_permutations": n_permutations,
    }


def run_experiment(activation_dir: Path, output_dir: Path, test_mode: bool = False):
    """Run the full Week 2 probing experiment.

    Experiment matrix:
      - 4 pipeline stages × {mass, friction, hue} targets
      - Permutation baselines for mass at each stage
      - Bootstrap CIs for key comparisons
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "experiment": "week2_day8_9_real_vlm_probes",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stages": {},
    }

    target_variables = {
        "mass": LABEL_COLUMNS["mass"],
        "friction": LABEL_COLUMNS["friction"],
        "hue": LABEL_COLUMNS["hue"],  # visual control
    }

    for stage in STAGE_NAMES:
        logger.info(f"\n{'='*60}")
        logger.info(f"Stage: {stage}")
        logger.info(f"{'='*60}")

        stage_results = {}

        for var_name, var_col in target_variables.items():
            logger.info(f"\n  Target: {var_name} (col {var_col})")

            try:
                X, y = load_activations_and_labels(
                    activation_dir, stage, var_col, exclude_background=True
                )
            except RuntimeError as e:
                logger.warning(f"  Skipping {var_name} at {stage}: {e}")
                stage_results[var_name] = {"error": str(e)}
                continue

            if len(y) < 50:
                logger.warning(f"  Too few samples ({len(y)}) for {var_name}")
                stage_results[var_name] = {"error": f"too_few_samples ({len(y)})"}
                continue

            # Train probe
            metrics = train_and_evaluate_probe(X, y)
            logger.info(f"    R²={metrics['r2']:.4f}, Pearson r={metrics['pearson_r']:.4f}")

            stage_results[var_name] = {
                "metrics": metrics,
                "n_samples": len(y),
                "feature_dim": X.shape[1],
            }

            # Permutation baseline (only for physics variables, not hue)
            if var_name in ("mass", "friction"):
                logger.info(f"    Running permutation baseline...")
                perm = run_permutation_baseline(X, y, n_permutations=5)
                stage_results[var_name]["permutation"] = perm
                logger.info(f"    Permutation R²: {perm['permutation_r2_mean']:.4f} ± {perm['permutation_r2_std']:.4f}")

            # Bootstrap CI for mass (the key variable)
            if var_name == "mass" and len(y) > 100:
                logger.info(f"    Computing bootstrap CI...")
                r2_mean, ci_low, ci_high = bootstrap_r2_ci(
                    X, y, alpha=1.0, n_resamples=200, seed=42
                )
                stage_results[var_name]["bootstrap_ci"] = {
                    "r2_mean": r2_mean,
                    "ci_95_low": ci_low,
                    "ci_95_high": ci_high,
                }
                logger.info(f"    Bootstrap: R²={r2_mean:.4f} [{ci_low:.4f}, {ci_high:.4f}]")

        results["stages"][stage] = stage_results

    # ---------------------------------------------------------------
    # Cross-stage analysis: projection layer differential degradation
    # ---------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("Cross-Stage Analysis: Projection Differential Degradation")
    logger.info(f"{'='*60}")

    analysis = {}

    for var_name in ("mass", "friction", "hue"):
        s1_key = "stage_1_enc_out"
        s2_key = "stage_2_post_proj"

        s1_data = results["stages"].get(s1_key, {}).get(var_name, {})
        s2_data = results["stages"].get(s2_key, {}).get(var_name, {})

        if "metrics" in s1_data and "metrics" in s2_data:
            r2_s1 = s1_data["metrics"]["r2"]
            r2_s2 = s2_data["metrics"]["r2"]
            delta = r2_s2 - r2_s1
            # Retention ratio: only meaningful when stage1 R² > 0
            if r2_s1 > 0.01:
                retention = r2_s2 / r2_s1
            else:
                retention = float("nan")
            analysis[var_name] = {
                "stage1_r2": r2_s1,
                "stage2_r2": r2_s2,
                "delta_r2": delta,
                "retention_ratio": retention,
            }
            ret_str = f"{retention:.2%}" if not np.isnan(retention) else "N/A (stage1 <= 0)"
            logger.info(f"  {var_name}: Stage1 R2={r2_s1:.4f} -> Stage2 R2={r2_s2:.4f} "
                        f"(delta={delta:+.4f}, retention={ret_str})")

    # Check if physics degrades more than appearance through projection
    if "mass" in analysis and "hue" in analysis:
        mass_ret = analysis["mass"]["retention_ratio"]
        hue_ret = analysis["hue"]["retention_ratio"]

        # Only compute differential if both retentions are valid
        if not np.isnan(mass_ret) and not np.isnan(hue_ret):
            diff_degradation = mass_ret - hue_ret
            interpretation = (
                "Physics PRESERVED better through projection (physics-aware merger)"
                if diff_degradation > 0.05
                else "Physics DEGRADED more through projection (physics-blind merger)"
                if diff_degradation < -0.05
                else "Similar degradation for physics and appearance"
            )
        else:
            diff_degradation = float("nan")
            # Use delta R2 comparison instead
            mass_delta = analysis["mass"]["delta_r2"]
            hue_delta = analysis["hue"]["delta_r2"]
            interpretation = (
                f"Stage1 mass R2 too low for retention analysis. "
                f"Delta comparison: mass delta={mass_delta:+.4f}, hue delta={hue_delta:+.4f}"
            )

        analysis["differential_degradation"] = {
            "mass_retention": mass_ret,
            "hue_retention": hue_ret,
            "physics_minus_appearance": diff_degradation,
            "interpretation": interpretation,
        }
        logger.info(f"\n  Differential degradation:")
        mass_str = f"{mass_ret:.2%}" if not np.isnan(mass_ret) else "N/A"
        hue_str = f"{hue_ret:.2%}" if not np.isnan(hue_ret) else "N/A"
        logger.info(f"    Mass retention: {mass_str}")
        logger.info(f"    Hue retention:  {hue_str}")
        logger.info(f"    Interpretation: {interpretation}")

    results["cross_stage_analysis"] = analysis

    # ---------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"{'Stage':<25} {'Mass R²':>10} {'Friction R²':>12} {'Hue R²':>10} {'Perm R²':>10}")
    logger.info("-" * 70)

    for stage in STAGE_NAMES:
        stage_data = results["stages"].get(stage, {})
        mass_r2 = stage_data.get("mass", {}).get("metrics", {}).get("r2", float("nan"))
        fric_r2 = stage_data.get("friction", {}).get("metrics", {}).get("r2", float("nan"))
        hue_r2 = stage_data.get("hue", {}).get("metrics", {}).get("r2", float("nan"))
        perm_r2 = stage_data.get("mass", {}).get("permutation", {}).get("permutation_r2_mean", float("nan"))
        logger.info(f"{stage:<25} {mass_r2:>10.4f} {fric_r2:>12.4f} {hue_r2:>10.4f} {perm_r2:>10.4f}")

    # Save results
    results_path = output_dir / "week2_probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    return results


def generate_test_activations(output_dir: Path, n_scenes: int = 100):
    """Generate test activations using ViT-base (CPU) for testing the probe pipeline.

    This allows running the full probe analysis without a GPU.
    """
    from src.data.deconfounded_physion import DeconfoundedPhysicsDataset
    from src.data.patch_label_assigner import PatchLabelAssigner
    from src.models.activation_extractor import LightweightViTExtractor

    logger.info("Generating test activations with ViT-base (CPU)...")

    dataset = DeconfoundedPhysicsDataset(n_scenes=n_scenes, image_size=224, seed=42)
    corr = dataset.verify_deconfounding()
    logger.info(f"Deconfounding: mass-hue r={corr['mass_hue_r']:.4f}, "
                f"mass-brightness r={corr['mass_brightness_r']:.4f}")

    extractor = LightweightViTExtractor(device="cpu")
    extractor.load()
    assigner = PatchLabelAssigner(patch_grid_size=14)

    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(range(len(dataset)), desc="Extracting ViT-base activations"):
        sample = dataset[idx]
        vis_features = dataset.get_visual_features(idx)
        h5_path = output_dir / f"scene_{idx:04d}.h5"

        if h5_path.exists():
            continue

        activations = extractor.extract(sample.image)
        patch_labels = assigner.assign(sample.object_masks, sample.physics_labels)

        # Add hue column
        hue_labels = np.full(assigner.n_patches, float("nan"), dtype=np.float32)
        patch_assignments = assigner._assign_patches_to_objects(sample.object_masks)
        for p_idx in range(assigner.n_patches):
            obj_id = patch_assignments[p_idx]
            if obj_id > 0 and (obj_id - 1) < len(vis_features["hue"]):
                hue_labels[p_idx] = vis_features["hue"][obj_id - 1]

        patch_labels_ext = np.column_stack([patch_labels, hue_labels])

        with h5py.File(h5_path, "w") as f:
            for stage_name in STAGE_NAMES:
                if stage_name in activations:
                    f.create_dataset(
                        stage_name, data=activations[stage_name].numpy(),
                        compression="gzip", compression_opts=4
                    )
            f.create_dataset("patch_labels", data=patch_labels_ext)
            f.attrs["scenario_id"] = sample.scenario_id
            f.attrs["model_name"] = "vit_base_patch16_224"
            f.attrs["patch_grid_size"] = 14
            f.attrs["physics_labels"] = json.dumps(
                {k: v.tolist() for k, v in sample.physics_labels.items()}
            )
            f.attrs["visual_features"] = json.dumps(
                {k: v.tolist() for k, v in vis_features.items()}
            )

    info = {
        "model": "vit_base_patch16_224",
        "n_scenes": n_scenes,
        "dataset_type": "deconfounded_synthetic",
        "patch_labels_columns": ["mass", "friction", "elasticity", "stability", "hue"],
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved {n_scenes} ViT-base activation files to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Week 2 Day 8-9 probing experiments")
    parser.add_argument("--activation-dir", type=str, default=None,
                        help="Directory with HDF5 activation files")
    parser.add_argument("--output-dir", type=str, default="results/week2_probes",
                        help="Output directory for probe results")
    parser.add_argument("--test-mode", action="store_true",
                        help="Use ViT-base (CPU) instead of Qwen for testing")
    parser.add_argument("--n-scenes", type=int, default=200,
                        help="Number of scenes for test mode")
    args = parser.parse_args()

    output_dir = PROJECT_ROOT / args.output_dir

    if args.test_mode or args.activation_dir is None:
        # Generate test activations with ViT-base
        act_dir = PROJECT_ROOT / "results" / "activations" / "vit_base_deconfounded"
        generate_test_activations(act_dir, n_scenes=args.n_scenes)
        activation_dir = act_dir
    else:
        activation_dir = Path(args.activation_dir)

    if not activation_dir.exists():
        logger.error(f"Activation directory not found: {activation_dir}")
        sys.exit(1)

    results = run_experiment(activation_dir, output_dir)

    # Print final verdict
    print("\n" + "=" * 60)
    print("WEEK 2 DAY 8-9: KEY FINDINGS")
    print("=" * 60)

    for stage in STAGE_NAMES:
        stage_data = results.get("stages", {}).get(stage, {})
        mass_data = stage_data.get("mass", {})
        hue_data = stage_data.get("hue", {})

        if "metrics" in mass_data and "metrics" in hue_data:
            mass_r2 = mass_data["metrics"]["r2"]
            hue_r2 = hue_data["metrics"]["r2"]
            perm_r2 = mass_data.get("permutation", {}).get("permutation_r2_mean", 0)

            is_above_chance = mass_r2 > perm_r2 + 0.05
            is_physics_specific = mass_r2 > hue_r2 + 0.02

            print(f"\n{stage}:")
            print(f"  Mass R2: {mass_r2:.4f} {'[ABOVE CHANCE]' if is_above_chance else '[AT CHANCE]'}")
            print(f"  Hue R2:  {hue_r2:.4f}")
            print(f"  Perm R2: {perm_r2:.4f}")
            if is_physics_specific:
                print(f"  -> PHYSICS-SPECIFIC encoding detected (mass >> hue)")
            elif mass_r2 > 0.05:
                print(f"  -> Signal detected but may reflect appearance, not physics")
            else:
                print(f"  -> No meaningful physics signal at this stage")

    cross = results.get("cross_stage_analysis", {})
    if "differential_degradation" in cross:
        dd = cross["differential_degradation"]
        print(f"\nProjection layer analysis:")
        print(f"  {dd['interpretation']}")


if __name__ == "__main__":
    main()
