"""
Extract activations from the ViT model and save to HDF5 files.

Generates or loads the synthetic dataset, runs all images through
google/vit-base-patch16-224, and saves per-stage activations + physics
labels to disk so probe training can be done without re-running the model.

Output layout:
    results/activations/vit_base/
        stage_1_enc_out.h5    — [N, 196, 768] activations + [N, 196, 4] labels
        stage_2_post_proj.h5
        stage_3_llm_8.h5
        stage_4_llm_16.h5
        dataset_info.json     — n_scenes, seed, properties

Usage:
    python scripts/extract_and_save_activations.py [--n-scenes 1000] [--force]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

import h5py
import numpy as np

from src.data.synthetic_physion import SyntheticPhysicsDataset
from src.data.patch_label_assigner import PatchLabelAssigner
from src.models.activation_extractor import LightweightViTExtractor, STAGE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROPERTY_NAMES = ["mass", "friction", "elasticity", "stability"]
PATCH_GRID_SIZE = 14
IMAGE_SIZE = 224
OUTPUT_DIR = Path("results/activations/vit_base")


def extract_all(
    n_scenes: int = 1000,
    seed: int = 42,
    device: str = "cpu",
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Generate dataset and extract activations for all scenes.

    Returns:
        stage_arrays: dict[stage_name → np.ndarray [N, 196, 768]]
        label_array:  np.ndarray [N, 196, 4]  (NaN for background patches)
    """
    logger.info("Generating %d synthetic scenes (seed=%d)...", n_scenes, seed)
    dataset = SyntheticPhysicsDataset(n_scenes=n_scenes, image_size=IMAGE_SIZE, seed=seed)
    logger.info("Dataset generated: %d scenes, %d properties per object",
                len(dataset), len(PROPERTY_NAMES))

    logger.info("Loading google/vit-base-patch16-224 on %s...", device)
    extractor = LightweightViTExtractor(device=device)
    extractor.load()

    assigner = PatchLabelAssigner(patch_grid_size=PATCH_GRID_SIZE)

    per_stage: dict[str, list[np.ndarray]] = {s: [] for s in STAGE_NAMES}
    label_list: list[np.ndarray] = []

    t0 = time.time()
    for i, sample in enumerate(dataset):
        acts = extractor.extract(sample.image)
        for stage in STAGE_NAMES:
            per_stage[stage].append(acts[stage].numpy())

        # Assign all 4 physics properties (including stability) per patch
        patch_labels = assigner.assign(
            sample.object_masks,
            sample.physics_labels,
            stability_scores=_compute_geom_stability_per_patch(
                sample.object_masks, sample.physics_labels, assigner
            ),
        )
        label_list.append(patch_labels)

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n_scenes - i - 1) / rate
            logger.info(
                "  Scene %d/%d  (%.1f scenes/s, ~%.0fs remaining)",
                i + 1, n_scenes, rate, remaining,
            )

    stage_arrays = {s: np.stack(v) for s, v in per_stage.items()}
    label_array = np.stack(label_list)

    logger.info(
        "Extraction complete: activation shape %s, label shape %s",
        next(iter(stage_arrays.values())).shape,
        label_array.shape,
    )
    valid_frac = float(np.mean(~np.isnan(label_array[:, :, 0])))
    logger.info("Valid (non-background) patch fraction: %.1f%%", valid_frac * 100)

    return stage_arrays, label_array


def _compute_geom_stability_per_patch(
    object_masks: np.ndarray,
    physics_labels: dict,
    assigner: PatchLabelAssigner,
) -> np.ndarray:
    """Map per-object stability values to per-patch stability scores."""
    if "stability" not in physics_labels:
        return None
    stab_vals = physics_labels["stability"]  # [N_objects]
    patch_assignments = assigner._assign_patches_to_objects(object_masks)
    stability_scores = np.full(assigner.n_patches, float("nan"), dtype=np.float32)
    for patch_idx in range(assigner.n_patches):
        obj_id = patch_assignments[patch_idx]
        if obj_id > 0:
            obj_array_idx = obj_id - 1
            if obj_array_idx < len(stab_vals):
                stability_scores[patch_idx] = stab_vals[obj_array_idx]
    return stability_scores


def save_to_hdf5(
    stage_arrays: dict[str, np.ndarray],
    label_array: np.ndarray,
    output_dir: Path,
    n_scenes: int,
    seed: int,
) -> None:
    """Save per-stage activations and labels to HDF5 files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for stage_name, acts in stage_arrays.items():
        h5_path = output_dir / f"{stage_name}.h5"
        logger.info("Saving %s → %s  shape=%s", stage_name, h5_path, acts.shape)
        with h5py.File(h5_path, "w") as f:
            f.create_dataset(
                "activations", data=acts,
                compression="gzip", compression_opts=4,
                chunks=(min(50, n_scenes), acts.shape[1], acts.shape[2]),
            )
            f.create_dataset(
                "labels", data=label_array,
                compression="gzip", compression_opts=4,
                chunks=(min(50, n_scenes), label_array.shape[1], label_array.shape[2]),
            )
            f.attrs["stage_name"] = stage_name
            f.attrs["n_scenes"] = n_scenes
            f.attrs["model_name"] = "vit_base_patch16_224"
            f.attrs["patch_grid_size"] = PATCH_GRID_SIZE
            f.attrs["properties"] = json.dumps(PROPERTY_NAMES)
            f.attrs["seed"] = seed

    info_path = output_dir / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump({
            "n_scenes": n_scenes,
            "seed": seed,
            "model": "vit_base_patch16_224",
            "patch_grid_size": PATCH_GRID_SIZE,
            "image_size": IMAGE_SIZE,
            "properties": PROPERTY_NAMES,
            "stages": STAGE_NAMES,
            "activation_shape": list(next(iter(stage_arrays.values())).shape),
            "label_shape": list(label_array.shape),
        }, f, indent=2)
    logger.info("Dataset info saved: %s", info_path)


def load_from_hdf5(
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Load previously saved activations and labels from HDF5."""
    stage_arrays = {}
    label_array = None

    for stage_name in STAGE_NAMES:
        h5_path = output_dir / f"{stage_name}.h5"
        if not h5_path.exists():
            raise FileNotFoundError(f"Missing HDF5 file: {h5_path}")
        with h5py.File(h5_path, "r") as f:
            stage_arrays[stage_name] = f["activations"][:]
            if label_array is None:
                label_array = f["labels"][:]

    logger.info(
        "Loaded activations: %s stages, shape %s",
        len(stage_arrays),
        next(iter(stage_arrays.values())).shape,
    )
    return stage_arrays, label_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and save ViT activations to HDF5")
    parser.add_argument("--n-scenes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if HDF5 files already exist",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    info_path = output_dir / "dataset_info.json"

    # Check if already extracted
    if not args.force and info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        if info.get("n_scenes") == args.n_scenes and info.get("seed") == args.seed:
            logger.info(
                "HDF5 files already exist for n_scenes=%d seed=%d. "
                "Use --force to re-extract.",
                args.n_scenes, args.seed,
            )
            return

    t_start = time.time()
    stage_arrays, label_array = extract_all(
        n_scenes=args.n_scenes,
        seed=args.seed,
        device=args.device,
    )
    save_to_hdf5(stage_arrays, label_array, output_dir, args.n_scenes, args.seed)
    logger.info(
        "Done. Total time: %.1fs. Files in %s",
        time.time() - t_start, output_dir,
    )


if __name__ == "__main__":
    main()
