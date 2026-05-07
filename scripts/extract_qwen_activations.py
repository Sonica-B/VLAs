#!/usr/bin/env python3
"""
Extract activations from Qwen2.5-VL-7B (4-bit) at all 4 pipeline stages.

Pipeline stages for Qwen2.5-VL-7B:
  Stage 1: model.visual.blocks.31      — ViT final layer output (before projection)
  Stage 2: model.visual.merger          — After visual merger/projection (maps to LLM space)
  Stage 3: model.model.layers.8         — LLM layer 8 hidden states
  Stage 4: model.model.layers.16        — LLM layer 16 hidden states

For each image in the dataset:
  1. Prepare input using Qwen's chat template
  2. Run forward pass with hooks registered at all 4 stages
  3. Extract visual patch activations
  4. Assign per-patch physics labels using segmentation masks
  5. Save to HDF5

Output: results/activations/qwen2_5_vl_7b/scene_XXXX.h5
"""

from __future__ import annotations

import gc
import json
import logging
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deconfounded_physion import DeconfoundedPhysicsDataset
from src.data.patch_label_assigner import PatchLabelAssigner
from src.models.activation_extractor import ActivationExtractor, STAGE_NAMES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_model():
    """Load Qwen2.5-VL-7B in 4-bit."""
    from transformers import AutoProcessor, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Try pre-quantized first
    model_ids = [
        ("unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit", False),
        ("Qwen/Qwen2.5-VL-7B-Instruct", True),
    ]

    model = None
    used_id = None

    for model_id, needs_bnb in model_ids:
        try:
            from transformers import Qwen2_5VLForConditionalGeneration
            load_kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16}
            if needs_bnb:
                load_kwargs["quantization_config"] = bnb_config

            logger.info(f"Loading {model_id}...")
            model = Qwen2_5VLForConditionalGeneration.from_pretrained(model_id, **load_kwargs)
            used_id = model_id
            break
        except Exception as e:
            logger.warning(f"Failed to load {model_id}: {e}")
            continue

    if model is None:
        raise RuntimeError("Could not load Qwen2.5-VL-7B. See load_qwen_4bit.py for setup.")

    model.eval()
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    vram = torch.cuda.memory_allocated() / 1024**3
    logger.info(f"Model loaded: {used_id} | VRAM: {vram:.2f}GB")

    return model, processor


def extract_all_activations(
    model,
    processor,
    dataset: DeconfoundedPhysicsDataset,
    output_dir: Path,
    patch_grid_size: int = 14,
    text_prompt: str = "Describe the physics of this scene.",
):
    """Extract activations for all scenes and save to HDF5."""
    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = ActivationExtractor(
        model=model,
        model_name="qwen2_5_vl_7b",
        patch_grid_size=patch_grid_size,
        device="cuda",
    )
    assigner = PatchLabelAssigner(patch_grid_size=patch_grid_size)

    n_extracted = 0
    n_failed = 0
    t_start = time.time()

    for idx in tqdm(range(len(dataset)), desc="Extracting activations"):
        sample = dataset[idx]
        vis_features = dataset.get_visual_features(idx)
        h5_path = output_dir / f"scene_{idx:04d}.h5"

        if h5_path.exists():
            n_extracted += 1
            continue

        try:
            # Extract activations at all 4 stages
            activations = extractor.extract(
                image=sample.image,
                processor=processor,
                text_prompt=text_prompt,
            )

            if not activations:
                logger.warning(f"Scene {idx}: no activations captured")
                n_failed += 1
                continue

            # Assign per-patch physics labels
            patch_labels = assigner.assign(
                sample.object_masks, sample.physics_labels
            )

            # Assign per-patch visual features (hue) for control probing
            # We add hue as an extra column (index 4) in the patch labels
            hue_labels = np.full(assigner.n_patches, float("nan"), dtype=np.float32)
            patch_assignments = assigner._assign_patches_to_objects(sample.object_masks)
            for p_idx in range(assigner.n_patches):
                obj_id = patch_assignments[p_idx]
                if obj_id > 0 and (obj_id - 1) < len(vis_features["hue"]):
                    hue_labels[p_idx] = vis_features["hue"][obj_id - 1]

            # Extend patch_labels with hue column
            patch_labels_extended = np.column_stack([patch_labels, hue_labels])

            # Save to HDF5
            with h5py.File(h5_path, "w") as f:
                for stage_name in STAGE_NAMES:
                    if stage_name in activations:
                        arr = activations[stage_name].numpy()
                        f.create_dataset(
                            stage_name, data=arr,
                            compression="gzip", compression_opts=4
                        )

                f.create_dataset("patch_labels", data=patch_labels_extended)

                f.attrs["scenario_id"] = sample.scenario_id
                f.attrs["model_name"] = "qwen2_5_vl_7b"
                f.attrs["patch_grid_size"] = patch_grid_size
                f.attrs["n_visual_tokens"] = activations.get(
                    "stage_2_post_proj", activations.get("stage_1_enc_out", torch.zeros(1))
                ).shape[0]
                f.attrs["physics_labels"] = json.dumps(
                    {k: v.tolist() for k, v in sample.physics_labels.items()}
                )
                f.attrs["visual_features"] = json.dumps(
                    {k: v.tolist() for k, v in vis_features.items()}
                )

            n_extracted += 1

        except Exception as e:
            logger.error(f"Scene {idx} failed: {e}")
            n_failed += 1
            # Clear CUDA cache on failure
            torch.cuda.empty_cache()
            gc.collect()
            continue

        # Periodic VRAM report
        if (idx + 1) % 50 == 0:
            vram = torch.cuda.memory_allocated() / 1024**3
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            logger.info(f"  Progress: {idx+1}/{len(dataset)} | VRAM: {vram:.2f}GB | {rate:.1f} img/s")

    elapsed = time.time() - t_start
    logger.info(f"\nExtraction complete: {n_extracted} succeeded, {n_failed} failed in {elapsed:.1f}s")

    # Save dataset info
    info = {
        "model": "qwen2_5_vl_7b",
        "n_scenes": len(dataset),
        "n_extracted": n_extracted,
        "n_failed": n_failed,
        "patch_grid_size": patch_grid_size,
        "dataset_type": "deconfounded_synthetic",
        "patch_labels_columns": ["mass", "friction", "elasticity", "stability", "hue"],
        "elapsed_seconds": elapsed,
    }
    with open(output_dir / "dataset_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return n_extracted


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Qwen2.5-VL activations")
    parser.add_argument("--n-scenes", type=int, default=200, help="Number of scenes")
    parser.add_argument("--image-size", type=int, default=224, help="Image size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str,
                        default="results/activations/qwen2_5_vl_7b",
                        help="Output directory for HDF5 files")
    args = parser.parse_args()

    # Generate deconfounded dataset
    logger.info(f"Generating {args.n_scenes} deconfounded scenes...")
    dataset = DeconfoundedPhysicsDataset(
        n_scenes=args.n_scenes,
        image_size=args.image_size,
        seed=args.seed,
    )

    # Verify deconfounding
    corr = dataset.verify_deconfounding()
    logger.info(f"Deconfounding check: mass-hue r={corr['mass_hue_r']:.4f}, "
                f"mass-brightness r={corr['mass_brightness_r']:.4f}")

    # Load model
    model, processor = load_model()

    # Extract activations
    output_dir = PROJECT_ROOT / args.output_dir
    n = extract_all_activations(
        model, processor, dataset, output_dir,
        patch_grid_size=14,
    )

    logger.info(f"Done. {n} activation files saved to {output_dir}")


if __name__ == "__main__":
    main()
