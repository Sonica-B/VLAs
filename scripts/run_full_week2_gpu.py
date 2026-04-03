#!/usr/bin/env python3
"""
Week 2 Day 10-14: Master GPU script for real VLM physics probing experiments.

Runs the full pipeline:
  1. Generate deconfounded physics scenes
  2. Load Qwen2.5-VL-7B in 4-bit
  3. Extract activations at 4 pipeline stages
  4. Train probes on real VLM activations
  5. Compute spatial metrics (Moran's I)
  6. Differential degradation analysis
  7. Generate publication figures
  8. Save comprehensive results JSON

Usage:
    # Full run on GPU with Qwen2.5-VL-7B:
    python scripts/run_full_week2_gpu.py --model qwen --num-scenes 500

    # Test mode (ViT-base on CPU, no download needed):
    python scripts/run_full_week2_gpu.py --test-mode --num-scenes 100

    # Resume from cached activations:
    python scripts/run_full_week2_gpu.py --model qwen --num-scenes 500 --resume
"""

import argparse
import gc
import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Project root setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deconfounded_physion import (
    DeconfoundedPhysicsDataset,
    save_deconfounded_dataset,
)
from src.probing.linear_probe import LinearProbe

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAGE_NAMES = [
    "stage_1_enc_out",
    "stage_2_post_proj",
    "stage_3_llm_8",
    "stage_4_llm_16",
]
STAGE_LABELS = [
    "Visual Encoder",
    "Post-Merger",
    "LLM Layer 8",
    "LLM Layer 16",
]
PHYSICS_VARS = ["mass", "friction", "elasticity", "stability"]
PHYSICS_COL = {"mass": 0, "friction": 1, "elasticity": 2, "stability": 3}
VISUAL_VARS = ["hue"]

MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen-4bit": "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
}

ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]
N_PERMUTATIONS = 10
N_BOOTSTRAP = 1000
CONFIDENCE_LEVEL = 0.95


def parse_args():
    p = argparse.ArgumentParser(description="Week 2 full VLM probing pipeline")
    p.add_argument(
        "--model",
        default="qwen",
        choices=["qwen", "qwen-4bit"],
        help="Model to use (default: qwen)",
    )
    p.add_argument(
        "--num-scenes", type=int, default=500, help="Number of scenes to generate"
    )
    p.add_argument(
        "--test-mode",
        action="store_true",
        help="Use ViT-base on CPU for pipeline validation",
    )
    p.add_argument(
        "--resume", action="store_true", help="Resume from cached HDF5 activations"
    )
    p.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for activation extraction"
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: results/week2_qwen or results/week2_test)",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


# ===========================================================================
# Utilities
# ===========================================================================

class Timer:
    """Context manager for timing blocks with ETA tracking."""

    _step_times: Dict[str, float] = {}

    def __init__(self, name: str, total_steps: int = 8):
        self.name = name
        self.total_steps = total_steps
        self.start = None

    def __enter__(self):
        self.start = time.time()
        step_num = len(Timer._step_times) + 1
        print(f"\n{'='*70}")
        print(f"  STEP {step_num}/{self.total_steps}: {self.name}")
        print(f"{'='*70}")
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.start
        Timer._step_times[self.name] = elapsed
        done = len(Timer._step_times)
        avg = sum(Timer._step_times.values()) / done
        remaining = (self.total_steps - done) * avg
        print(f"  -> Completed in {elapsed:.1f}s | "
              f"ETA for remaining steps: ~{remaining:.0f}s")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def report_vram():
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  VRAM: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.1f}GB total")
    else:
        print("  No CUDA device available")


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP,
                 ci: float = CONFIDENCE_LEVEL) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.RandomState(42)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1 - ci) / 2
    return float(np.percentile(boot_means, 100 * alpha)), float(
        np.percentile(boot_means, 100 * (1 - alpha))
    )


def morans_i(values: np.ndarray, grid_h: int, grid_w: int) -> float:
    """Compute Moran's I spatial autocorrelation on a grid."""
    if values.ndim == 1:
        values = values.reshape(grid_h, grid_w)
    n = grid_h * grid_w
    mean_val = np.nanmean(values)
    flat = values.flatten()
    valid = ~np.isnan(flat)
    if valid.sum() < 4:
        return 0.0

    # Queen contiguity weights
    w_sum = 0.0
    numerator = 0.0
    denominator = np.nansum((flat - mean_val) ** 2)
    if denominator == 0:
        return 0.0

    for i in range(grid_h):
        for j in range(grid_w):
            idx = i * grid_w + j
            if not valid[idx]:
                continue
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue
                    ni, nj = i + di, j + dj
                    if 0 <= ni < grid_h and 0 <= nj < grid_w:
                        nidx = ni * grid_w + nj
                        if valid[nidx]:
                            w_sum += 1.0
                            numerator += (flat[idx] - mean_val) * (flat[nidx] - mean_val)

    if w_sum == 0:
        return 0.0
    return float((n / w_sum) * (numerator / denominator))


def spatial_precision_recall_f1(
    r2_map: np.ndarray, object_mask: np.ndarray, threshold: float = 0.1
) -> Dict[str, float]:
    """Compute spatial precision/recall/F1 from R² map vs object mask."""
    pred = (r2_map > threshold).astype(float).flatten()
    gt = (object_mask > 0).astype(float).flatten()
    valid = ~np.isnan(r2_map.flatten())
    pred, gt = pred[valid], gt[valid]
    tp = np.sum(pred * gt)
    fp = np.sum(pred * (1 - gt))
    fn = np.sum((1 - pred) * gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


# ===========================================================================
# Step 1: Generate deconfounded data
# ===========================================================================

def step1_generate_data(num_scenes: int, output_dir: Path, seed: int) -> Tuple[np.ndarray, Dict]:
    """Generate deconfounded physics scenes."""
    print(f"  Generating {num_scenes} deconfounded scenes...")
    dataset = DeconfoundedPhysicsDataset(
        n_scenes=num_scenes, image_size=224, seed=seed
    )

    # Collect images, labels, visual features
    images = []
    all_labels = []  # per-patch physics labels
    all_visual = []  # per-object visual features
    all_masks = []

    for i in range(len(dataset)):
        sample = dataset[i]
        vis_feat = dataset.get_visual_features(i)

        images.append(sample.image)
        all_labels.append(sample.physics_labels)
        all_visual.append(vis_feat)
        all_masks.append(sample.object_masks)

        if (i + 1) % 100 == 0:
            print(f"    Generated {i+1}/{num_scenes} scenes")

    # Verify deconfounding
    deconf = dataset.verify_deconfounding()
    print(f"  Deconfounding check:")
    print(f"    mass-hue r = {deconf['mass_hue_r']:.4f} (should be ~0)")
    print(f"    mass-brightness r = {deconf['mass_brightness_r']:.4f} (should be ~0)")

    metadata = {
        "n_scenes": num_scenes,
        "seed": seed,
        "deconfounding": {k: float(v) for k, v in deconf.items()},
        "image_size": 224,
    }

    scene_data = {
        "images": images,
        "labels": all_labels,
        "visual_features": all_visual,
        "masks": all_masks,
    }
    return scene_data, metadata


def _load_cached_scenes(data_dir: Path, metadata: dict) -> Dict:
    """Load cached scenes from disk."""
    from PIL import Image as PILImage

    n = metadata["n_scenes"]
    images = []
    masks = []
    for i in range(n):
        img_path = data_dir / "images" / f"scene_{i:04d}.png"
        mask_path = data_dir / "masks" / f"scene_{i:04d}.png"
        if img_path.exists():
            images.append(np.array(PILImage.open(img_path).convert("RGB")))
        if mask_path.exists():
            masks.append(np.array(PILImage.open(mask_path)))

    with open(data_dir / "metadata.json", "r") as f:
        full_meta = json.load(f)

    # Reconstruct labels and visual features from metadata
    labels_list = full_meta.get("scenes", [])
    all_labels = []
    all_visual = []
    for scene in labels_list:
        all_labels.append(scene.get("physics_labels", {}))
        all_visual.append(scene.get("visual_features", {}))

    return {
        "images": images,
        "labels": all_labels,
        "visual_features": all_visual,
        "masks": masks,
    }


# ===========================================================================
# Step 2: Load model
# ===========================================================================

def step2_load_model_test_mode():
    """Load ViT-base for test mode (CPU)."""
    from src.models.activation_extractor import LightweightViTExtractor

    print("  Loading ViT-base (test mode, CPU)...")
    extractor = LightweightViTExtractor()
    print("  ViT-base loaded successfully")
    return extractor, "vit_base"


def step2_load_model_gpu(model_key: str):
    """Load Qwen2.5-VL-7B in 4-bit quantization."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"  Loading model: {MODEL_IDS[model_key]}")
    report_vram()

    # Try pre-quantized first
    try:
        print("  Attempting pre-quantized model (unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit)...")
        from transformers import BitsAndBytesConfig

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-7B-Instruct",
            trust_remote_code=True,
        )
        print("  Pre-quantized model loaded!")
    except Exception as e:
        print(f"  Pre-quantized failed ({e}), loading with BitsAndBytesConfig...")
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_IDS["qwen"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_IDS["qwen"],
            trust_remote_code=True,
        )
        print("  Model loaded with 4-bit quantization!")

    report_vram()
    model.eval()
    return model, processor


# ===========================================================================
# Step 3: Extract activations
# ===========================================================================

def step3_extract_activations_test_mode(
    extractor, scene_data: Dict, output_dir: Path, num_scenes: int
) -> Dict[str, np.ndarray]:
    """Extract activations with ViT-base (test mode)."""
    from src.data.patch_label_assigner import PatchLabelAssigner

    cache_path = output_dir / "activations" / "test_activations.h5"
    label_cache_path = output_dir / "activations" / "test_labels.h5"

    # Check cache
    if cache_path.exists() and label_cache_path.exists():
        print(f"  Loading cached activations from {cache_path}")
        return _load_hdf5_activations(cache_path, label_cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    assigner = PatchLabelAssigner(patch_grid_size=14)
    all_activations = {s: [] for s in STAGE_NAMES}
    all_patch_labels = []
    all_visual_labels = []  # hue per patch

    images = scene_data["images"]
    labels_list = scene_data["labels"]
    masks_list = scene_data["masks"]
    visual_list = scene_data["visual_features"]

    for i in range(min(num_scenes, len(images))):
        img = images[i]
        if isinstance(img, np.ndarray):
            from PIL import Image as PILImage
            img_pil = PILImage.fromarray(img.astype(np.uint8))
        else:
            img_pil = img

        # Extract activations at 4 stages
        acts = extractor.extract(img_pil)  # dict of stage -> [1, n_patches, dim]
        for stage_name in STAGE_NAMES:
            if stage_name in acts:
                all_activations[stage_name].append(acts[stage_name].squeeze(0))

        # Assign per-patch physics labels
        physics = labels_list[i] if i < len(labels_list) else {}
        mask = masks_list[i] if i < len(masks_list) else None
        if mask is not None and physics:
            patch_labels = assigner.assign(mask, physics)
            all_patch_labels.append(patch_labels)
        else:
            n_patches = 196  # 14x14
            all_patch_labels.append(np.full((n_patches, 4), np.nan))

        # Assign per-patch hue labels
        vis = visual_list[i] if i < len(visual_list) else {}
        hue_arr = vis.get("hue", None)
        if hue_arr is not None and mask is not None:
            hue_labels = assigner.assign(mask, {"hue": hue_arr})
            if isinstance(hue_labels, dict):
                hue_labels = hue_labels.get("hue", np.full(196, np.nan))
            all_visual_labels.append(hue_labels)
        else:
            all_visual_labels.append(np.full(196, np.nan))

        if (i + 1) % 50 == 0:
            print(f"    Extracted {i+1}/{num_scenes} scenes")

    # Stack arrays
    result = {}
    for stage in STAGE_NAMES:
        if all_activations[stage]:
            stacked = np.concatenate(all_activations[stage], axis=0)
            result[stage] = stacked
            print(f"    {stage}: {stacked.shape}")

    if all_patch_labels:
        result["physics_labels"] = np.concatenate(
            [pl if pl.ndim == 2 else pl.reshape(-1, 4) for pl in all_patch_labels], axis=0
        )
    if all_visual_labels:
        vis_arr = np.concatenate(
            [v if v.ndim == 1 else v.flatten() for v in all_visual_labels], axis=0
        )
        result["hue_labels"] = vis_arr

    # Cache to HDF5
    _save_hdf5_activations(result, cache_path)
    print(f"  Saved activations to {cache_path}")
    return result


def step3_extract_activations_gpu(
    model, processor, scene_data: Dict, output_dir: Path,
    num_scenes: int, batch_size: int, resume: bool
) -> Dict[str, np.ndarray]:
    """Extract activations from Qwen2.5-VL-7B at 4 pipeline stages."""
    from PIL import Image as PILImage
    from src.data.patch_label_assigner import PatchLabelAssigner

    cache_path = output_dir / "activations" / "qwen_activations.h5"

    if resume and cache_path.exists():
        print(f"  Resuming from cached activations at {cache_path}")
        return _load_hdf5_activations(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    assigner = PatchLabelAssigner(patch_grid_size=14)

    # -----------------------------------------------------------------------
    # Register hooks at 4 pipeline stages
    # -----------------------------------------------------------------------
    hook_storage = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hook_storage[name] = output[0].detach().cpu()
            elif isinstance(output, torch.Tensor):
                hook_storage[name] = output.detach().cpu()
        return hook_fn

    # Stage 1: Visual encoder last block output
    try:
        enc_hook = model.visual.blocks[-1].register_forward_hook(
            make_hook("stage_1_enc_out")
        )
        hooks.append(enc_hook)
        print("  Hooked: visual.blocks[-1] (encoder output)")
    except (AttributeError, IndexError) as e:
        print(f"  WARNING: Could not hook encoder: {e}")

    # Stage 2: Merger/projection output
    try:
        merger_hook = model.visual.merger.register_forward_hook(
            make_hook("stage_2_post_proj")
        )
        hooks.append(merger_hook)
        print("  Hooked: visual.merger (post-projection)")
    except AttributeError as e:
        print(f"  WARNING: Could not hook merger: {e}")

    # Stage 3: LLM layer 8
    try:
        llm_l8_hook = model.model.layers[8].register_forward_hook(
            make_hook("stage_3_llm_8")
        )
        hooks.append(llm_l8_hook)
        print("  Hooked: model.layers[8] (LLM layer 8)")
    except (AttributeError, IndexError) as e:
        print(f"  WARNING: Could not hook LLM layer 8: {e}")

    # Stage 4: LLM layer 16
    try:
        llm_l16_hook = model.model.layers[16].register_forward_hook(
            make_hook("stage_4_llm_16")
        )
        hooks.append(llm_l16_hook)
        print("  Hooked: model.layers[16] (LLM layer 16)")
    except (AttributeError, IndexError) as e:
        print(f"  WARNING: Could not hook LLM layer 16: {e}")

    # -----------------------------------------------------------------------
    # Extract activations for each scene
    # -----------------------------------------------------------------------
    all_activations = {s: [] for s in STAGE_NAMES}
    all_patch_labels = []
    all_hue_labels = []
    images = scene_data["images"]
    labels_list = scene_data["labels"]
    masks_list = scene_data["masks"]
    visual_list = scene_data["visual_features"]

    device = next(model.parameters()).device
    print(f"  Model device: {device}")
    print(f"  Extracting activations from {min(num_scenes, len(images))} scenes...")

    for i in range(min(num_scenes, len(images))):
        hook_storage.clear()

        img = images[i]
        if isinstance(img, np.ndarray):
            img_pil = PILImage.fromarray(img.astype(np.uint8))
        else:
            img_pil = img

        # Prepare input using Qwen2.5-VL chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_pil},
                    {"type": "text", "text": "Describe the objects."},
                ],
            }
        ]

        try:
            from qwen_vl_utils import process_vision_info

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        except ImportError:
            # Fallback without qwen_vl_utils
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[img_pil], return_tensors="pt", padding=True
            )

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        # Forward pass (no generation, just activations)
        with torch.no_grad():
            try:
                outputs = model(**inputs, output_hidden_states=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at scene {i}, clearing cache and retrying...")
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    outputs = model(**inputs, output_hidden_states=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"  Skipping scene {i} due to OOM")
                    continue

        # ---------------------------------------------------------------
        # Extract visual token activations from hooks
        # ---------------------------------------------------------------
        # Identify visual token positions for LLM stages
        # Qwen2.5-VL uses image_grid_thw to define the visual token grid
        image_grid_thw = inputs.get("image_grid_thw", None)
        if image_grid_thw is not None:
            # Total visual tokens = product of grid dimensions
            grid = image_grid_thw[0]  # [T, H, W] for first image
            n_visual_tokens = int(grid.prod().item())
        else:
            n_visual_tokens = None

        for stage_name in STAGE_NAMES:
            if stage_name not in hook_storage:
                continue

            act = hook_storage[stage_name]  # [1, seq_len, hidden_dim] or [1, n_patches, dim]

            if act.ndim == 3:
                act = act.squeeze(0)  # [seq_len, dim]
            elif act.ndim == 2:
                pass  # already [seq_len, dim]

            # For encoder/merger stages, all tokens are visual
            if stage_name in ("stage_1_enc_out", "stage_2_post_proj"):
                # Take all tokens (they're all visual patches)
                visual_act = act.numpy()
            else:
                # For LLM stages, extract only visual token positions
                # Visual tokens are typically the first n_visual_tokens after
                # any system/BOS tokens. We find them by looking at the
                # input_ids for image placeholder tokens.
                if n_visual_tokens is not None and act.shape[0] > n_visual_tokens:
                    # Find image token positions in input_ids
                    input_ids = inputs.get("input_ids", None)
                    if input_ids is not None:
                        # Qwen2.5-VL uses token ID 151655 for <|image_pad|>
                        img_token_id = 151655
                        ids = input_ids[0].cpu()
                        img_positions = (ids == img_token_id).nonzero(as_tuple=True)[0]
                        if len(img_positions) >= n_visual_tokens:
                            visual_act = act[img_positions[:n_visual_tokens]].numpy()
                        else:
                            # Fallback: take first n_visual_tokens
                            visual_act = act[:n_visual_tokens].numpy()
                    else:
                        visual_act = act[:n_visual_tokens].numpy()
                else:
                    visual_act = act.numpy()

            all_activations[stage_name].append(visual_act)

        # Assign per-patch physics labels
        physics = labels_list[i] if i < len(labels_list) else {}
        mask = masks_list[i] if i < len(masks_list) else None
        if mask is not None and physics:
            patch_labels = assigner.assign(mask, physics)
            if isinstance(patch_labels, dict):
                combined = np.stack(
                    [patch_labels.get(v, np.full(196, np.nan)) for v in PHYSICS_VARS],
                    axis=-1,
                )
                all_patch_labels.append(combined)
            elif isinstance(patch_labels, np.ndarray):
                if patch_labels.ndim == 1:
                    all_patch_labels.append(patch_labels.reshape(-1, 1))
                else:
                    all_patch_labels.append(patch_labels)
        else:
            n_patches = 196
            all_patch_labels.append(np.full((n_patches, 4), np.nan))

        # Hue labels
        vis = visual_list[i] if i < len(visual_list) else {}
        hue_arr = vis.get("hue", None)
        if hue_arr is not None and mask is not None:
            hue_labels = assigner.assign(mask, {"hue": hue_arr})
            if isinstance(hue_labels, dict):
                hue_labels = hue_labels.get("hue", np.full(196, np.nan))
            all_hue_labels.append(hue_labels.flatten())
        else:
            all_hue_labels.append(np.full(196, np.nan))

        if (i + 1) % 10 == 0:
            print(f"    Extracted {i+1}/{num_scenes} scenes")
            report_vram()

        # Free memory
        del outputs, inputs
        torch.cuda.empty_cache()

    # Remove hooks
    for h in hooks:
        h.remove()

    # Stack results
    result = {}
    for stage in STAGE_NAMES:
        if all_activations[stage]:
            stacked = np.concatenate(all_activations[stage], axis=0)
            result[stage] = stacked
            print(f"    {stage}: {stacked.shape}")
        else:
            print(f"    WARNING: No activations for {stage}")

    if all_patch_labels:
        result["physics_labels"] = np.concatenate(all_patch_labels, axis=0)
    if all_hue_labels:
        result["hue_labels"] = np.concatenate(all_hue_labels, axis=0)

    # Cache
    _save_hdf5_activations(result, cache_path)
    print(f"  Saved activations to {cache_path}")
    return result


def _save_hdf5_activations(data: Dict[str, np.ndarray], path: Path):
    """Save activation dict to HDF5 with compression."""
    with h5py.File(str(path), "w") as f:
        for key, arr in data.items():
            f.create_dataset(key, data=arr, compression="gzip", compression_opts=4)


def _load_hdf5_activations(path: Path, label_path: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Load activation dict from HDF5."""
    result = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            result[key] = f[key][:]
    if label_path and label_path.exists():
        with h5py.File(str(label_path), "r") as f:
            for key in f.keys():
                result[key] = f[key][:]
    return result


# ===========================================================================
# Step 4: Train probes
# ===========================================================================

def step4_train_probes(
    activations: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Train linear ridge regression probes for physics + visual control variables."""
    physics_labels = activations.get("physics_labels")  # [N, 4]
    hue_labels = activations.get("hue_labels")  # [N]

    results = {}
    all_vars = PHYSICS_VARS + VISUAL_VARS

    for stage in STAGE_NAMES:
        if stage not in activations:
            continue

        X = activations[stage]  # [N_patches, D]
        print(f"\n  {stage}: X.shape = {X.shape}")
        stage_results = {}

        for var in all_vars:
            if var in PHYSICS_VARS and physics_labels is not None:
                col = PHYSICS_COL[var]
                y = physics_labels[:, col] if physics_labels.ndim == 2 else physics_labels
            elif var == "hue" and hue_labels is not None:
                y = hue_labels
            else:
                continue

            # Align shapes
            n_samples = min(X.shape[0], len(y))
            X_var = X[:n_samples]
            y_var = y[:n_samples]

            # Remove NaN labels
            valid = ~np.isnan(y_var)
            if valid.sum() < 20:
                print(f"    {var}: too few valid samples ({valid.sum()}), skipping")
                continue
            X_valid = X_var[valid]
            y_valid = y_var[valid]

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_valid, y_valid, test_size=0.2, random_state=42
            )

            # Normalize
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            # Ridge regression
            ridge = RidgeCV(alphas=ALPHA_CANDIDATES)
            ridge.fit(X_train_sc, y_train)
            y_pred = ridge.predict(X_test_sc)

            r2 = float(1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            pearson_r = float(np.corrcoef(y_test, y_pred)[0, 1]) if len(y_test) > 2 else 0.0
            mse = float(np.mean((y_test - y_pred) ** 2))
            mae = float(np.mean(np.abs(y_test - y_pred)))

            # Bootstrap CI for R²
            boot_r2s = []
            rng = np.random.RandomState(42)
            for _ in range(N_BOOTSTRAP):
                idx = rng.choice(len(y_test), size=len(y_test), replace=True)
                ss_res = np.sum((y_test[idx] - y_pred[idx]) ** 2)
                ss_tot = np.sum((y_test[idx] - np.mean(y_test[idx])) ** 2)
                if ss_tot > 0:
                    boot_r2s.append(1 - ss_res / ss_tot)
            ci_low, ci_high = np.percentile(boot_r2s, [2.5, 97.5]) if boot_r2s else (0, 0)

            # Permutation baseline
            perm_r2s = []
            for p in range(N_PERMUTATIONS):
                y_perm = rng.permutation(y_train)
                ridge_perm = RidgeCV(alphas=ALPHA_CANDIDATES)
                ridge_perm.fit(X_train_sc, y_perm)
                y_perm_pred = ridge_perm.predict(X_test_sc)
                ss_res = np.sum((y_test - y_perm_pred) ** 2)
                ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
                perm_r2s.append(float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0)

            var_result = {
                "r2": r2,
                "pearson_r": pearson_r,
                "mse": mse,
                "mae": mae,
                "ci_95": [float(ci_low), float(ci_high)],
                "permutation_r2_mean": float(np.mean(perm_r2s)),
                "permutation_r2_std": float(np.std(perm_r2s)),
                "best_alpha": float(ridge.alpha_),
                "n_train": len(y_train),
                "n_test": len(y_test),
            }
            stage_results[var] = var_result
            print(f"    {var}: R²={r2:.4f} [{ci_low:.4f}, {ci_high:.4f}] | "
                  f"Perm R²={np.mean(perm_r2s):.4f}±{np.std(perm_r2s):.4f}")

        results[stage] = stage_results

    return results


# ===========================================================================
# Step 5: Spatial metrics
# ===========================================================================

def step5_spatial_metrics(
    activations: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Compute spatial metrics: Moran's I, spatial precision/recall/F1."""
    physics_labels = activations.get("physics_labels")
    hue_labels = activations.get("hue_labels")
    grid_h, grid_w = 14, 14
    n_patches = grid_h * grid_w

    results = {}

    for stage in STAGE_NAMES:
        if stage not in activations:
            continue

        X = activations[stage]
        stage_metrics = {}

        # Compute per-patch R² for spatial maps
        for var in ["mass", "hue"]:
            if var in PHYSICS_VARS and physics_labels is not None:
                col = PHYSICS_COL[var]
                y_full = physics_labels[:, col] if physics_labels.ndim == 2 else physics_labels
            elif var == "hue" and hue_labels is not None:
                y_full = hue_labels
            else:
                continue

            n_samples = min(X.shape[0], len(y_full))
            X_use = X[:n_samples]
            y_use = y_full[:n_samples]

            # Compute per-patch R² by grouping patches
            n_scenes = n_samples // n_patches
            if n_scenes < 10:
                continue

            X_scenes = X_use[: n_scenes * n_patches].reshape(n_scenes, n_patches, -1)
            y_scenes = y_use[: n_scenes * n_patches].reshape(n_scenes, n_patches)

            r2_per_patch = np.full(n_patches, np.nan)
            for p in range(n_patches):
                x_p = X_scenes[:, p, :]
                y_p = y_scenes[:, p]
                valid = ~np.isnan(y_p)
                if valid.sum() < 10:
                    continue
                scaler = StandardScaler()
                x_fit = scaler.fit_transform(x_p[valid])
                ridge = RidgeCV(alphas=ALPHA_CANDIDATES)
                try:
                    ridge.fit(x_fit, y_p[valid])
                    y_pred = ridge.predict(x_fit)
                    ss_res = np.sum((y_p[valid] - y_pred) ** 2)
                    ss_tot = np.sum((y_p[valid] - np.mean(y_p[valid])) ** 2)
                    r2_per_patch[p] = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                except Exception:
                    pass

            # Moran's I
            mi = morans_i(r2_per_patch, grid_h, grid_w)

            # Spatial precision/recall using mean object mask
            obj_mask = np.zeros(n_patches)
            y_mean = np.nanmean(y_scenes, axis=0)
            obj_mask[~np.isnan(y_mean) & (np.abs(y_mean) > 1e-6)] = 1.0
            sprf = spatial_precision_recall_f1(r2_per_patch, obj_mask.reshape(grid_h, grid_w))

            stage_metrics[var] = {
                "morans_i": mi,
                "r2_map": r2_per_patch.tolist(),
                "spatial_precision": sprf["precision"],
                "spatial_recall": sprf["recall"],
                "spatial_f1": sprf["f1"],
                "r2_map_mean": float(np.nanmean(r2_per_patch)),
                "r2_map_std": float(np.nanstd(r2_per_patch)),
            }
            print(f"    {stage}/{var}: Moran's I = {mi:.4f}, F1 = {sprf['f1']:.4f}")

        # Cross-property correlation
        if "mass" in stage_metrics and "hue" in stage_metrics:
            mass_map = np.array(stage_metrics["mass"]["r2_map"])
            hue_map = np.array(stage_metrics["hue"]["r2_map"])
            valid = ~(np.isnan(mass_map) | np.isnan(hue_map))
            if valid.sum() > 5:
                corr, pval = stats.pearsonr(mass_map[valid], hue_map[valid])
                stage_metrics["cross_property"] = {
                    "mass_hue_r2_correlation": float(corr),
                    "mass_hue_r2_pvalue": float(pval),
                }

        results[stage] = stage_metrics

    return results


# ===========================================================================
# Step 6: Differential degradation analysis
# ===========================================================================

def step6_differential_degradation(probe_results: Dict) -> Dict[str, Any]:
    """Analyze how physics vs appearance degrade through the projection layer."""
    analysis = {}

    # R² at each stage for mass and hue
    mass_r2 = {}
    hue_r2 = {}
    for stage in STAGE_NAMES:
        if stage in probe_results:
            if "mass" in probe_results[stage]:
                mass_r2[stage] = probe_results[stage]["mass"]["r2"]
            if "hue" in probe_results[stage]:
                hue_r2[stage] = probe_results[stage]["hue"]["r2"]

    analysis["mass_r2_by_stage"] = mass_r2
    analysis["hue_r2_by_stage"] = hue_r2

    # Key comparison: Stage 1 → Stage 2 (encoder → post-merger)
    s1, s2 = "stage_1_enc_out", "stage_2_post_proj"
    if s1 in mass_r2 and s2 in mass_r2 and s1 in hue_r2 and s2 in hue_r2:
        mass_s1, mass_s2 = mass_r2[s1], mass_r2[s2]
        hue_s1, hue_s2 = hue_r2[s1], hue_r2[s2]

        mass_retention = mass_s2 / mass_s1 if mass_s1 > 0 else 0
        hue_retention = hue_s2 / hue_s1 if hue_s1 > 0 else 0
        mass_drop = mass_s1 - mass_s2
        hue_drop = hue_s1 - hue_s2

        analysis["projection_bottleneck"] = {
            "mass_retention": float(mass_retention),
            "hue_retention": float(hue_retention),
            "mass_drop": float(mass_drop),
            "hue_drop": float(hue_drop),
            "differential_drop": float(mass_drop - hue_drop),
        }

        if mass_drop > hue_drop + 0.05:
            verdict = "PHYSICS-BLIND MERGER: mass degrades more than hue at projection"
        elif hue_drop > mass_drop + 0.05:
            verdict = "PHYSICS-AWARE MERGER: hue degrades more than mass at projection"
        else:
            verdict = "NEUTRAL MERGER: mass and hue degrade similarly at projection"

        analysis["projection_bottleneck"]["verdict"] = verdict
        print(f"\n  === PROJECTION BOTTLENECK ANALYSIS ===")
        print(f"  Mass: {mass_s1:.4f} → {mass_s2:.4f} (retention: {mass_retention:.2%})")
        print(f"  Hue:  {hue_s1:.4f} → {hue_s2:.4f} (retention: {hue_retention:.2%})")
        print(f"  Verdict: {verdict}")

    # Full degradation across all stages
    for var_name, var_r2 in [("mass", mass_r2), ("hue", hue_r2)]:
        stages_ordered = [s for s in STAGE_NAMES if s in var_r2]
        if len(stages_ordered) >= 2:
            total_drop = var_r2[stages_ordered[0]] - var_r2[stages_ordered[-1]]
            analysis[f"{var_name}_total_degradation"] = float(total_drop)

    return analysis


# ===========================================================================
# Step 7: Generate figures
# ===========================================================================

def step7_generate_figures(
    probe_results: Dict,
    spatial_results: Dict,
    degradation: Dict,
    output_dir: Path,
):
    """Generate publication-quality figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # --- Figure 1: Degradation curves with CI and permutation bands ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x_positions = list(range(len(STAGE_NAMES)))

    colors = {"mass": "#d62728", "friction": "#2ca02c", "elasticity": "#1f77b4",
              "stability": "#9467bd", "hue": "#ff7f0e"}
    linestyles = {"mass": "-", "friction": "-", "elasticity": "-",
                  "stability": "-", "hue": "--"}

    for var in PHYSICS_VARS + VISUAL_VARS:
        r2_vals = []
        ci_lows, ci_highs = [], []
        perm_means, perm_stds = [], []
        valid_x = []

        for idx, stage in enumerate(STAGE_NAMES):
            if stage in probe_results and var in probe_results[stage]:
                res = probe_results[stage][var]
                r2_vals.append(res["r2"])
                ci_lows.append(res["ci_95"][0])
                ci_highs.append(res["ci_95"][1])
                perm_means.append(res["permutation_r2_mean"])
                perm_stds.append(res["permutation_r2_std"])
                valid_x.append(idx)

        if not r2_vals:
            continue

        color = colors.get(var, "#333333")
        ls = linestyles.get(var, "-")
        ax.plot(valid_x, r2_vals, f"o{ls}", color=color, label=var, linewidth=2, markersize=8)
        ax.fill_between(valid_x, ci_lows, ci_highs, alpha=0.15, color=color)

        # Permutation band
        if perm_means:
            pm = np.array(perm_means)
            ps = np.array(perm_stds)
            ax.fill_between(valid_x, pm - 2 * ps, pm + 2 * ps,
                            alpha=0.08, color=color, hatch="//")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(STAGE_LABELS, fontsize=11)
    ax.set_ylabel("R² Score", fontsize=13)
    ax.set_title("Physics Probing: R² Across Pipeline Stages", fontsize=14)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    # Highlight projection bottleneck
    ax.axvspan(0.5, 1.5, alpha=0.05, color="red")
    ax.text(1.0, ax.get_ylim()[1] * 0.95, "Projection\nBottleneck",
            ha="center", va="top", fontsize=9, color="red", alpha=0.7)

    plt.tight_layout()
    fig.savefig(str(fig_dir / "degradation_curves_with_ci.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved: degradation_curves_with_ci.png")

    # --- Figure 2: Saliency maps (R² heatmaps) at all stages ---
    vars_to_plot = ["mass", "hue"]
    n_vars = len(vars_to_plot)
    n_stages = len(STAGE_NAMES)

    fig, axes = plt.subplots(n_vars, n_stages, figsize=(4 * n_stages, 4 * n_vars))
    if n_vars == 1:
        axes = axes[np.newaxis, :]

    for vi, var in enumerate(vars_to_plot):
        for si, stage in enumerate(STAGE_NAMES):
            ax = axes[vi, si]
            if (stage in spatial_results and
                    var in spatial_results[stage] and
                    "r2_map" in spatial_results[stage][var]):
                r2_map = np.array(spatial_results[stage][var]["r2_map"]).reshape(14, 14)
                r2_map = np.nan_to_num(r2_map, nan=0.0)
                im = ax.imshow(r2_map, cmap="hot", vmin=0, vmax=max(0.5, np.nanmax(r2_map)))
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                mi = spatial_results[stage][var].get("morans_i", 0)
                ax.set_title(f"{STAGE_LABELS[si]}\nMoran's I={mi:.3f}", fontsize=10)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(STAGE_LABELS[si], fontsize=10)

            if si == 0:
                ax.set_ylabel(var.upper(), fontsize=12, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Per-Patch R² Saliency Maps", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(str(fig_dir / "saliency_maps_all_stages.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: saliency_maps_all_stages.png")

    # --- Figure 3: Cross-property correlation scatter ---
    fig, axes = plt.subplots(1, n_stages, figsize=(4 * n_stages, 4))
    for si, stage in enumerate(STAGE_NAMES):
        ax = axes[si]
        if (stage in spatial_results and
                "mass" in spatial_results[stage] and
                "hue" in spatial_results[stage]):
            mass_map = np.array(spatial_results[stage]["mass"]["r2_map"])
            hue_map = np.array(spatial_results[stage]["hue"]["r2_map"])
            valid = ~(np.isnan(mass_map) | np.isnan(hue_map))
            if valid.sum() > 0:
                ax.scatter(mass_map[valid], hue_map[valid], alpha=0.5, s=20, c="steelblue")
                # Fit line
                if valid.sum() > 5:
                    z = np.polyfit(mass_map[valid], hue_map[valid], 1)
                    x_line = np.linspace(mass_map[valid].min(), mass_map[valid].max(), 50)
                    ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.7)
                    corr_info = spatial_results[stage].get("cross_property", {})
                    r_val = corr_info.get("mass_hue_r2_correlation", 0)
                    ax.text(0.05, 0.95, f"r={r_val:.3f}", transform=ax.transAxes,
                            fontsize=10, va="top")
        ax.set_xlabel("Mass R²", fontsize=10)
        ax.set_ylabel("Hue R²", fontsize=10)
        ax.set_title(STAGE_LABELS[si], fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Cross-Property Correlation: Mass vs Hue R² (per patch)", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(fig_dir / "cross_property_scatter.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: cross_property_scatter.png")

    # --- Figure 4: Differential degradation bar chart ---
    bottleneck = degradation.get("projection_bottleneck", {})
    if bottleneck:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        bars = ["Mass", "Hue"]
        drops = [bottleneck.get("mass_drop", 0), bottleneck.get("hue_drop", 0)]
        retentions = [bottleneck.get("mass_retention", 0), bottleneck.get("hue_retention", 0)]

        x = np.arange(len(bars))
        width = 0.35
        ax.bar(x - width / 2, drops, width, label="R² Drop", color=["#d62728", "#ff7f0e"])
        ax.bar(x + width / 2, retentions, width, label="R² Retention",
               color=["#d62728", "#ff7f0e"], alpha=0.5)

        ax.set_xlabel("Variable", fontsize=12)
        ax.set_ylabel("Value", fontsize=12)
        ax.set_title(f"Projection Bottleneck\n{bottleneck.get('verdict', '')}", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(bars)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        fig.savefig(str(fig_dir / "differential_degradation.png"), dpi=300)
        plt.close(fig)
        print(f"  Saved: differential_degradation.png")


# ===========================================================================
# Step 8: Save comprehensive results
# ===========================================================================

def step8_save_results(
    probe_results: Dict,
    spatial_results: Dict,
    degradation: Dict,
    metadata: Dict,
    output_dir: Path,
):
    """Save all results to a comprehensive JSON file."""
    comprehensive = {
        "experiment": "week2_day10-14_vlm_probing",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {k: v for k, v in metadata.items()
                     if not isinstance(v, (np.ndarray, np.generic))},
        "probe_results": probe_results,
        "spatial_metrics": {
            stage: {
                var: {k: v for k, v in metrics.items() if k != "r2_map"}
                for var, metrics in stage_data.items()
            }
            for stage, stage_data in spatial_results.items()
        },
        "differential_degradation": degradation,
        "summary": _make_summary(probe_results, degradation),
    }

    results_path = output_dir / "comprehensive_results.json"
    with open(str(results_path), "w") as f:
        json.dump(comprehensive, f, indent=2, default=str)
    print(f"  Saved comprehensive results to {results_path}")
    return comprehensive


def _make_summary(probe_results: Dict, degradation: Dict) -> Dict:
    """Create a human-readable summary of key findings."""
    summary = {"key_findings": []}

    # Best stage for mass
    best_mass_stage = None
    best_mass_r2 = -1
    for stage in STAGE_NAMES:
        if stage in probe_results and "mass" in probe_results[stage]:
            r2 = probe_results[stage]["mass"]["r2"]
            if r2 > best_mass_r2:
                best_mass_r2 = r2
                best_mass_stage = stage

    if best_mass_stage:
        summary["key_findings"].append(
            f"Mass best decoded at {best_mass_stage} (R²={best_mass_r2:.4f})"
        )

    # Projection verdict
    bottleneck = degradation.get("projection_bottleneck", {})
    if "verdict" in bottleneck:
        summary["key_findings"].append(bottleneck["verdict"])

    # Physics vs appearance comparison
    for stage in STAGE_NAMES:
        if stage in probe_results:
            mass_r2 = probe_results[stage].get("mass", {}).get("r2", None)
            hue_r2 = probe_results[stage].get("hue", {}).get("r2", None)
            if mass_r2 is not None and hue_r2 is not None:
                if mass_r2 > hue_r2 + 0.1:
                    summary["key_findings"].append(
                        f"{stage}: physics (R²={mass_r2:.3f}) > appearance (R²={hue_r2:.3f})"
                    )
                    break

    return summary


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    set_seed(args.seed)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.test_mode:
        output_dir = PROJECT_ROOT / "results" / "week2_test"
    else:
        output_dir = PROJECT_ROOT / "results" / "week2_qwen"
    output_dir.mkdir(parents=True, exist_ok=True)

    mode_str = "TEST MODE (ViT-base, CPU)" if args.test_mode else f"GPU MODE ({args.model})"
    print(f"\n{'#'*70}")
    print(f"  WEEK 2 DAY 10-14: VLM PHYSICS PROBING PIPELINE")
    print(f"  Mode: {mode_str}")
    print(f"  Scenes: {args.num_scenes}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*70}")

    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        report_vram()
    else:
        print("  No CUDA GPU available" + (" (expected for test mode)" if args.test_mode else ""))

    # Step 1: Generate data
    with Timer("Generate deconfounded data"):
        scene_data, metadata = step1_generate_data(args.num_scenes, output_dir, args.seed)

    # Step 2: Load model
    with Timer("Load model"):
        if args.test_mode:
            extractor, model_name = step2_load_model_test_mode()
            metadata["model"] = "vit-base-patch16-224 (test)"
        else:
            model, processor = step2_load_model_gpu(args.model)
            model_name = args.model
            metadata["model"] = MODEL_IDS[args.model]

    # Step 3: Extract activations
    with Timer("Extract activations"):
        if args.test_mode:
            activations = step3_extract_activations_test_mode(
                extractor, scene_data, output_dir, args.num_scenes
            )
        else:
            activations = step3_extract_activations_gpu(
                model, processor, scene_data, output_dir,
                args.num_scenes, args.batch_size, args.resume
            )
            # Free model memory
            del model, processor
            torch.cuda.empty_cache()
            gc.collect()

    # Step 4: Train probes
    with Timer("Train probes"):
        probe_results = step4_train_probes(activations)

    # Step 5: Spatial metrics
    with Timer("Compute spatial metrics"):
        spatial_results = step5_spatial_metrics(activations)

    # Step 6: Differential degradation
    with Timer("Differential degradation analysis"):
        degradation = step6_differential_degradation(probe_results)

    # Step 7: Figures
    with Timer("Generate figures"):
        step7_generate_figures(probe_results, spatial_results, degradation, output_dir)

    # Step 8: Save results
    with Timer("Save comprehensive results"):
        comprehensive = step8_save_results(
            probe_results, spatial_results, degradation, metadata, output_dir
        )

    # Final summary
    print(f"\n{'#'*70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'#'*70}")
    print(f"\n  Key findings:")
    for finding in comprehensive.get("summary", {}).get("key_findings", []):
        print(f"    - {finding}")
    print(f"\n  Results saved to: {output_dir}")
    print(f"  Total time: {sum(Timer._step_times.values()):.1f}s")


if __name__ == "__main__":
    main()
