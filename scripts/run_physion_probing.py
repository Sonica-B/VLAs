#!/usr/bin/env python3
"""
Physion++ real-data probing: realistic materials + global pooling + alternative probes.

This script runs the comprehensive probing experiments that go beyond the
deconfounded synthetic shapes. Key additions over run_full_week2_gpu.py:

1. REALISTIC MATERIAL DATA: Objects with material-specific appearances (metal,
   wood, rubber, foam) where physics correlates with material type — the way
   VLMs learn physics from web data.

2. GLOBAL POOLING: Mean-pool all patch activations per scene → single vector →
   probe for physics. This replicates the Pixels-to-Principles methodology and
   should show higher R² than per-patch probing.

3. BINARY CLASSIFICATION: Heavy vs light (above/below median mass) instead of
   continuous regression — easier task, higher signal-to-noise.

4. MLP PROBES: Non-linear probes on global-pooled representations.

5. MATERIAL CLASSIFICATION: Probe for material type as a visual control.

Usage:
    # Full run with Qwen2.5-VL-7B:
    python scripts/run_physion_probing.py --model qwen --num-scenes 300

    # Test mode (ViT-base on CPU):
    python scripts/run_physion_probing.py --test-mode --num-scenes 100

    # Resume from cached activations:
    python scripts/run_physion_probing.py --model qwen --resume
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
from sklearn.linear_model import RidgeCV, LogisticRegressionCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.realistic_physion import (
    RealisticPhysicsDataset,
    MATERIAL_NAMES,
    MATERIAL_TO_IDX,
)
from src.data.patch_label_assigner import PatchLabelAssigner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAGE_NAMES = [
    "stage_1_enc_out",
    "stage_2_post_proj",
    "stage_3_llm_8",
    "stage_4_llm_16",
]
STAGE_LABELS = ["Visual Encoder", "Post-Merger", "LLM Layer 8", "LLM Layer 16"]
PHYSICS_VARS = ["mass", "friction", "elasticity"]
PHYSICS_COL = {"mass": 0, "friction": 1, "elasticity": 2, "stability": 3}

MODEL_IDS = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen-4bit": "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
}

ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]
N_PERMUTATIONS = 20
N_BOOTSTRAP = 500


def parse_args():
    p = argparse.ArgumentParser(description="Physion++ real-data probing")
    p.add_argument("--model", default="qwen", choices=["qwen", "qwen-4bit"])
    p.add_argument("--num-scenes", type=int, default=300)
    p.add_argument("--test-mode", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ===========================================================================
# Utilities
# ===========================================================================

class Timer:
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
              f"ETA remaining: ~{remaining:.0f}s")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def report_vram():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {allocated:.2f}GB / {reserved:.2f}GB / {total:.1f}GB")


# ===========================================================================
# Step 1: Generate realistic material-based scenes
# ===========================================================================

def step1_generate_data(num_scenes: int, seed: int) -> Dict[str, Any]:
    """Generate realistic physics scenes with material-correlated properties."""
    print(f"  Generating {num_scenes} realistic material-based scenes...")
    dataset = RealisticPhysicsDataset(
        n_scenes=num_scenes, image_size=224, seed=seed
    )

    images, all_labels, all_masks, all_materials = [], [], [], []

    for i in range(len(dataset)):
        sample = dataset[i]
        extra = dataset.get_extra_info(i)
        images.append(sample.image)
        all_labels.append(sample.physics_labels)
        all_masks.append(sample.object_masks)
        all_materials.append(extra)

        if (i + 1) % 100 == 0:
            print(f"    Generated {i+1}/{num_scenes} scenes")

    # Verify material-physics correlation
    corr = dataset.get_material_physics_correlation()
    print(f"  Material-physics correlations:")
    print(f"    material-mass r = {corr['material_mass_r']:.4f}")
    print(f"    material-friction r = {corr['material_friction_r']:.4f}")
    print(f"    material-elasticity r = {corr['material_elasticity_r']:.4f}")
    print(f"    Total objects: {corr['n_objects_total']}")

    return {
        "images": images,
        "labels": all_labels,
        "masks": all_masks,
        "materials": all_materials,
        "correlation_stats": corr,
    }


# ===========================================================================
# Step 2: Load model (reuses run_full_week2_gpu logic)
# ===========================================================================

def step2_load_model_test():
    from src.models.activation_extractor import LightweightViTExtractor
    print("  Loading ViT-base (test mode, CPU)...")
    extractor = LightweightViTExtractor()
    extractor.load()
    print("  ViT-base loaded")
    return extractor, None, "vit_base"


def step2_load_model_gpu(model_key: str):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"  Loading model: {MODEL_IDS[model_key]}")
    report_vram()

    try:
        from transformers import BitsAndBytesConfig
        print("  Attempting 4-bit quantization...")
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
            MODEL_IDS["qwen"], trust_remote_code=True,
        )
        print("  Model loaded with 4-bit quantization!")
    except Exception as e:
        print(f"  4-bit failed: {e}")
        print("  Attempting float16...")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_IDS["qwen"],
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        processor = AutoProcessor.from_pretrained(
            MODEL_IDS["qwen"], trust_remote_code=True,
        )

    report_vram()
    model.eval()
    return model, processor, "qwen"


# ===========================================================================
# Step 3: Extract activations
# ===========================================================================

def step3_extract_activations_test(
    extractor, scene_data: Dict, output_dir: Path, num_scenes: int
) -> Dict[str, Any]:
    """Extract activations with ViT-base (test mode)."""
    cache_path = output_dir / "activations" / "realistic_test_activations.h5"

    if cache_path.exists():
        print(f"  Loading cached activations from {cache_path}")
        return _load_hdf5(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    assigner = PatchLabelAssigner(patch_grid_size=14)

    all_act = {s: [] for s in STAGE_NAMES}
    all_patch_labels = []
    all_material_labels = []
    # Per-scene global labels (for global pooling)
    scene_physics = []  # list of per-scene mean physics
    scene_materials = []  # list of per-scene material names

    images = scene_data["images"]
    labels_list = scene_data["labels"]
    masks_list = scene_data["masks"]
    materials_list = scene_data["materials"]

    for i in range(min(num_scenes, len(images))):
        img = images[i]
        if isinstance(img, np.ndarray):
            from PIL import Image as PILImage
            img = PILImage.fromarray(img.astype(np.uint8))

        acts = extractor.extract(img)
        for stage in STAGE_NAMES:
            if stage in acts:
                act_np = acts[stage].numpy() if isinstance(acts[stage], torch.Tensor) else acts[stage]
                all_act[stage].append(act_np)

        # Per-patch physics labels
        physics = labels_list[i]
        mask = masks_list[i]
        if mask is not None and physics:
            patch_labels = assigner.assign(mask, physics)
            if isinstance(patch_labels, np.ndarray):
                if patch_labels.ndim == 2 and patch_labels.shape == (196, 4):
                    all_patch_labels.append(patch_labels)
                elif patch_labels.ndim == 1:
                    all_patch_labels.append(patch_labels.reshape(196, -1))
                else:
                    # Ensure 196x4
                    pl = np.full((196, 4), np.nan, dtype=np.float32)
                    pl[:min(196, patch_labels.shape[0]), :min(4, patch_labels.shape[-1])] = \
                        patch_labels[:min(196, patch_labels.shape[0]), :min(4, patch_labels.shape[-1])]
                    all_patch_labels.append(pl)
            elif isinstance(patch_labels, dict):
                combined = np.stack(
                    [patch_labels.get(v, np.full(196, np.nan)) for v in
                     ["mass", "friction", "elasticity", "stability"]],
                    axis=-1,
                )
                all_patch_labels.append(combined)
            else:
                all_patch_labels.append(np.full((196, 4), np.nan))
        else:
            all_patch_labels.append(np.full((196, 4), np.nan))

        # Per-patch material labels (manual assignment since PatchLabelAssigner
        # only knows standard physics properties)
        mat_info = materials_list[i]
        mat_idx = mat_info["material_idx"]  # [n_objects]
        mat_patch = _assign_material_to_patches(mask, mat_idx, patch_grid=14)
        all_material_labels.append(mat_patch)

        # Per-scene global physics (mean over objects)
        scene_physics.append({
            k: float(np.mean(v)) for k, v in physics.items()
        })
        scene_materials.append(mat_info["material_names"])

        if (i + 1) % 50 == 0:
            print(f"    Extracted {i+1}/{num_scenes}", flush=True)

    # Stack
    result = {}
    for stage in STAGE_NAMES:
        if all_act[stage]:
            result[stage] = np.stack(all_act[stage], axis=0)  # [N_scenes, 196, D]
            print(f"    {stage}: {result[stage].shape}", flush=True)

    result["physics_labels"] = np.stack(all_patch_labels, axis=0)  # [N_scenes, 196, 4]
    result["material_labels"] = np.stack(all_material_labels, axis=0)  # [N_scenes, 196]
    result["scene_physics"] = scene_physics
    result["scene_materials"] = scene_materials

    _save_hdf5(result, cache_path)
    return result


def step3_extract_activations_gpu(
    model, processor, scene_data: Dict, output_dir: Path,
    num_scenes: int, resume: bool
) -> Dict[str, Any]:
    """Extract activations from Qwen2.5-VL-7B."""
    from PIL import Image as PILImage

    cache_path = output_dir / "activations" / "realistic_qwen_activations.h5"
    if resume and cache_path.exists():
        print(f"  Resuming from cached activations at {cache_path}")
        return _load_hdf5(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    assigner = PatchLabelAssigner(patch_grid_size=14)

    # Register hooks
    hook_storage = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hook_storage[name] = output[0].detach().cpu()
            elif isinstance(output, torch.Tensor):
                hook_storage[name] = output.detach().cpu()
        return hook_fn

    visual = getattr(model, 'visual', None) or getattr(model.model, 'visual', None)
    try:
        hooks.append(visual.blocks[-1].register_forward_hook(make_hook("stage_1_enc_out")))
        print(f"  Hooked: visual.blocks[-1] ({len(visual.blocks)} blocks)")
    except Exception as e:
        print(f"  WARNING: encoder hook failed: {e}")

    try:
        hooks.append(visual.merger.register_forward_hook(make_hook("stage_2_post_proj")))
        print("  Hooked: visual.merger")
    except Exception as e:
        print(f"  WARNING: merger hook failed: {e}")

    llm_layers = getattr(model.model, 'layers', None) or \
                 getattr(getattr(model.model, 'language_model', None), 'layers', None)
    try:
        hooks.append(llm_layers[8].register_forward_hook(make_hook("stage_3_llm_8")))
        hooks.append(llm_layers[16].register_forward_hook(make_hook("stage_4_llm_16")))
        print(f"  Hooked: LLM layers 8, 16 (of {len(llm_layers)})")
    except Exception as e:
        print(f"  WARNING: LLM hooks failed: {e}")

    all_act = {s: [] for s in STAGE_NAMES}
    all_patch_labels = []
    all_material_labels = []
    scene_physics = []
    scene_materials = []

    images = scene_data["images"]
    labels_list = scene_data["labels"]
    masks_list = scene_data["masks"]
    materials_list = scene_data["materials"]
    device = next(model.parameters()).device
    print(f"  Device: {device}")
    print(f"  Extracting activations from {min(num_scenes, len(images))} scenes...")

    for i in range(min(num_scenes, len(images))):
        hook_storage.clear()

        img = images[i]
        if isinstance(img, np.ndarray):
            img_pil = PILImage.fromarray(img.astype(np.uint8))
        else:
            img_pil = img

        # Prepare Qwen2.5-VL input
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_pil},
                {"type": "text", "text": "Describe the objects."},
            ],
        }]

        try:
            from qwen_vl_utils import process_vision_info
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                               padding=True, return_tensors="pt")
        except ImportError:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[img_pil], return_tensors="pt", padding=True)

        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            try:
                outputs = model(**inputs, output_hidden_states=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM at scene {i}, clearing cache...")
                torch.cuda.empty_cache()
                gc.collect()
                try:
                    outputs = model(**inputs, output_hidden_states=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"  Skipping scene {i}")
                    continue

        # Get visual token count
        image_grid_thw = inputs.get("image_grid_thw", None)
        n_visual_tokens = int(image_grid_thw[0].prod().item()) if image_grid_thw is not None else None

        for stage_name in STAGE_NAMES:
            if stage_name not in hook_storage:
                continue
            act = hook_storage[stage_name]
            if act.ndim == 3:
                act = act.squeeze(0)

            if stage_name in ("stage_1_enc_out", "stage_2_post_proj"):
                visual_act = act.float().numpy()
            else:
                if n_visual_tokens is not None and act.shape[0] > n_visual_tokens:
                    input_ids = inputs.get("input_ids", None)
                    if input_ids is not None:
                        img_token_id = 151655
                        ids = input_ids[0].cpu()
                        img_positions = (ids == img_token_id).nonzero(as_tuple=True)[0]
                        if len(img_positions) >= n_visual_tokens:
                            visual_act = act[img_positions[:n_visual_tokens]].float().numpy()
                        else:
                            visual_act = act[:n_visual_tokens].float().numpy()
                    else:
                        visual_act = act[:n_visual_tokens].float().numpy()
                else:
                    visual_act = act.float().numpy()

            all_act[stage_name].append(visual_act)

        # Physics + material labels
        physics = labels_list[i]
        mask = masks_list[i]
        if mask is not None and physics:
            patch_labels = assigner.assign(mask, physics)
            if isinstance(patch_labels, dict):
                combined = np.stack(
                    [patch_labels.get(v, np.full(196, np.nan))
                     for v in ["mass", "friction", "elasticity", "stability"]],
                    axis=-1,
                )
                all_patch_labels.append(combined)
            elif isinstance(patch_labels, np.ndarray):
                all_patch_labels.append(
                    patch_labels if patch_labels.ndim == 2 else patch_labels.reshape(-1, 1)
                )
        else:
            all_patch_labels.append(np.full((196, 4), np.nan))

        # Material labels
        mat_info = materials_list[i]
        mat_idx = mat_info["material_idx"]
        mat_patch = _assign_material_to_patches(mask, mat_idx, patch_grid=14)
        all_material_labels.append(mat_patch)

        scene_physics.append({k: float(np.mean(v)) for k, v in physics.items()})
        scene_materials.append(mat_info["material_names"])

        if (i + 1) % 10 == 0:
            print(f"    Extracted {i+1}/{num_scenes}")
            report_vram()

        del outputs, inputs
        torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    # Stack — keep as [N_scenes, N_patches, D] for global pooling
    result = {}
    for stage in STAGE_NAMES:
        if all_act[stage]:
            # Each entry may have different n_patches for Qwen, so we need to handle that
            # For global pooling, we'll mean-pool each scene separately
            shapes = [a.shape for a in all_act[stage]]
            min_patches = min(s[0] for s in shapes)
            # Truncate to min_patches for stacking
            truncated = [a[:min_patches] for a in all_act[stage]]
            stacked = np.stack(truncated, axis=0)  # [N_scenes, min_patches, D]
            result[stage] = stacked
            print(f"    {stage}: {stacked.shape}")

    if all_patch_labels:
        result["physics_labels"] = np.stack(
            [pl[:196] if len(pl) >= 196 else np.pad(pl, ((0, 196 - len(pl)), (0, 0)),
                                                      constant_values=np.nan)
             for pl in all_patch_labels], axis=0
        )  # [N_scenes, 196, 4]
    if all_material_labels:
        result["material_labels"] = np.stack(all_material_labels, axis=0)  # [N_scenes, 196]

    result["scene_physics"] = scene_physics
    result["scene_materials"] = scene_materials

    _save_hdf5(result, cache_path)
    return result


def _assign_material_to_patches(
    mask: np.ndarray, mat_idx: np.ndarray, patch_grid: int = 14
) -> np.ndarray:
    """Assign material index to each patch position based on segmentation mask."""
    mat_patch = np.full(patch_grid * patch_grid, np.nan, dtype=np.float32)
    if mask is None:
        return mat_patch
    H, W = mask.shape
    patch_h = H // patch_grid
    patch_w = W // patch_grid
    for pi in range(patch_grid):
        for pj in range(patch_grid):
            r0, r1 = pi * patch_h, (pi + 1) * patch_h
            c0, c1 = pj * patch_w, (pj + 1) * patch_w
            patch_region = mask[r0:r1, c0:c1]
            obj_ids = patch_region[patch_region > 0]
            if len(obj_ids) > 0:
                counts = np.bincount(obj_ids)
                dominant = counts.argmax()
                if dominant > 0 and dominant <= len(mat_idx):
                    mat_patch[pi * patch_grid + pj] = float(mat_idx[dominant - 1])
    return mat_patch


def _save_hdf5(data: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w") as f:
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                f.create_dataset(key, data=val, compression="gzip", compression_opts=4)
            elif isinstance(val, list) and len(val) > 0:
                try:
                    # Try to save as JSON string
                    f.attrs[key] = json.dumps(val, default=str)
                except Exception:
                    pass


def _load_hdf5(path: Path) -> Dict:
    result = {}
    with h5py.File(str(path), "r") as f:
        for key in f.keys():
            result[key] = f[key][:]
        for key in f.attrs:
            try:
                result[key] = json.loads(f.attrs[key])
            except Exception:
                result[key] = f.attrs[key]
    return result


# ===========================================================================
# Step 4: Per-patch probing (standard — matches run_full_week2_gpu.py)
# ===========================================================================

def step4_per_patch_probing(data: Dict) -> Dict[str, Any]:
    """Standard per-patch probing with Ridge regression."""
    results = {}

    for stage in STAGE_NAMES:
        if stage not in data:
            continue

        X = data[stage]  # [N_scenes, N_patches, D] or [N*N_patches, D]
        stage_results = {}

        # Flatten to [N_total_patches, D]
        if X.ndim == 3:
            N_scenes, N_patches, D = X.shape
            X_flat = X.reshape(-1, D)
        else:
            X_flat = X
            N_patches = 196

        physics_labels = data.get("physics_labels")  # [N_scenes, 196, 4] or [N*196, 4]

        for var_idx, var in enumerate(PHYSICS_VARS):
            col = PHYSICS_COL[var]
            if physics_labels is not None:
                if physics_labels.ndim == 3:
                    y = physics_labels[:, :, col].flatten()
                elif physics_labels.ndim == 2:
                    y = physics_labels[:, col]
                else:
                    continue
            else:
                continue

            n_samples = min(X_flat.shape[0], len(y))
            X_var = X_flat[:n_samples]
            y_var = y[:n_samples]

            valid = ~np.isnan(y_var)
            if valid.sum() < 30:
                print(f"    {var}: too few valid samples ({valid.sum()})")
                continue
            X_v, y_v = X_var[valid], y_var[valid]

            X_train, X_test, y_train, y_test = train_test_split(
                X_v, y_v, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)

            ridge = RidgeCV(alphas=ALPHA_CANDIDATES)
            ridge.fit(X_train_sc, y_train)
            y_pred = ridge.predict(X_test_sc)

            ss_res = np.sum((y_test - y_pred) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            # Permutation baseline
            rng = np.random.RandomState(42)
            perm_r2s = []
            for _ in range(N_PERMUTATIONS):
                y_perm = rng.permutation(y_train)
                ridge_p = RidgeCV(alphas=ALPHA_CANDIDATES)
                ridge_p.fit(X_train_sc, y_perm)
                yp = ridge_p.predict(X_test_sc)
                ss_r = np.sum((y_test - yp) ** 2)
                perm_r2s.append(float(1 - ss_r / ss_tot) if ss_tot > 0 else 0.0)

            # Bootstrap CI
            boot_r2s = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.choice(len(y_test), size=len(y_test), replace=True)
                ss_r = np.sum((y_test[idx] - y_pred[idx]) ** 2)
                ss_t = np.sum((y_test[idx] - np.mean(y_test[idx])) ** 2)
                if ss_t > 0:
                    boot_r2s.append(1 - ss_r / ss_t)
            ci_low, ci_high = np.percentile(boot_r2s, [2.5, 97.5]) if boot_r2s else (0, 0)

            stage_results[var] = {
                "r2": r2,
                "ci_95": [float(ci_low), float(ci_high)],
                "permutation_r2_mean": float(np.mean(perm_r2s)),
                "permutation_r2_std": float(np.std(perm_r2s)),
                "n_train": len(y_train),
                "n_test": len(y_test),
            }
            print(f"    {stage}/{var} (per-patch): R²={r2:.4f} [{ci_low:.4f}, {ci_high:.4f}] "
                  f"| Perm={np.mean(perm_r2s):.4f}")

        results[stage] = stage_results

    return results


# ===========================================================================
# Step 5: GLOBAL POOLING probing (Pixels-to-Principles methodology)
# ===========================================================================

def step5_global_pooling_probes(data: Dict) -> Dict[str, Any]:
    """Mean-pool all patches per scene → single vector → probe.

    This replicates the Pixels-to-Principles methodology. Instead of probing
    per-patch, we aggregate the entire scene representation.
    """
    results = {}
    scene_physics = data.get("scene_physics", [])

    if not scene_physics:
        print("  WARNING: No scene_physics data, computing from patch labels...")
        physics_labels = data.get("physics_labels")
        if physics_labels is not None and physics_labels.ndim == 3:
            N_scenes = physics_labels.shape[0]
            scene_physics = []
            for i in range(N_scenes):
                scene_dict = {}
                for var_idx, var in enumerate(["mass", "friction", "elasticity", "stability"]):
                    vals = physics_labels[i, :, var_idx]
                    valid_vals = vals[~np.isnan(vals)]
                    scene_dict[var] = float(np.mean(valid_vals)) if len(valid_vals) > 0 else np.nan
                scene_physics.append(scene_dict)

    for stage in STAGE_NAMES:
        if stage not in data:
            continue

        X = data[stage]  # [N_scenes, N_patches, D]
        if X.ndim != 3:
            print(f"    {stage}: skipping global pooling (not 3D: {X.shape})")
            continue

        # Global mean pooling: [N_scenes, D]
        X_global = np.mean(X, axis=1)
        N_scenes = X_global.shape[0]
        print(f"\n  {stage}: global-pooled shape = {X_global.shape}")

        stage_results = {}

        for var in PHYSICS_VARS:
            y = np.array([sp.get(var, np.nan) for sp in scene_physics[:N_scenes]])
            valid = ~np.isnan(y)
            if valid.sum() < 20:
                continue

            X_v, y_v = X_global[valid], y[valid]
            X_train, X_test, y_train, y_test = train_test_split(
                X_v, y_v, test_size=0.2, random_state=42
            )

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)

            # Linear (Ridge)
            ridge = RidgeCV(alphas=ALPHA_CANDIDATES)
            ridge.fit(X_tr, y_train)
            y_pred_linear = ridge.predict(X_te)
            ss_res = np.sum((y_test - y_pred_linear) ** 2)
            ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
            r2_linear = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

            # MLP probe
            r2_mlp = _train_mlp_probe(X_tr, y_train, X_te, y_test)

            # Permutation baseline
            rng = np.random.RandomState(42)
            perm_r2s = []
            for _ in range(N_PERMUTATIONS):
                y_perm = rng.permutation(y_train)
                ridge_p = RidgeCV(alphas=ALPHA_CANDIDATES)
                ridge_p.fit(X_tr, y_perm)
                yp = ridge_p.predict(X_te)
                ss_r = np.sum((y_test - yp) ** 2)
                perm_r2s.append(float(1 - ss_r / ss_tot) if ss_tot > 0 else 0.0)

            # Bootstrap CI
            boot_r2s = []
            for _ in range(N_BOOTSTRAP):
                idx = rng.choice(len(y_test), size=len(y_test), replace=True)
                ss_r = np.sum((y_test[idx] - y_pred_linear[idx]) ** 2)
                ss_t = np.sum((y_test[idx] - np.mean(y_test[idx])) ** 2)
                if ss_t > 0:
                    boot_r2s.append(1 - ss_r / ss_t)
            ci_low, ci_high = np.percentile(boot_r2s, [2.5, 97.5]) if boot_r2s else (0, 0)

            stage_results[var] = {
                "r2_linear": r2_linear,
                "r2_mlp": r2_mlp,
                "ci_95_linear": [float(ci_low), float(ci_high)],
                "permutation_r2_mean": float(np.mean(perm_r2s)),
                "permutation_r2_std": float(np.std(perm_r2s)),
                "n_train": len(y_train),
                "n_test": len(y_test),
            }
            print(f"    {var} (global): Linear R²={r2_linear:.4f} | MLP R²={r2_mlp:.4f} "
                  f"| Perm={np.mean(perm_r2s):.4f}")

        results[stage] = stage_results

    return results


def _train_mlp_probe(X_train, y_train, X_test, y_test) -> float:
    """Train a single-layer MLP probe and return R²."""
    try:
        from src.probing.mlp_probe import MLPProbe
        D = X_train.shape[1]
        probe = MLPProbe(
            input_dim=D,
            hidden_dim=min(512, D),
            dropout=0.1,
            epochs=50,
            batch_size=min(256, len(X_train)),
            early_stopping_patience=10,
            normalize_features=False,  # already scaled
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        # Split train for validation
        n_val = max(1, len(X_train) // 5)
        probe.fit(
            X_train[n_val:], y_train[n_val:],
            X_val=X_train[:n_val], y_val=y_train[:n_val],
        )
        metrics = probe.score(X_test, y_test)
        return float(metrics["r2"])
    except Exception as e:
        print(f"      MLP probe failed: {e}")
        return 0.0


# ===========================================================================
# Step 6: Binary classification (heavy vs light)
# ===========================================================================

def step6_binary_classification(data: Dict) -> Dict[str, Any]:
    """Binary classification: above/below median for each physics property."""
    results = {}
    scene_physics = data.get("scene_physics", [])

    if not scene_physics:
        print("  No scene_physics for binary classification, computing...")
        physics_labels = data.get("physics_labels")
        if physics_labels is not None and physics_labels.ndim == 3:
            N_scenes = physics_labels.shape[0]
            scene_physics = []
            for i in range(N_scenes):
                d = {}
                for vi, var in enumerate(["mass", "friction", "elasticity", "stability"]):
                    vals = physics_labels[i, :, vi]
                    valid_vals = vals[~np.isnan(vals)]
                    d[var] = float(np.mean(valid_vals)) if len(valid_vals) > 0 else np.nan
                scene_physics.append(d)

    for stage in STAGE_NAMES:
        if stage not in data:
            continue

        X = data[stage]
        if X.ndim != 3:
            continue

        X_global = np.mean(X, axis=1)  # [N_scenes, D]
        N_scenes = X_global.shape[0]
        stage_results = {}

        for var in PHYSICS_VARS:
            y_cont = np.array([sp.get(var, np.nan) for sp in scene_physics[:N_scenes]])
            valid = ~np.isnan(y_cont)
            if valid.sum() < 20:
                continue

            X_v, y_cont_v = X_global[valid], y_cont[valid]
            median = np.median(y_cont_v)
            y_binary = (y_cont_v > median).astype(np.int32)

            # Check class balance
            n_pos = y_binary.sum()
            n_neg = len(y_binary) - n_pos
            if min(n_pos, n_neg) < 5:
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X_v, y_binary, test_size=0.2, random_state=42, stratify=y_binary
            )

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_train)
            X_te = scaler.transform(X_test)

            # Logistic regression
            clf = LogisticRegressionCV(
                Cs=[0.01, 0.1, 1.0, 10.0, 100.0],
                cv=3,
                max_iter=1000,
                random_state=42,
            )
            clf.fit(X_tr, y_train)
            y_pred = clf.predict(X_te)
            y_prob = clf.predict_proba(X_te)[:, 1]

            acc = float(accuracy_score(y_test, y_pred))
            try:
                auc = float(roc_auc_score(y_test, y_prob))
            except ValueError:
                auc = 0.5

            # Permutation baseline
            rng = np.random.RandomState(42)
            perm_accs = []
            for _ in range(N_PERMUTATIONS):
                y_perm = rng.permutation(y_train)
                try:
                    clf_p = LogisticRegressionCV(Cs=[0.1, 1.0, 10.0], cv=3, max_iter=500, random_state=42)
                    clf_p.fit(X_tr, y_perm)
                    perm_accs.append(float(accuracy_score(y_test, clf_p.predict(X_te))))
                except Exception:
                    perm_accs.append(0.5)

            stage_results[var] = {
                "accuracy": acc,
                "auc": auc,
                "permutation_acc_mean": float(np.mean(perm_accs)),
                "permutation_acc_std": float(np.std(perm_accs)),
                "median_threshold": float(median),
                "n_pos": int(n_pos),
                "n_neg": int(n_neg),
                "n_train": len(y_train),
                "n_test": len(y_test),
            }
            print(f"    {stage}/{var} (binary): Acc={acc:.4f} AUC={auc:.4f} | "
                  f"Perm={np.mean(perm_accs):.4f}")

        results[stage] = stage_results

    return results


# ===========================================================================
# Step 7: Material classification (visual control)
# ===========================================================================

def step7_material_classification(data: Dict) -> Dict[str, Any]:
    """Probe for material type — should be high if model encodes materials."""
    results = {}

    for stage in STAGE_NAMES:
        if stage not in data:
            continue

        X = data[stage]
        material_labels = data.get("material_labels")

        if material_labels is None:
            continue

        # Flatten for per-patch classification
        if X.ndim == 3:
            X_flat = X.reshape(-1, X.shape[-1])
        else:
            X_flat = X

        if material_labels.ndim == 2:
            y_flat = material_labels.flatten()
        else:
            y_flat = material_labels

        n_samples = min(X_flat.shape[0], len(y_flat))
        X_v = X_flat[:n_samples]
        y_v = y_flat[:n_samples]

        valid = ~np.isnan(y_v)
        if valid.sum() < 50:
            continue

        X_v, y_v = X_v[valid], y_v[valid].astype(np.int32)

        # Subsample if too many samples (speed)
        if len(y_v) > 10000:
            idx = np.random.RandomState(42).choice(len(y_v), 10000, replace=False)
            X_v, y_v = X_v[idx], y_v[idx]

        X_train, X_test, y_train, y_test = train_test_split(
            X_v, y_v, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        )
        clf.fit(X_tr, y_train)
        acc = float(accuracy_score(y_test, clf.predict(X_te)))

        # Chance level
        n_classes = len(np.unique(y_train))
        chance = 1.0 / max(n_classes, 1)

        results[stage] = {
            "material_accuracy": acc,
            "chance_level": chance,
            "n_classes": n_classes,
            "n_train": len(y_train),
            "n_test": len(y_test),
        }
        print(f"    {stage}: material acc={acc:.4f} (chance={chance:.4f}, {n_classes} classes)")

    return results


# ===========================================================================
# Step 8: Generate figures
# ===========================================================================

def step8_generate_figures(
    per_patch_results: Dict,
    global_results: Dict,
    binary_results: Dict,
    material_results: Dict,
    output_dir: Path,
):
    """Generate comprehensive visualization figures."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ----- Figure 1: Per-patch vs Global pooling R² across stages -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, var in enumerate(PHYSICS_VARS):
        ax = axes[ax_idx]
        per_patch_r2s = []
        global_linear_r2s = []
        global_mlp_r2s = []
        stage_labels_used = []

        for si, stage in enumerate(STAGE_NAMES):
            if stage in per_patch_results and var in per_patch_results[stage]:
                per_patch_r2s.append(per_patch_results[stage][var]["r2"])
            else:
                per_patch_r2s.append(np.nan)

            if stage in global_results and var in global_results[stage]:
                global_linear_r2s.append(global_results[stage][var]["r2_linear"])
                global_mlp_r2s.append(global_results[stage][var]["r2_mlp"])
            else:
                global_linear_r2s.append(np.nan)
                global_mlp_r2s.append(np.nan)

            stage_labels_used.append(STAGE_LABELS[si])

        x = np.arange(len(STAGE_NAMES))
        w = 0.25
        ax.bar(x - w, per_patch_r2s, w, label="Per-patch Ridge", color="#2196F3", alpha=0.8)
        ax.bar(x, global_linear_r2s, w, label="Global Ridge", color="#4CAF50", alpha=0.8)
        ax.bar(x + w, global_mlp_r2s, w, label="Global MLP", color="#FF9800", alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
        ax.set_title(var.capitalize(), fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(stage_labels_used, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("R²")
        if ax_idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle("Per-Patch vs Global Pooling: Physics Probing R² by Stage",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "per_patch_vs_global_r2.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ----- Figure 2: Binary classification accuracy across stages -----
    fig, ax = plt.subplots(figsize=(10, 5))
    for var in PHYSICS_VARS:
        accs = []
        for stage in STAGE_NAMES:
            if stage in binary_results and var in binary_results[stage]:
                accs.append(binary_results[stage][var]["accuracy"])
            else:
                accs.append(np.nan)
        ax.plot(STAGE_LABELS, accs, "o-", label=var.capitalize(), linewidth=2, markersize=8)

    ax.axhline(y=0.5, color="gray", linewidth=1, linestyle="--", label="Chance (50%)")
    ax.set_ylabel("Binary Classification Accuracy")
    ax.set_title("Heavy vs Light Classification Across Pipeline Stages", fontweight="bold")
    ax.legend()
    ax.set_ylim(0.3, 1.0)
    plt.tight_layout()
    plt.savefig(fig_dir / "binary_classification.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ----- Figure 3: Material classification accuracy -----
    fig, ax = plt.subplots(figsize=(8, 5))
    mat_accs = []
    chance_levels = []
    for stage in STAGE_NAMES:
        if stage in material_results:
            mat_accs.append(material_results[stage]["material_accuracy"])
            chance_levels.append(material_results[stage]["chance_level"])
        else:
            mat_accs.append(np.nan)
            chance_levels.append(np.nan)

    ax.bar(STAGE_LABELS, mat_accs, color="#9C27B0", alpha=0.8)
    if chance_levels:
        ax.axhline(y=np.nanmean(chance_levels), color="gray", linestyle="--",
                    label=f"Chance ({np.nanmean(chance_levels):.2f})")
    ax.set_ylabel("Material Classification Accuracy")
    ax.set_title("Material Type Classification (Visual Control)", fontweight="bold")
    ax.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(fig_dir / "material_classification.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ----- Figure 4: Comprehensive summary heatmap -----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 4a: Global R² heatmap
    r2_matrix = np.full((len(PHYSICS_VARS), len(STAGE_NAMES)), np.nan)
    for vi, var in enumerate(PHYSICS_VARS):
        for si, stage in enumerate(STAGE_NAMES):
            if stage in global_results and var in global_results[stage]:
                r2_matrix[vi, si] = global_results[stage][var]["r2_linear"]

    im = axes[0].imshow(r2_matrix, cmap="RdYlGn", aspect="auto", vmin=-0.5, vmax=0.5)
    axes[0].set_xticks(range(len(STAGE_NAMES)))
    axes[0].set_xticklabels(STAGE_LABELS, rotation=30, ha="right", fontsize=9)
    axes[0].set_yticks(range(len(PHYSICS_VARS)))
    axes[0].set_yticklabels([v.capitalize() for v in PHYSICS_VARS])
    axes[0].set_title("Global Pooling R² (Linear Probe)", fontweight="bold")
    for vi in range(len(PHYSICS_VARS)):
        for si in range(len(STAGE_NAMES)):
            val = r2_matrix[vi, si]
            if not np.isnan(val):
                axes[0].text(si, vi, f"{val:.3f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=axes[0])

    # 4b: Binary accuracy heatmap
    acc_matrix = np.full((len(PHYSICS_VARS), len(STAGE_NAMES)), np.nan)
    for vi, var in enumerate(PHYSICS_VARS):
        for si, stage in enumerate(STAGE_NAMES):
            if stage in binary_results and var in binary_results[stage]:
                acc_matrix[vi, si] = binary_results[stage][var]["accuracy"]

    im2 = axes[1].imshow(acc_matrix, cmap="RdYlGn", aspect="auto", vmin=0.3, vmax=0.9)
    axes[1].set_xticks(range(len(STAGE_NAMES)))
    axes[1].set_xticklabels(STAGE_LABELS, rotation=30, ha="right", fontsize=9)
    axes[1].set_yticks(range(len(PHYSICS_VARS)))
    axes[1].set_yticklabels([v.capitalize() for v in PHYSICS_VARS])
    axes[1].set_title("Binary Classification Accuracy", fontweight="bold")
    for vi in range(len(PHYSICS_VARS)):
        for si in range(len(STAGE_NAMES)):
            val = acc_matrix[vi, si]
            if not np.isnan(val):
                axes[1].text(si, vi, f"{val:.3f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im2, ax=axes[1])

    plt.suptitle("Comprehensive Physics Probing Results (Realistic Materials)",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(fig_dir / "comprehensive_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved figures to {fig_dir}")


# ===========================================================================
# Step 9: Save results
# ===========================================================================

def step9_save_results(
    output_dir: Path,
    per_patch_results: Dict,
    global_results: Dict,
    binary_results: Dict,
    material_results: Dict,
    correlation_stats: Dict,
    total_time: float,
):
    """Save comprehensive results JSON."""
    comprehensive = {
        "experiment": "realistic_material_probing",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_seconds": total_time,
        "data": {
            "type": "realistic_materials",
            "material_physics_correlations": correlation_stats,
        },
        "per_patch_probing": per_patch_results,
        "global_pooling_probing": global_results,
        "binary_classification": binary_results,
        "material_classification": material_results,
        "summary": _compute_summary(per_patch_results, global_results, binary_results, material_results),
    }

    results_path = output_dir / "comprehensive_results.json"
    with open(results_path, "w") as f:
        json.dump(comprehensive, f, indent=2, default=_json_default)

    print(f"\n  Results saved to {results_path}")
    _print_summary(comprehensive["summary"])

    return comprehensive


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _compute_summary(per_patch, global_pool, binary, material):
    """Compute summary statistics for results table."""
    summary = {
        "per_patch": {},
        "global_linear": {},
        "global_mlp": {},
        "binary_acc": {},
        "material": {},
    }

    for stage in STAGE_NAMES:
        if stage in per_patch:
            for var in PHYSICS_VARS:
                if var in per_patch[stage]:
                    key = f"{stage}/{var}"
                    summary["per_patch"][key] = per_patch[stage][var]["r2"]

        if stage in global_pool:
            for var in PHYSICS_VARS:
                if var in global_pool[stage]:
                    key = f"{stage}/{var}"
                    summary["global_linear"][key] = global_pool[stage][var]["r2_linear"]
                    summary["global_mlp"][key] = global_pool[stage][var]["r2_mlp"]

        if stage in binary:
            for var in PHYSICS_VARS:
                if var in binary[stage]:
                    key = f"{stage}/{var}"
                    summary["binary_acc"][key] = binary[stage][var]["accuracy"]

        if stage in material:
            summary["material"][stage] = material[stage]["material_accuracy"]

    return summary


def _print_summary(summary):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("  COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 80)

    # Print per-method table
    print(f"\n  {'Stage/Property':<35} {'Per-Patch':>10} {'Global Lin':>12} {'Global MLP':>12} {'Binary Acc':>12}")
    print(f"  {'-'*35} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

    for stage in STAGE_NAMES:
        for var in PHYSICS_VARS:
            key = f"{stage}/{var}"
            pp = summary["per_patch"].get(key, None)
            gl = summary["global_linear"].get(key, None)
            gm = summary["global_mlp"].get(key, None)
            ba = summary["binary_acc"].get(key, None)

            pp_s = f"{pp:.4f}" if pp is not None else "N/A"
            gl_s = f"{gl:.4f}" if gl is not None else "N/A"
            gm_s = f"{gm:.4f}" if gm is not None else "N/A"
            ba_s = f"{ba:.4f}" if ba is not None else "N/A"

            label = f"{STAGE_LABELS[STAGE_NAMES.index(stage)]}/{var}"
            print(f"  {label:<35} {pp_s:>10} {gl_s:>12} {gm_s:>12} {ba_s:>12}")

    print(f"\n  Material Classification Accuracy by Stage:")
    for stage in STAGE_NAMES:
        acc = summary["material"].get(stage, None)
        if acc is not None:
            label = STAGE_LABELS[STAGE_NAMES.index(stage)]
            print(f"    {label}: {acc:.4f}")

    print("=" * 80)


# ===========================================================================
# Main pipeline
# ===========================================================================

def main():
    args = parse_args()
    set_seed(args.seed)
    Timer._step_times.clear()

    total_steps = 8
    output_dir = Path(args.output_dir) if args.output_dir else (
        PROJECT_ROOT / "results" / "week2_physion"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("  PHYSION++ REAL-DATA PROBING EXPERIMENT")
    print(f"  Mode: {'TEST (ViT-base, CPU)' if args.test_mode else f'GPU ({args.model})'}")
    print(f"  Scenes: {args.num_scenes}")
    print(f"  Output: {output_dir}")
    print("#" * 70)

    overall_start = time.time()

    # Step 1: Generate data
    with Timer("Generate realistic material-based scenes", total_steps):
        scene_data = step1_generate_data(args.num_scenes, args.seed)
        correlation_stats = scene_data["correlation_stats"]

    # Step 2: Load model
    with Timer("Load model", total_steps):
        if args.test_mode:
            model_or_extractor, processor, model_name = step2_load_model_test()
        else:
            model_or_extractor, processor, model_name = step2_load_model_gpu(args.model)

    # Step 3: Extract activations
    with Timer("Extract activations", total_steps):
        if args.test_mode:
            data = step3_extract_activations_test(
                model_or_extractor, scene_data, output_dir, args.num_scenes
            )
        else:
            data = step3_extract_activations_gpu(
                model_or_extractor, processor, scene_data, output_dir,
                args.num_scenes, args.resume
            )

    # Free model memory for probe training
    if not args.test_mode:
        del model_or_extractor, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("  Model freed, VRAM available for probe training")

    # Step 4: Per-patch probing
    with Timer("Per-patch probing (standard)", total_steps):
        per_patch_results = step4_per_patch_probing(data)

    # Step 5: Global pooling probes
    with Timer("Global pooling probes (Pixels-to-Principles)", total_steps):
        global_results = step5_global_pooling_probes(data)

    # Step 6: Binary classification
    with Timer("Binary classification (heavy vs light)", total_steps):
        binary_results = step6_binary_classification(data)

    # Step 7: Material classification
    with Timer("Material classification (visual control)", total_steps):
        material_results = step7_material_classification(data)

    # Step 8: Figures + results
    total_time = time.time() - overall_start
    with Timer("Generate figures and save results", total_steps):
        step8_generate_figures(
            per_patch_results, global_results, binary_results,
            material_results, output_dir
        )
        step9_save_results(
            output_dir, per_patch_results, global_results,
            binary_results, material_results, correlation_stats, total_time
        )

    print(f"\n  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  Results at: {output_dir}")


if __name__ == "__main__":
    main()
