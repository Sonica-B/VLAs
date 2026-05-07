#!/usr/bin/env python3
"""
Multi-model probing script for NeurIPS 2026 paper.
Extracts activations from 4 pipeline stages across 4 VLMs and trains
linear probes for physics properties (mass, friction, elasticity, stability).

Key design:
  - Auto-discovers architecture by printing model.named_modules() tree
  - Registers forward hooks at 4 stages: encoder, projection, LLM-8, LLM-16
  - Falls back to best-guess hook points if auto-discovery fails
  - Saves activations to HDF5 for reproducibility

Usage:
    # Single model
    python scripts/run_multi_model_probing.py --model qwen3-vl-8b --num-scenes 300

    # All models
    python scripts/run_multi_model_probing.py --model all --num-scenes 300

    # Print architecture tree only (no probing)
    python scripts/run_multi_model_probing.py --model internvl3-8b --print-arch-only
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.realistic_physion import RealisticPhysicsDataset, MATERIAL_NAMES
from src.data.patch_label_assigner import PatchLabelAssigner

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STAGE_NAMES = ["stage_1_enc_out", "stage_2_post_proj", "stage_3_llm_8", "stage_4_llm_16"]
STAGE_LABELS = ["Visual Encoder", "Post-Projection", "LLM Layer 8", "LLM Layer 16"]
PHYSICS_VARS = ["mass", "friction", "elasticity", "stability"]
PHYSICS_COL = {"mass": 0, "friction": 1, "elasticity": 2, "stability": 3}

ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]

# ---------------------------------------------------------------------------
# Model registry — best-guess hook paths per model
# These are starting points; the script auto-discovers actual paths.
# ---------------------------------------------------------------------------
HOOK_HINTS = {
    "qwen3-vl-8b": {
        "encoder_patterns": ["visual", "vision_model", "vit"],
        "projection_patterns": ["merger", "multi_modal_projector", "mlp_proj"],
        "llm_layer_prefix": "model.layers",
        "llm_early": 7,
        "llm_mid": 15,
    },
    "internvl3-8b": {
        "encoder_patterns": ["vision_model", "visual"],
        "projection_patterns": ["mlp1", "projector", "multi_modal_projector"],
        "llm_layer_prefix": "language_model.model.layers",
        "llm_early": 7,
        "llm_mid": 15,
    },
    "gemma3-12b": {
        "encoder_patterns": ["vision_tower", "vision_model"],
        "projection_patterns": ["multi_modal_projector", "projector"],
        "llm_layer_prefix": "language_model.model.layers",
        "llm_early": 7,
        "llm_mid": 15,
    },
    "glm-4.5v": {
        "encoder_patterns": ["vision", "visual", "vit", "encoder"],
        "projection_patterns": ["adapter", "projector", "mlp_proj", "connector"],
        "llm_layer_prefix": "transformer.encoder.layers",
        "llm_early": 7,
        "llm_mid": 15,
    },
}

# Model IDs (same as eval script)
MODEL_IDS = {
    "qwen3-vl-8b": "Qwen/Qwen3-VL-8B-Instruct",
    "internvl3-8b": "OpenGVLab/InternVL3-8B",
    "gemma3-12b": "google/gemma-3-12b-it",
    "glm-4.5v": "zai-org/GLM-4.5V",
}


# ---------------------------------------------------------------------------
# Architecture discovery
# ---------------------------------------------------------------------------

def print_architecture_tree(model, model_key: str, max_depth: int = 4):
    """Print the module tree to help identify hook points."""
    print(f"\n{'='*70}")
    print(f"ARCHITECTURE TREE: {model_key}")
    print(f"{'='*70}")

    for name, module in model.named_modules():
        depth = name.count(".")
        if depth > max_depth:
            continue
        # Skip individual parameters, show structure
        class_name = module.__class__.__name__
        indent = "  " * depth
        # Count children
        n_children = sum(1 for _ in module.children())
        n_params = sum(p.numel() for p in module.parameters(recurse=False))
        suffix = ""
        if n_params > 0:
            suffix = f" [{n_params/1e6:.1f}M params]"
        if n_children > 0:
            suffix += f" ({n_children} children)"
        print(f"{indent}{name or '(root)'}: {class_name}{suffix}")


def discover_hook_points(model, model_key: str) -> Dict[str, str]:
    """
    Auto-discover the 4 hook points by analyzing model.named_modules().
    Returns dict mapping stage names to module paths.
    """
    hints = HOOK_HINTS.get(model_key, HOOK_HINTS["qwen3-vl-8b"])
    all_names = [name for name, _ in model.named_modules()]

    hooks = {}

    # Stage 1: Vision encoder output — find the top-level vision module
    encoder_path = _find_module(all_names, hints["encoder_patterns"], prefer_toplevel=True)
    if encoder_path:
        hooks["stage_1_enc_out"] = encoder_path
        print(f"  Stage 1 (encoder):    {encoder_path}")

    # Stage 2: Post-projection — find merger/projector MLP
    proj_path = _find_module(all_names, hints["projection_patterns"], prefer_toplevel=True)
    if proj_path:
        hooks["stage_2_post_proj"] = proj_path
        print(f"  Stage 2 (projection): {proj_path}")

    # Stage 3 & 4: LLM layers
    llm_prefix = hints["llm_layer_prefix"]
    llm_early = hints["llm_early"]
    llm_mid = hints["llm_mid"]

    # Try exact match first, then fuzzy
    for stage_name, layer_idx in [("stage_3_llm_8", llm_early), ("stage_4_llm_16", llm_mid)]:
        exact = f"{llm_prefix}.{layer_idx}"
        if exact in all_names:
            hooks[stage_name] = exact
            print(f"  {stage_name}: {exact}")
        else:
            # Fuzzy: find any module matching "layers.{idx}" pattern
            pattern = re.compile(rf".*layers?\.{layer_idx}$")
            matches = [n for n in all_names if pattern.match(n)]
            if matches:
                hooks[stage_name] = matches[0]
                print(f"  {stage_name}: {matches[0]} (fuzzy)")
            else:
                # Try nearby layers
                for offset in [0, -1, 1, -2, 2]:
                    alt_pattern = re.compile(rf".*layers?\.{layer_idx + offset}$")
                    alt_matches = [n for n in all_names if alt_pattern.match(n)]
                    if alt_matches:
                        hooks[stage_name] = alt_matches[0]
                        print(f"  {stage_name}: {alt_matches[0]} (nearby layer {layer_idx + offset})")
                        break

    # Report missing hooks
    for stage in STAGE_NAMES:
        if stage not in hooks:
            print(f"  WARNING: Could not find hook for {stage}")

    return hooks


def _find_module(all_names: list, patterns: list, prefer_toplevel: bool = True) -> Optional[str]:
    """Find a module name matching one of the patterns."""
    candidates = []
    for pattern in patterns:
        for name in all_names:
            parts = name.split(".")
            if pattern in parts or pattern in name:
                candidates.append(name)

    if not candidates:
        return None

    if prefer_toplevel:
        # Prefer shorter (higher-level) module paths
        candidates.sort(key=lambda x: x.count("."))

    return candidates[0]


# ---------------------------------------------------------------------------
# Activation extraction with forward hooks
# ---------------------------------------------------------------------------

class MultiModelActivationExtractor:
    """Registers hooks on discovered modules and captures outputs."""

    def __init__(self, model, hook_points: Dict[str, str]):
        self.model = model
        self.hook_points = hook_points
        self.activations: Dict[str, torch.Tensor] = {}
        self._handles = []

    def _make_hook(self, stage_name: str):
        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, torch.Tensor):
                self.activations[stage_name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                # Many transformer layers return (hidden_states, ...) tuples
                tensor = output[0] if isinstance(output[0], torch.Tensor) else None
                if tensor is not None:
                    self.activations[stage_name] = tensor.detach().cpu()
            elif hasattr(output, "last_hidden_state"):
                self.activations[stage_name] = output.last_hidden_state.detach().cpu()
        return hook_fn

    def register_hooks(self):
        """Attach forward hooks to all discovered modules."""
        for stage_name, module_path in self.hook_points.items():
            module = dict(self.model.named_modules()).get(module_path)
            if module is not None:
                handle = module.register_forward_hook(self._make_hook(stage_name))
                self._handles.append(handle)
            else:
                print(f"  WARNING: Module '{module_path}' not found, skipping hook for {stage_name}")

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def clear(self):
        self.activations.clear()


# ---------------------------------------------------------------------------
# Model loading (reuse from eval script where possible)
# ---------------------------------------------------------------------------

def load_model_for_probing(model_key: str, quantize: str = "4bit"):
    """Load model with quantization for probing (activation extraction)."""
    # Import the loaders from the eval script
    from scripts.run_multi_model_eval import (
        load_model, get_available_vram, choose_quantization, MODELS,
    )

    if quantize == "auto":
        quantize = choose_quantization(model_key)

    if quantize == "skip":
        vram = get_available_vram()
        print(f"  SKIPPING {model_key}: needs {MODELS[model_key]['vram_fp16']}GB, have {vram:.1f}GB")
        return None, None, "skip"

    model, processor = load_model(model_key, quantize)
    return model, processor, quantize


# ---------------------------------------------------------------------------
# Data generation (reuse Physion++ realistic data)
# ---------------------------------------------------------------------------

def generate_scenes(num_scenes: int, seed: int = 42):
    """Generate realistic physics scenes for probing."""
    print(f"\nGenerating {num_scenes} realistic material-based scenes...")
    dataset = RealisticPhysicsDataset(n_scenes=num_scenes, image_size=448, seed=seed)

    images, all_labels, all_masks = [], [], []
    for i in range(len(dataset)):
        sample = dataset[i]
        images.append(sample.image)
        all_labels.append(sample.physics_labels)
        all_masks.append(sample.object_masks)
        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{num_scenes}")

    return {"images": images, "labels": all_labels, "masks": all_masks}


# ---------------------------------------------------------------------------
# Forward pass to extract activations
# ---------------------------------------------------------------------------

def extract_activations(
    model_key: str,
    model,
    processor,
    extractor: MultiModelActivationExtractor,
    scene_data: dict,
    output_dir: Path,
    num_scenes: int,
):
    """Run forward passes and save activations to HDF5."""
    cache_path = output_dir / "activations" / f"{model_key}_activations.h5"

    if cache_path.exists():
        print(f"  Loading cached activations from {cache_path}")
        return _load_hdf5(cache_path)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    assigner = PatchLabelAssigner(patch_grid_size=14)

    all_acts = {s: [] for s in STAGE_NAMES}
    all_patch_labels = []

    images = scene_data["images"]
    labels_list = scene_data["labels"]
    masks_list = scene_data["masks"]

    extractor.register_hooks()

    for i in range(min(num_scenes, len(images))):
        img = images[i]
        if isinstance(img, np.ndarray):
            from PIL import Image as PILImage
            img = PILImage.fromarray(img.astype(np.uint8))

        # Build a simple prompt with the image
        _run_forward_pass(model_key, model, processor, img)

        # Collect activations
        for stage in STAGE_NAMES:
            if stage in extractor.activations:
                act = extractor.activations[stage]
                # Flatten to [N_tokens, D]
                if act.dim() == 3:
                    act = act.squeeze(0)  # remove batch
                elif act.dim() == 1:
                    act = act.unsqueeze(0)
                all_acts[stage].append(act.float().numpy())

        # Assign patch labels
        physics = labels_list[i]
        mask = masks_list[i]
        if mask is not None and physics:
            patch_labels = assigner.assign(mask, physics)
            if isinstance(patch_labels, np.ndarray):
                all_patch_labels.append(patch_labels)

        extractor.clear()

        if (i + 1) % 50 == 0:
            print(f"  [{model_key}] Extracted {i+1}/{num_scenes} scenes")

    extractor.remove_hooks()

    # Save to HDF5
    _save_hdf5(cache_path, all_acts, all_patch_labels)
    print(f"  Saved activations to {cache_path}")

    return all_acts, all_patch_labels


def _run_forward_pass(model_key: str, model, processor, image):
    """Run a single forward pass with a dummy prompt to trigger hooks."""
    from PIL import Image

    prompt = "Describe this image briefly."

    if model_key == "qwen3-vl-8b":
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image, "max_pixels": 448 * 448, "min_pixels": 28 * 28},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
    elif model_key == "internvl3-8b":
        # InternVL3: use model.chat for a single forward or tokenizer
        if hasattr(processor, "encode"):
            # processor is a tokenizer
            inputs = processor(f"<image>\n{prompt}", return_tensors="pt").to(model.device)
        else:
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    else:
        # Gemma3 / GLM-4.5V: use processor with chat template
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        try:
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
        except Exception:
            # Fallback: direct processor call
            inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Just forward pass, no generation needed for activation extraction
        try:
            model(**inputs)
        except Exception:
            # Some models need generate() to trigger all hooks
            model.generate(**inputs, max_new_tokens=1, do_sample=False)

    del inputs
    torch.cuda.empty_cache()


def _save_hdf5(path: Path, all_acts: dict, all_patch_labels: list):
    """Save activations and labels to HDF5."""
    with h5py.File(str(path), "w") as f:
        for stage, act_list in all_acts.items():
            if act_list:
                # Global-pool each scene: mean across tokens → [D]
                pooled = []
                for act in act_list:
                    if act.ndim == 2:
                        pooled.append(act.mean(axis=0))
                    else:
                        pooled.append(act)
                f.create_dataset(f"{stage}_global", data=np.stack(pooled))
                # Also save raw (variable-length) — store first scene shape for reference
                f.attrs[f"{stage}_token_dim"] = act_list[0].shape[-1] if act_list else 0

        if all_patch_labels:
            # Store per-scene mean physics (for global probing)
            scene_physics = []
            for pl in all_patch_labels:
                if pl.ndim == 2:
                    # Mean over non-NaN patches
                    with np.errstate(all="ignore"):
                        scene_mean = np.nanmean(pl, axis=0)
                    scene_physics.append(scene_mean)
            if scene_physics:
                f.create_dataset("scene_physics", data=np.stack(scene_physics))


def _load_hdf5(path: Path):
    """Load cached activations from HDF5."""
    all_acts = {s: [] for s in STAGE_NAMES}
    all_patch_labels = []

    with h5py.File(str(path), "r") as f:
        for stage in STAGE_NAMES:
            key = f"{stage}_global"
            if key in f:
                data = f[key][:]
                # Convert stacked array back to list of per-scene vectors
                all_acts[stage] = [data[i] for i in range(data.shape[0])]

        if "scene_physics" in f:
            physics = f["scene_physics"][:]
            # Store as list of arrays for compatibility
            all_patch_labels = [physics]

    return all_acts, all_patch_labels


# ---------------------------------------------------------------------------
# Linear probing
# ---------------------------------------------------------------------------

def run_probing(
    model_key: str,
    all_acts: dict,
    all_patch_labels: list,
    output_dir: Path,
) -> Dict[str, Any]:
    """Train linear probes at each stage for each physics variable."""
    print(f"\n{'='*60}")
    print(f"PROBING: {model_key}")
    print(f"{'='*60}")

    # Build global-pooled features [N_scenes, D] and labels [N_scenes, 4]
    # Labels from scene_physics (already global-pooled)
    if len(all_patch_labels) == 1 and all_patch_labels[0].ndim == 2:
        labels = all_patch_labels[0]  # [N_scenes, 4]
    elif all_patch_labels:
        labels = np.stack([np.nanmean(pl, axis=0) if pl.ndim == 2 else pl for pl in all_patch_labels])
    else:
        print("  No labels available, skipping probing")
        return {}

    results = {}

    for stage in STAGE_NAMES:
        acts = all_acts.get(stage, [])
        if not acts:
            print(f"  {stage}: no activations, skipping")
            continue

        X = np.stack(acts) if isinstance(acts, list) else acts  # [N, D]
        n_samples = min(X.shape[0], labels.shape[0])
        X = X[:n_samples]
        y_all = labels[:n_samples]

        print(f"\n  {stage}: X={X.shape}, labels={y_all.shape}")

        stage_results = {}
        for var in PHYSICS_VARS:
            col = PHYSICS_COL[var]
            if col >= y_all.shape[1]:
                continue
            y = y_all[:, col]

            # Remove NaN
            valid = ~np.isnan(y)
            if valid.sum() < 20:
                print(f"    {var}: too few valid samples ({valid.sum()})")
                continue

            X_v = X[valid]
            y_v = y[valid]

            # Standardize
            scaler = StandardScaler()
            X_v = scaler.fit_transform(X_v)

            # Ridge regression with cross-validation
            ridge = RidgeCV(alphas=ALPHA_CANDIDATES, cv=5)
            ridge.fit(X_v, y_v)
            r2 = ridge.score(X_v, y_v)

            # Cross-validated R²
            cv_scores = cross_val_score(
                RidgeCV(alphas=ALPHA_CANDIDATES),
                X_v, y_v, cv=5, scoring="r2",
            )
            cv_r2 = cv_scores.mean()

            stage_results[var] = {
                "r2_train": round(float(r2), 4),
                "r2_cv": round(float(cv_r2), 4),
                "r2_cv_std": round(float(cv_scores.std()), 4),
                "n_samples": int(valid.sum()),
                "best_alpha": float(ridge.alpha_),
            }
            print(f"    {var}: R²(train)={r2:.4f}, R²(CV)={cv_r2:.4f} ± {cv_scores.std():.4f}")

        results[stage] = stage_results

    # Compute degradation curve (mass)
    if all(s in results and "mass" in results[s] for s in STAGE_NAMES):
        mass_curve = [results[s]["mass"]["r2_cv"] for s in STAGE_NAMES]
        degradation = mass_curve[0] - mass_curve[-1]
        results["degradation_mass"] = {
            "curve": mass_curve,
            "encoder_r2": mass_curve[0],
            "llm16_r2": mass_curve[-1],
            "total_drop": round(degradation, 4),
        }
        print(f"\n  Mass degradation: {mass_curve[0]:.4f} → {mass_curve[-1]:.4f} (drop={degradation:.4f})")

    # Save
    metrics_dir = output_dir / "probe_metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{model_key}_probe_results.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Probe results saved to: {metrics_path}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-model probing for physics understanding")
    parser.add_argument("--model", default="all",
                        choices=list(MODEL_IDS.keys()) + ["all"])
    parser.add_argument("--num-scenes", type=int, default=300)
    parser.add_argument("--quantize", default="auto", choices=["none", "4bit", "8bit", "auto"])
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "multi_model_probing"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-arch-only", action="store_true",
                        help="Print architecture tree and exit (no probing)")
    args = parser.parse_args()

    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "all":
        model_keys = list(MODEL_IDS.keys())
    else:
        model_keys = [args.model]

    # Generate scenes once (shared across models)
    if not args.print_arch_only:
        scene_data = generate_scenes(args.num_scenes, seed=args.seed)

    all_results = {}

    for model_key in model_keys:
        print(f"\n{'#'*70}")
        print(f"# MODEL: {model_key} ({MODEL_IDS[model_key]})")
        print(f"{'#'*70}")

        try:
            model, processor, quant = load_model_for_probing(model_key, args.quantize)
            if quant == "skip":
                all_results[model_key] = {"status": "skipped"}
                continue

            # Print architecture tree
            print_architecture_tree(model, model_key)

            if args.print_arch_only:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                del model, processor
                continue

            # Discover hook points
            print(f"\nDiscovering hook points for {model_key}...")
            hook_points = discover_hook_points(model, model_key)

            if len(hook_points) < 2:
                print(f"  WARNING: Only found {len(hook_points)} hooks, results may be incomplete")

            # Extract activations
            extractor = MultiModelActivationExtractor(model, hook_points)
            all_acts, all_labels = extract_activations(
                model_key, model, processor, extractor,
                scene_data, output_dir, args.num_scenes,
            )

            # Run probing
            results = run_probing(model_key, all_acts, all_labels, output_dir)
            all_results[model_key] = results

            # Cleanup
            extractor.remove_hooks()
            del model, processor, extractor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(2)

        except Exception as e:
            print(f"\n*** FAILED {model_key}: {e} ***")
            import traceback
            traceback.print_exc()
            all_results[model_key] = {"status": "failed", "error": str(e)}
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if args.print_arch_only:
        return

    # Print comparison
    print(f"\n{'='*70}")
    print("MULTI-MODEL PROBING COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<18} {'Enc R²':>8} {'Proj R²':>8} {'LLM-8 R²':>9} {'LLM-16 R²':>10} {'Drop':>7}")
    print("-" * 65)

    for key in model_keys:
        r = all_results.get(key, {})
        if "degradation_mass" in r:
            d = r["degradation_mass"]
            curve = d["curve"]
            print(f"{key:<18} {curve[0]:>8.4f} {curve[1]:>8.4f} {curve[2]:>9.4f} {curve[3]:>10.4f} {d['total_drop']:>7.4f}")
        elif "status" in r:
            print(f"{key:<18} {'— ' + r.get('status', 'unknown'):>50}")
        else:
            # Partial results
            vals = []
            for s in STAGE_NAMES:
                if s in r and "mass" in r[s]:
                    vals.append(f"{r[s]['mass']['r2_cv']:.4f}")
                else:
                    vals.append("  —")
            print(f"{key:<18} {'  '.join(vals)}")

    # Save combined
    combined_path = output_dir / "multi_model_probe_summary.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nCombined probe results: {combined_path}")


if __name__ == "__main__":
    main()
