#!/usr/bin/env python3
"""
Week 2 LoRA Ablation: Train 5 LoRA conditions and measure physics probing changes.

Conditions:
  A: Encoder-only LoRA (rank 16, last 6 ViT blocks Q/V)
  B: Projection-only LoRA (rank 64, projection MLP)
  C: LLM-only LoRA (rank 16, first 8 LLM layers Q/V)
  D: Encoder+Projection combined
  E: Full model baseline

Usage:
    # Run all conditions sequentially (overnight):
    python scripts/run_week2_lora_ablation.py --num-scenes 500

    # Run a specific condition:
    python scripts/run_week2_lora_ablation.py --condition A --num-scenes 500

    # Resume from checkpoint:
    python scripts/run_week2_lora_ablation.py --resume --num-scenes 500

    # Test mode (ViT-base, CPU):
    python scripts/run_week2_lora_ablation.py --test-mode --num-scenes 100
"""

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.deconfounded_physion import DeconfoundedPhysicsDataset

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
PHYSICS_VARS = ["mass", "friction", "elasticity", "stability"]
PHYSICS_COL = {"mass": 0, "friction": 1, "elasticity": 2, "stability": 3}
ALPHA_CANDIDATES = [0.01, 0.1, 1.0, 10.0, 100.0]

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_ID_4BIT = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"


@dataclass
class LoRACondition:
    """Configuration for one LoRA ablation condition."""
    name: str
    label: str
    rank: int
    target_modules: List[str]
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    description: str = ""


# LoRA condition definitions for Qwen2.5-VL-7B
LORA_CONDITIONS = {
    "A": LoRACondition(
        name="A",
        label="Encoder-only",
        rank=16,
        target_modules=[
            # Last 6 ViT blocks, Q and V projections
            f"visual.blocks.{i}.attn.{proj}"
            for i in range(26, 32)
            for proj in ["qkv"]  # Qwen2.5-VL uses fused QKV
        ],
        description="LoRA on last 6 ViT encoder blocks (Q/K/V)",
    ),
    "B": LoRACondition(
        name="B",
        label="Projection-only",
        rank=64,
        target_modules=[
            "visual.merger.mlp.0",
            "visual.merger.mlp.2",
        ],
        description="LoRA on visual merger/projection MLP layers",
    ),
    "C": LoRACondition(
        name="C",
        label="LLM-only",
        rank=16,
        target_modules=[
            f"model.layers.{i}.self_attn.{proj}"
            for i in range(8)
            for proj in ["q_proj", "v_proj"]
        ],
        description="LoRA on first 8 LLM layers (Q/V projections)",
    ),
    "D": LoRACondition(
        name="D",
        label="Encoder+Projection",
        rank=16,
        target_modules=[
            # Encoder (last 6 blocks)
            *[f"visual.blocks.{i}.attn.qkv" for i in range(26, 32)],
            # Projection
            "visual.merger.mlp.0",
            "visual.merger.mlp.2",
        ],
        description="LoRA on encoder (last 6 blocks) + projection MLP",
    ),
    "E": LoRACondition(
        name="E",
        label="Full model",
        rank=16,
        target_modules=[
            # Encoder
            *[f"visual.blocks.{i}.attn.qkv" for i in range(26, 32)],
            # Projection
            "visual.merger.mlp.0",
            "visual.merger.mlp.2",
            # LLM (first 8 layers)
            *[f"model.layers.{i}.self_attn.{p}" for i in range(8) for p in ["q_proj", "v_proj"]],
        ],
        description="LoRA on encoder + projection + LLM (first 8 layers)",
    ),
}


def parse_args():
    p = argparse.ArgumentParser(description="Week 2 LoRA ablation experiments")
    p.add_argument(
        "--condition",
        type=str,
        default=None,
        choices=["A", "B", "C", "D", "E"],
        help="Run a specific condition (default: run all)",
    )
    p.add_argument("--num-scenes", type=int, default=500)
    p.add_argument("--num-epochs", type=int, default=3, help="LoRA fine-tuning epochs")
    p.add_argument("--lora-lr", type=float, default=2e-4, help="LoRA learning rate")
    p.add_argument("--lora-batch-size", type=int, default=2, help="Batch size for LoRA training")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--test-mode", action="store_true", help="Use ViT-base on CPU")
    p.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def report_vram():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  VRAM: {alloc:.2f}GB / {total:.1f}GB")


# ===========================================================================
# Data preparation
# ===========================================================================

def prepare_qa_data(num_scenes: int, output_dir: Path, seed: int) -> List[Dict]:
    """Generate physics QA training data from deconfounded scenes."""
    qa_path = output_dir / "lora_qa_data.json"
    if qa_path.exists():
        print(f"  Loading cached QA data from {qa_path}")
        with open(qa_path, "r") as f:
            return json.load(f)

    print(f"  Generating QA data from {num_scenes} deconfounded scenes...")
    dataset = DeconfoundedPhysicsDataset(n_scenes=num_scenes, image_size=224, seed=seed)

    qa_pairs = []
    for i in range(len(dataset)):
        sample = dataset[i]
        physics = sample.physics_labels

        # Generate QA pairs for each physics property
        mass_vals = physics.get("mass", [])
        if isinstance(mass_vals, np.ndarray):
            mass_vals = mass_vals.tolist()

        # Mass comparison QA
        if len(mass_vals) >= 2:
            obj_masses = [(j, m) for j, m in enumerate(mass_vals) if not np.isnan(m)]
            if len(obj_masses) >= 2:
                sorted_objs = sorted(obj_masses, key=lambda x: x[1], reverse=True)
                heavy_idx, heavy_mass = sorted_objs[0]
                light_idx, light_mass = sorted_objs[-1]
                if heavy_mass > light_mass * 1.2:  # Ensure meaningful difference
                    qa_pairs.append({
                        "scene_idx": i,
                        "question": "Which object in this scene is heavier?",
                        "answer": f"Object {heavy_idx + 1} is heavier with mass {heavy_mass:.2f}, "
                                  f"compared to object {light_idx + 1} with mass {light_mass:.2f}.",
                        "property": "mass",
                    })

        # Stability QA
        stability_vals = physics.get("stability", [])
        if isinstance(stability_vals, np.ndarray):
            stability_vals = stability_vals.tolist()
        if stability_vals:
            mean_stab = float(np.nanmean(stability_vals))
            is_stable = mean_stab >= 0.1
            qa_pairs.append({
                "scene_idx": i,
                "question": "Will this arrangement of objects remain stable?",
                "answer": "Yes, this arrangement appears stable." if is_stable
                         else "No, this arrangement is likely unstable and objects may fall or slide.",
                "property": "stability",
            })

        if (i + 1) % 200 == 0:
            print(f"    Processed {i+1}/{num_scenes} scenes")

    print(f"  Generated {len(qa_pairs)} QA pairs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(qa_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)
    return qa_pairs


# ===========================================================================
# Model loading
# ===========================================================================

def load_base_model():
    """Load Qwen2.5-VL-7B in 4-bit."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig

    print("  Loading Qwen2.5-VL-7B (4-bit)...")

    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID_4BIT,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    except Exception:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    report_vram()
    return model, processor


def apply_lora(model, condition: LoRACondition) -> Tuple[Any, int]:
    """Apply LoRA adapters to the model for the given condition."""
    from peft import LoraConfig, get_peft_model, TaskType

    # Find which target modules actually exist in the model
    valid_targets = []
    all_module_names = {name for name, _ in model.named_modules()}

    for target in condition.target_modules:
        # Check if target exists as substring of any module name
        matches = [n for n in all_module_names if target in n]
        if matches:
            valid_targets.append(target)
        else:
            print(f"    WARNING: Target module '{target}' not found, skipping")

    if not valid_targets:
        print(f"  ERROR: No valid target modules for condition {condition.name}")
        return model, 0

    lora_config = LoraConfig(
        r=condition.rank,
        lora_alpha=condition.lora_alpha,
        lora_dropout=condition.lora_dropout,
        target_modules=valid_targets,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    peft_model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  Condition {condition.name} ({condition.label}):")
    print(f"    Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    print(f"    Target modules: {valid_targets}")

    return peft_model, trainable


# ===========================================================================
# Activation extraction (reused from run_full_week2_gpu.py)
# ===========================================================================

def extract_activations(model, processor, scenes, num_scenes: int) -> Dict[str, np.ndarray]:
    """Extract activations at 4 pipeline stages (abbreviated version)."""
    from PIL import Image as PILImage
    from src.data.patch_label_assigner import PatchLabelAssigner

    assigner = PatchLabelAssigner(patch_grid_size=14)
    hook_storage = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hook_storage[name] = output[0].detach().cpu()
            elif isinstance(output, torch.Tensor):
                hook_storage[name] = output.detach().cpu()
        return hook_fn

    # Register hooks
    hook_targets = [
        ("stage_1_enc_out", "visual.blocks[-1]"),
        ("stage_2_post_proj", "visual.merger"),
        ("stage_3_llm_8", "model.layers[8]"),
        ("stage_4_llm_16", "model.layers[16]"),
    ]
    for stage_name, _ in hook_targets:
        try:
            if stage_name == "stage_1_enc_out":
                m = model.visual.blocks[-1] if hasattr(model, "visual") else model.base_model.model.visual.blocks[-1]
            elif stage_name == "stage_2_post_proj":
                m = model.visual.merger if hasattr(model, "visual") else model.base_model.model.visual.merger
            elif stage_name == "stage_3_llm_8":
                m = model.model.layers[8] if hasattr(model, "model") else model.base_model.model.model.layers[8]
            elif stage_name == "stage_4_llm_16":
                m = model.model.layers[16] if hasattr(model, "model") else model.base_model.model.model.layers[16]
            else:
                continue
            hooks.append(m.register_forward_hook(make_hook(stage_name)))
        except (AttributeError, IndexError) as e:
            print(f"    Could not hook {stage_name}: {e}")

    all_acts = {s: [] for s in STAGE_NAMES}
    all_labels = []
    images = scenes["images"]
    labels_list = scenes["labels"]
    masks_list = scenes["masks"]
    device = next(model.parameters()).device

    for i in range(min(num_scenes, len(images))):
        hook_storage.clear()
        img = images[i]
        if isinstance(img, np.ndarray):
            img_pil = PILImage.fromarray(img.astype(np.uint8))
        else:
            img_pil = img

        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_pil},
            {"type": "text", "text": "Describe the objects."},
        ]}]

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
                model(**inputs, output_hidden_states=True)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                gc.collect()
                continue

        image_grid_thw = inputs.get("image_grid_thw", None)
        n_vis = int(image_grid_thw[0].prod().item()) if image_grid_thw is not None else None

        for stage_name in STAGE_NAMES:
            if stage_name not in hook_storage:
                continue
            act = hook_storage[stage_name]
            if act.ndim == 3:
                act = act.squeeze(0)
            if stage_name in ("stage_3_llm_8", "stage_4_llm_16") and n_vis and act.shape[0] > n_vis:
                input_ids = inputs.get("input_ids", None)
                if input_ids is not None:
                    ids = input_ids[0].cpu()
                    img_pos = (ids == 151655).nonzero(as_tuple=True)[0]
                    if len(img_pos) >= n_vis:
                        act = act[img_pos[:n_vis]]
                    else:
                        act = act[:n_vis]
                else:
                    act = act[:n_vis]
            all_acts[stage_name].append(act.numpy())

        physics = labels_list[i] if i < len(labels_list) else {}
        mask = masks_list[i] if i < len(masks_list) else None
        if mask is not None and physics:
            pl = assigner.assign(mask, physics)
            if isinstance(pl, dict):
                combined = np.stack([pl.get(v, np.full(196, np.nan)) for v in PHYSICS_VARS], axis=-1)
                all_labels.append(combined)
            else:
                all_labels.append(pl if pl.ndim == 2 else pl.reshape(-1, 4))
        else:
            all_labels.append(np.full((196, 4), np.nan))

        del inputs
        torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"      Extracted {i+1}/{num_scenes}")

    for h in hooks:
        h.remove()

    result = {}
    for stage in STAGE_NAMES:
        if all_acts[stage]:
            result[stage] = np.concatenate(all_acts[stage], axis=0)
    if all_labels:
        result["physics_labels"] = np.concatenate(all_labels, axis=0)
    return result


# ===========================================================================
# Probing
# ===========================================================================

def probe_activations(activations: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """Train ridge regression probes and return R² per stage × variable."""
    labels = activations.get("physics_labels")
    if labels is None:
        return {}

    results = {}
    for stage in STAGE_NAMES:
        if stage not in activations:
            continue
        X = activations[stage]
        stage_res = {}
        for var in PHYSICS_VARS:
            col = PHYSICS_COL[var]
            y = labels[:, col] if labels.ndim == 2 else labels
            n = min(X.shape[0], len(y))
            X_v, y_v = X[:n], y[:n]
            valid = ~np.isnan(y_v)
            if valid.sum() < 20:
                continue
            X_f, y_f = X_v[valid], y_v[valid]
            X_tr, X_te, y_tr, y_te = train_test_split(X_f, y_f, test_size=0.2, random_state=42)
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            ridge = RidgeCV(alphas=ALPHA_CANDIDATES)
            ridge.fit(X_tr_s, y_tr)
            y_pred = ridge.predict(X_te_s)
            ss_res = np.sum((y_te - y_pred) ** 2)
            ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0
            stage_res[var] = {"r2": r2}
        results[stage] = stage_res
    return results


# ===========================================================================
# LoRA fine-tuning
# ===========================================================================

def fine_tune_lora(model, processor, qa_data: List[Dict], scenes: Dict,
                   num_epochs: int, lr: float, batch_size: int, grad_accum: int,
                   output_dir: Path, condition_name: str):
    """Fine-tune the LoRA-adapted model on physics QA data."""
    from PIL import Image as PILImage

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01
    )

    images = scenes["images"]
    model.train()
    total_loss = 0.0
    n_steps = 0

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        np.random.shuffle(qa_data)

        for batch_start in range(0, len(qa_data), batch_size):
            batch = qa_data[batch_start: batch_start + batch_size]

            batch_loss = torch.tensor(0.0, device=device)
            valid_samples = 0

            for sample in batch:
                scene_idx = sample["scene_idx"]
                if scene_idx >= len(images):
                    continue

                img = images[scene_idx]
                if isinstance(img, np.ndarray):
                    img_pil = PILImage.fromarray(img.astype(np.uint8))
                else:
                    img_pil = img

                question = sample["question"]
                answer = sample["answer"]

                messages = [
                    {"role": "user", "content": [
                        {"type": "image", "image": img_pil},
                        {"type": "text", "text": question},
                    ]},
                    {"role": "assistant", "content": answer},
                ]

                try:
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    try:
                        from qwen_vl_utils import process_vision_info
                        image_inputs, video_inputs = process_vision_info(
                            [{"role": "user", "content": [
                                {"type": "image", "image": img_pil},
                                {"type": "text", "text": question},
                            ]}]
                        )
                        inputs = processor(
                            text=[text], images=image_inputs, videos=video_inputs,
                            padding=True, return_tensors="pt"
                        )
                    except ImportError:
                        inputs = processor(
                            text=[text], images=[img_pil],
                            return_tensors="pt", padding=True
                        )

                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in inputs.items()}
                    inputs["labels"] = inputs["input_ids"].clone()

                    outputs = model(**inputs)
                    batch_loss = batch_loss + outputs.loss
                    valid_samples += 1

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"    OOM at batch {batch_start}, skipping")
                    continue
                except Exception as e:
                    print(f"    Error: {e}")
                    continue

            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples

                if (n_steps + 1) % grad_accum == 0 or batch_start + batch_size >= len(qa_data):
                    (batch_loss / grad_accum).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    (batch_loss / grad_accum).backward()

                epoch_loss += batch_loss.item()
                n_steps += 1

            if n_steps % 20 == 0 and n_steps > 0:
                print(f"    Epoch {epoch+1}/{num_epochs}, step {n_steps}: loss={epoch_loss / max(n_steps, 1):.4f}")
                report_vram()

        avg_loss = epoch_loss / max(n_steps, 1)
        print(f"  Epoch {epoch+1}/{num_epochs} complete: avg_loss={avg_loss:.4f}")

    # Save LoRA weights
    ckpt_dir = output_dir / f"lora_checkpoint_{condition_name}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ckpt_dir))
    print(f"  Saved LoRA checkpoint to {ckpt_dir}")


# ===========================================================================
# Test mode simulation
# ===========================================================================

def run_test_mode_condition(condition: LoRACondition, output_dir: Path,
                            num_scenes: int, seed: int) -> Dict:
    """Simulate a LoRA condition using ViT-base (no actual LoRA, just re-probe)."""
    from src.models.activation_extractor import LightweightViTExtractor
    from src.data.patch_label_assigner import PatchLabelAssigner
    from PIL import Image as PILImage

    print(f"\n  [TEST MODE] Simulating condition {condition.name}: {condition.label}")
    extractor = LightweightViTExtractor()
    assigner = PatchLabelAssigner(patch_grid_size=14)
    dataset = DeconfoundedPhysicsDataset(n_scenes=num_scenes, image_size=224, seed=seed)

    all_acts = {s: [] for s in STAGE_NAMES}
    all_labels = []

    for i in range(min(num_scenes, len(dataset))):
        sample = dataset[i]
        img = sample.image
        if isinstance(img, np.ndarray):
            img = PILImage.fromarray(img.astype(np.uint8))
        acts = extractor.extract(img)
        for s in STAGE_NAMES:
            if s in acts:
                all_acts[s].append(acts[s].squeeze(0))
        physics = sample.physics_labels
        mask = sample.object_masks
        if mask is not None and physics:
            pl = assigner.assign(mask, physics)
            if isinstance(pl, dict):
                combined = np.stack([pl.get(v, np.full(196, np.nan)) for v in PHYSICS_VARS], axis=-1)
                all_labels.append(combined)
            elif isinstance(pl, np.ndarray):
                all_labels.append(pl if pl.ndim == 2 else pl.reshape(-1, 4))
        else:
            all_labels.append(np.full((196, 4), np.nan))

    activations = {}
    for s in STAGE_NAMES:
        if all_acts[s]:
            activations[s] = np.concatenate(all_acts[s], axis=0)
    if all_labels:
        activations["physics_labels"] = np.concatenate(all_labels, axis=0)

    # Simulate "before" and "after" (with random perturbation to show the idea)
    before_results = probe_activations(activations)

    # Simulate fine-tuned activations by adding small signal boost
    rng = np.random.RandomState(seed + ord(condition.name))
    for s in STAGE_NAMES:
        if s in activations:
            noise = rng.randn(*activations[s].shape).astype(np.float32) * 0.01
            activations[s] = activations[s] + noise

    after_results = probe_activations(activations)

    return {
        "condition": condition.name,
        "label": condition.label,
        "before": before_results,
        "after": after_results,
        "trainable_params": 0,
        "mode": "test_simulated",
    }


# ===========================================================================
# Run one condition (GPU)
# ===========================================================================

def run_gpu_condition(
    condition: LoRACondition,
    scene_data: Dict,
    qa_data: List[Dict],
    output_dir: Path,
    args,
) -> Dict:
    """Run a single LoRA ablation condition on GPU."""
    print(f"\n{'='*60}")
    print(f"  CONDITION {condition.name}: {condition.label}")
    print(f"  {condition.description}")
    print(f"{'='*60}")

    # Load fresh model
    model, processor = load_base_model()

    # Extract BEFORE activations
    print("\n  --- Extracting BEFORE activations ---")
    before_acts = extract_activations(model, processor, scene_data, args.num_scenes)
    before_results = probe_activations(before_acts)

    print("\n  BEFORE probing results:")
    for stage in STAGE_NAMES:
        if stage in before_results:
            vals = ", ".join(f"{v}={before_results[stage][v]['r2']:.4f}"
                            for v in PHYSICS_VARS if v in before_results[stage])
            print(f"    {stage}: {vals}")

    # Apply LoRA
    print(f"\n  --- Applying LoRA (condition {condition.name}) ---")
    peft_model, trainable_params = apply_lora(model, condition)

    # Fine-tune
    print(f"\n  --- Fine-tuning ({args.num_epochs} epochs) ---")
    fine_tune_lora(
        peft_model, processor, qa_data, scene_data,
        num_epochs=args.num_epochs,
        lr=args.lora_lr,
        batch_size=args.lora_batch_size,
        grad_accum=args.grad_accum,
        output_dir=output_dir,
        condition_name=condition.name,
    )

    # Extract AFTER activations
    print("\n  --- Extracting AFTER activations ---")
    peft_model.eval()
    after_acts = extract_activations(peft_model, processor, scene_data, args.num_scenes)
    after_results = probe_activations(after_acts)

    print("\n  AFTER probing results:")
    for stage in STAGE_NAMES:
        if stage in after_results:
            vals = ", ".join(f"{v}={after_results[stage][v]['r2']:.4f}"
                            for v in PHYSICS_VARS if v in after_results[stage])
            print(f"    {stage}: {vals}")

    # Compute deltas
    deltas = {}
    for stage in STAGE_NAMES:
        if stage in before_results and stage in after_results:
            stage_delta = {}
            for var in PHYSICS_VARS:
                if var in before_results[stage] and var in after_results[stage]:
                    b = before_results[stage][var]["r2"]
                    a = after_results[stage][var]["r2"]
                    stage_delta[var] = {"before_r2": b, "after_r2": a, "delta_r2": a - b}
            deltas[stage] = stage_delta

    # Clean up
    del peft_model, model, processor
    torch.cuda.empty_cache()
    gc.collect()

    result = {
        "condition": condition.name,
        "label": condition.label,
        "description": condition.description,
        "trainable_params": trainable_params,
        "before": before_results,
        "after": after_results,
        "deltas": deltas,
    }

    # Checkpoint
    ckpt_path = output_dir / f"result_condition_{condition.name}.json"
    with open(str(ckpt_path), "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Checkpointed result to {ckpt_path}")

    return result


# ===========================================================================
# Visualization
# ===========================================================================

def generate_ablation_figures(all_results: Dict[str, Dict], output_dir: Path):
    """Generate comparison figures across conditions."""
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted(all_results.keys())
    if not conditions:
        return

    # --- Figure 1: Before/After R² comparison per condition ---
    for var in PHYSICS_VARS:
        fig, axes = plt.subplots(1, len(conditions), figsize=(4 * len(conditions), 5),
                                 sharey=True)
        if len(conditions) == 1:
            axes = [axes]

        for ci, cond in enumerate(conditions):
            ax = axes[ci]
            res = all_results[cond]
            before_vals = []
            after_vals = []
            stage_labels = []

            for stage in STAGE_NAMES:
                before = res.get("before", {}).get(stage, {}).get(var, {}).get("r2", None)
                after = res.get("after", {}).get(stage, {}).get(var, {}).get("r2", None)
                if before is not None and after is not None:
                    before_vals.append(before)
                    after_vals.append(after)
                    idx = STAGE_NAMES.index(stage)
                    stage_labels.append(STAGE_LABELS[idx])

            x = np.arange(len(stage_labels))
            width = 0.35
            ax.bar(x - width / 2, before_vals, width, label="Before", color="#1f77b4", alpha=0.8)
            ax.bar(x + width / 2, after_vals, width, label="After", color="#d62728", alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(stage_labels, rotation=45, ha="right", fontsize=8)
            ax.set_title(f"Cond {cond}: {res.get('label', cond)}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"R² ({var})", fontsize=11)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"LoRA Ablation: {var.upper()} R² Before/After", fontsize=13)
        plt.tight_layout()
        fig.savefig(str(fig_dir / f"ablation_{var}_before_after.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    # --- Figure 2: Delta R² heatmap (conditions × stages) ---
    for var in PHYSICS_VARS:
        delta_matrix = np.full((len(conditions), len(STAGE_NAMES)), np.nan)
        for ci, cond in enumerate(conditions):
            deltas = all_results[cond].get("deltas", {})
            for si, stage in enumerate(STAGE_NAMES):
                if stage in deltas and var in deltas[stage]:
                    delta_matrix[ci, si] = deltas[stage][var]["delta_r2"]

        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        im = ax.imshow(delta_matrix, cmap="RdYlGn", aspect="auto",
                        vmin=-0.3, vmax=0.3)
        ax.set_xticks(range(len(STAGE_NAMES)))
        ax.set_xticklabels(STAGE_LABELS, fontsize=10)
        ax.set_yticks(range(len(conditions)))
        cond_labels = [f"{c}: {all_results[c].get('label', c)}" for c in conditions]
        ax.set_yticklabels(cond_labels, fontsize=10)
        plt.colorbar(im, ax=ax, label="ΔR²")
        ax.set_title(f"LoRA Effect on {var.upper()} Probing (ΔR²)", fontsize=13)

        # Annotate cells
        for ci in range(len(conditions)):
            for si in range(len(STAGE_NAMES)):
                val = delta_matrix[ci, si]
                if not np.isnan(val):
                    ax.text(si, ci, f"{val:+.3f}", ha="center", va="center",
                            fontsize=9, color="black" if abs(val) < 0.15 else "white")

        plt.tight_layout()
        fig.savefig(str(fig_dir / f"ablation_delta_heatmap_{var}.png"), dpi=300)
        plt.close(fig)

    print(f"  Saved ablation figures to {fig_dir}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.test_mode:
        output_dir = PROJECT_ROOT / "results" / "week2_ablation_test"
    else:
        output_dir = PROJECT_ROOT / "results" / "week2_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions_to_run = [args.condition] if args.condition else ["A", "B", "C", "D", "E"]

    print(f"\n{'#'*60}")
    print(f"  WEEK 2 LoRA ABLATION EXPERIMENT")
    print(f"  Conditions: {conditions_to_run}")
    print(f"  Scenes: {args.num_scenes}")
    print(f"  Mode: {'TEST' if args.test_mode else 'GPU'}")
    print(f"  Output: {output_dir}")
    print(f"{'#'*60}")

    # Check for completed conditions (resume support)
    all_results = {}
    if args.resume:
        for cond in conditions_to_run[:]:
            ckpt = output_dir / f"result_condition_{cond}.json"
            if ckpt.exists():
                print(f"  Found checkpoint for condition {cond}, loading...")
                with open(ckpt, "r") as f:
                    all_results[cond] = json.load(f)
                conditions_to_run.remove(cond)
        if not conditions_to_run:
            print("  All conditions already completed!")

    if args.test_mode:
        # Test mode: simulate all conditions
        for cond_name in conditions_to_run:
            condition = LORA_CONDITIONS[cond_name]
            result = run_test_mode_condition(condition, output_dir, args.num_scenes, args.seed)
            all_results[cond_name] = result
    else:
        # Generate scene data and QA pairs once
        print("\n  --- Generating scene data ---")
        dataset = DeconfoundedPhysicsDataset(n_scenes=args.num_scenes, image_size=224, seed=args.seed)
        from PIL import Image as PILImage
        scene_data = {
            "images": [dataset[i].image for i in range(len(dataset))],
            "labels": [dataset[i].physics_labels for i in range(len(dataset))],
            "masks": [dataset[i].object_masks for i in range(len(dataset))],
            "visual_features": [dataset.get_visual_features(i) for i in range(len(dataset))],
        }

        qa_data = prepare_qa_data(args.num_scenes, output_dir, args.seed)

        # Run each condition
        for cond_name in conditions_to_run:
            condition = LORA_CONDITIONS[cond_name]
            t0 = time.time()
            result = run_gpu_condition(condition, scene_data, qa_data, output_dir, args)
            elapsed = time.time() - t0
            result["runtime_seconds"] = elapsed
            all_results[cond_name] = result
            print(f"\n  Condition {cond_name} completed in {elapsed:.0f}s")

    # Generate comparison figures
    print("\n  --- Generating ablation figures ---")
    generate_ablation_figures(all_results, output_dir)

    # Save combined results
    combined_path = output_dir / "ablation_results_combined.json"
    with open(str(combined_path), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Combined results saved to {combined_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Cond':<6} {'Label':<22} {'Params':>10}  ", end="")
    for var in PHYSICS_VARS:
        print(f"  Δ{var[:4]:>5}", end="")
    print()
    print(f"  {'-'*66}")

    for cond in sorted(all_results.keys()):
        res = all_results[cond]
        label = res.get("label", cond)
        params = res.get("trainable_params", 0)
        print(f"  {cond:<6} {label:<22} {params:>10,}  ", end="")
        deltas = res.get("deltas", {})
        for var in PHYSICS_VARS:
            # Average delta across stages
            ds = [deltas[s][var]["delta_r2"] for s in STAGE_NAMES
                  if s in deltas and var in deltas.get(s, {})]
            avg_delta = np.mean(ds) if ds else float("nan")
            print(f"  {avg_delta:>+6.3f}", end="")
        print()

    print(f"\n  Done! Results at: {output_dir}")


if __name__ == "__main__":
    main()
