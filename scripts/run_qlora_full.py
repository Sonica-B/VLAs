#!/usr/bin/env python3
"""
Phase 3 Production: Full Image-Conditioned QLoRA Ablation Experiment.

CRITICAL DIFFERENCE from run_qlora_ablation.py:
  Every training sample includes an ACTUAL IMAGE from Physion++.
  No text-only shortcuts. Full VLM fine-tuning with vision inputs.

Tests H3: "The visual-language merger is the primary bottleneck for physics
understanding in VLMs. Fine-tuning the merger alone improves physics
performance more than fine-tuning the LLM backbone."

5 QLoRA Conditions (Qwen2.5-VL-7B-Instruct, 4-bit):
  A (Merger only):     visual.merger MLP layers (rank=64)
  B (LLM only):        First 8 LLM decoder layers Q/V (rank=16)
  C (Encoder only):    Last 6 ViT blocks QKV (rank=16)
  D (Merger+Encoder):  A + C combined (rank=16)
  E (Full):            All of the above (rank=16)

Per condition:
  1. Generate image-conditioned physics QA from Physion++ (if not cached)
  2. Train QLoRA with actual images through the VLM processor
  3. Evaluate on PhysBench val set (200 samples)
  4. Run probing at 4 pipeline stages
  5. Report deltas, save figures

Usage:
    python scripts/run_qlora_full.py --condition merger --epochs 3
    python scripts/run_qlora_full.py --condition all --epochs 3 --resume
    python scripts/run_qlora_full.py --generate-data-only
    python scripts/run_qlora_full.py --evaluate-only --adapter-path results/qlora_full/merger/adapter

Hardware target: RTX 5070 Ti (12GB VRAM)
"""

import argparse
import gc
import json
import os
import pickle
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_ID_4BIT = "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit"

PHYSION_DATA_ROOT = PROJECT_ROOT / "data" / "physion_readout" / "readout_data_v1"
PHYSION_ZIP_PATH = PROJECT_ROOT / "data" / "physion_readout.zip"

PHYSBENCH_BASELINE_VAL = 60.31  # Measured baseline (val set, 200 samples)
PHYSBENCH_BASELINE_BY_TYPE = {
    "dynamics": 61.54,
    "property": 63.89,
    "relationships": 66.00,
    "scene": 48.84,
}

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

CONDITION_CLI_MAP = {
    "merger": "A",
    "llm": "B",
    "encoder": "C",
    "merger+encoder": "D",
    "full": "E",
    "all": "all",
}


# ---------------------------------------------------------------------------
# QLoRA Condition Definitions
# ---------------------------------------------------------------------------
@dataclass
class QLoRACondition:
    id: str
    name: str
    label: str
    rank: int
    target_modules: List[str]
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    description: str = ""


QLORA_CONDITIONS: Dict[str, QLoRACondition] = {
    "A": QLoRACondition(
        id="A",
        name="merger",
        label="Merger only",
        rank=64,
        target_modules=[
            "visual.merger.mlp.0",
            "visual.merger.mlp.2",
        ],
        description=(
            "QLoRA on visual.merger MLP layers. Projects visual tokens "
            "from encoder space (1280-dim) to LLM space (3584-dim). "
            "Hypothesis: this is the physics bottleneck."
        ),
    ),
    "B": QLoRACondition(
        id="B",
        name="llm",
        label="LLM only",
        rank=16,
        target_modules=[
            *[f"model.layers.{i}.self_attn.{proj}"
              for i in range(8)
              for proj in ["q_proj", "v_proj"]],
        ],
        description=(
            "QLoRA on first 8 LLM layers (Q/V projections). "
            "Tests whether LLM reasoning layers are the bottleneck."
        ),
    ),
    "C": QLoRACondition(
        id="C",
        name="encoder",
        label="Encoder only",
        rank=16,
        target_modules=[
            *[f"visual.blocks.{i}.attn.qkv" for i in range(26, 32)],
        ],
        description=(
            "QLoRA on last 6 ViT encoder blocks (QKV projections). "
            "Tests whether encoder already loses physics information."
        ),
    ),
    "D": QLoRACondition(
        id="D",
        name="merger+encoder",
        label="Merger + Encoder",
        rank=16,
        target_modules=[
            "visual.merger.mlp.0",
            "visual.merger.mlp.2",
            *[f"visual.blocks.{i}.attn.qkv" for i in range(26, 32)],
        ],
        description=(
            "QLoRA on merger + last 6 encoder blocks. "
            "Expected strongest improvement if merger IS the bottleneck."
        ),
    ),
    "E": QLoRACondition(
        id="E",
        name="full",
        label="Full (all components)",
        rank=16,
        target_modules=[
            *[f"visual.blocks.{i}.attn.qkv" for i in range(26, 32)],
            "visual.merger.mlp.0",
            "visual.merger.mlp.2",
            *[f"model.layers.{i}.self_attn.{p}"
              for i in range(8)
              for p in ["q_proj", "v_proj"]],
        ],
        description=(
            "QLoRA on all components: encoder + merger + LLM. "
            "More trainable params but diluted across the pipeline."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def report_vram(prefix: str = ""):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  {prefix}VRAM: {alloc:.2f}GB / {total:.1f}GB")


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def timestamp():
    return time.strftime("%H:%M:%S")


# ===========================================================================
# 1. IMAGE-CONDITIONED Physics QA Data Generation
# ===========================================================================

# --- Question template banks ---
MASS_COMPARISON_TEMPLATES = [
    "Looking at this physics scene, which object appears to be heavier — {obj_a} or {obj_b}?",
    "In this scene, compare the masses of {obj_a} and {obj_b}. Which has greater mass?",
    "Based on the objects visible in this image, which is heavier: {obj_a} or {obj_b}?",
]

MASS_RATIO_TEMPLATES = [
    "Estimate the relative mass ratio between the heaviest and lightest objects in this scene.",
    "What is the approximate mass ratio between the two main objects in this physics scenario?",
]

MASS_INERTIA_TEMPLATES = [
    "Which object in this scene has greater inertia and would be harder to accelerate?",
    "If you pushed both objects with the same force, which would accelerate less?",
]

FRICTION_COMPARISON_TEMPLATES = [
    "Which object in this scene has more surface friction?",
    "If both objects were sliding, which would slow down faster due to friction?",
    "Compare the friction properties of the objects in this scene.",
]

FRICTION_PREDICTION_TEMPLATES = [
    "If given the same initial push, which object in this scene would slide farther on a flat surface?",
    "Based on the surface properties visible, how much friction does the contact surface have?",
]

BOUNCE_PREDICTION_TEMPLATES = [
    "Will the objects in this scene bounce significantly after collision?",
    "If these objects collide, how elastic will the collision be?",
    "Predict whether the collision in this scenario will be mostly elastic or inelastic.",
]

BOUNCE_COMPARISON_TEMPLATES = [
    "Which object is more elastic and will bounce more after a collision?",
    "Compare the elasticity (restitution) of the objects in this scene.",
]

COMBINED_PREDICTION_TEMPLATES = [
    "Considering both mass and friction, which object requires the most force to start moving?",
    "Based on the physical properties of these objects, predict what will happen when they interact.",
    "If the heavier object is pushed toward the lighter one, describe the expected outcome.",
]

COLLISION_OUTCOME_TEMPLATES = [
    "If these two objects collide, which one will experience a greater change in velocity?",
    "Predict the collision dynamics: will both objects move, or will one dominate?",
]


def _object_description(model_name: bytes, color: np.ndarray, idx: int) -> str:
    """Create a natural-language object description from metadata."""
    name = model_name.decode("utf-8") if isinstance(model_name, bytes) else str(model_name)

    # Convert RGB to color name
    r, g, b = color[:3]
    color_name = _rgb_to_color_name(r, g, b)

    return f"the {color_name} {name}"


def _rgb_to_color_name(r: float, g: float, b: float) -> str:
    """Approximate RGB → color name."""
    color_map = [
        ((1.0, 0.0, 0.0), "red"),
        ((0.0, 1.0, 0.0), "green"),
        ((0.0, 0.0, 1.0), "blue"),
        ((1.0, 1.0, 0.0), "yellow"),
        ((1.0, 0.5, 0.0), "orange"),
        ((0.5, 0.0, 0.5), "purple"),
        ((0.0, 1.0, 1.0), "cyan"),
        ((1.0, 0.75, 0.8), "pink"),
        ((0.5, 0.5, 0.5), "gray"),
        ((1.0, 1.0, 1.0), "white"),
        ((0.0, 0.0, 0.0), "black"),
        ((0.6, 0.3, 0.0), "brown"),
    ]
    best_name = "colored"
    best_dist = float("inf")
    for (cr, cg, cb), name in color_map:
        dist = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def generate_image_conditioned_qa(
    output_dir: Path,
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[str, str]:
    """
    Generate image-conditioned physics QA pairs from Physion++ readout data.

    Each sample includes:
      - image_path: path to the actual Physion++ scene frame (_map.png)
      - question: physics reasoning question about the scene
      - answer: detailed answer with ground-truth values
      - physics_property: mass | friction | elasticity
      - ground_truth_values: dict of actual physics values

    Returns (train_jsonl_path, val_jsonl_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "physics_qa_train.jsonl"
    val_path = output_dir / "physics_qa_val.jsonl"

    # Check cache
    if train_path.exists() and val_path.exists():
        n_train = sum(1 for _ in open(train_path, "r", encoding="utf-8"))
        n_val = sum(1 for _ in open(val_path, "r", encoding="utf-8"))
        print(f"  QA data already exists: {n_train} train, {n_val} val")
        return str(train_path), str(val_path)

    rng = np.random.RandomState(seed)
    data_root = PHYSION_DATA_ROOT

    if not data_root.exists():
        raise FileNotFoundError(
            f"Physion++ data not found at {data_root}. "
            f"No fallback — image-conditioned training requires real data."
        )

    print(f"  Generating image-conditioned QA from {data_root}...")
    qa_pairs = []
    trials_parsed = 0
    images_verified = 0
    images_missing = 0

    for scenario_dir in sorted(data_root.iterdir()):
        if not scenario_dir.is_dir():
            continue
        scenario_type = scenario_dir.name  # e.g., "mass_collision_pp"

        for variant_dir in sorted(scenario_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            variant_name = variant_dir.name

            for pkl_file in sorted(variant_dir.glob("*.pkl")):
                frame_id = pkl_file.stem  # e.g., "0000"
                map_png = variant_dir / f"{frame_id}_map.png"

                if not map_png.exists():
                    images_missing += 1
                    continue
                images_verified += 1

                # Load physics metadata
                try:
                    with open(pkl_file, "rb") as f:
                        data = pickle.load(f)
                except Exception:
                    continue

                static = data.get("static", {})
                masses = static.get("mass", np.array([]))
                dyn_friction = static.get("dynamic_friction", np.array([]))
                bounciness = static.get("bounciness", np.array([]))
                model_names = static.get("model_names", np.array([]))
                colors = static.get("color", np.array([]))
                n_objects = len(masses) if hasattr(masses, "__len__") else 0

                if n_objects < 2:
                    continue

                # Absolute image path for training
                image_path = str(map_png.resolve())

                # Build object descriptions
                obj_descs = []
                for oi in range(n_objects):
                    if oi < len(model_names) and oi < len(colors):
                        obj_descs.append(
                            _object_description(model_names[oi], colors[oi], oi)
                        )
                    else:
                        obj_descs.append(f"object {oi + 1}")

                trial_id = f"{scenario_type}/{variant_name}/{frame_id}"

                # ---- MASS questions ----
                real_masses = [
                    (j, float(m))
                    for j, m in enumerate(masses)
                    if not np.isnan(m) and 0.001 < m < 10000
                ]
                if len(real_masses) >= 2:
                    sorted_mass = sorted(real_masses, key=lambda x: x[1], reverse=True)
                    heavy_idx, heavy_m = sorted_mass[0]
                    light_idx, light_m = sorted_mass[-1]

                    if heavy_m > light_m * 1.2:
                        # Q: Which is heavier?
                        tmpl = rng.choice(MASS_COMPARISON_TEMPLATES)
                        qa_pairs.append({
                            "id": f"{trial_id}/mass_compare",
                            "image_path": image_path,
                            "question": tmpl.format(
                                obj_a=obj_descs[heavy_idx],
                                obj_b=obj_descs[light_idx],
                            ),
                            "answer": (
                                f"{obj_descs[heavy_idx].capitalize()} is heavier "
                                f"(mass ~ {heavy_m:.2f} kg) compared to "
                                f"{obj_descs[light_idx]} (mass ~ {light_m:.2f} kg). "
                                f"The mass ratio is approximately {heavy_m / light_m:.1f}:1."
                            ),
                            "physics_property": "mass",
                            "scenario": scenario_type,
                            "ground_truth_values": {
                                f"obj_{heavy_idx}_mass": heavy_m,
                                f"obj_{light_idx}_mass": light_m,
                            },
                        })

                        # Q: Mass ratio
                        tmpl = rng.choice(MASS_RATIO_TEMPLATES)
                        qa_pairs.append({
                            "id": f"{trial_id}/mass_ratio",
                            "image_path": image_path,
                            "question": tmpl,
                            "answer": (
                                f"The mass ratio between the heaviest object "
                                f"({obj_descs[heavy_idx]}, {heavy_m:.2f} kg) and the lightest "
                                f"({obj_descs[light_idx]}, {light_m:.2f} kg) is approximately "
                                f"{heavy_m / light_m:.1f}:1."
                            ),
                            "physics_property": "mass",
                            "scenario": scenario_type,
                            "ground_truth_values": {
                                "ratio": heavy_m / light_m,
                                "heavy_mass": heavy_m,
                                "light_mass": light_m,
                            },
                        })

                        # Q: Collision outcome (if large mass difference)
                        if heavy_m > light_m * 3:
                            tmpl = rng.choice(COLLISION_OUTCOME_TEMPLATES)
                            qa_pairs.append({
                                "id": f"{trial_id}/collision_outcome",
                                "image_path": image_path,
                                "question": tmpl,
                                "answer": (
                                    f"{obj_descs[light_idx].capitalize()} (lighter, "
                                    f"~{light_m:.2f} kg) will experience a much greater "
                                    f"velocity change than {obj_descs[heavy_idx]} "
                                    f"(~{heavy_m:.2f} kg). By conservation of momentum, "
                                    f"the lighter object's velocity changes "
                                    f"~{heavy_m / light_m:.1f}x more."
                                ),
                                "physics_property": "mass",
                                "scenario": scenario_type,
                                "ground_truth_values": {
                                    "heavy_mass": heavy_m,
                                    "light_mass": light_m,
                                },
                            })

                        # Q: Inertia
                        tmpl = rng.choice(MASS_INERTIA_TEMPLATES)
                        qa_pairs.append({
                            "id": f"{trial_id}/inertia",
                            "image_path": image_path,
                            "question": tmpl,
                            "answer": (
                                f"{obj_descs[heavy_idx].capitalize()} has greater inertia "
                                f"(mass ~ {heavy_m:.2f} kg) and requires ~{heavy_m / light_m:.1f}x "
                                f"more force to achieve the same acceleration as "
                                f"{obj_descs[light_idx]} (~{light_m:.2f} kg), per Newton's F=ma."
                            ),
                            "physics_property": "mass",
                            "scenario": scenario_type,
                            "ground_truth_values": {
                                "heavy_mass": heavy_m,
                                "light_mass": light_m,
                            },
                        })

                # ---- FRICTION questions ----
                valid_friction = [
                    (j, float(f_val))
                    for j, f_val in enumerate(dyn_friction)
                    if not np.isnan(f_val)
                ] if hasattr(dyn_friction, "__len__") and len(dyn_friction) >= 1 else []

                if len(valid_friction) >= 2:
                    sorted_fric = sorted(valid_friction, key=lambda x: x[1], reverse=True)
                    rough_idx, rough_val = sorted_fric[0]
                    smooth_idx, smooth_val = sorted_fric[-1]

                    if abs(rough_val - smooth_val) > 0.1:
                        # Q: Compare friction
                        tmpl = rng.choice(FRICTION_COMPARISON_TEMPLATES)
                        qa_pairs.append({
                            "id": f"{trial_id}/friction_compare",
                            "image_path": image_path,
                            "question": tmpl,
                            "answer": (
                                f"{obj_descs[rough_idx].capitalize()} has higher friction "
                                f"(dynamic friction coefficient ~ {rough_val:.3f}) compared to "
                                f"{obj_descs[smooth_idx]} (~ {smooth_val:.3f}). "
                                f"The rougher object decelerates ~{rough_val / max(smooth_val, 0.001):.1f}x faster."
                            ),
                            "physics_property": "friction",
                            "scenario": scenario_type,
                            "ground_truth_values": {
                                f"obj_{rough_idx}_friction": rough_val,
                                f"obj_{smooth_idx}_friction": smooth_val,
                            },
                        })

                        # Q: Sliding prediction
                        tmpl = rng.choice(FRICTION_PREDICTION_TEMPLATES)
                        qa_pairs.append({
                            "id": f"{trial_id}/friction_predict",
                            "image_path": image_path,
                            "question": tmpl,
                            "answer": (
                                f"{obj_descs[smooth_idx].capitalize()} would slide farther "
                                f"because it has lower friction (mu ~ {smooth_val:.3f}) compared "
                                f"to {obj_descs[rough_idx]} (mu ~ {rough_val:.3f}). "
                                f"Friction-induced deceleration a = mu*g is "
                                f"~{rough_val / max(smooth_val, 0.001):.1f}x lower for the smoother object."
                            ),
                            "physics_property": "friction",
                            "scenario": scenario_type,
                            "ground_truth_values": {
                                "smooth_friction": smooth_val,
                                "rough_friction": rough_val,
                            },
                        })

                # ---- BOUNCINESS / ELASTICITY questions ----
                valid_bounce = [
                    (j, float(b_val))
                    for j, b_val in enumerate(bounciness)
                    if not np.isnan(b_val)
                ] if hasattr(bounciness, "__len__") and len(bounciness) >= 2 else []

                if len(valid_bounce) >= 2:
                    sorted_bounce = sorted(valid_bounce, key=lambda x: x[1], reverse=True)
                    bouncy_idx, bouncy_val = sorted_bounce[0]
                    flat_idx, flat_val = sorted_bounce[-1]

                    # Q: Will they bounce?
                    tmpl = rng.choice(BOUNCE_PREDICTION_TEMPLATES)
                    if bouncy_val > 0.5:
                        qa_pairs.append({
                            "id": f"{trial_id}/bounce_predict",
                            "image_path": image_path,
                            "question": tmpl,
                            "answer": (
                                f"Yes, at least {obj_descs[bouncy_idx]} has high elasticity "
                                f"(restitution ~ {bouncy_val:.2f}), meaning it retains "
                                f"~{bouncy_val * 100:.0f}% of relative velocity after collision. "
                                f"The collision will be significantly elastic."
                            ),
                            "physics_property": "elasticity",
                            "scenario": scenario_type,
                            "ground_truth_values": {
                                "max_bounciness": bouncy_val,
                                "min_bounciness": flat_val,
                            },
                        })
                    elif bouncy_val < 0.3:
                        qa_pairs.append({
                            "id": f"{trial_id}/bounce_predict",
                            "image_path": image_path,
                            "question": tmpl,
                            "answer": (
                                f"No, the objects have low elasticity (max restitution "
                                f"~ {bouncy_val:.2f}). The collision will be largely inelastic, "
                                f"with most kinetic energy absorbed on contact."
                            ),
                            "physics_property": "elasticity",
                            "scenario": scenario_type,
                            "ground_truth_values": {
                                "max_bounciness": bouncy_val,
                                "min_bounciness": flat_val,
                            },
                        })

                    # Q: Compare bounciness
                    if abs(bouncy_val - flat_val) > 0.15:
                        tmpl = rng.choice(BOUNCE_COMPARISON_TEMPLATES)
                        qa_pairs.append({
                            "id": f"{trial_id}/bounce_compare",
                            "image_path": image_path,
                            "question": tmpl,
                            "answer": (
                                f"{obj_descs[bouncy_idx].capitalize()} is more elastic "
                                f"(restitution ~ {bouncy_val:.2f}) compared to "
                                f"{obj_descs[flat_idx]} (~ {flat_val:.2f}). "
                                f"In a collision, {obj_descs[bouncy_idx]} retains more "
                                f"kinetic energy and bounces back faster."
                            ),
                            "physics_property": "elasticity",
                            "scenario": scenario_type,
                            "ground_truth_values": {
                                f"obj_{bouncy_idx}_bounce": bouncy_val,
                                f"obj_{flat_idx}_bounce": flat_val,
                            },
                        })

                # ---- COMBINED physics reasoning ----
                if len(real_masses) >= 2 and len(valid_friction) >= 1:
                    tmpl = rng.choice(COMBINED_PREDICTION_TEMPLATES)
                    heaviest_idx, heaviest_m = max(real_masses, key=lambda x: x[1])
                    max_fric_idx, max_fric = max(valid_friction, key=lambda x: x[1])
                    qa_pairs.append({
                        "id": f"{trial_id}/combined",
                        "image_path": image_path,
                        "question": tmpl,
                        "answer": (
                            f"Starting force F = mu_s * m * g. "
                            f"{obj_descs[heaviest_idx].capitalize()} has mass ~ {heaviest_m:.2f} kg. "
                            f"The highest friction object has mu ~ {max_fric:.3f}. "
                            f"The product of mass and friction coefficient determines "
                            f"which object needs more force to move."
                        ),
                        "physics_property": "mass",
                        "scenario": scenario_type,
                        "ground_truth_values": {
                            "heaviest_mass": heaviest_m,
                            "max_friction": max_fric,
                        },
                    })

                trials_parsed += 1
                if trials_parsed % 100 == 0:
                    print(f"    Parsed {trials_parsed} trials, {len(qa_pairs)} QA pairs so far")

    print(f"\n  Generated {len(qa_pairs)} image-conditioned QA pairs from {trials_parsed} trials")
    print(f"  Images verified: {images_verified}, missing: {images_missing}")

    # Property distribution
    prop_counts = defaultdict(int)
    for qa in qa_pairs:
        prop_counts[qa["physics_property"]] += 1
    for prop, count in sorted(prop_counts.items()):
        print(f"    {prop}: {count}")

    # Scenario distribution
    scenario_counts = defaultdict(int)
    for qa in qa_pairs:
        scenario_counts[qa["scenario"]] += 1
    print(f"  Scenarios: {dict(sorted(scenario_counts.items()))}")

    # Shuffle and split 80/20
    rng.shuffle(qa_pairs)
    split_idx = int(len(qa_pairs) * (1 - val_fraction))
    train_pairs = qa_pairs[:split_idx]
    val_pairs = qa_pairs[split_idx:]

    # Save
    for path, pairs in [(train_path, train_pairs), (val_path, val_pairs)]:
        with open(path, "w", encoding="utf-8") as f:
            for qa in pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"  Train: {len(train_pairs)} pairs -> {train_path}")
    print(f"  Val:   {len(val_pairs)} pairs -> {val_path}")

    return str(train_path), str(val_path)


def load_qa_data(qa_path: str, max_samples: Optional[int] = None) -> List[Dict]:
    pairs = []
    with open(qa_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    if max_samples:
        pairs = pairs[:max_samples]
    print(f"  Loaded {len(pairs)} QA pairs from {qa_path}")
    return pairs


# ===========================================================================
# 2. Model Loading
# ===========================================================================

def load_base_model(quantize: str = "4bit"):
    """Load Qwen2.5-VL-7B with quantization. Returns (model, processor)."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"\n  [{timestamp()}] Loading Qwen2.5-VL-7B-Instruct ({quantize})...")

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID_4BIT, **model_kwargs
            )
            print(f"  Loaded pre-quantized model from {MODEL_ID_4BIT}")
        except Exception:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                MODEL_ID, **model_kwargs
            )
            print(f"  Loaded with on-the-fly 4-bit quantization")
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, **model_kwargs
        )
        print(f"  Loaded with 8-bit quantization")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, **model_kwargs
        )

    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e9:.1f}B")
    report_vram("After load: ")

    return model, processor


# ===========================================================================
# 3. QLoRA Application
# ===========================================================================

def apply_qlora(model, condition: QLoRACondition):
    """Apply QLoRA adapters. Returns (peft_model, trainable_count)."""
    from peft import LoraConfig, TaskType, get_peft_model

    all_module_names = {name for name, _ in model.named_modules()}
    valid_targets = []
    missing_targets = []

    for target in condition.target_modules:
        matches = [n for n in all_module_names if target in n]
        if matches:
            valid_targets.append(target)
        else:
            missing_targets.append(target)

    if missing_targets:
        print(f"  WARNING: {len(missing_targets)} target modules not found:")
        for t in missing_targets[:5]:
            print(f"    - {t}")

    if not valid_targets:
        raise RuntimeError(
            f"No valid target modules for condition {condition.id} ({condition.name}). "
            f"Tried: {condition.target_modules}"
        )

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

    print(f"\n  QLoRA Condition {condition.id} ({condition.label}):")
    print(f"    Description: {condition.description}")
    print(f"    Rank: {condition.rank}, Alpha: {condition.lora_alpha}")
    print(f"    Target modules ({len(valid_targets)}):")
    for t in valid_targets:
        print(f"      - {t}")
    print(f"    Trainable params: {trainable:,} ({trainable / 1e6:.2f}M)")
    print(f"    Total params: {total:,} ({total / 1e9:.2f}B)")
    print(f"    Trainable %: {100 * trainable / total:.4f}%")
    report_vram("After QLoRA: ")

    return peft_model, trainable


# ===========================================================================
# 4. IMAGE-CONDITIONED QLoRA Training
# ===========================================================================

def train_qlora_condition(
    model,
    processor,
    condition: QLoRACondition,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation: int = 16,
    learning_rate: float = 2e-4,
    patience: int = 3,
) -> Tuple[str, float]:
    """
    Train a QLoRA condition with ACTUAL IMAGES through the VLM processor.

    Key difference from run_qlora_ablation.py: every training forward pass
    includes an image processed through Qwen2.5-VL's vision encoder.

    Args:
        patience: Early stopping patience (epochs without val loss improvement).
    """
    from PIL import Image as PILImage

    train_data = load_qa_data(train_data_path)
    val_data = load_qa_data(val_data_path)
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    total_steps = (len(train_data) // batch_size) * epochs
    warmup_steps = min(100, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    adapter_dir = Path(output_dir) / condition.name / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'=' * 60}")
    print(f"  TRAINING Condition {condition.id}: {condition.label}")
    print(f"  {'=' * 60}")
    print(f"  Train pairs: {len(train_data)} (IMAGE-CONDITIONED)")
    print(f"  Val pairs: {len(val_data)}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Grad Accum: {gradient_accumulation}")
    print(f"  Effective batch: {batch_size * gradient_accumulation}")
    print(f"  Total steps: ~{total_steps}")
    print(f"  Early stopping patience: {patience} epochs")
    print(f"  LR: {learning_rate}, Warmup: {warmup_steps} steps")

    model.train()
    global_step = 0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    training_losses = []
    val_losses = []
    t_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        epoch_skipped = 0
        np.random.shuffle(train_data)

        for batch_start in range(0, len(train_data), batch_size):
            batch = train_data[batch_start : batch_start + batch_size]
            batch_loss_accum = 0.0
            valid_count = 0

            for sample in batch:
                question = sample["question"]
                answer = sample["answer"]
                image_path = sample.get("image_path", "")

                # Load the ACTUAL IMAGE
                img_pil = None
                if image_path and os.path.exists(image_path):
                    try:
                        img_pil = PILImage.open(image_path).convert("RGB")
                    except Exception:
                        pass

                if img_pil is None:
                    epoch_skipped += 1
                    continue

                # Build chat messages WITH IMAGE
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img_pil},
                            {"type": "text", "text": question},
                        ],
                    },
                    {"role": "assistant", "content": answer},
                ]

                try:
                    # Process with Qwen's VLM processor (image goes through vision encoder)
                    from qwen_vl_utils import process_vision_info

                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        return_tensors="pt",
                        padding=True,
                    )
                    inputs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                    inputs["labels"] = inputs["input_ids"].clone()

                    outputs = model(**inputs)
                    loss = outputs.loss

                    # Gradient accumulation
                    scaled_loss = loss / gradient_accumulation
                    scaled_loss.backward()

                    batch_loss_accum += loss.item()
                    valid_count += 1

                    del inputs, outputs, loss, scaled_loss
                    torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"    [{timestamp()}] OOM at step {global_step}, skipping")
                    epoch_skipped += 1
                    continue
                except Exception as e:
                    if global_step < 5:
                        print(f"    [{timestamp()}] Error at step {global_step}: {e}")
                    epoch_skipped += 1
                    continue

            if valid_count > 0:
                avg_batch_loss = batch_loss_accum / valid_count

                if (global_step + 1) % gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += avg_batch_loss
                epoch_steps += 1
                global_step += 1

                if global_step % 25 == 0:
                    avg = epoch_loss / max(epoch_steps, 1)
                    elapsed = time.time() - t_start
                    rate = global_step / elapsed
                    eta = (total_steps - global_step) / max(rate, 0.001)
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"    [{timestamp()}] Epoch {epoch + 1}/{epochs} | "
                        f"Step {global_step}/{total_steps} | "
                        f"Loss: {avg:.4f} | LR: {lr_now:.2e} | "
                        f"{rate:.1f} step/s | ETA: {eta / 60:.1f}min"
                    )
                    report_vram("    ")

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        training_losses.append(avg_epoch_loss)

        # --- Validation loss ---
        val_loss = _compute_val_loss(model, processor, val_data, device)
        val_losses.append(val_loss)

        elapsed = time.time() - t_start
        print(f"\n  Epoch {epoch + 1}/{epochs} complete:")
        print(f"    Train loss: {avg_epoch_loss:.4f}")
        print(f"    Val loss:   {val_loss:.4f}")
        print(f"    Skipped:    {epoch_skipped} samples (image load failures)")
        print(f"    Elapsed:    {elapsed / 60:.1f}min")
        report_vram("    ")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            model.save_pretrained(str(adapter_dir))
            print(f"    -> Saved BEST adapter (val_loss={val_loss:.4f}) to {adapter_dir}")
        else:
            epochs_without_improvement += 1
            print(f"    -> No improvement ({epochs_without_improvement}/{patience})")
            if epochs_without_improvement >= patience:
                print(f"    -> EARLY STOPPING at epoch {epoch + 1}")
                break

    # Final save (always save last checkpoint too)
    final_dir = Path(output_dir) / condition.name / "adapter_final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(final_dir))

    total_time = time.time() - t_start
    print(f"\n  Training complete: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Training losses: {[f'{l:.4f}' for l in training_losses]}")
    print(f"  Val losses:      {[f'{l:.4f}' for l in val_losses]}")

    # Save loss curves data
    loss_data = {
        "training_losses": training_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "total_time_sec": total_time,
        "total_steps": global_step,
    }
    loss_path = Path(output_dir) / condition.name / "loss_curves.json"
    with open(loss_path, "w") as f:
        json.dump(loss_data, f, indent=2)

    return str(adapter_dir), total_time


def _compute_val_loss(
    model,
    processor,
    val_data: List[Dict],
    device: torch.device,
    max_samples: int = 50,
) -> float:
    """Compute average validation loss on a subset of val data WITH images."""
    from PIL import Image as PILImage

    model.eval()
    total_loss = 0.0
    count = 0

    subset = val_data[:max_samples]

    for sample in subset:
        image_path = sample.get("image_path", "")
        if not image_path or not os.path.exists(image_path):
            continue

        try:
            img_pil = PILImage.open(image_path).convert("RGB")
        except Exception:
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_pil},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {"role": "assistant", "content": sample["answer"]},
        ]

        try:
            from qwen_vl_utils import process_vision_info

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            )
            inputs = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            inputs["labels"] = inputs["input_ids"].clone()

            with torch.no_grad():
                outputs = model(**inputs)
            total_loss += outputs.loss.item()
            count += 1

            del inputs, outputs
            torch.cuda.empty_cache()

        except Exception:
            continue

    model.train()
    return total_loss / max(count, 1)


# ===========================================================================
# 5. PhysBench Evaluation
# ===========================================================================

def evaluate_after_training(
    model,
    processor,
    condition_name: str,
    output_dir: str,
    max_samples: int = 200,
    split: str = "val",
) -> Dict:
    """Run PhysBench evaluation with trained model."""
    from qwen_vl_utils import process_vision_info

    data_dir = PROJECT_ROOT / "data" / "physbench"
    json_path = data_dir / f"{split}.json"

    if not json_path.exists():
        print(f"  WARNING: PhysBench {split} data not found at {json_path}")
        return {"overall_accuracy": None, "error": f"Data not found: {json_path}"}

    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        data = json.loads(content)
    else:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]

    if max_samples:
        data = data[:max_samples]

    print(f"\n  {'=' * 60}")
    print(f"  PHYSBENCH EVALUATION ({condition_name})")
    print(f"  {'=' * 60}")
    print(f"  Split: {split}, Samples: {len(data)}")

    model.eval()
    correct = 0
    total = 0
    errors = 0
    by_task_type = defaultdict(lambda: {"correct": 0, "total": 0})
    t_start = time.time()

    for i, item in enumerate(data):
        file_names = item.get("file_name", [])
        if isinstance(file_names, str):
            file_names = [file_names]

        media_paths = []
        for fname in file_names:
            for subdir in ["", "image", "video", "images", "videos"]:
                candidate = data_dir / subdir / fname if subdir else data_dir / fname
                if candidate.exists():
                    media_paths.append(str(candidate))
                    break
            else:
                media_paths.append(None)

        question_text = item.get("question", "")
        content_list = []
        media_idx = 0
        parts = re.split(r"(<image>|<video>)", question_text)

        for part in parts:
            if part == "<image>" and media_idx < len(media_paths):
                path = media_paths[media_idx]
                media_idx += 1
                if path and os.path.exists(path):
                    content_list.append({
                        "type": "image", "image": path,
                        "max_pixels": 256 * 256, "min_pixels": 28 * 28,
                    })
                else:
                    content_list.append({"type": "text", "text": "[image unavailable]"})
            elif part == "<video>" and media_idx < len(media_paths):
                path = media_paths[media_idx]
                media_idx += 1
                if path and os.path.exists(path):
                    content_list.append({"type": "video", "video": path, "nframes": 4})
                else:
                    content_list.append({"type": "text", "text": "[video unavailable]"})
            elif part.strip():
                content_list.append({"type": "text", "text": part})

        content_list.append({
            "type": "text",
            "text": "\nAnswer with ONLY the letter (A, B, C, or D) of the correct option.",
        })

        messages = [{"role": "user", "content": content_list}]

        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(model.device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs, max_new_tokens=32, do_sample=False,
                    temperature=None, top_p=None,
                )
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)

            response_clean = response.strip()
            predicted = "X"
            if response_clean in ("A", "B", "C", "D"):
                predicted = response_clean
            else:
                match = re.search(r"(?:answer|option)\s*(?:is|:)\s*([A-D])", response_clean, re.I)
                if match:
                    predicted = match.group(1).upper()
                else:
                    match = re.search(r"\b([A-D])\b", response_clean)
                    if match:
                        predicted = match.group(1).upper()

            gt = item.get("answer", "").strip().upper()
            if predicted == gt:
                correct += 1
            total += 1

            task_type = item.get("task_type", "unknown")
            by_task_type[task_type]["total"] += 1
            if predicted == gt:
                by_task_type[task_type]["correct"] += 1

            del inputs, output_ids
            torch.cuda.empty_cache()

        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Eval error on item {i}: {e}")

        if (i + 1) % 50 == 0:
            acc = correct / max(total, 1) * 100
            print(f"    [{timestamp()}] [{i + 1}/{len(data)}] Acc: {acc:.1f}% ({correct}/{total})")

    elapsed = time.time() - t_start
    accuracy = correct / max(total, 1) * 100
    delta = accuracy - PHYSBENCH_BASELINE_VAL

    per_domain = {}
    for k, v in sorted(by_task_type.items()):
        domain_acc = v["correct"] / max(v["total"], 1) * 100
        baseline_domain = PHYSBENCH_BASELINE_BY_TYPE.get(k, PHYSBENCH_BASELINE_VAL)
        per_domain[k] = {
            "accuracy": round(domain_acc, 2),
            "correct": v["correct"],
            "total": v["total"],
            "baseline": baseline_domain,
            "delta": round(domain_acc - baseline_domain, 2),
        }

    result = {
        "condition": condition_name,
        "split": split,
        "overall_accuracy": round(accuracy, 2),
        "baseline_accuracy": PHYSBENCH_BASELINE_VAL,
        "delta": round(delta, 2),
        "correct": correct,
        "total": total,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "by_task_type": per_domain,
    }

    eval_dir = Path(output_dir) / condition_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"physbench_{split}_results.json"
    with open(eval_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  PhysBench Result ({condition_name}):")
    print(f"    Overall: {accuracy:.2f}% (baseline: {PHYSBENCH_BASELINE_VAL}%, delta: {delta:+.2f}%)")
    print(f"    Per domain:")
    for k, v in per_domain.items():
        print(f"      {k:20s}: {v['accuracy']:.1f}% (was {v['baseline']:.1f}%, delta {v['delta']:+.1f}%)")

    return result


# ===========================================================================
# 6. Probing After Training
# ===========================================================================

def probe_after_training(
    model,
    processor,
    condition_name: str,
    output_dir: str,
    num_scenes: int = 100,
    seed: int = 42,
) -> Dict:
    """Extract activations at 4 pipeline stages and run linear probing."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print(f"\n  {'=' * 60}")
    print(f"  PROBING DIAGNOSTICS ({condition_name})")
    print(f"  {'=' * 60}")

    hook_storage = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hook_storage[name] = output[0].detach().cpu()
            elif isinstance(output, torch.Tensor):
                hook_storage[name] = output.detach().cpu()
        return hook_fn

    hook_defs = [
        ("stage_1_enc_out", ["visual", "blocks", "-1"]),
        ("stage_2_post_proj", ["visual", "merger"]),
        ("stage_3_llm_8", ["model", "layers", "8"]),
        ("stage_4_llm_16", ["model", "layers", "16"]),
    ]

    for stage_name, path_parts in hook_defs:
        try:
            obj = model
            if hasattr(obj, "base_model"):
                obj = obj.base_model
                if hasattr(obj, "model"):
                    obj = obj.model

            for p in path_parts:
                if p == "-1":
                    obj = obj[-1]
                elif p.isdigit():
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p)

            hooks.append(obj.register_forward_hook(make_hook(stage_name)))
        except (AttributeError, IndexError) as e:
            print(f"    Could not hook {stage_name}: {e}")

    if not hooks:
        print("  WARNING: No hooks registered, skipping probing")
        return {}

    try:
        from src.data.deconfounded_physion import DeconfoundedPhysicsDataset
        from src.data.patch_label_assigner import PatchLabelAssigner
        from PIL import Image as PILImage

        dataset = DeconfoundedPhysicsDataset(
            n_scenes=num_scenes, image_size=224, seed=seed
        )
        assigner = PatchLabelAssigner(patch_grid_size=14)
    except ImportError as e:
        print(f"  WARNING: Could not import probing dependencies: {e}")
        for h in hooks:
            h.remove()
        return {}

    model.eval()
    all_acts = {s: [] for s in STAGE_NAMES}
    all_labels = []

    for i in range(min(num_scenes, len(dataset))):
        hook_storage.clear()
        sample = dataset[i]
        img = sample.image
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
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            )
        except ImportError:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[img_pil], return_tensors="pt", padding=True
            )

        device = next(model.parameters()).device
        inputs = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        try:
            with torch.no_grad():
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

        physics = sample.physics_labels
        mask = sample.object_masks
        if mask is not None and physics:
            pl = assigner.assign(mask, physics)
            if isinstance(pl, dict):
                combined = np.stack(
                    [pl.get(v, np.full(196, np.nan)) for v in PHYSICS_VARS],
                    axis=-1,
                )
                all_labels.append(combined)
            elif isinstance(pl, np.ndarray):
                all_labels.append(pl if pl.ndim == 2 else pl.reshape(-1, 4))
        else:
            all_labels.append(np.full((196, 4), np.nan))

        del inputs
        torch.cuda.empty_cache()

        if (i + 1) % 25 == 0:
            print(f"    [{timestamp()}] Extracted {i + 1}/{num_scenes} scenes")

    for h in hooks:
        h.remove()

    activations = {}
    for s in STAGE_NAMES:
        if all_acts[s]:
            activations[s] = np.concatenate(all_acts[s], axis=0)
    if all_labels:
        activations["physics_labels"] = np.concatenate(all_labels, axis=0)

    labels = activations.get("physics_labels")
    if labels is None:
        print("  WARNING: No physics labels available for probing")
        return {}

    probe_results = {}
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
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_f, y_f, test_size=0.2, random_state=42
            )
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            ridge = RidgeCV(alphas=ALPHA_CANDIDATES)
            ridge.fit(X_tr_s, y_tr)
            y_pred = ridge.predict(X_te_s)
            ss_res = np.sum((y_te - y_pred) ** 2)
            ss_tot = np.sum((y_te - np.mean(y_te)) ** 2)
            r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
            pearson = float(np.corrcoef(y_te, y_pred)[0, 1]) if len(y_te) > 2 else 0.0
            stage_res[var] = {"r2": round(r2, 4), "pearson_r": round(pearson, 4)}
        probe_results[stage] = stage_res

    probe_dir = Path(output_dir) / condition_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / "probe_results.json"
    with open(probe_path, "w") as f:
        json.dump(probe_results, f, indent=2)

    print(f"\n  Probing Results ({condition_name}):")
    for stage in STAGE_NAMES:
        if stage in probe_results:
            vals = ", ".join(
                f"{v}={probe_results[stage][v]['r2']:.3f}"
                for v in PHYSICS_VARS
                if v in probe_results[stage]
            )
            label = STAGE_LABELS[STAGE_NAMES.index(stage)]
            print(f"    {label:20s}: {vals}")

    return probe_results


# ===========================================================================
# 7. Hypothesis Tracker — Comprehensive Output
# ===========================================================================

def print_hypothesis_tracker(all_results: Dict[str, Dict]):
    """Print the full hypothesis tracking summary."""
    print()
    print("=" * 70)
    print("  OPTION C HYPOTHESIS TRACKER")
    print("=" * 70)
    print()
    print(f"  BASELINE PhysBench: {PHYSBENCH_BASELINE_VAL}% (val)")
    print(f"    Property:      {PHYSBENCH_BASELINE_BY_TYPE.get('property', '?')}%")
    print(f"    Dynamics:      {PHYSBENCH_BASELINE_BY_TYPE.get('dynamics', '?')}%")
    print(f"    Scene:         {PHYSBENCH_BASELINE_BY_TYPE.get('scene', '?')}%")
    print(f"    Relationships: {PHYSBENCH_BASELINE_BY_TYPE.get('relationships', '?')}%")
    print()
    print(f"  H3 Prediction: Merger QLoRA > LLM QLoRA for physics")
    print(f"  Expected: Condition A (merger) improves PhysBench by 3-5%")
    print(f"            Condition B (LLM) improves PhysBench by 0-2%")
    print()

    for cond_id in ["A", "B", "C", "D", "E"]:
        if cond_id not in all_results:
            continue
        res = all_results[cond_id]
        cond_def = QLORA_CONDITIONS[cond_id]

        print(f"  CONDITION {cond_id}: {cond_def.label}")
        print(f"    Trainable params: {res.get('trainable_params', '?'):,}")
        train_time = res.get("train_time_sec", 0)
        print(f"    Training time: {train_time / 3600:.1f}h ({train_time / 60:.0f}min)")

        eval_res = res.get("physbench_eval", {})
        if eval_res.get("overall_accuracy") is not None:
            acc = eval_res["overall_accuracy"]
            delta = eval_res.get("delta", acc - PHYSBENCH_BASELINE_VAL)
            print(f"    PhysBench val accuracy: {acc:.2f}% (delta: {delta:+.2f}%)")
            print(f"    Per-domain:")
            for k, v in sorted(eval_res.get("by_task_type", {}).items()):
                baseline_d = v.get("baseline", PHYSBENCH_BASELINE_VAL)
                delta_d = v.get("delta", v["accuracy"] - baseline_d)
                print(f"      {k:20s}: {v['accuracy']:.1f}% (was {baseline_d:.1f}%, delta {delta_d:+.1f}%)")
        else:
            print(f"    PhysBench AFTER: NOT YET RUN")

        probe_res = res.get("probe_results", {})
        if probe_res:
            print(f"    Probing R²:")
            for stage in STAGE_NAMES:
                if stage in probe_res:
                    vals = ", ".join(
                        f"{v}={probe_res[stage][v]['r2']:.3f}"
                        for v in PHYSICS_VARS
                        if v in probe_res[stage]
                    )
                    label = STAGE_LABELS[STAGE_NAMES.index(stage)]
                    print(f"      {label:20s}: {vals}")

        print()

    # Verdict
    a_acc = all_results.get("A", {}).get("physbench_eval", {}).get("overall_accuracy")
    b_acc = all_results.get("B", {}).get("physbench_eval", {}).get("overall_accuracy")

    if a_acc is not None and b_acc is not None:
        if a_acc > b_acc + 1.0:
            verdict = "SUPPORTS H3"
            reason = f"Merger ({a_acc:.1f}%) > LLM ({b_acc:.1f}%)"
        elif b_acc > a_acc + 1.0:
            verdict = "CONTRADICTS H3"
            reason = f"LLM ({b_acc:.1f}%) > Merger ({a_acc:.1f}%)"
        else:
            verdict = "INCONCLUSIVE"
            reason = f"Merger ({a_acc:.1f}%) ~ LLM ({b_acc:.1f}%)"
        print(f"  === H3 VERDICT: {verdict} ===")
        print(f"  Reason: {reason}")
    else:
        print("  === H3 VERDICT: PENDING (need both A and B results) ===")

    print("=" * 70)


# ===========================================================================
# 8. Visualization — 300 DPI Publication Figures
# ===========================================================================

def generate_figures(all_results: Dict[str, Dict], output_dir: str):
    """Generate 300 DPI comparison figures."""
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted(all_results.keys())
    if not conditions:
        return

    color_map = {
        "A": "#e74c3c", "B": "#3498db", "C": "#2ecc71",
        "D": "#9b59b6", "E": "#f39c12",
    }

    # --- Figure 1: PhysBench accuracy comparison (grouped bar) ---
    _fig_physbench_accuracy(conditions, all_results, color_map, fig_dir)

    # --- Figure 2: Probing degradation comparison ---
    _fig_probing_degradation(conditions, all_results, color_map, fig_dir)

    # --- Figure 3: Delta heatmap ---
    _fig_delta_heatmap(conditions, all_results, fig_dir)

    # --- Figure 4: Training loss curves ---
    _fig_training_loss_curves(conditions, all_results, color_map, fig_dir)

    print(f"  All figures saved to {fig_dir}")


def _fig_physbench_accuracy(conditions, all_results, color_map, fig_dir):
    """Grouped bar chart: baseline vs each condition, per domain."""
    domains = ["dynamics", "property", "relationships", "scene"]
    n_groups = len(domains)
    n_bars = 1 + len(conditions)  # baseline + each condition
    bar_width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Baseline bars
    baseline_accs = [PHYSBENCH_BASELINE_BY_TYPE.get(d, PHYSBENCH_BASELINE_VAL) for d in domains]
    x = np.arange(n_groups)
    ax.bar(
        x - (n_bars - 1) * bar_width / 2,
        baseline_accs,
        bar_width,
        label="Baseline",
        color="#95a5a6",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.5,
    )

    for ci, cid in enumerate(conditions):
        eval_res = all_results[cid].get("physbench_eval", {}).get("by_task_type", {})
        accs = []
        for d in domains:
            if d in eval_res:
                accs.append(eval_res[d]["accuracy"])
            else:
                accs.append(0)

        label = f"{cid}: {QLORA_CONDITIONS[cid].label}"
        ax.bar(
            x - (n_bars - 1) * bar_width / 2 + (ci + 1) * bar_width,
            accs,
            bar_width,
            label=label,
            color=color_map.get(cid, "#7f8c8d"),
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([d.title() for d in domains], fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("PhysBench Accuracy: Baseline vs QLoRA Conditions (per domain)", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(str(fig_dir / "physbench_accuracy_comparison.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved physbench_accuracy_comparison.png")


def _fig_probing_degradation(conditions, all_results, color_map, fig_dir):
    """R² curves overlaid for each condition."""
    for var in ["mass", "friction", "elasticity"]:
        fig, ax = plt.subplots(figsize=(8, 5))

        for cid in conditions:
            probe = all_results[cid].get("probe_results", {})
            r2_vals = []
            stage_lbls = []

            for si, stage in enumerate(STAGE_NAMES):
                r2 = probe.get(stage, {}).get(var, {}).get("r2")
                if r2 is not None:
                    r2_vals.append(r2)
                    stage_lbls.append(STAGE_LABELS[si])

            if r2_vals:
                label = f"{cid}: {QLORA_CONDITIONS[cid].label}"
                ax.plot(
                    range(len(r2_vals)), r2_vals, "o-",
                    color=color_map.get(cid, "gray"),
                    linewidth=2, markersize=8, label=label,
                )

        if stage_lbls:
            ax.set_xticks(range(len(stage_lbls)))
            ax.set_xticklabels(stage_lbls, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(f"R² ({var})", fontsize=11)
        ax.set_title(f"Probing R² After QLoRA: {var.title()}", fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(str(fig_dir / f"probing_degradation_{var}.png"), dpi=300)
        plt.close(fig)

    print(f"  Saved probing_degradation_comparison.png (3 vars)")


def _fig_delta_heatmap(conditions, all_results, fig_dir):
    """Delta heatmap: which PhysBench categories changed most per condition."""
    eval_categories = set()
    for cid in conditions:
        cats = all_results[cid].get("physbench_eval", {}).get("by_task_type", {})
        eval_categories.update(cats.keys())
    eval_categories = sorted(eval_categories)

    if not eval_categories or len(conditions) < 2:
        return

    delta_matrix = np.full((len(conditions), len(eval_categories)), np.nan)

    for ci, cid in enumerate(conditions):
        eval_res = all_results[cid].get("physbench_eval", {}).get("by_task_type", {})
        for cati, cat in enumerate(eval_categories):
            if cat in eval_res:
                delta_matrix[ci, cati] = eval_res[cat].get(
                    "delta", eval_res[cat]["accuracy"] - PHYSBENCH_BASELINE_VAL
                )

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(delta_matrix, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)
    ax.set_xticks(range(len(eval_categories)))
    ax.set_xticklabels(eval_categories, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(conditions)))
    ylabels = [f"{c}: {QLORA_CONDITIONS[c].label}" for c in conditions]
    ax.set_yticklabels(ylabels, fontsize=10)
    plt.colorbar(im, ax=ax, label="Accuracy Delta (%)")
    ax.set_title("PhysBench Accuracy Delta by Category and Condition", fontsize=13)

    for ci in range(len(conditions)):
        for cati in range(len(eval_categories)):
            val = delta_matrix[ci, cati]
            if not np.isnan(val):
                ax.text(cati, ci, f"{val:+.1f}", ha="center", va="center",
                        fontsize=9, color="black" if abs(val) < 5 else "white")

    plt.tight_layout()
    fig.savefig(str(fig_dir / "delta_heatmap.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved delta_heatmap.png")


def _fig_training_loss_curves(conditions, all_results, color_map, fig_dir):
    """Training and validation loss curves for all conditions overlaid."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for cid in conditions:
        loss_data = all_results[cid].get("loss_curves", {})
        train_losses = loss_data.get("training_losses", [])
        val_losses = loss_data.get("val_losses", [])
        label = f"{cid}: {QLORA_CONDITIONS[cid].label}"
        color = color_map.get(cid, "gray")

        if train_losses:
            ax1.plot(
                range(1, len(train_losses) + 1), train_losses,
                "o-", color=color, linewidth=2, markersize=6, label=label,
            )
        if val_losses:
            ax2.plot(
                range(1, len(val_losses) + 1), val_losses,
                "o-", color=color, linewidth=2, markersize=6, label=label,
            )

    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Training Loss", fontsize=11)
    ax1.set_title("Training Loss per Epoch", fontsize=13)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Validation Loss", fontsize=11)
    ax2.set_title("Validation Loss per Epoch", fontsize=13)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(fig_dir / "training_loss_curves.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved training_loss_curves.png")


# ===========================================================================
# 9. Run Single Condition (Full Pipeline)
# ===========================================================================

def run_single_condition(
    condition: QLoRACondition,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 1,
    gradient_accumulation: int = 16,
    learning_rate: float = 2e-4,
    physbench_samples: int = 200,
    probe_scenes: int = 100,
    patience: int = 3,
    seed: int = 42,
) -> Dict:
    """Run the full pipeline for one QLoRA condition."""
    print()
    print("#" * 70)
    print(f"  CONDITION {condition.id}: {condition.label}")
    print(f"  {condition.description}")
    print(f"  [{timestamp()}] Starting...")
    print("#" * 70)

    result = {
        "condition_id": condition.id,
        "condition_name": condition.name,
        "label": condition.label,
        "description": condition.description,
    }

    # Step 1: Load base model
    model, processor = load_base_model()

    # Step 2: Apply QLoRA
    peft_model, trainable = apply_qlora(model, condition)
    result["trainable_params"] = trainable

    # Step 3: Train with images
    adapter_path, train_time = train_qlora_condition(
        peft_model, processor, condition,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
        patience=patience,
    )
    result["adapter_path"] = adapter_path
    result["train_time_sec"] = train_time

    # Load loss curves
    loss_path = Path(output_dir) / condition.name / "loss_curves.json"
    if loss_path.exists():
        with open(loss_path, "r") as f:
            result["loss_curves"] = json.load(f)

    # Step 4: PhysBench evaluation
    eval_result = evaluate_after_training(
        peft_model, processor,
        condition_name=condition.name,
        output_dir=output_dir,
        max_samples=physbench_samples,
    )
    result["physbench_eval"] = eval_result

    # Step 5: Probing diagnostics
    probe_result = probe_after_training(
        peft_model, processor,
        condition_name=condition.name,
        output_dir=output_dir,
        num_scenes=probe_scenes,
        seed=seed,
    )
    result["probe_results"] = probe_result

    # Condition verdict
    acc = eval_result.get("overall_accuracy")
    if acc is not None:
        delta = acc - PHYSBENCH_BASELINE_VAL
        if delta > 2:
            verdict = "IMPROVED"
        elif delta < -2:
            verdict = "DEGRADED"
        else:
            verdict = "NEUTRAL"
        print(f"\n  [{timestamp()}] Condition {condition.id} VERDICT: {verdict} ({delta:+.2f}%)")

    result["total_time_sec"] = time.time()

    # Cleanup
    del peft_model, model, processor
    cleanup_gpu()

    # Checkpoint
    ckpt_path = Path(output_dir) / f"result_condition_{condition.id}.json"
    with open(ckpt_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Checkpointed to {ckpt_path}")

    return result


# ===========================================================================
# 10. Evaluate-Only Mode
# ===========================================================================

def evaluate_only(adapter_path: str, output_dir: str, max_samples: int = 200):
    """Load a saved adapter and run PhysBench + probing evaluation."""
    print(f"\n  === Evaluate-Only Mode ===")
    print(f"  Adapter: {adapter_path}")

    model, processor = load_base_model()

    from peft import PeftModel
    print(f"  Loading adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()
    print(f"  Adapter merged.")

    condition_name = Path(adapter_path).parent.name

    eval_result = evaluate_after_training(
        model, processor,
        condition_name=condition_name,
        output_dir=output_dir,
        max_samples=max_samples,
    )

    probe_result = probe_after_training(
        model, processor,
        condition_name=condition_name,
        output_dir=output_dir,
    )

    del model, processor
    cleanup_gpu()

    return {"physbench_eval": eval_result, "probe_results": probe_result}


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 3 Production: Full Image-Conditioned QLoRA Ablation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Conditions:
  merger          Condition A: QLoRA on visual.merger MLP only (rank=64)
  llm             Condition B: QLoRA on first 8 LLM layers Q/V (rank=16)
  encoder         Condition C: QLoRA on last 6 ViT blocks QKV (rank=16)
  merger+encoder  Condition D: A + C combined (rank=16)
  full            Condition E: All components (rank=16)
  all             Run all conditions sequentially

Examples:
  python scripts/run_qlora_full.py --condition merger --epochs 3
  python scripts/run_qlora_full.py --condition all --epochs 3 --resume
  python scripts/run_qlora_full.py --generate-data-only
  python scripts/run_qlora_full.py --evaluate-only --adapter-path results/qlora_full/merger/adapter
        """,
    )
    p.add_argument(
        "--condition", type=str, default="all",
        choices=["merger", "llm", "encoder", "merger+encoder", "full", "all"],
    )
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch size (default: 1 for 12GB VRAM with images)")
    p.add_argument("--gradient-accumulation", type=int, default=16,
                   help="Gradient accumulation steps (default: 16)")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=3,
                   help="Early stopping patience in epochs (default: 3)")
    p.add_argument("--physbench-samples", type=int, default=200)
    p.add_argument("--probe-scenes", type=int, default=100)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quantize", type=str, default="4bit", choices=["4bit", "8bit", "none"],
                   help="Model quantization (default: 4bit)")

    # Special modes
    p.add_argument("--generate-data-only", action="store_true",
                   help="Only generate physics QA data, then exit")
    p.add_argument("--evaluate-only", action="store_true",
                   help="Only evaluate a saved adapter")
    p.add_argument("--adapter-path", type=str, default=None)
    p.add_argument("--resume", action="store_true",
                   help="Skip conditions with existing checkpoints")

    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "qlora_full"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate-data-only mode ---
    if args.generate_data_only:
        data_dir = output_dir if args.output_dir else PROJECT_ROOT / "data" / "physics_qa_full"
        train_path, val_path = generate_image_conditioned_qa(data_dir)
        print(f"\n  Done! Train: {train_path}, Val: {val_path}")
        return

    # --- Evaluate-only mode ---
    if args.evaluate_only:
        if not args.adapter_path:
            print("ERROR: --evaluate-only requires --adapter-path")
            sys.exit(1)
        result = evaluate_only(args.adapter_path, str(output_dir), args.physbench_samples)
        acc = result.get("physbench_eval", {}).get("overall_accuracy", "?")
        print(f"\n  Done! Accuracy: {acc}%")
        return

    # --- Full experiment mode ---
    print()
    print("#" * 70)
    print("  PHASE 3 PRODUCTION: FULL IMAGE-CONDITIONED QLORA ABLATION")
    print("  Testing H3: Merger is the physics bottleneck")
    print("  Training: EVERY sample includes actual Physion++ scene image")
    print("#" * 70)

    # Step 1: Generate image-conditioned QA data
    qa_dir = PROJECT_ROOT / "data" / "physics_qa_full"
    train_path, val_path = generate_image_conditioned_qa(qa_dir)

    # Determine conditions
    if args.condition == "all":
        conditions_to_run = ["A", "B", "C", "D", "E"]
    else:
        cond_id = CONDITION_CLI_MAP.get(args.condition, args.condition)
        conditions_to_run = [cond_id]

    print(f"\n  Conditions: {conditions_to_run}")
    print(f"  Output dir: {output_dir}")
    print(f"  Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"  Grad accum: {args.gradient_accumulation}")
    print(f"  Effective batch: {args.batch_size * args.gradient_accumulation}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  PhysBench eval samples: {args.physbench_samples}")
    print(f"  Probe scenes: {args.probe_scenes}")
    print(f"  Quantization: {args.quantize}")

    # Resume support
    all_results = {}
    if args.resume:
        for cid in conditions_to_run[:]:
            ckpt = output_dir / f"result_condition_{cid}.json"
            if ckpt.exists():
                print(f"  Resuming: found checkpoint for condition {cid}")
                with open(ckpt, "r") as f:
                    all_results[cid] = json.load(f)
                conditions_to_run.remove(cid)
        if not conditions_to_run:
            print("  All conditions already completed!")

    # Run each condition
    for cid in conditions_to_run:
        condition = QLORA_CONDITIONS[cid]
        result = run_single_condition(
            condition=condition,
            train_data_path=train_path,
            val_data_path=val_path,
            output_dir=str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            physbench_samples=args.physbench_samples,
            probe_scenes=args.probe_scenes,
            patience=args.patience,
            seed=args.seed,
        )
        all_results[cid] = result

    # Hypothesis tracking
    print_hypothesis_tracker(all_results)

    # Visualization
    print("\n  --- Generating figures ---")
    generate_figures(all_results, str(output_dir))

    # Save combined results
    combined_path = output_dir / "qlora_full_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Final summary table
    print(f"\n{'=' * 70}")
    print(f"  QLORA FULL ABLATION SUMMARY (IMAGE-CONDITIONED)")
    print(f"{'=' * 70}")
    print(f"  {'Cond':<4} {'Label':<20} {'Params':>10} {'PhysBench':>10} {'Delta':>8} {'Time':>8}")
    print(f"  {'-' * 64}")
    print(f"  {'BASE':<4} {'Baseline':<20} {'---':>10} {PHYSBENCH_BASELINE_VAL:>9.2f}% {'---':>8} {'---':>8}")

    for cid in sorted(all_results.keys()):
        res = all_results[cid]
        label = QLORA_CONDITIONS[cid].label
        params = res.get("trainable_params", 0)
        acc = res.get("physbench_eval", {}).get("overall_accuracy")
        delta = res.get("physbench_eval", {}).get("delta")
        t = res.get("train_time_sec", 0)
        acc_str = f"{acc:.2f}%" if acc else "---"
        delta_str = f"{delta:+.2f}%" if delta is not None else "---"
        time_str = f"{t / 60:.0f}min" if t else "---"
        print(f"  {cid:<4} {label:<20} {params:>10,} {acc_str:>10} {delta_str:>8} {time_str:>8}")

    print(f"\n  Results: {combined_path}")
    print(f"  Figures: {output_dir / 'figures'}")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
