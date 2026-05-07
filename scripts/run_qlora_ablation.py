#!/usr/bin/env python3
"""
Phase 3: QLoRA Ablation Experiment — Option C Core Script.

Tests H3: "The visual-language merger is the primary bottleneck for physics
understanding in VLMs. Fine-tuning the merger alone improves physics
performance more than fine-tuning the LLM backbone."

5 QLoRA Conditions (Qwen2.5-VL-7B-Instruct):
  A (Merger only):     visual.merger MLP layers
  B (LLM only):        First 8 LLM decoder layers (Q/V projections)
  C (Encoder only):    Last 6 ViT blocks (Q/V projections)
  D (Merger+Encoder):  Conditions A + C combined
  E (Full):            All of the above

Pipeline per condition:
  1. Load base model (4-bit BitsAndBytes)
  2. Apply QLoRA with condition-specific target modules
  3. Train on physics QA data generated from Physion++ metadata
  4. Run PhysBench evaluation (val set, 200 samples)
  5. Run probing diagnostics (activations → ridge regression)
  6. Report deltas vs baseline and hypothesis verdict

Usage:
    # Run single condition
    python scripts/run_qlora_ablation.py --condition merger --epochs 3 --output-dir results/qlora_ablation

    # Run all conditions sequentially
    python scripts/run_qlora_ablation.py --condition all --epochs 3

    # Just generate training data
    python scripts/run_qlora_ablation.py --generate-data-only --output-dir data/physics_qa

    # Just evaluate a saved adapter
    python scripts/run_qlora_ablation.py --evaluate-only --adapter-path results/qlora_ablation/merger/adapter

Requirements:
    pip install torch transformers accelerate bitsandbytes peft qwen-vl-utils pillow
    pip install scikit-learn matplotlib h5py
"""

import argparse
import gc
import io
import json
import os
import pickle
import re
import sys
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
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

PHYSBENCH_BASELINE_VAL = 60.31  # Our measured baseline (val set, 200 samples)

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

# Condition name mapping
CONDITION_NAMES = {
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
    """Configuration for a single QLoRA ablation condition."""
    id: str                      # "A", "B", "C", "D", "E"
    name: str                    # "merger", "llm", "encoder", etc.
    label: str                   # Human-readable label
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
            # Qwen2.5-VL merger: 2-layer MLP that projects 1280→3584
            "visual.merger.mlp.0",
            "visual.merger.mlp.2",
        ],
        description=(
            "QLoRA on visual.merger MLP layers. These project visual tokens "
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
            # First 8 of 28 LLM decoder layers, Q and V projections
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
            # Last 6 of 32 ViT blocks, fused QKV projection
            # Qwen2.5-VL uses fused qkv in the encoder
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
            # Merger MLP
            "visual.merger.mlp.0",
            "visual.merger.mlp.2",
            # Last 6 encoder blocks
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
            # Encoder (last 6 blocks)
            *[f"visual.blocks.{i}.attn.qkv" for i in range(26, 32)],
            # Merger
            "visual.merger.mlp.0",
            "visual.merger.mlp.2",
            # LLM (first 8 layers)
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
    """Print current VRAM usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  {prefix}VRAM: {alloc:.2f}GB / {total:.1f}GB")


def cleanup_gpu():
    """Aggressive GPU memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ===========================================================================
# 1. Physics QA Training Data Generation
# ===========================================================================

def generate_physics_qa_from_physion(output_dir: Path) -> str:
    """
    Generate physics QA training pairs from Physion++ readout metadata.

    Reads .pkl files from physion_readout.zip, creates diverse QA pairs
    targeting mass, friction, and bounciness (elasticity) understanding.

    Returns path to the saved JSONL file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    qa_path = output_dir / "physics_qa_train.jsonl"

    if qa_path.exists():
        n_lines = sum(1 for _ in open(qa_path, "r", encoding="utf-8"))
        print(f"  Physics QA data already exists: {qa_path} ({n_lines} pairs)")
        return str(qa_path)

    zip_path = PROJECT_ROOT / "data" / "physion_readout.zip"
    if not zip_path.exists():
        print(f"  WARNING: Physion++ zip not found at {zip_path}")
        print(f"  Falling back to synthetic physics QA generation")
        return _generate_synthetic_physics_qa(output_dir)

    print(f"  Parsing Physion++ metadata from {zip_path}...")
    zf = zipfile.ZipFile(str(zip_path), "r")
    names = zf.namelist()
    pkl_files = sorted([n for n in names if n.endswith(".pkl")])
    map_files = sorted([n for n in names if n.endswith("_map.png")])

    # Build lookup: trial_prefix → map_png path
    map_lookup = {}
    for mf in map_files:
        prefix = mf.rsplit("/", 1)[0] if "/" in mf else ""
        if prefix not in map_lookup:
            map_lookup[prefix] = mf

    qa_pairs = []
    trials_parsed = 0

    for pkl_path in pkl_files:
        parts = pkl_path.split("/")
        scenario_type = parts[1] if len(parts) > 1 else "unknown"
        trial_name = parts[2] if len(parts) > 2 else "unknown"
        frame_num = Path(pkl_path).stem

        try:
            with zf.open(pkl_path) as f:
                data = pickle.load(io.BytesIO(f.read()))
        except Exception:
            continue

        static = data.get("static", {})
        masses = static.get("mass", np.array([]))
        dyn_friction = static.get("dynamic_friction", np.array([]))
        bounciness = static.get("bounciness", np.array([]))
        model_names = static.get("model_names", np.array([]))
        n_objects = len(masses) if hasattr(masses, '__len__') else 0

        if n_objects < 2:
            continue

        # Find corresponding frame image
        trial_prefix = pkl_path.rsplit("/", 1)[0] if "/" in pkl_path else ""
        image_ref = f"physion_readout/{trial_prefix}/{frame_num}_map.png"

        # --- Mass comparison questions ---
        if hasattr(masses, '__len__') and len(masses) >= 2:
            real_masses = [(j, float(m)) for j, m in enumerate(masses)
                          if not np.isnan(m) and 0.001 < m < 500]
            if len(real_masses) >= 2:
                sorted_by_mass = sorted(real_masses, key=lambda x: x[1], reverse=True)
                heavy_idx, heavy_m = sorted_by_mass[0]
                light_idx, light_m = sorted_by_mass[-1]

                # Only if there's a meaningful mass difference (>20%)
                if heavy_m > light_m * 1.2:
                    # Q1: Which is heavier?
                    qa_pairs.append({
                        "image_path": image_ref,
                        "question": "Which object in this scene is heavier?",
                        "answer": (
                            f"Object {heavy_idx + 1} is heavier (mass ≈ {heavy_m:.2f} kg) "
                            f"compared to object {light_idx + 1} (mass ≈ {light_m:.2f} kg). "
                            f"The mass ratio is approximately {heavy_m / light_m:.1f}:1."
                        ),
                        "physics_property": "mass",
                        "scenario": scenario_type,
                        "trial": trial_name,
                    })

                    # Q2: What happens if they collide? (mass-dependent)
                    if heavy_m > light_m * 3:
                        qa_pairs.append({
                            "image_path": image_ref,
                            "question": (
                                "If these two objects collide, which one will experience "
                                "a greater change in velocity?"
                            ),
                            "answer": (
                                f"Object {light_idx + 1} (the lighter object, ~{light_m:.2f} kg) "
                                f"will experience a much greater velocity change than "
                                f"object {heavy_idx + 1} (~{heavy_m:.2f} kg). By conservation "
                                f"of momentum, the lighter object's velocity changes inversely "
                                f"with its mass ratio ({heavy_m / light_m:.1f}x more)."
                            ),
                            "physics_property": "mass",
                            "scenario": scenario_type,
                            "trial": trial_name,
                        })

                    # Q3: Relative inertia
                    qa_pairs.append({
                        "image_path": image_ref,
                        "question": "Which object in this scene has greater inertia and is harder to accelerate?",
                        "answer": (
                            f"Object {heavy_idx + 1} has greater inertia (mass ≈ {heavy_m:.2f} kg) "
                            f"and requires approximately {heavy_m / light_m:.1f}x more force to "
                            f"achieve the same acceleration as object {light_idx + 1} "
                            f"(mass ≈ {light_m:.2f} kg), according to Newton's second law F=ma."
                        ),
                        "physics_property": "mass",
                        "scenario": scenario_type,
                        "trial": trial_name,
                    })

        # --- Bounciness / elasticity questions ---
        if hasattr(bounciness, '__len__') and len(bounciness) >= 2:
            valid_bounce = [(j, float(b)) for j, b in enumerate(bounciness)
                           if not np.isnan(b)]
            if len(valid_bounce) >= 2:
                sorted_bounce = sorted(valid_bounce, key=lambda x: x[1], reverse=True)
                bouncy_idx, bouncy_val = sorted_bounce[0]
                flat_idx, flat_val = sorted_bounce[-1]

                # Q4: Will they bounce?
                if bouncy_val > 0.5:
                    qa_pairs.append({
                        "image_path": image_ref,
                        "question": "Will the objects bounce significantly after collision?",
                        "answer": (
                            f"Yes, at least object {bouncy_idx + 1} has high elasticity "
                            f"(restitution coefficient ≈ {bouncy_val:.2f}), meaning it will "
                            f"bounce significantly. A coefficient of {bouncy_val:.2f} means "
                            f"it retains ~{bouncy_val * 100:.0f}% of its relative velocity "
                            f"after collision."
                        ),
                        "physics_property": "elasticity",
                        "scenario": scenario_type,
                        "trial": trial_name,
                    })
                elif bouncy_val < 0.3:
                    qa_pairs.append({
                        "image_path": image_ref,
                        "question": "Will the objects bounce significantly after collision?",
                        "answer": (
                            f"No, the objects have low elasticity (max restitution ≈ "
                            f"{bouncy_val:.2f}). The collision will be largely inelastic, "
                            f"with most kinetic energy absorbed. The objects will decelerate "
                            f"quickly on contact rather than bouncing."
                        ),
                        "physics_property": "elasticity",
                        "scenario": scenario_type,
                        "trial": trial_name,
                    })

                # Q5: Compare bounciness
                if abs(bouncy_val - flat_val) > 0.2:
                    qa_pairs.append({
                        "image_path": image_ref,
                        "question": "Which object is more elastic and will bounce more?",
                        "answer": (
                            f"Object {bouncy_idx + 1} is more elastic (restitution ≈ "
                            f"{bouncy_val:.2f}) compared to object {flat_idx + 1} "
                            f"(restitution ≈ {flat_val:.2f}). In a collision, object "
                            f"{bouncy_idx + 1} will retain more kinetic energy and "
                            f"bounce back faster."
                        ),
                        "physics_property": "elasticity",
                        "scenario": scenario_type,
                        "trial": trial_name,
                    })

        # --- Friction questions ---
        if hasattr(dyn_friction, '__len__') and len(dyn_friction) >= 1:
            valid_friction = [(j, float(f)) for j, f in enumerate(dyn_friction)
                             if not np.isnan(f)]
            if valid_friction:
                sorted_fric = sorted(valid_friction, key=lambda x: x[1], reverse=True)
                rough_idx, rough_val = sorted_fric[0]
                smooth_idx, smooth_val = sorted_fric[-1]

                # Q6: Surface friction
                if rough_val > 0.5:
                    qa_pairs.append({
                        "image_path": image_ref,
                        "question": "How much friction does the surface have?",
                        "answer": (
                            f"The surface has high friction (dynamic friction coefficient ≈ "
                            f"{rough_val:.2f}). Objects sliding on this surface will decelerate "
                            f"quickly. A coefficient above 0.5 indicates a rough surface "
                            f"similar to rubber on concrete."
                        ),
                        "physics_property": "friction",
                        "scenario": scenario_type,
                        "trial": trial_name,
                    })
                elif smooth_val < 0.2:
                    qa_pairs.append({
                        "image_path": image_ref,
                        "question": "How much friction does the surface have?",
                        "answer": (
                            f"The surface has low friction (dynamic friction coefficient ≈ "
                            f"{smooth_val:.2f}). Objects will slide easily with minimal "
                            f"deceleration, similar to ice or a polished surface."
                        ),
                        "physics_property": "friction",
                        "scenario": scenario_type,
                        "trial": trial_name,
                    })

                # Q7: Compare friction between objects
                if len(valid_friction) >= 2 and abs(rough_val - smooth_val) > 0.15:
                    qa_pairs.append({
                        "image_path": image_ref,
                        "question": (
                            "Which object will slide farther on a flat surface "
                            "if given the same initial push?"
                        ),
                        "answer": (
                            f"Object {smooth_idx + 1} will slide farther because it has "
                            f"lower friction (μ ≈ {smooth_val:.2f}) compared to "
                            f"object {rough_idx + 1} (μ ≈ {rough_val:.2f}). "
                            f"The deceleration due to friction is a = μg, so object "
                            f"{smooth_idx + 1} decelerates ~{rough_val / max(smooth_val, 0.01):.1f}x "
                            f"slower."
                        ),
                        "physics_property": "friction",
                        "scenario": scenario_type,
                        "trial": trial_name,
                    })

        # --- Combined physics reasoning questions ---
        if (hasattr(masses, '__len__') and len(masses) >= 2 and
                hasattr(dyn_friction, '__len__') and len(dyn_friction) >= 2):
            real_masses = [(j, float(m)) for j, m in enumerate(masses)
                          if not np.isnan(m) and 0.001 < m < 500]
            valid_friction = [(j, float(f)) for j, f in enumerate(dyn_friction)
                             if not np.isnan(f)]
            if real_masses and valid_friction:
                heaviest_idx, heaviest_m = max(real_masses, key=lambda x: x[1])
                max_fric_idx, max_fric = max(valid_friction, key=lambda x: x[1])

                # Q8: Combined mass + friction
                qa_pairs.append({
                    "image_path": image_ref,
                    "question": (
                        "Considering both mass and friction, which object requires "
                        "the most force to start moving?"
                    ),
                    "answer": (
                        f"To start an object moving, you need force F = μ_s × m × g. "
                        f"Object {heaviest_idx + 1} has mass ≈ {heaviest_m:.2f} kg. "
                        f"The object with highest friction has μ ≈ {max_fric:.2f}. "
                        f"The required force depends on the product of mass and friction "
                        f"coefficient — heavier objects on rougher surfaces need more force."
                    ),
                    "physics_property": "mass",
                    "scenario": scenario_type,
                    "trial": trial_name,
                })

        trials_parsed += 1
        if trials_parsed % 50 == 0:
            print(f"    Parsed {trials_parsed} trials, {len(qa_pairs)} QA pairs so far")

    zf.close()

    print(f"  Generated {len(qa_pairs)} QA pairs from {trials_parsed} Physion++ trials")

    # Property distribution
    prop_counts = defaultdict(int)
    for qa in qa_pairs:
        prop_counts[qa["physics_property"]] += 1
    print(f"  Distribution: {dict(prop_counts)}")

    # Save as JSONL
    with open(qa_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")
    print(f"  Saved to {qa_path}")

    return str(qa_path)


def _generate_synthetic_physics_qa(output_dir: Path) -> str:
    """Fallback: generate synthetic physics QA when Physion++ is unavailable."""
    qa_path = output_dir / "physics_qa_train.jsonl"
    rng = np.random.RandomState(42)

    qa_pairs = []
    for i in range(800):
        mass_a = rng.uniform(0.1, 50.0)
        mass_b = rng.uniform(0.1, 50.0)
        friction = rng.uniform(0.05, 1.5)
        bounce = rng.uniform(0.0, 1.0)

        # Mass question
        heavier = "A" if mass_a > mass_b else "B"
        qa_pairs.append({
            "image_path": f"synthetic/scene_{i:04d}.png",
            "question": "Which object is heavier, object A or object B?",
            "answer": (
                f"Object {heavier} is heavier. Object A has mass {mass_a:.1f} kg "
                f"and object B has mass {mass_b:.1f} kg, making {heavier} "
                f"{max(mass_a, mass_b) / min(mass_a, mass_b):.1f}x heavier."
            ),
            "physics_property": "mass",
            "scenario": "synthetic",
            "trial": f"scene_{i:04d}",
        })

        # Friction question
        if i % 3 == 0:
            desc = "very rough" if friction > 1.0 else "moderate" if friction > 0.3 else "very smooth"
            qa_pairs.append({
                "image_path": f"synthetic/scene_{i:04d}.png",
                "question": "How much friction does the surface have in this scene?",
                "answer": (
                    f"The surface friction coefficient is approximately {friction:.2f}, "
                    f"which is {desc}. Objects will "
                    f"{'decelerate quickly' if friction > 0.5 else 'slide easily'}."
                ),
                "physics_property": "friction",
                "scenario": "synthetic",
                "trial": f"scene_{i:04d}",
            })

        # Bounce question
        if i % 3 == 1:
            qa_pairs.append({
                "image_path": f"synthetic/scene_{i:04d}.png",
                "question": "Will these objects bounce after collision?",
                "answer": (
                    f"The restitution coefficient is {bounce:.2f}. "
                    f"{'Yes, they will bounce significantly' if bounce > 0.5 else 'No, the collision is mostly inelastic'}. "
                    f"About {bounce * 100:.0f}% of relative velocity is preserved after impact."
                ),
                "physics_property": "elasticity",
                "scenario": "synthetic",
                "trial": f"scene_{i:04d}",
            })

    with open(qa_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    print(f"  Generated {len(qa_pairs)} synthetic QA pairs → {qa_path}")
    return str(qa_path)


def load_qa_data(qa_path: str, max_samples: int = None) -> List[Dict]:
    """Load QA pairs from JSONL file."""
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
    """Load Qwen2.5-VL-7B with optional quantization. Returns (model, processor)."""
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"\n  Loading Qwen2.5-VL-7B-Instruct ({quantize})...")

    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        # Try pre-quantized first (faster), fall back to on-the-fly quantization
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

def apply_qlora(model, condition: QLoRACondition) -> Tuple[Any, int]:
    """Apply QLoRA adapters for the given condition. Returns (peft_model, trainable_count)."""
    from peft import LoraConfig, TaskType, get_peft_model

    # Validate which target modules exist in the model
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
    print(f"    Rank: {condition.rank}")
    print(f"    Target modules ({len(valid_targets)}):")
    for t in valid_targets:
        print(f"      - {t}")
    print(f"    Trainable params: {trainable:,} ({trainable / 1e6:.2f}M)")
    print(f"    Total params: {total:,} ({total / 1e9:.2f}B)")
    print(f"    Trainable %: {100 * trainable / total:.4f}%")
    report_vram("After QLoRA: ")

    return peft_model, trainable


# ===========================================================================
# 4. QLoRA Training
# ===========================================================================

def train_qlora_condition(
    model,
    processor,
    condition: QLoRACondition,
    train_data_path: str,
    output_dir: str,
    rank: int = 8,
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation: int = 8,
    learning_rate: float = 2e-4,
):
    """
    Train a QLoRA condition on physics QA data.

    Uses manual training loop (not HF Trainer) for maximum control over
    VRAM and progress reporting on a 12GB GPU.
    """
    from PIL import Image as PILImage

    qa_data = load_qa_data(train_data_path)
    device = next(model.parameters()).device

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=0.01,
    )

    # Linear warmup + cosine decay
    total_steps = (len(qa_data) // batch_size) * epochs
    warmup_steps = min(100, total_steps // 10)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    model.train()
    global_step = 0
    total_loss = 0.0
    best_loss = float("inf")

    print(f"\n  === Training Condition {condition.id}: {condition.label} ===")
    print(f"  QA pairs: {len(qa_data)}")
    print(f"  Epochs: {epochs}, Batch: {batch_size}, Grad Accum: {gradient_accumulation}")
    print(f"  Effective batch: {batch_size * gradient_accumulation}")
    print(f"  Total steps: ~{total_steps}")

    adapter_dir = Path(output_dir) / condition.name / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        np.random.shuffle(qa_data)

        for batch_start in range(0, len(qa_data), batch_size):
            batch = qa_data[batch_start:batch_start + batch_size]
            batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            valid_count = 0

            for sample in batch:
                question = sample["question"]
                answer = sample["answer"]

                # Build chat messages (text-only for now; images optional)
                messages = [
                    {"role": "user", "content": [
                        {"type": "text", "text": question},
                    ]},
                    {"role": "assistant", "content": answer},
                ]

                try:
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=False
                    )
                    inputs = processor(
                        text=[text], return_tensors="pt", padding=True
                    )
                    inputs = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in inputs.items()
                    }
                    inputs["labels"] = inputs["input_ids"].clone()

                    outputs = model(**inputs)
                    batch_loss = batch_loss + outputs.loss
                    valid_count += 1

                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    gc.collect()
                    print(f"    OOM at step {global_step}, skipping sample")
                    continue
                except Exception as e:
                    if global_step < 5:
                        print(f"    Training error at step {global_step}: {e}")
                    continue

            if valid_count > 0:
                loss = batch_loss / valid_count

                # Gradient accumulation
                scaled_loss = loss / gradient_accumulation
                scaled_loss.backward()

                if (global_step + 1) % gradient_accumulation == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                epoch_loss += loss.item()
                total_loss += loss.item()
                epoch_steps += 1
                global_step += 1

                # Progress reporting
                if global_step % 25 == 0:
                    avg = epoch_loss / max(epoch_steps, 1)
                    elapsed = time.time() - t_start
                    rate = global_step / elapsed
                    eta = (total_steps - global_step) / max(rate, 0.001)
                    lr_now = scheduler.get_last_lr()[0]
                    print(
                        f"    [Epoch {epoch + 1}/{epochs}] Step {global_step}/{total_steps} | "
                        f"Loss: {avg:.4f} | LR: {lr_now:.2e} | "
                        f"{rate:.1f} step/s | ETA: {eta / 60:.1f}min"
                    )
                    report_vram("    ")

            del batch_loss
            torch.cuda.empty_cache()

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch + 1}/{epochs} complete: avg_loss={avg_epoch_loss:.4f}")

        # Save best checkpoint
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_pretrained(str(adapter_dir))
            print(f"    Saved best adapter to {adapter_dir}")

    # Final save
    model.save_pretrained(str(adapter_dir))
    elapsed = time.time() - t_start
    print(f"  Training complete: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    print(f"  Best loss: {best_loss:.4f}")
    print(f"  Adapter saved to: {adapter_dir}")

    return str(adapter_dir), elapsed


# ===========================================================================
# 5. PhysBench Evaluation After Training
# ===========================================================================

def evaluate_after_training(
    model,
    processor,
    condition_name: str,
    output_dir: str,
    max_samples: int = 200,
    split: str = "val",
) -> Dict:
    """
    Run PhysBench evaluation with the trained model.

    Uses the same evaluation logic as run_physbench_eval.py but with
    a model already in memory (avoids reload).
    """
    from qwen_vl_utils import process_vision_info

    data_dir = PROJECT_ROOT / "data" / "physbench"
    json_path = data_dir / f"{split}.json"

    if not json_path.exists():
        print(f"  WARNING: PhysBench {split} data not found at {json_path}")
        return {"overall_accuracy": None, "error": f"Data not found: {json_path}"}

    # Load data
    with open(json_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if content.startswith("["):
        data = json.loads(content)
    else:
        data = [json.loads(line) for line in content.splitlines() if line.strip()]

    if max_samples:
        data = data[:max_samples]

    print(f"\n  === PhysBench Evaluation ({condition_name}) ===")
    print(f"  Split: {split}, Samples: {len(data)}")

    model.eval()
    correct = 0
    total = 0
    errors = 0
    by_task_type = defaultdict(lambda: {"correct": 0, "total": 0})

    t_start = time.time()

    for i, item in enumerate(data):
        # Resolve media
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

        # Format question
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

            # Extract answer
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
            print(f"    [{i + 1}/{len(data)}] Acc: {acc:.1f}% ({correct}/{total})")

    elapsed = time.time() - t_start
    accuracy = correct / max(total, 1) * 100
    delta = accuracy - PHYSBENCH_BASELINE_VAL

    per_domain = {}
    for k, v in sorted(by_task_type.items()):
        per_domain[k] = {
            "accuracy": v["correct"] / max(v["total"], 1) * 100,
            "correct": v["correct"],
            "total": v["total"],
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

    # Save
    eval_dir = Path(output_dir) / condition_name
    eval_dir.mkdir(parents=True, exist_ok=True)
    eval_path = eval_dir / f"physbench_{split}_results.json"
    with open(eval_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n  PhysBench Result ({condition_name}):")
    print(f"    Accuracy: {accuracy:.2f}% (baseline: {PHYSBENCH_BASELINE_VAL}%)")
    print(f"    Delta: {delta:+.2f}%")
    print(f"    Per domain:")
    for k, v in per_domain.items():
        print(f"      {k:20s}: {v['accuracy']:.1f}% ({v['correct']}/{v['total']})")

    return result


# ===========================================================================
# 6. Probing After Training (Diagnostic)
# ===========================================================================

def probe_after_training(
    model,
    processor,
    condition_name: str,
    output_dir: str,
    num_scenes: int = 100,
    seed: int = 42,
) -> Dict:
    """
    Extract activations at 4 pipeline stages and run linear probing.

    Compares R² for mass/friction/elasticity before and after training.
    This tells us whether fine-tuning changed internal representations.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    print(f"\n  === Probing Diagnostics ({condition_name}) ===")

    # Register hooks at 4 stages
    hook_storage = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hook_storage[name] = output[0].detach().cpu()
            elif isinstance(output, torch.Tensor):
                hook_storage[name] = output.detach().cpu()
        return hook_fn

    # Navigate model hierarchy (handle both bare and PEFT-wrapped models)
    def get_module(path_parts):
        """Navigate model by attribute path, handling PEFT wrapping."""
        obj = model
        # If PEFT-wrapped, try base_model.model first
        if hasattr(obj, "base_model"):
            obj = obj.base_model
            if hasattr(obj, "model"):
                obj = obj.model
        for part in path_parts:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    hook_defs = [
        ("stage_1_enc_out", ["visual", "blocks", "-1"]),
        ("stage_2_post_proj", ["visual", "merger"]),
        ("stage_3_llm_8", ["model", "layers", "8"]),
        ("stage_4_llm_16", ["model", "layers", "16"]),
    ]

    for stage_name, path_parts in hook_defs:
        try:
            # Handle negative indexing
            adjusted = []
            for p in path_parts:
                if p == "-1":
                    adjusted.append("-1")
                else:
                    adjusted.append(p)

            obj = model
            if hasattr(obj, "base_model"):
                obj = obj.base_model
                if hasattr(obj, "model"):
                    obj = obj.model

            for p in adjusted:
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

    # Use deconfounded scenes for probing (consistent with Phase 2)
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

        # Collect activations
        image_grid_thw = inputs.get("image_grid_thw", None)
        n_vis = int(image_grid_thw[0].prod().item()) if image_grid_thw is not None else None

        for stage_name in STAGE_NAMES:
            if stage_name not in hook_storage:
                continue
            act = hook_storage[stage_name]
            if act.ndim == 3:
                act = act.squeeze(0)
            # For LLM stages, extract only visual token positions
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

        # Physics labels
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
            print(f"    Extracted {i + 1}/{num_scenes} scenes")

    for h in hooks:
        h.remove()

    # Concatenate
    activations = {}
    for s in STAGE_NAMES:
        if all_acts[s]:
            activations[s] = np.concatenate(all_acts[s], axis=0)
    if all_labels:
        activations["physics_labels"] = np.concatenate(all_labels, axis=0)

    # Run probing
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

    # Save
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
# 7. Hypothesis Tracker
# ===========================================================================

def print_hypothesis_tracker(all_results: Dict[str, Dict]):
    """Print the hypothesis tracking summary."""
    print("\n" + "=" * 70)
    print("  === HYPOTHESIS TRACKER ===")
    print("=" * 70)
    print()
    print("  H3 Prediction: Merger QLoRA > LLM QLoRA for physics")
    print(f"  Expected: Condition A (merger) improves PhysBench by 3-5%")
    print(f"           Condition B (LLM) improves PhysBench by 0-2%")
    print()
    print(f"  BASELINE: {PHYSBENCH_BASELINE_VAL}% (val set)")
    print()

    # Per-condition summary
    for cond_id in ["A", "B", "C", "D", "E"]:
        if cond_id not in all_results:
            continue
        res = all_results[cond_id]
        cond_def = QLORA_CONDITIONS[cond_id]

        print(f"  Condition {cond_id} ({cond_def.label}):")
        print(f"    Training params: {res.get('trainable_params', '?'):,}")
        print(f"    Training time: {res.get('train_time_sec', 0) / 60:.1f} min")

        eval_res = res.get("physbench_eval", {})
        if eval_res.get("overall_accuracy") is not None:
            acc = eval_res["overall_accuracy"]
            delta = eval_res.get("delta", acc - PHYSBENCH_BASELINE_VAL)
            print(f"    PhysBench AFTER: {acc:.2f}%")
            print(f"    Delta: {delta:+.2f}%")
        else:
            print(f"    PhysBench AFTER: NOT YET RUN")

        probe_res = res.get("probe_results", {})
        for var in ["mass", "friction", "elasticity"]:
            enc_r2 = probe_res.get("stage_1_enc_out", {}).get(var, {}).get("r2", "?")
            llm_r2 = probe_res.get("stage_4_llm_16", {}).get(var, {}).get("r2", "?")
            if isinstance(enc_r2, float) and isinstance(llm_r2, float):
                print(f"    R² {var} (encoder→LLM-16): {enc_r2:.3f} → {llm_r2:.3f}")

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
            reason = f"Merger ({a_acc:.1f}%) ≈ LLM ({b_acc:.1f}%)"
        print(f"  === H3 VERDICT: {verdict} ===")
        print(f"  Reason: {reason}")
    else:
        print("  === H3 VERDICT: PENDING (need both A and B results) ===")

    print("=" * 70)


# ===========================================================================
# 8. Visualization
# ===========================================================================

def generate_figures(all_results: Dict[str, Dict], output_dir: str):
    """Generate comparison figures across conditions."""
    fig_dir = Path(output_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    conditions = sorted(all_results.keys())
    if not conditions:
        return

    # --- Figure 1: PhysBench accuracy bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    cond_labels = []
    accuracies = []
    colors = []
    color_map = {"A": "#e74c3c", "B": "#3498db", "C": "#2ecc71", "D": "#9b59b6", "E": "#f39c12"}

    # Baseline bar
    cond_labels.append("Baseline")
    accuracies.append(PHYSBENCH_BASELINE_VAL)
    colors.append("#95a5a6")

    for cid in conditions:
        res = all_results[cid]
        acc = res.get("physbench_eval", {}).get("overall_accuracy")
        if acc is not None:
            cond_def = QLORA_CONDITIONS.get(cid)
            label = f"{cid}: {cond_def.label}" if cond_def else cid
            cond_labels.append(label)
            accuracies.append(acc)
            colors.append(color_map.get(cid, "#7f8c8d"))

    x = np.arange(len(cond_labels))
    bars = ax.bar(x, accuracies, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=PHYSBENCH_BASELINE_VAL, color="gray", linestyle="--", alpha=0.5, label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(cond_labels, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("PhysBench Accuracy (%)", fontsize=12)
    ax.set_title("QLoRA Ablation: PhysBench Accuracy per Condition", fontsize=14)
    ax.set_ylim(max(0, min(accuracies) - 5), max(accuracies) + 5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(str(fig_dir / "physbench_accuracy_comparison.png"), dpi=300)
    plt.close(fig)
    print(f"  Saved PhysBench accuracy chart")

    # --- Figure 2: Probing R² degradation curves (before vs after) ---
    for var in ["mass", "friction", "elasticity"]:
        fig, axes = plt.subplots(1, len(conditions), figsize=(4 * len(conditions), 5), sharey=True)
        if len(conditions) == 1:
            axes = [axes]

        for ci, cid in enumerate(conditions):
            ax = axes[ci]
            probe = all_results[cid].get("probe_results", {})
            r2_vals = []
            stage_lbls = []

            for si, stage in enumerate(STAGE_NAMES):
                r2 = probe.get(stage, {}).get(var, {}).get("r2")
                if r2 is not None:
                    r2_vals.append(r2)
                    stage_lbls.append(STAGE_LABELS[si])

            if r2_vals:
                ax.plot(range(len(r2_vals)), r2_vals, "o-", color=color_map.get(cid, "gray"),
                        linewidth=2, markersize=8)
                ax.set_xticks(range(len(stage_lbls)))
                ax.set_xticklabels(stage_lbls, rotation=45, ha="right", fontsize=8)

            cond_def = QLORA_CONDITIONS.get(cid)
            ax.set_title(f"{cid}: {cond_def.label if cond_def else cid}", fontsize=10)
            if ci == 0:
                ax.set_ylabel(f"R² ({var})", fontsize=11)
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Probing R² After QLoRA: {var.title()}", fontsize=13)
        plt.tight_layout()
        fig.savefig(str(fig_dir / f"probe_r2_{var}.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"  Saved probing degradation curves")

    # --- Figure 3: Delta heatmap (conditions × physics categories) ---
    eval_categories = set()
    for cid in conditions:
        cats = all_results[cid].get("physbench_eval", {}).get("by_task_type", {})
        eval_categories.update(cats.keys())
    eval_categories = sorted(eval_categories)

    if eval_categories and len(conditions) >= 2:
        delta_matrix = np.full((len(conditions), len(eval_categories)), np.nan)

        # Get baseline per-category accuracies (from val results file)
        baseline_path = (
            PROJECT_ROOT / "results" / "physbench" /
            "Qwen2.5-VL-7B-Instruct_baseline" / "physbench_summary.json"
        )
        baseline_by_type = {}
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                baseline_summary = json.load(f)
            baseline_by_type = {
                k: v["accuracy"]
                for k, v in baseline_summary.get("by_task_type", {}).items()
            }

        for ci, cid in enumerate(conditions):
            eval_res = all_results[cid].get("physbench_eval", {}).get("by_task_type", {})
            for cati, cat in enumerate(eval_categories):
                if cat in eval_res:
                    after_acc = eval_res[cat]["accuracy"]
                    base_acc = baseline_by_type.get(cat, PHYSBENCH_BASELINE_VAL)
                    delta_matrix[ci, cati] = after_acc - base_acc

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
        fig.savefig(str(fig_dir / "delta_heatmap_categories.png"), dpi=300)
        plt.close(fig)
        print(f"  Saved delta heatmap")

    print(f"  All figures saved to {fig_dir}")


# ===========================================================================
# 9. Run Single Condition (Full Pipeline)
# ===========================================================================

def run_single_condition(
    condition: QLoRACondition,
    train_data_path: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    gradient_accumulation: int = 8,
    learning_rate: float = 2e-4,
    physbench_samples: int = 200,
    probe_scenes: int = 100,
    seed: int = 42,
) -> Dict:
    """Run the full pipeline for one QLoRA condition."""
    print(f"\n{'#' * 70}")
    print(f"  CONDITION {condition.id}: {condition.label}")
    print(f"  {condition.description}")
    print(f"{'#' * 70}")

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

    # Step 3: Train
    t0 = time.time()
    adapter_path, train_time = train_qlora_condition(
        peft_model, processor, condition,
        train_data_path=train_data_path,
        output_dir=output_dir,
        rank=condition.rank,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        learning_rate=learning_rate,
    )
    result["adapter_path"] = adapter_path
    result["train_time_sec"] = train_time

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

    # Step 6: Print condition verdict
    acc = eval_result.get("overall_accuracy")
    if acc is not None:
        delta = acc - PHYSBENCH_BASELINE_VAL
        if delta > 2:
            verdict = "IMPROVED"
        elif delta < -2:
            verdict = "DEGRADED"
        else:
            verdict = "NEUTRAL"
        print(f"\n  Condition {condition.id} VERDICT: {verdict} ({delta:+.2f}%)")
    result["total_time_sec"] = time.time() - t0

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
    """Load a saved adapter and run PhysBench evaluation."""
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
    result = evaluate_after_training(
        model, processor,
        condition_name=condition_name,
        output_dir=output_dir,
        max_samples=max_samples,
    )

    del model, processor
    cleanup_gpu()

    return result


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 3: QLoRA Ablation Experiment for Physics VLM Probing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Conditions:
  merger          Condition A: QLoRA on visual.merger MLP only
  llm             Condition B: QLoRA on first 8 LLM layers (Q/V)
  encoder         Condition C: QLoRA on last 6 ViT blocks (QKV)
  merger+encoder  Condition D: Conditions A + C combined
  full            Condition E: All components
  all             Run all conditions sequentially

Examples:
  python scripts/run_qlora_ablation.py --condition merger --epochs 3
  python scripts/run_qlora_ablation.py --condition all --epochs 3
  python scripts/run_qlora_ablation.py --generate-data-only
  python scripts/run_qlora_ablation.py --evaluate-only --adapter-path results/qlora_ablation/merger/adapter
        """,
    )
    p.add_argument(
        "--condition",
        type=str,
        default="all",
        choices=["merger", "llm", "encoder", "merger+encoder", "full", "all"],
        help="Which condition to run (default: all)",
    )
    p.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    p.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    p.add_argument("--gradient-accumulation", type=int, default=8, help="Gradient accumulation steps (default: 8)")
    p.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate (default: 2e-4)")
    p.add_argument("--physbench-samples", type=int, default=200, help="PhysBench val samples (default: 200)")
    p.add_argument("--probe-scenes", type=int, default=100, help="Probing scenes (default: 100)")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Special modes
    p.add_argument("--generate-data-only", action="store_true",
                   help="Only generate physics QA training data, then exit")
    p.add_argument("--evaluate-only", action="store_true",
                   help="Only evaluate a saved adapter (requires --adapter-path)")
    p.add_argument("--adapter-path", type=str, default=None,
                   help="Path to saved LoRA adapter (for --evaluate-only)")
    p.add_argument("--resume", action="store_true",
                   help="Skip conditions that already have checkpoints")

    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Default output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "results" / "qlora_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate-data-only mode ---
    if args.generate_data_only:
        data_dir = output_dir if args.output_dir else PROJECT_ROOT / "data" / "physics_qa"
        qa_path = generate_physics_qa_from_physion(data_dir)
        print(f"\n  Done! QA data at: {qa_path}")
        return

    # --- Evaluate-only mode ---
    if args.evaluate_only:
        if not args.adapter_path:
            print("ERROR: --evaluate-only requires --adapter-path")
            sys.exit(1)
        result = evaluate_only(args.adapter_path, str(output_dir), args.physbench_samples)
        print(f"\n  Done! Accuracy: {result.get('overall_accuracy', '?')}%")
        return

    # --- Full experiment mode ---
    # Step 1: Generate training data
    print("\n" + "#" * 70)
    print("  PHASE 3: QLORA ABLATION EXPERIMENT")
    print("  Testing H3: Merger is the physics bottleneck")
    print("#" * 70)

    qa_dir = PROJECT_ROOT / "data" / "physics_qa"
    qa_path = generate_physics_qa_from_physion(qa_dir)

    # Determine conditions to run
    if args.condition == "all":
        conditions_to_run = ["A", "B", "C", "D", "E"]
    else:
        cond_id = CONDITION_NAMES.get(args.condition, args.condition)
        conditions_to_run = [cond_id]

    print(f"\n  Conditions: {conditions_to_run}")
    print(f"  Output dir: {output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  PhysBench eval samples: {args.physbench_samples}")
    print(f"  Probe scenes: {args.probe_scenes}")

    # Check for existing checkpoints (resume support)
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
            train_data_path=qa_path,
            output_dir=str(output_dir),
            epochs=args.epochs,
            batch_size=args.batch_size,
            gradient_accumulation=args.gradient_accumulation,
            learning_rate=args.learning_rate,
            physbench_samples=args.physbench_samples,
            probe_scenes=args.probe_scenes,
            seed=args.seed,
        )
        all_results[cid] = result

    # Hypothesis tracking
    print_hypothesis_tracker(all_results)

    # Visualization
    print("\n  --- Generating figures ---")
    generate_figures(all_results, str(output_dir))

    # Save combined results
    combined_path = output_dir / "qlora_ablation_combined.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Final summary table
    print(f"\n{'=' * 70}")
    print(f"  QLORA ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Cond':<4} {'Label':<20} {'Params':>10} {'PhysBench':>10} {'Delta':>8} {'Time':>8}")
    print(f"  {'-' * 64}")
    print(f"  {'BASE':<4} {'Baseline':<20} {'—':>10} {PHYSBENCH_BASELINE_VAL:>9.2f}% {'—':>8} {'—':>8}")

    for cid in sorted(all_results.keys()):
        res = all_results[cid]
        label = QLORA_CONDITIONS[cid].label
        params = res.get("trainable_params", 0)
        acc = res.get("physbench_eval", {}).get("overall_accuracy")
        delta = res.get("physbench_eval", {}).get("delta")
        t = res.get("total_time_sec", 0)
        acc_str = f"{acc:.2f}%" if acc else "—"
        delta_str = f"{delta:+.2f}%" if delta is not None else "—"
        time_str = f"{t / 60:.0f}min" if t else "—"
        print(f"  {cid:<4} {label:<20} {params:>10,} {acc_str:>10} {delta_str:>8} {time_str:>8}")

    print(f"\n  Results: {combined_path}")
    print(f"  Figures: {output_dir / 'figures'}")
    print(f"\n  Done!")


if __name__ == "__main__":
    main()
