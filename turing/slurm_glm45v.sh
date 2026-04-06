#!/bin/bash
# ============================================================================
# SLURM Job: GLM-4.5V — 5th architecture data point for compression curve
# ============================================================================
# This was BLOCKED on the 12.8GB laptop (48B model needs ~24GB at 4-bit).
# On H100 80GB it's trivial. Adds the 5th and LARGEST model to the
# compression-vs-H3 monotonic predictor curve.
#
# Expected outcome: GLM-4.5V uses a different merger architecture (GLM family).
# We predict its compression ratio and H3 behavior will fit the existing
# monotonic curve.
#
# Usage: sbatch turing/slurm_glm45v.sh
# ============================================================================

#SBATCH --job-name=glm45v-physics
#SBATCH --output=logs/turing/glm45v_%j.out
#SBATCH --error=logs/turing/glm45v_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -e

# Activate environment.
source activate vla_physics || conda activate vla_physics
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/turing

echo "=== GLM-4.5V Physics Probing ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# ---- Step 1: Add GLM-4.5V to the model registry ----
# GLM-4.5V is not in the week1 script's registry yet. We'll run it
# with a custom config. First, check if we need to add it.
python -c "
import sys
sys.path.insert(0, '.')
# Test if GLM can be imported
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print('transformers OK')
except Exception as e:
    print(f'ERROR: {e}')
"

# ---- Step 2: Run the probing pipeline ----
# Since GLM-4.5V isn't in the MODEL_REGISTRY, we'll use a standalone
# script that handles GLM specifically.
python -u -c "
import sys, json, time, os
sys.path.insert(0, '.')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
import numpy as np
from pathlib import Path
from src.optim.vram import build_bnb_config, snapshot_vram, hard_cleanup
from src.optim.compute import pick_attn_impl, inference_ctx
from src.optim.features import FeatureCache, register_probe_hooks, ProbeSites, _resolve_module
from src.optim.resilience import JsonlAppender, configure_traceback_logging
from src.optim.physbench_split import classify_quantitative, split_physbench
from scripts.run_physbench_eval import load_physbench_data, resolve_media_paths

logger = configure_traceback_logging(Path('logs/turing'), 'glm45v_probe')

# --- Load GLM-4.5V ---
MODEL_ID = 'THUDM/glm-4v-9b'  # or 'zai-org/GLM-4.5V' if available
# Try the smaller GLM-4V-9B first (fits easily on H100).
# If GLM-4.5V (48B) is needed, change MODEL_ID.

print(f'Loading {MODEL_ID}...')
from transformers import AutoModelForCausalLM, AutoTokenizer

t0 = time.time()
attn = pick_attn_impl(prefer_flash=True)
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=build_bnb_config(load_in_4bit=True),
        device_map='auto',
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
except Exception as e:
    print(f'Failed to load {MODEL_ID}: {e}')
    print('Trying alternative: zai-org/GLM-4.5V-Thinking-9B')
    MODEL_ID = 'zai-org/GLM-4.5V-Thinking-9B'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=build_bnb_config(load_in_4bit=True),
        device_map='auto',
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

print(f'Loaded in {time.time()-t0:.1f}s')
snap = snapshot_vram()
print(f'VRAM: {snap}')

# --- Print module tree for probe site discovery ---
print('\\n=== Module tree (depth 3) ===')
for name, module in model.named_modules():
    depth = name.count('.')
    if depth <= 3:
        indent = '  ' * depth
        cls = type(module).__name__
        if depth == 0:
            print(cls)
        else:
            short = name.split('.')[-1]
            print(f'{indent}{short}: {cls}')

# --- Clean up ---
hard_cleanup(model)
print('\\nDone. Use the module tree above to configure ProbeSites for GLM-4.5V.')
print('Then re-run with the full probing pipeline.')
" 2>&1

echo "=== GLM-4.5V Job Complete ($(date)) ==="
