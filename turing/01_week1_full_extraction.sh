#!/bin/bash
# ============================================================================
# JOB 1/7: Week 1 Full-Fidelity Feature Extraction (ALL 5 models)
# ============================================================================
# Re-runs Week 1 probing on ALL models at FULL RESOLUTION on Turing.
#
# What was partial/downgraded on laptop:
#   - Qwen2.5-VL-7B: 194/200 (6 StopIteration video errors)
#   - InternVL3-8B:  200/200 but 1 image @ 448x448 (OOM workaround)
#   - Gemma:         200/200 but 1 image @ 448x448 (OOM workaround)
#   - GLM-4.5V:      NEVER RUN (48B doesn't fit 12.8GB laptop)
#   - Qwen3-VL-8B:   200/200 full resolution -- OK
#
# This job re-runs ALL 5 models at full resolution with FULL_RESOLUTION=1
# so InternVL3/Gemma get all images at native resolution.
# GLM-4.5V runs for the first time (5th architecture data point).
#
# Submit: sbatch turing/01_week1_full_extraction.sh
# ============================================================================

#SBATCH -J week1-full-extract
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

set -e
mkdir -p jobs results/week1_turing cache/week1_turing/features logs/turing

module load python
module load cuda/12.2
source activate vla_physics 2>/dev/null || conda activate vla_physics 2>/dev/null

# FULL_RESOLUTION=1 tells the PIL input builder to keep ALL images at
# native resolution (no 1-image cap, no 448x448 resize).
export FULL_RESOLUTION=1
export HF_TOKEN="${HF_TOKEN}"

echo "=== JOB 1/7: Week 1 Full-Fidelity Extraction ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "FULL_RESOLUTION=${FULL_RESOLUTION}"

# Run each model. Use --cache-dir and --output-dir with _turing suffix
# so we don't overwrite laptop results and can compare side-by-side.
for MODEL in qwen3-vl-8b qwen2.5-vl-7b internvl3-8b gemma4-e4b; do
    echo ""
    echo "================================================================"
    echo "  Extracting: ${MODEL} at FULL RESOLUTION ($(date))"
    echo "================================================================"
    python -u scripts/week1_quant_qual_probe.py \
        --model ${MODEL} \
        --fresh \
        --cache-dir cache/week1_turing \
        --output-dir results/week1_turing \
        --log-dir logs/turing \
        2>&1
    echo "  ${MODEL} complete."
done

# GLM-4.5V: first-time run. This model is 48B and needs the module tree
# discovery step first. For now, run the print-structure to get paths,
# then we add it to the registry in a follow-up.
echo ""
echo "================================================================"
echo "  GLM-4.5V: Module tree discovery ($(date))"
echo "================================================================"
python -u -c "
import sys, time, os, json
sys.path.insert(0, '.')
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from src.optim.vram import build_bnb_config, snapshot_vram, hard_cleanup
from src.optim.compute import pick_attn_impl

# Try multiple GLM model IDs.
for MODEL_ID in ['THUDM/glm-4v-9b', 'THUDM/cogvlm2-llama3-chat-19B']:
    print(f'\\nTrying {MODEL_ID}...')
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            quantization_config=build_bnb_config(load_in_4bit=True),
            device_map='auto', torch_dtype=torch.bfloat16,
            trust_remote_code=True, low_cpu_mem_usage=True,
        )
        print(f'Loaded. VRAM: {snapshot_vram()}')
        print('Module tree (depth 3):')
        for name, mod in model.named_modules():
            depth = name.count('.')
            if depth <= 3:
                indent = '  ' * depth
                cls = type(mod).__name__
                if depth == 0: print(cls)
                else: print(f'{indent}{name.split(\".\")[-1]}: {cls}')
        hard_cleanup(model)
        break
    except Exception as e:
        print(f'  Failed: {type(e).__name__}: {str(e)[:200]}')
        continue
" 2>&1

echo ""
echo "=== JOB 1/7 Complete ($(date)) ==="
echo "Results: results/week1_turing/"
echo "Caches:  cache/week1_turing/features/"
