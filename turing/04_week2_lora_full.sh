#!/bin/bash
# ============================================================================
# JOB 4/7: Week 2 LoRA Intervention at FULL SCALE
# ============================================================================
# Re-runs LoRA Conditions B and C on Qwen3-VL-8B with:
#   - Full 1798 training samples (laptop used 500)
#   - Full resolution images (laptop used 256x256 caps)
#   - Full video frames via nframes=4 (laptop converted to single frame)
#   - Larger batch with 80GB VRAM headroom
#
# This resolves the concern that the laptop LoRA results were
# artifacts of the training data reduction.
#
# Submit AFTER Job 1 (needs feature cache for PCA): sbatch turing/04_week2_lora_full.sh
# ============================================================================

#SBATCH -J week2-lora-full
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

set -e
mkdir -p jobs results/week2_turing cache/week2/training_data logs/turing

module load python
module load cuda/12.2
source activate vla_physics 2>/dev/null || conda activate vla_physics 2>/dev/null

export FULL_RESOLUTION=1
export HF_TOKEN="${HF_TOKEN}"

echo "=== JOB 4/7: Week 2 LoRA Full Scale ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "FULL_RESOLUTION=${FULL_RESOLUTION}"

# Step 1: Prepare full training data (all 1798 balanced samples).
echo "Preparing training data..."
python -u scripts/week2_prepare_training_data.py \
    --output-dir cache/week2/training_data 2>&1

# Step 2: Baseline eval (full resolution).
echo ""
echo "Running baseline eval..."
python -u scripts/week2_lora_intervention.py \
    --stage baseline \
    --output-dir results/week2_turing \
    --force 2>&1

# Step 3: Condition B (merger-only LoRA) — FULL 1798 samples, FULL resolution.
echo ""
echo "Training Condition B (merger LoRA) at FULL scale..."
python -u scripts/week2_lora_intervention.py \
    --stage train \
    --condition B \
    --output-dir results/week2_turing \
    --force 2>&1

# Step 4: Condition C (LLM-only LoRA) — FULL 1798 samples, FULL resolution.
echo ""
echo "Training Condition C (LLM LoRA) at FULL scale..."
python -u scripts/week2_lora_intervention.py \
    --stage train \
    --condition C \
    --output-dir results/week2_turing \
    --force 2>&1

# Step 5: Aggregate.
echo ""
echo "Aggregating Week 2 results..."
python -u scripts/week2_lora_intervention.py \
    --stage aggregate \
    --output-dir results/week2_turing 2>&1

echo ""
echo "=== JOB 4/7 Complete ($(date)) ==="
echo "Results: results/week2_turing/"
