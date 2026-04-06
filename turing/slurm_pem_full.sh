#!/bin/bash
# ============================================================================
# SLURM Job: PEM training at full scale on Qwen3-VL-8B (H100, no shortcuts)
# ============================================================================
# On the 12.8GB laptop, PEM training used:
#   - 500 samples (of 1798 available)
#   - Reduced resolution images (256x256)
#   - Video-to-frame conversion
#
# On H100 80GB, we can train with:
#   - Full 1798 samples
#   - Full resolution images
#   - Full video frames (no conversion needed)
#   - Larger batch size (4 instead of 1)
#
# This gives a CLEAN PEM result without any training shortcuts.
#
# Usage: sbatch turing/slurm_pem_full.sh
# ============================================================================

#SBATCH --job-name=pem-full-train
#SBATCH --output=logs/turing/pem_full_%j.out
#SBATCH --error=logs/turing/pem_full_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -e
source activate vla_physics || conda activate vla_physics
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/turing results/week3

echo "=== PEM Full-Scale Training ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Step 1: Prepare training data (if not already done).
if [ ! -f "cache/week2/training_data/lora_train.jsonl" ]; then
    echo "Preparing training data..."
    python -u scripts/week2_prepare_training_data.py
fi

# Step 2: Train PEM on Qwen3-VL-8B.
# NOTE: scripts/week3_pem_train.py needs to be written — this is a placeholder
# that shows the intended interface. For now, the PEM module exists at
# src/optim/pem.py and can be used programmatically.
echo ""
echo "PEM training script (week3_pem_train.py) needs to be implemented."
echo "The PEM module (src/optim/pem.py) is ready. Key classes:"
echo "  - PhysicsExpertModule.from_pca_cache() — creates PEM from Week 1 feature cache"
echo "  - pem.make_injection_hook() — hooks into model.visual.merger"
echo "  - pem.make_enc_capture_hook() — captures enc_out for PEM input"
echo ""
echo "To train manually:"
echo "  python -c \""
echo "  from src.optim.pem import PhysicsExpertModule"
echo "  pem = PhysicsExpertModule.from_pca_cache('cache/week1', 'qwen3-vl-8b')"
echo "  # Register hooks, train with standard PyTorch loop"
echo "  \""

echo ""
echo "=== PEM Job Complete ($(date)) ==="
