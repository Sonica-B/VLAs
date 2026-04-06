#!/bin/bash
# ============================================================================
# JOB 6/7: PEM Training at Full Scale
# ============================================================================
# Trains the Physics Expert Module on Qwen3-VL-8B with:
#   - Full 1798 training samples
#   - Full resolution images/videos
#   - PCA basis from full-resolution Turing feature cache
#
# Submit AFTER Jobs 1 and 4: sbatch turing/06_pem_train.sh
# ============================================================================

#SBATCH -J pem-full-train
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

set -e
mkdir -p jobs results/week3_turing logs/turing

# Load environment (modules + pip packages).
source /home/ssboyane/VLAs/.turing_env


export FULL_RESOLUTION=1
export HF_TOKEN="${HF_TOKEN}"

echo "=== JOB 6/7: PEM Full-Scale Training ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# Verify prerequisites.
if [ ! -d "cache/week1_turing/features/qwen3-vl-8b_val" ]; then
    echo "ERROR: Week 1 Turing cache not found. Run Job 1 first."
    exit 1
fi
if [ ! -f "cache/week2/training_data/lora_train.jsonl" ]; then
    echo "Preparing training data..."
    python -u scripts/week2_prepare_training_data.py 2>&1
fi

echo ""
echo "Initializing PEM from Turing PCA basis..."
python -u -c "
import sys
sys.path.insert(0, '.')
from src.optim.pem import PhysicsExpertModule

pem = PhysicsExpertModule.from_pca_cache(
    cache_dir='cache/week1_turing',
    model_name='qwen3-vl-8b',
    low_var_k=64,
    llm_dim=4096,
    hidden_dim=512,
)
n = sum(p.numel() for p in pem.parameters() if p.requires_grad)
print(f'PEM ready: {n:,} trainable params')
print()
print('NOTE: Full PEM training loop (week3_pem_train.py) is being developed.')
print('The PEM architecture + hooks are fully implemented in src/optim/pem.py.')
print('Manual training: use pem.make_injection_hook() + pem.make_enc_capture_hook()')
" 2>&1

echo ""
echo "=== JOB 6/7 Complete ($(date)) ==="
