#!/bin/bash
#SBATCH --job-name=qlora_cond
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/qlora_%j.out
#SBATCH --error=logs/qlora_%j.err

# Usage: sbatch turing/slurm_qlora_condition.sh merger
#        sbatch turing/slurm_qlora_condition.sh llm
#        sbatch turing/slurm_qlora_condition.sh encoder
#        sbatch turing/slurm_qlora_condition.sh merger+encoder
#        sbatch turing/slurm_qlora_condition.sh full

CONDITION=${1:-merger}

source ~/vlas_env/bin/activate
cd ~/VLAs
mkdir -p logs results/qlora_ablation

echo "=== QLoRA Ablation: Condition ${CONDITION} ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"

# A100: bf16 (no quantization), larger batch size, less gradient accumulation
python3 scripts/run_qlora_full.py \
    --condition ${CONDITION} \
    --epochs 3 \
    --batch-size 4 \
    --gradient-accumulation 4 \
    --learning-rate 2e-4 \
    --quantize none \
    --output-dir results/qlora_ablation

echo "End time: $(date)"
