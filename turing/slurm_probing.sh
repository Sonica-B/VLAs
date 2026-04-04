#!/bin/bash
#SBATCH --job-name=physion_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/probing_%j.out
#SBATCH --error=logs/probing_%j.err

source ~/vlas_env/bin/activate
cd ~/VLAs
mkdir -p logs

echo "=== Full Physion++ Probing Study ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"

# Run full probing on Qwen2.5-VL-7B (no quantization on A100)
python3 scripts/run_full_week2_gpu.py \
    --model qwen \
    --num-scenes 800 \
    --output-dir results/physion_a100_full

echo "End time: $(date)"
