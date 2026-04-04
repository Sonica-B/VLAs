#!/bin/bash
#SBATCH --job-name=physbench_baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/baseline_%j.out
#SBATCH --error=logs/baseline_%j.err

source ~/vlas_env/bin/activate
cd ~/VLAs
mkdir -p logs results

echo "=== PhysBench Baseline Evaluation ==="
echo "Model: Qwen2.5-VL-7B-Instruct"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"

# On A100 with 80GB VRAM: no quantization needed, use bf16 for best accuracy
python3 scripts/run_physbench_eval.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --quantize none \
    --data-dir data/physbench \
    --split test \
    --output-dir results/physbench_a100_baseline

echo "End time: $(date)"
