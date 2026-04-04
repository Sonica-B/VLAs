#!/bin/bash
#SBATCH --job-name=multi_probe
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/multi_probe_%j.out
#SBATCH --error=logs/multi_probe_%j.err

source ~/vlas_env/bin/activate
cd ~/VLAs
mkdir -p logs results/multi_model_probing

echo "=== Multi-Model Physion++ Probing Study ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start time: $(date)"

# First pass: print architecture trees for all models (useful for debugging hooks)
echo ""
echo "=========================================="
echo "=== Architecture Discovery ==="
echo "=========================================="
for model in qwen3-vl-8b internvl3-8b gemma3-12b glm-4.5v; do
    echo "--- ${model} ---"
    python3 scripts/run_multi_model_probing.py \
        --model "${model}" \
        --print-arch-only \
        --quantize auto
    sleep 5
done

# Second pass: extract activations and run probing
echo ""
echo "=========================================="
echo "=== Activation Extraction & Probing ==="
echo "=========================================="
for model in qwen3-vl-8b internvl3-8b gemma3-12b glm-4.5v; do
    echo ""
    echo "=== Probing ${model} ==="
    echo "Start: $(date)"

    python3 scripts/run_multi_model_probing.py \
        --model "${model}" \
        --num-scenes 300 \
        --quantize auto \
        --output-dir results/multi_model_probing

    echo "Finished ${model}: $(date)"
    echo ""
    sleep 5
done

echo ""
echo "=== All probing complete ==="
echo "End time: $(date)"
