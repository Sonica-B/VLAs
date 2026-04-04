#!/bin/bash
#SBATCH --job-name=multi_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/multi_eval_%j.out
#SBATCH --error=logs/multi_eval_%j.err

source ~/vlas_env/bin/activate
cd ~/VLAs
mkdir -p logs results/multi_model

echo "=== Optimized Multi-Model PhysBench Evaluation ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start time: $(date)"

# ---------------------------------------------------------------------------
# LOCAL MODELS (run on any GPU with >= 16GB VRAM)
# These use the optimized pipeline: AWQ/bnb-4bit + FlashAttn/SDPA + torch.compile
# ---------------------------------------------------------------------------
for model in qwen3-vl-8b internvl3-8b gemma4-e4b; do
    echo ""
    echo "=========================================="
    echo "=== Evaluating ${model} (optimized) ==="
    echo "=========================================="
    echo "Start: $(date)"

    python3 scripts/run_optimized_eval.py \
        --model "${model}" \
        --split test \
        --output-dir results/multi_model

    echo "Finished ${model}: $(date)"
    echo ""

    # Brief pause between models for VRAM cleanup
    sleep 5
done

# ---------------------------------------------------------------------------
# GLM-4.5V (Turing only — requires A100 48GB+ in bf16)
# Uses the older run_multi_model_eval.py since GLM needs full precision
# ---------------------------------------------------------------------------
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "${GPU_MEM}" -ge 45000 ]; then
    echo ""
    echo "=========================================="
    echo "=== Evaluating GLM-4.5V (A100/H100) ==="
    echo "=========================================="
    echo "Start: $(date)"

    python3 scripts/run_multi_model_eval.py \
        --model glm-4.5v \
        --split test \
        --quantize auto \
        --output-dir results/multi_model

    echo "Finished GLM-4.5V: $(date)"
else
    echo ""
    echo "=== SKIPPING GLM-4.5V ==="
    echo "  GPU has ${GPU_MEM}MB VRAM — GLM-4.5V requires >= 48GB (A100/H100)"
    echo "  To run GLM-4.5V, request: #SBATCH --gres=gpu:a100:1"
fi

echo ""
echo "=== All evaluations complete ==="
echo "End time: $(date)"

# Print summary
echo ""
echo "=== Results ==="
for f in results/multi_model/*/summary.json results/multi_model/*/*_summary.json; do
    if [ -f "$f" ]; then
        echo "$f:"
        python3 -c "import json; d=json.load(open('$f')); print(f'  Accuracy: {d.get(\"overall_accuracy\", \"N/A\")}%')"
    fi
done
