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

echo "=== Multi-Model PhysBench Evaluation ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "Start time: $(date)"

# Run all 4 models sequentially — auto-detects VRAM and quantization
# On A100 80GB: all models run in bf16
# On smaller GPUs: Qwen3/InternVL3 in 4-bit, Gemma/GLM skipped
for model in qwen3-vl-8b internvl3-8b gemma3-12b glm-4.5v; do
    echo ""
    echo "=========================================="
    echo "=== Evaluating ${model} ==="
    echo "=========================================="
    echo "Start: $(date)"

    python3 scripts/run_multi_model_eval.py \
        --model "${model}" \
        --split test \
        --quantize auto \
        --output-dir results/multi_model

    echo "Finished ${model}: $(date)"
    echo ""

    # Brief pause between models for VRAM cleanup
    sleep 5
done

echo ""
echo "=== All evaluations complete ==="
echo "End time: $(date)"

# Print summary
echo ""
echo "=== Results ==="
for f in results/multi_model/*/summary.json; do
    if [ -f "$f" ]; then
        echo "$f:"
        python3 -c "import json; d=json.load(open('$f')); print(f'  Accuracy: {d.get(\"overall_accuracy\", \"N/A\")}%')"
    fi
done
