#!/bin/bash
# ============================================================================
# SLURM Job: SCAS alpha-sweep on ALL 4 models (full resolution, full 200 samples)
# ============================================================================
# While local laptop runs Qwen3-VL-8B SCAS sweep, Turing runs the other 3
# models. On H100 80GB we can use full-resolution images (no caps needed).
#
# This completes the cross-model SCAS story: does the improvement scale
# with compression ratio as predicted by the theory?
#
# Usage: sbatch turing/slurm_scas_all_models.sh
# ============================================================================

#SBATCH --job-name=scas-sweep-all
#SBATCH --output=logs/turing/scas_all_%j.out
#SBATCH --error=logs/turing/scas_all_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -e
source activate vla_physics || conda activate vla_physics
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs/turing results/week3

echo "=== SCAS All-Model Sweep ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

# NOTE: Before running, ensure the Week 1 feature caches exist for each model.
# If they don't exist on Turing, run the Week 1 extraction first:
#   python scripts/week1_quant_qual_probe.py --model qwen2.5-vl-7b
#   python scripts/week1_quant_qual_probe.py --model internvl3-8b
#   python scripts/week1_quant_qual_probe.py --model gemma4-e4b

# Run SCAS sweep on each model sequentially (they share one GPU).
# Alphas: 0 (baseline), 25, 50, 75, 100, 200
ALPHAS="0 25 50 75 100 200"

for MODEL in qwen2.5-vl-7b internvl3-8b gemma4-e4b; do
    echo ""
    echo "================================================================"
    echo "  SCAS sweep: ${MODEL}"
    echo "================================================================"

    # Check if cache exists.
    if [ ! -d "cache/week1/features/${MODEL}_val" ]; then
        echo "  No feature cache for ${MODEL}, running extraction first..."
        python -u scripts/week1_quant_qual_probe.py --model ${MODEL} --fresh
    fi

    # Run SCAS sweep.
    python -u scripts/week3_scas_sweep.py \
        --model ${MODEL} \
        --alphas ${ALPHAS} \
        --method contrast \
        --low-var-k 64 \
        --output-dir results/week3

    echo "  ${MODEL} sweep complete."
done

echo ""
echo "=== All SCAS sweeps complete ($(date)) ==="
echo "Results in results/week3/scas_sweep_*.json"
