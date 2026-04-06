#!/bin/bash
# ============================================================================
# JOB 5/7: SCAS Amplify Sweep on ALL Models (Full Resolution)
# ============================================================================
# Runs the SCAS amplify-method alpha sweep on all models using
# FULL-RESOLUTION Turing-extracted features.
#
# Local laptop result (Qwen3-VL-8B, full res eval):
#   alpha=3: quant +3.64pp, qual +0.00pp (exact theoretical prediction)
#
# This job tests whether the improvement REPLICATES across all 4 models
# and SCALES with the compression ratio as predicted by Theorem 3.
#
# Submit AFTER Job 1 (needs feature caches): sbatch turing/05_scas_amplify_all.sh
# ============================================================================

#SBATCH -J scas-amplify-all
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 6:00:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

set -e
mkdir -p jobs results/week3_turing logs/turing

module load python
module load cuda/12.2
source activate vla_physics 2>/dev/null || conda activate vla_physics 2>/dev/null

export FULL_RESOLUTION=1
export HF_TOKEN="${HF_TOKEN}"

echo "=== JOB 5/7: SCAS Amplify All Models ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

ALPHAS="0 1 2 3 4 5"

for MODEL in qwen3-vl-8b qwen2.5-vl-7b internvl3-8b gemma4-e4b; do
    echo ""
    echo "================================================================"
    echo "  SCAS amplify sweep: ${MODEL} ($(date))"
    echo "================================================================"

    # Use Turing caches for steering vector computation.
    python -u scripts/week3_scas_sweep.py \
        --model ${MODEL} \
        --alphas ${ALPHAS} \
        --method amplify \
        --low-var-k 64 \
        --cache-dir cache/week1_turing \
        --output-dir results/week3_turing \
        --log-dir logs/turing \
        2>&1

    echo "  ${MODEL} complete."
done

echo ""
echo "=== JOB 5/7 Complete ($(date)) ==="
echo "Results: results/week3_turing/scas_sweep_*.json"
