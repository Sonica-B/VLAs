#!/bin/bash
# ============================================================================
# JOB 2/7: Permutation Baseline on Full-Fidelity Features (Raw + PCA-128)
# ============================================================================
# Re-runs permutation baseline on the Turing-extracted features.
# Now uses the full-resolution features from Job 1 instead of the
# downgraded laptop features.
#
# Also runs permutation on RAW features (no PCA) for the critical
# task_type quantitative slots — this was too slow on laptop (2+ hours)
# but tractable on Turing with faster CPUs.
#
# Submit AFTER Job 1 completes: sbatch turing/02_week1_permutation.sh
# ============================================================================

#SBATCH -J week1-permutation
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 16
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

set -e
mkdir -p jobs

# Load environment (modules + pip packages).
source /home/ssboyane/VLAs/.turing_env


echo "=== JOB 2/7: Permutation Baseline ($(date)) ==="

# Run on Turing-extracted features.
python -u scripts/week1_permutation_check.py \
    --cache-dir cache/week1_turing \
    --output-dir results/week1_turing \
    --n-permutations 200 \
    --models qwen3-vl-8b qwen2.5-vl-7b internvl3-8b gemma4-e4b \
    2>&1

echo ""
echo "=== JOB 2/7 Complete ($(date)) ==="
