#!/bin/bash
# ============================================================================
# JOB 3/7: Multi-Model Aggregation on Full-Fidelity Features
# ============================================================================
# Runs the aggregation script on Turing results to produce the
# compression-vs-H3 curve from full-resolution features.
#
# Submit AFTER Job 2: sbatch turing/03_week1_aggregate.sh
# ============================================================================

#SBATCH -J week1-aggregate
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

set -e
mkdir -p jobs

# Load environment (modules + pip packages).
source /home/ssboyane/VLAs/.turing_env


echo "=== JOB 3/7: Week 1 Aggregation ($(date)) ==="

# Point the aggregation at Turing results.
# The aggregate script reads from results/week1/ by default;
# we'll run it after copying Turing results into that path.
cp results/week1_turing/*_quant_qual_probe.json results/week1/ 2>/dev/null || true

python -u scripts/week1_aggregate.py 2>&1

echo ""
echo "=== JOB 3/7 Complete ($(date)) ==="
