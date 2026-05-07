#!/bin/bash
# =============================================================================
# Permutation H3 hit-rate test for the 3 ACTIVE models
# =============================================================================
# Runs scripts/week1_permutation_check.py for LLaVA-OV, Phi-3.5, and
# Granite-Vision-3.2-2B. These are the models whose features were extracted
# in vla_physics_v2 but whose H3 hit-rate values are still missing from
# MODEL_H3_HITS in scripts/phys_lens_predict.py.
#
# Without these values, the LOO regression in phys_lens_predict only sees
# the 4 baseline models (verdict: ANECDOTAL, kill_gate fired). Adding the
# 3 active model H3 values lifts LOO to n=7.
#
# Runtime: ~1-2 min per model on CPU (PCA-128, lbfgs LR, 200 permutations).
# Total: ~5-10 min for all 3.
#
# Output:
#   results/week1_turing/<model>_permutation_check.json   (per-model)
#   stdout: H3 hit-rate per model -> paste into MODEL_H3_HITS
#
# After this completes:
#   1. Read printed H3 values
#   2. Update scripts/phys_lens_predict.py:MODEL_H3_HITS for the 3 models
#   3. Resubmit aggregator: sbatch turing/09_weekb_aggregate.sh
#
# Submit:
#     sbatch turing/15_permutation_active.sh
# =============================================================================

#SBATCH -J perm-active
#SBATCH -p quick
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32G
#SBATCH -t 1:00:00
#SBATCH --account=${SLURM_ACCOUNT:-default}
#SBATCH --export=ALL
#SBATCH -D ${HOME}/VLAs
#SBATCH -o jobs/%x.%j.out
# (no --gres=gpu -- permutation tests are CPU-only)

set -uo pipefail
mkdir -p jobs results/week1_turing logs/turing

set +u
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate vla_physics_v2 2>/dev/null \
    || source activate vla_physics_v2 2>/dev/null \
    || { echo "FATAL: cannot activate vla_physics_v2"; exit 99; }
set -u

echo "================================================================"
echo "[$(date)] PERMUTATION TESTS for ACTIVE models"
echo "  job:          ${SLURM_JOB_ID}"
echo "  node:         $(hostname)"
echo "  transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "================================================================"

# Run permutation for the 5 ACTIVE+EXPANSION models that have features in cache/week1_turing.
# (idefics3-8b, idefics2-8b, blip2-opt-2.7b added 2026-05-03 for n=10 expansion)
ACTIVE_MODELS="llava-onevision-7b phi3.5-vision granite-vision-3.2-2b idefics3-8b idefics2-8b blip2-opt-2.7b"
echo ""
echo "Running permutation check for: ${ACTIVE_MODELS}"
echo ""

python -u scripts/week1_permutation_check.py \
    --cache-dir cache/week1_turing \
    --output-dir results/week1_turing \
    --n-permutations 200 \
    --models ${ACTIVE_MODELS} \
    2>&1

PERM_RC=$?
if [ ${PERM_RC} -ne 0 ]; then
    echo ""
    echo "WARN [permutation] script exited with rc=${PERM_RC}"
    echo "  Some models may have completed. Continuing to extract H3 hit-rates."
fi

echo ""
echo "================================================================"
echo "Extracting H3 hit-rates from permutation_check JSONs"
echo "================================================================"

# Auto-extract H3 hit-rates and print MODEL_H3_HITS-ready Python lines.
python -u scripts/compute_h3_hits.py \
    --results-dir results/week1_turing \
    --models ${ACTIVE_MODELS} \
    2>&1

echo ""
echo "[$(date)] PERMUTATION + H3 EXTRACTION COMPLETE"
echo ""
echo "NEXT STEPS:"
echo "  1. Open scripts/phys_lens_predict.py and update MODEL_H3_HITS"
echo "     with the 3 lines printed above."
echo "  2. git commit + git push to physics-steering."
echo "  3. On Turing: git pull && sbatch turing/09_weekb_aggregate.sh"
echo "  4. Read final Gate 5 verdict from"
echo "     results/week1_turing/phys_lens_predict_weekb.json"
