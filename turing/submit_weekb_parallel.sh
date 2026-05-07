#!/bin/bash
# ============================================================================
# Submit Week B in PARALLEL: array job + dependent aggregator.
# ============================================================================
# Usage (on Turing, after `git pull origin physics-steering`):
#     bash turing/submit_weekb_parallel.sh
#
# What it does:
#   1. Submits 08b_weekb_array.sh as a SLURM array job (3 parallel tasks,
#      one A100 per task — one model per task).
#   2. Submits 09_weekb_aggregate.sh with --dependency=afterany:<array_job>
#      so it runs after ALL array tasks complete (success OR failure).
#
# Wall-clock estimate:
#   Array tasks: ~30-40 min each, all in parallel → ~35 min total
#   Aggregator: ~1 min
#   Total: ~36 min vs ~100 min for the sequential 08_weekb_extract_new_models.sh
#
# Monitor:
#     squeue -u $USER             # see queued/running tasks
#     tail -f jobs/weekb-array.<JOB>_<TASK>.out
#     tail -f jobs/weekb-agg.<JOB>.out
# ============================================================================

set -e

cd ${HOME}/VLAs

# Sanity check: branch is up to date
echo "Branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "Latest commit:"
git log --oneline -1
echo ""

# 1) Submit array job
ARRAY_JOB=$(sbatch --parsable turing/08b_weekb_array.sh)
echo "Submitted array job: ${ARRAY_JOB} (3 tasks)"

# 2) Submit aggregator with dependency
AGG_JOB=$(sbatch --parsable --dependency=afterany:${ARRAY_JOB} turing/09_weekb_aggregate.sh)
echo "Submitted aggregator: ${AGG_JOB} (waits for ${ARRAY_JOB})"

echo ""
echo "Queue status:"
squeue -u $USER -j ${ARRAY_JOB},${AGG_JOB} 2>&1 || squeue -u $USER

echo ""
echo "Logs will appear at:"
echo "  jobs/weekb-array.${ARRAY_JOB}_0.out  (llava-onevision-7b)"
echo "  jobs/weekb-array.${ARRAY_JOB}_1.out  (pixtral-12b)"
echo "  jobs/weekb-array.${ARRAY_JOB}_2.out  (phi3.5-vision)"
echo "  jobs/weekb-agg.${AGG_JOB}.out        (PhysLens-Predict LOO summary)"
echo ""
echo "Wait ~35 min, then check:"
echo "  cat results/week1_turing/phys_lens_predict_weekb.json | python -m json.tool"
