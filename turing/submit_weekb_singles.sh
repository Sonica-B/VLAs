#!/bin/bash
# =============================================================================
# Submit Week B as 3 INDEPENDENT single-model jobs + dependent aggregator.
# =============================================================================
# Use when the array job (08b_weekb_array.sh) keeps hitting QoS or cluster
# resource limits. Lower per-job CPU count (4 vs 8) and independent SLURM
# dependencies make these easier for the scheduler to slot in.
#
#     bash turing/submit_weekb_singles.sh
#
# What it does:
#   1. Submits 10/11/12_probe_*.sh independently (each gets its own A100).
#   2. Submits aggregator with --dependency=afterany on ALL THREE job IDs,
#      so it runs after all three finish (success OR failure — picks up
#      whichever JSONs landed).
#
# Usage tip: if one model job is stuck PENDING for hours, you can scancel
# just that one and the aggregator will still run when the others finish.
# =============================================================================

set -e

cd ${HOME}/VLAs

echo "================================================================"
echo "Submit Week B singles ($(date))"
echo "  branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "================================================================"
git log --oneline -1
echo ""

# --- Submit 3 single-model jobs ---
LLAVA_JOB=$(sbatch --parsable turing/10_probe_llava_ov.sh)
echo "Submitted llava-onevision-7b: ${LLAVA_JOB}"

PIXTRAL_JOB=$(sbatch --parsable turing/11_probe_pixtral.sh)
echo "Submitted pixtral-12b:        ${PIXTRAL_JOB}"

PHI_JOB=$(sbatch --parsable turing/12_probe_phi35v.sh)
echo "Submitted phi3.5-vision:      ${PHI_JOB}"

# --- Submit aggregator dependent on ALL three ---
# afterany = run after ALL listed jobs finish, regardless of exit status.
# Aggregator uses whichever JSONs exist at run time.
DEP="afterany:${LLAVA_JOB}:${PIXTRAL_JOB}:${PHI_JOB}"
AGG_JOB=$(sbatch --parsable --dependency="${DEP}" turing/09_weekb_aggregate.sh)
echo "Submitted aggregator:         ${AGG_JOB} (waits on all 3)"

echo ""
echo "Queue status:"
squeue -u "$USER" -j "${LLAVA_JOB},${PIXTRAL_JOB},${PHI_JOB},${AGG_JOB}" 2>&1 \
    || squeue -u "$USER"

echo ""
echo "Logs will appear at:"
echo "  jobs/probe-llava.${LLAVA_JOB}.out      (llava-onevision-7b)"
echo "  jobs/probe-pixtral.${PIXTRAL_JOB}.out  (pixtral-12b)"
echo "  jobs/probe-phi35v.${PHI_JOB}.out       (phi3.5-vision)"
echo "  jobs/weekb-agg.${AGG_JOB}.out          (PhysLens-Predict LOO)"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER"
echo "  tail -f jobs/probe-llava.${LLAVA_JOB}.out"
echo ""
echo "If one model gets stuck PENDING, scancel just that job:"
echo "  scancel ${LLAVA_JOB}    # or ${PIXTRAL_JOB} or ${PHI_JOB}"
echo "Aggregator will still run after the remaining jobs finish."
echo ""
echo "Final result:"
echo "  cat results/week1_turing/phys_lens_predict_weekb.json | python -m json.tool"
