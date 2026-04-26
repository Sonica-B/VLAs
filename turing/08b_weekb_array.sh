#!/bin/bash
# ============================================================================
# JOB 8b: Week B PARALLEL — SLURM array job, one A100 per model
# ============================================================================
# Replaces sequential 08_weekb_extract_new_models.sh with a SLURM array.
# Each array task probes ONE model on its own A100 — all 3 run in parallel
# instead of sequentially. Wall-clock: ~100 min sequential → ~35 min parallel.
#
# After this job's array tasks complete, the dependent aggregator job
# (turing/09_weekb_aggregate.sh) runs the PhysLens-Predict LOO regression
# on whichever models succeeded.
#
# Submit:
#     ARRAY_JOB=$(sbatch --parsable turing/08b_weekb_array.sh)
#     echo "Array job: $ARRAY_JOB"
#     sbatch --dependency=afterany:$ARRAY_JOB turing/09_weekb_aggregate.sh
#
# Or one-liner:
#     bash turing/submit_weekb_parallel.sh
# ============================================================================

#SBATCH -J weekb-array
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH --array=0-2
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%A_%a.out

set -e
mkdir -p jobs results/week1_turing cache/week1_turing/features logs/turing

# Load environment.
source activate /home/ssboyane/VLAs/vla_physics 2>/dev/null \
    || conda activate vla_physics 2>/dev/null \
    || source activate vla_physics

export FULL_RESOLUTION=1
export HF_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Per-task model selection. Add models to this array AND bump --array=0-N above.
MODELS=(llava-onevision-7b pixtral-12b phi3.5-vision)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

if [ -z "$MODEL" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID has no model in MODELS array"
    exit 2
fi

echo "================================================================"
echo "[$(date)] WEEK B ARRAY TASK ${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_MAX}"
echo "  array job:   ${SLURM_ARRAY_JOB_ID}"
echo "  task id:     ${SLURM_ARRAY_TASK_ID}"
echo "  model:       ${MODEL}"
echo "  GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  branch:      $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "  transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "================================================================"

# ----- Step 1 (advisory): probe-site discovery + compression measurement -----
# Failure here is non-fatal; week1_quant_qual_probe uses its own MODEL_REGISTRY.
echo ""
echo "--- [${MODEL}] probe-site discovery (advisory) ---"
set +e
python -u scripts/discover_probe_sites.py --model ${MODEL} \
    2>&1 | tee logs/turing/discover_${MODEL}.log
DISCOVERY_RC=${PIPESTATUS[0]}
if [ ${DISCOVERY_RC} -ne 0 ]; then
    echo "NOTE [${MODEL}] discovery rc=${DISCOVERY_RC} — advisory only, continuing"
fi
set -e

# ----- Step 2: VAL-split probing (the actual data the predictor needs) -----
echo ""
echo "--- [${MODEL}] VAL-split probing ---"
python -u scripts/week1_quant_qual_probe.py \
    --model ${MODEL} \
    --cache-dir cache/week1_turing \
    --output-dir results/week1_turing \
    --log-dir logs/turing \
    2>&1 | tee logs/turing/probe_val_${MODEL}.log

PROBE_RC=${PIPESTATUS[0]}
if [ ${PROBE_RC} -ne 0 ]; then
    echo "FAIL [${MODEL}] val probing rc=${PROBE_RC}"
    exit ${PROBE_RC}
fi

# Sanity check: did probe extract ANY samples?
PROBE_JSON="results/week1_turing/${MODEL}_quant_qual_probe.json"
if [ ! -f "${PROBE_JSON}" ]; then
    echo "FAIL [${MODEL}] expected probe JSON not written: ${PROBE_JSON}"
    exit 3
fi
PROCESSED=$(python -c "
import json
d = json.load(open('${PROBE_JSON}'))
print(d.get('extraction_stats', {}).get('processed', 0))
")
if [ "${PROCESSED}" -lt 50 ]; then
    echo "FAIL [${MODEL}] probe extracted only ${PROCESSED}/200 samples — likely processor bug"
    exit 4
fi

echo ""
echo "[$(date)] OK [${MODEL}] processed=${PROCESSED}/200 — task ${SLURM_ARRAY_TASK_ID} complete"
