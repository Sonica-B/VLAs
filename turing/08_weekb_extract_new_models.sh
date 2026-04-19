#!/bin/bash
# ============================================================================
# JOB 8/N: Week B — Feature Extraction for 4 New VLM Families (PhysLens-Predict n=8)
# ============================================================================
# Extends the compression-ratio predictor from n=4 to n=8 model families
# to pass pre-registered Gate 5 (LOO median |error| < 0.20).
#
# Target models (safe picks — standard HF processor API):
#   - llava-onevision-7b    (SigLIP + MLP proj + Qwen2-7B)
#   - phi3.5-vision         (CLIP ViT-L + img_proj + Phi-3.5-mini)
#   - pixtral-12b           (CLIP ViT + MLP proj + Mistral-Nemo-12B)
#   - molmo-7b              (custom OpenCLIP + projector + Qwen2-7B)
#
# Dropped from Week B plan (custom-processor complications):
#   - MiniCPM-V-2.6, GLM-4.5V, DeepSeek-VL2 — these use `.chat()` or custom msgs=
#     formats not compatible with our standard processor() call path.
#
# IMPORTANT — before submitting:
#   1. Pull latest physics-steering on Turing: git pull origin physics-steering
#   2. Verify HF_TOKEN is set for any gated models (Phi-3.5 is public)
#   3. Confirm transformers >= 4.48 (needed for LlavaOnevision class)
#
# This script is DEFENSIVE: per-model failures do NOT kill the whole job.
# Each model is attempted independently; failures are logged and the next
# model is tried.
#
# Submit: sbatch turing/08_weekb_extract_new_models.sh
# ============================================================================

#SBATCH -J weekb-extract
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=96G
#SBATCH -t 8:00:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

# Fail-fast for setup, then switch to continue-on-error for the per-model loop.
set -e
mkdir -p jobs results/week1_turing cache/week1_turing/features logs/turing

# Load environment (modules + pip packages).
source /home/ssboyane/VLAs/.turing_env

export FULL_RESOLUTION=1
export HF_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== JOB 8/N: Week B Feature Extraction ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "FULL_RESOLUTION=${FULL_RESOLUTION}"
echo "Branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "transformers: $(python -c 'import transformers; print(transformers.__version__)')"

# Molmo-7B-D's processor requires tensorflow; install if not present so its
# per-model attempt can succeed. No-op if already installed.
if ! python -c 'import tensorflow' 2>/dev/null; then
    echo "Installing tensorflow (Molmo processor dependency)..."
    pip install --quiet 'tensorflow-cpu>=2.15' || echo "WARN: TF install failed; Molmo will fail gracefully"
fi

# Models to extract. Per-model failure does NOT abort the job.
NEW_MODELS=(llava-onevision-7b phi3.5-vision pixtral-12b molmo-7b)

# Switch off fail-fast for the main loop so one model failing does NOT kill the job.
set +e
SUCCESS_MODELS=()
FAILED_MODELS=()

for MODEL in "${NEW_MODELS[@]}"; do
    echo ""
    echo "################################################################"
    echo "# [${MODEL}] START ($(date))"
    echo "################################################################"
    MODEL_OK=1

    # ----- Step 1: discover probe sites + measure compression ratio -----
    echo ""
    echo "--- [${MODEL}] probe-site discovery ---"
    python -u scripts/discover_probe_sites.py --model ${MODEL} \
        2>&1 | tee logs/turing/discover_${MODEL}.log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "WARN [${MODEL}] probe-site discovery had issues — check log, continuing"
    fi

    # ----- Step 2: TRAIN-split feature extraction (for consistency with n=4) -----
    echo ""
    echo "--- [${MODEL}] TRAIN-split feature extraction ---"
    python -u scripts/extract_training_features.py \
        --model ${MODEL} \
        --cache-dir cache/week1_turing \
        --log-dir logs/turing \
        2>&1 | tee logs/turing/extract_train_${MODEL}.log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "FAIL [${MODEL}] train extraction failed — logging and continuing"
        FAILED_MODELS+=("${MODEL} (train_extract)")
        MODEL_OK=0
    fi

    # ----- Step 3: VAL-split probing (for PhysLens-Predict gaps) -----
    if [ ${MODEL_OK} -eq 1 ]; then
        echo ""
        echo "--- [${MODEL}] VAL-split probing ---"
        python -u scripts/week1_quant_qual_probe.py \
            --model ${MODEL} \
            --cache-dir cache/week1_turing \
            --output-dir results/week1_turing \
            --log-dir logs/turing \
            2>&1 | tee logs/turing/probe_val_${MODEL}.log
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            echo "FAIL [${MODEL}] val probing failed — logging and continuing"
            FAILED_MODELS+=("${MODEL} (val_probe)")
            MODEL_OK=0
        fi
    fi

    if [ ${MODEL_OK} -eq 1 ]; then
        SUCCESS_MODELS+=("${MODEL}")
        echo "OK [${MODEL}] complete ($(date))."
    fi
done

# Re-enable fail-fast for the aggregator
set -e

echo ""
echo "################################################################"
echo "# Week B per-model summary"
echo "################################################################"
echo "Succeeded (${#SUCCESS_MODELS[@]}/${#NEW_MODELS[@]}): ${SUCCESS_MODELS[@]:-<none>}"
echo "Failed    (${#FAILED_MODELS[@]}/${#NEW_MODELS[@]}): ${FAILED_MODELS[@]:-<none>}"

if [ ${#SUCCESS_MODELS[@]} -eq 0 ]; then
    echo ""
    echo "ERROR: ZERO models succeeded. Check logs/turing/ for details."
    echo "  The predictor regression cannot run without at least one new model."
    exit 1
fi

# ----- Step 4: PhysLens-Predict LOO regression on successful models + n=4 existing -----
echo ""
echo "################################################################"
echo "# Running PhysLens-Predict LOO regression"
echo "################################################################"
python -u scripts/phys_lens_predict.py \
    --week1-dir results/week1_turing \
    --output results/week1_turing/phys_lens_predict_weekb.json \
    2>&1 | tee logs/turing/phys_lens_predict_weekb.log

echo ""
echo "=== JOB 8/N Complete ($(date)) ==="
echo "Artifacts:"
echo "  Per-model probe JSONs: results/week1_turing/<model>_quant_qual_probe.json"
echo "  Predictor summary:     results/week1_turing/phys_lens_predict_weekb.json"
echo ""
echo "--- Gate 5 decision (open predictor summary) ---"
echo "PASS if:"
echo "  loo_regression.median_abs_error < 0.20"
echo "  loo_regression.spearman_rho     > 0.5"
echo "  loo_regression.kill_gate_fired  == false"
echo "FAIL means: paper reframes to D&B with controlled-negative-SCAS as headline."
