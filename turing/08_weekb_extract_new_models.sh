#!/bin/bash
# ============================================================================
# JOB 8/N: Week B — VAL Probing for New VLM Families (PhysLens-Predict n>=6)
# ============================================================================
# Runs val-split probing on new model families to extend the compression-ratio
# predictor from n=4 to n=6 (or n=7 if Phi-3.5-Vision works).
#
# Target models (Rev 2 — post-RCA 2026-04-19):
#   - llava-onevision-7b    (SigLIP + MLP proj + Qwen2-7B)     GUARANTEED
#   - pixtral-12b           (CLIP ViT + proj + Mistral-Nemo)   GUARANTEED (after regex fix)
#   - phi3.5-vision         (CLIP ViT-L + img_proj + Phi)      STRETCH (FA2-check issue fix)
#
# Dropped from Week B (investigated and dropped with reason):
#   - molmo-7b — transformers 5.x API drift: its remote code uses
#     _tied_weights_keys but the bnb-4bit quantizer expects
#     all_tied_weights_keys. Not worth 2-week engineering; re-add later.
#   - MiniCPM-V-2.6, GLM-4.5V, DeepSeek-VL2 — custom .chat()/msgs= APIs.
#
# Design changes in Rev 2 (vs Rev 1 which failed all 4 models):
#   - DROP extract_training_features.py step entirely. Training features are
#     only needed for SCAS (dead per Gate 2). The predictor only needs val
#     probing output. Removing this step unblocks all models from the
#     "lora_train_clean.jsonl not found on Turing" failure.
#   - Drop TF auto-install (Molmo is dropped).
#   - Trust PROBE_CANDIDATES paths from scripts/week1_quant_qual_probe.py
#     (not discover_probe_sites.py guesses) — week1 is battle-tested.
#     discover_probe_sites is now advisory only (still run for logging).
#
# IMPORTANT — before submitting:
#   1. Pull latest physics-steering on Turing: git pull origin physics-steering
#   2. Verify HF_TOKEN is set for any gated models (all 3 above are public)
#
# Per-model failures do NOT kill the whole job.
# Submit: sbatch turing/08_weekb_extract_new_models.sh
# ============================================================================

#SBATCH -J weekb-probe
#SBATCH -p quick
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

set -e
mkdir -p jobs results/week1_turing cache/week1_turing/features logs/turing

# Load environment (modules + pip packages).
source activate /home/ssboyane/VLAs/vla_physics || conda activate vla_physics || source activate vla_physics

export FULL_RESOLUTION=1
export HF_TOKEN="${HF_TOKEN}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== JOB 8/N: Week B Val Probing (Rev 2) ($(date)) ==="
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "FULL_RESOLUTION=${FULL_RESOLUTION}"
echo "Branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "transformers: $(python -c 'import transformers; print(transformers.__version__)')"

# Models to probe. Per-model failure does NOT abort the job.
# Molmo dropped due to transformers 5.x API drift (_tied_weights_keys vs
# all_tied_weights_keys mismatch in bnb-4bit quantizer).
NEW_MODELS=(llava-onevision-7b pixtral-12b phi3.5-vision)

set +e  # allow per-model failures
SUCCESS_MODELS=()
FAILED_MODELS=()

for MODEL in "${NEW_MODELS[@]}"; do
    echo ""
    echo "################################################################"
    echo "# [${MODEL}] START ($(date))"
    echo "################################################################"
    MODEL_OK=1

    # ----- Step 1 (advisory): discover probe sites + measure compression ratio -----
    # This is advisory only. The ACTUAL probe paths used by week1_quant_qual_probe.py
    # come from its MODEL_REGISTRY. Discovery here just confirms the paths resolve
    # on the real module tree and measures compression ratio for phys_lens_predict.
    echo ""
    echo "--- [${MODEL}] probe-site discovery (advisory) ---"
    python -u scripts/discover_probe_sites.py --model ${MODEL} \
        2>&1 | tee logs/turing/discover_${MODEL}.log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "NOTE [${MODEL}] discovery advisory failed — week1 probing uses its own"
        echo "     MODEL_REGISTRY paths, so this is not a hard blocker. Continuing."
    fi

    # ----- Step 2: VAL-split probing (THIS is what produces predictor inputs) -----
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
        FAILED_MODELS+=("${MODEL}")
        MODEL_OK=0
    fi

    if [ ${MODEL_OK} -eq 1 ]; then
        SUCCESS_MODELS+=("${MODEL}")
        echo "OK [${MODEL}] complete ($(date))."
    fi
done

set -e

echo ""
echo "################################################################"
echo "# Week B per-model summary"
echo "################################################################"
echo "Succeeded (${#SUCCESS_MODELS[@]}/${#NEW_MODELS[@]}): ${SUCCESS_MODELS[@]:-<none>}"
echo "Failed    (${#FAILED_MODELS[@]}/${#NEW_MODELS[@]}): ${FAILED_MODELS[@]:-<none>}"

if [ ${#SUCCESS_MODELS[@]} -eq 0 ]; then
    echo ""
    echo "ERROR: ZERO new models succeeded. Existing n=4 data still usable but"
    echo "       the predictor cannot expand. Check logs/turing/ for details."
    exit 1
fi

# ----- Step 3: PhysLens-Predict LOO regression on n=4 existing + new successes -----
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
