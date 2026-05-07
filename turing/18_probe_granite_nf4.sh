#!/bin/bash
# =============================================================================
# Granite-Vision-3.2-2B NF4 ablation (Q-LENS reviewer-blocker, May 2026)
# =============================================================================
# Purpose:
#   Re-run Granite-Vision-3.2-2B under bnb-NF4 4-bit quantization to remove the
#   bf16-only confound from the Q-LENS 10-VLM panel. Granite is currently the
#   only model loaded in bf16 (see scripts/week1_quant_qual_probe.py:829-884
#   for the original dtype-mismatch reason). Reviewers will read its
#   no-compression-cell H3 = 0.222 as a quantization artifact unless we run
#   this apples-to-apples ablation.
#
# Outputs:
#   results/week1_turing/granite-vision-3.2-2b_quant_qual_probe_NF4.json
#   logs/turing/probe_val_granite-vision-3.2-2b_NF4.log
#
# DOES NOT OVERWRITE the existing bf16 probe JSON
# (results/week1_turing/granite-vision-3.2-2b_quant_qual_probe.json) — both are
# preserved for the §7 limitation table.
#
# Pre-requisites:
#   1. Apply the patch in scripts/patches/granite_nf4_patch.md to
#      scripts/week1_quant_qual_probe.py (adds GRANITE_FORCE_NF4 env-var gate).
#   2. transformers >= 4.49.0 (run turing/upgrade_env_for_2025.sh once if not
#      already done — see turing/14_probe_granite_vision.sh header).
#
# Usage (from Turing login node):
#     sbatch turing/18_probe_granite_nf4.sh
#
# Estimated wall-time: ~30 min (Granite 2B is the smallest model in the panel).
# =============================================================================

#SBATCH -J probe-granite-nf4
#SBATCH -p quick
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH -t 02:00:00
#SBATCH --account=${SLURM_ACCOUNT:-default}
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D ${HOME}/VLAs
#SBATCH -o jobs/%x.%j.out

set -uo pipefail
mkdir -p jobs results/week1_turing cache/week1_turing/features logs/turing

readonly MODEL="granite-vision-3.2-2b"
readonly NF4_SUFFIX="_NF4"

# --- Conda environment ----------------------------------------------------
set +u
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate vla_physics_v2 2>/dev/null \
    || source activate vla_physics_v2 2>/dev/null \
    || { echo "FATAL: cannot activate vla_physics_v2"; exit 99; }
set -u

# --- Verify transformers >= 4.49 ------------------------------------------
TFM_VER=$(python -c 'import transformers; print(transformers.__version__)')
TFM_MAJOR=$(echo "$TFM_VER" | cut -d. -f1)
TFM_MINOR=$(echo "$TFM_VER" | cut -d. -f2)
if [ "$TFM_MAJOR" -lt 4 ] || { [ "$TFM_MAJOR" -eq 4 ] && [ "$TFM_MINOR" -lt 49 ]; }; then
    echo "FATAL: transformers $TFM_VER < 4.49 (Granite-Vision needs >=4.49)"
    echo "Run: bash turing/upgrade_env_for_2025.sh"
    exit 98
fi
echo "transformers $TFM_VER OK"

# --- Verify bitsandbytes available ----------------------------------------
BNB_VER=$(python -c 'import bitsandbytes; print(bitsandbytes.__version__)' 2>/dev/null)
if [ -z "${BNB_VER:-}" ]; then
    echo "FATAL: bitsandbytes not importable — required for NF4"
    exit 97
fi
echo "bitsandbytes ${BNB_VER} OK"

# --- Activate the NF4 ablation gate (added by patch in scripts/patches/) --
export GRANITE_FORCE_NF4=1
export FULL_RESOLUTION=1
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- GPU sanity -----------------------------------------------------------
python -u - <<'PYCHECK'
import sys, time, torch
if not torch.cuda.is_available():
    print("FATAL: torch.cuda.is_available() = False", file=sys.stderr)
    sys.exit(1)
print(f"GPU OK: {torch.cuda.get_device_name(0)} / torch {torch.__version__} / cuda {torch.version.cuda}")
x = torch.randn(2000, 2000, device='cuda')
torch.cuda.synchronize(); t0 = time.time()
_ = x @ x.T
torch.cuda.synchronize(); dt = (time.time() - t0) * 1000
print(f"GPU matmul 2kx2k: {dt:.1f}ms")
if dt > 500:
    print(f"FATAL: GPU op took {dt:.0f}ms (expected <100ms)", file=sys.stderr)
    sys.exit(2)
PYCHECK
[ $? -ne 0 ] && { echo "GPU SANITY CHECK FAILED"; exit 1; }

# --- Pre-flight NF4 sanity check (load 1 layer, confirm 4-bit dtype) ------
echo ""
echo "--- [${MODEL}] NF4 PRE-FLIGHT (load + verify 4-bit dtype) ---"
python -u - <<'PYNF4'
import os, sys, torch
from transformers import BitsAndBytesConfig
try:
    from transformers import LlavaNextForConditionalGeneration as Cls
except ImportError:
    from transformers import AutoModelForVision2Seq as Cls

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print("Building model under NF4...")
try:
    m = Cls.from_pretrained(
        "ibm-granite/granite-vision-3.2-2b",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb,
        low_cpu_mem_usage=True,
    )
except Exception as e:
    print(f"FATAL: Granite NF4 load raised: {type(e).__name__}: {e}", file=sys.stderr)
    print("FALLBACK: see scripts/patches/granite_nf4_patch.md Section 4a "
          "(force bf16 for all 10 models).", file=sys.stderr)
    sys.exit(3)

# Confirm at least one parameter is in a 4-bit storage class.
saw_4bit = False
sample_dtype = None
for name, p in m.named_parameters():
    cls_name = type(p).__name__.lower()
    if "4bit" in cls_name or "params4bit" in cls_name:
        saw_4bit = True
        sample_dtype = (name, str(p.dtype), cls_name)
        break

if not saw_4bit:
    print("WARNING: no 4-bit parameter detected — NF4 may have silently "
          "fallen back to bf16. Probe will continue but output JSON will "
          "be effectively bf16. Inspect logs.", file=sys.stderr)
else:
    print(f"NF4 verified: {sample_dtype}")

# Print one Linear weight dtype for the probe-site path the script uses.
try:
    proj = m.multi_modal_projector
    for n, p in proj.named_parameters():
        print(f"multi_modal_projector.{n}: dtype={p.dtype} cls={type(p).__name__}")
        break
except AttributeError:
    pass

# Free for the real probe job below.
del m
import gc; gc.collect()
torch.cuda.empty_cache()
print("Pre-flight OK — proceeding to probe.")
PYNF4
PREFLIGHT_RC=$?
if [ ${PREFLIGHT_RC} -ne 0 ]; then
    echo "FAIL [${MODEL}] NF4 pre-flight rc=${PREFLIGHT_RC}"
    echo "Consult scripts/patches/granite_nf4_patch.md Section 3 (failure modes)"
    echo "and Section 4 (fallback strategies)."
    exit ${PREFLIGHT_RC}
fi

echo "================================================================"
echo "[$(date)] PROBE ${MODEL} (NF4 ablation)"
echo "  job:           ${SLURM_JOB_ID}"
echo "  node:          $(hostname)"
echo "  GPU:           $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  branch:        $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "  transformers:  ${TFM_VER}"
echo "  bitsandbytes:  ${BNB_VER}"
echo "  GRANITE_FORCE_NF4=${GRANITE_FORCE_NF4}"
echo "  output suffix: ${NF4_SUFFIX}"
echo "================================================================"

# --- Probe site discovery (advisory) --------------------------------------
echo ""
echo "--- [${MODEL}] probe-site discovery (advisory) ---"
set +e
python -u scripts/discover_probe_sites.py --model ${MODEL} \
    2>&1 | tee logs/turing/discover_${MODEL}${NF4_SUFFIX}.log
DISCOVERY_RC=${PIPESTATUS[0]}
set -e
if [ ${DISCOVERY_RC} -ne 0 ]; then
    echo "NOTE [${MODEL}] discovery rc=${DISCOVERY_RC} - advisory only, continuing"
fi

# --- Run the probe with NF4 forced via env var ----------------------------
# NOTE: write into a sibling cache dir so the bf16 features are NOT
# overwritten. Output JSON gets the _NF4 suffix via --output-suffix (added
# downstream, see RUN_INSTRUCTIONS).
echo ""
echo "--- [${MODEL}] VAL-split probing (NF4) ---"
python -u scripts/week1_quant_qual_probe.py \
    --model ${MODEL} \
    --cache-dir cache/week1_turing_nf4 \
    --output-dir results/week1_turing \
    --output-suffix "${NF4_SUFFIX}" \
    --log-dir logs/turing \
    2>&1 | tee logs/turing/probe_val_${MODEL}${NF4_SUFFIX}.log

PROBE_RC=${PIPESTATUS[0]}
if [ ${PROBE_RC} -ne 0 ]; then
    echo "FAIL [${MODEL}] NF4 val probing rc=${PROBE_RC}"
    echo "If this is a dtype-mismatch failure (BFloat16 vs Byte), consult"
    echo "scripts/patches/granite_nf4_patch.md Section 3.1 — try"
    echo "llm_int8_skip_modules=['multi_modal_projector'] in build_bnb_config."
    exit ${PROBE_RC}
fi

# --- Verify output JSON written under NF4 suffix --------------------------
PROBE_JSON="results/week1_turing/${MODEL}_quant_qual_probe${NF4_SUFFIX}.json"
if [ ! -f "${PROBE_JSON}" ]; then
    echo "FAIL [${MODEL}] expected probe JSON not written: ${PROBE_JSON}"
    echo "(If your patched script doesn't honor --output-suffix, the file may"
    echo " have landed at the un-suffixed name. Check both before failing."
    UNSUFFIXED="results/week1_turing/${MODEL}_quant_qual_probe.json"
    if [ -f "${UNSUFFIXED}" ]; then
        echo "WARN: unsuffixed JSON exists at ${UNSUFFIXED} — DO NOT TRUST."
        echo "      The bf16 baseline may have been overwritten."
        echo "      Restore from git or re-run turing/14_probe_granite_vision.sh."
    fi
    exit 3
fi

# --- Summary --------------------------------------------------------------
read TOTAL PROCESSED SKIPPED ERRORS <<<"$(python -c "
import json
d = json.load(open('${PROBE_JSON}'))
s = d.get('extraction_stats', {})
p = s.get('processed', 0); k = s.get('skipped', 0); e = s.get('errors', 0)
print(p + k, p, k, e)
")"
if [ "${TOTAL:-0}" -lt 50 ]; then
    echo "FAIL [${MODEL}] only ${TOTAL}/200 usable samples (new=${PROCESSED} cached=${SKIPPED} err=${ERRORS})"
    exit 4
fi

echo ""
echo "[$(date)] OK [${MODEL} NF4] usable=${TOTAL}/200 (new=${PROCESSED} cached=${SKIPPED} err=${ERRORS})"
echo ""
echo "Next steps:"
echo "  1. Compute H3 from NF4 probe:"
echo "       python scripts/compute_h3_hits.py --probe-json ${PROBE_JSON}"
echo "  2. Compare bf16 vs NF4 H3:"
echo "       jq .h3 results/week1_turing/${MODEL}_quant_qual_probe.json"
echo "       jq .h3 ${PROBE_JSON}"
echo "  3. Re-fit LOO regression with NF4 H3 (see RUN_INSTRUCTIONS_GRANITE_NF4.md)."
