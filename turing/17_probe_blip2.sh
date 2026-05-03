#!/bin/bash
# =============================================================================
# Single-model probe: BLIP-2 OPT-2.7B (Salesforce, 2023)
# =============================================================================
# EVA-CLIP-g + Q-Former (32 q tokens) + OPT-2.7B. Compression ~8x.
# Smallest model in the panel (2.7B). MIT license. arxiv 2301.12597.
#
#     sbatch turing/17_probe_blip2.sh
# =============================================================================

#SBATCH -J probe-blip2
#SBATCH -p quick
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=24G
#SBATCH -t 12:00:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

set -uo pipefail
mkdir -p jobs results/week1_turing cache/week1_turing/features logs/turing

readonly MODEL="blip2-opt-2.7b"

set +u
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate vla_physics_v2 2>/dev/null \
    || source activate vla_physics_v2 2>/dev/null \
    || { echo "FATAL: cannot activate vla_physics_v2"; exit 99; }
set -u

export FULL_RESOLUTION=1
export HF_TOKEN="${HF_TOKEN:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -u - <<'PYCHECK'
import sys, time, torch
if not torch.cuda.is_available():
    print("FATAL: torch.cuda.is_available() = False", file=sys.stderr); sys.exit(1)
print(f"GPU OK: {torch.cuda.get_device_name(0)} / torch {torch.__version__} / cuda {torch.version.cuda}")
x = torch.randn(2000, 2000, device='cuda')
torch.cuda.synchronize(); t0 = time.time()
_ = x @ x.T
torch.cuda.synchronize(); dt = (time.time() - t0) * 1000
print(f"GPU matmul 2kx2k: {dt:.1f}ms")
if dt > 500:
    print(f"FATAL: GPU op took {dt:.0f}ms", file=sys.stderr); sys.exit(2)
PYCHECK
[ $? -ne 0 ] && { echo "GPU SANITY CHECK FAILED"; exit 1; }

echo "================================================================"
echo "[$(date)] PROBE ${MODEL}"
echo "  job:     ${SLURM_JOB_ID}"
echo "  node:    $(hostname)"
echo "  GPU:     $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  branch:  $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "  transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "================================================================"

echo ""
echo "--- [${MODEL}] probe-site discovery (advisory) ---"
set +e
python -u scripts/discover_probe_sites.py --model ${MODEL} \
    2>&1 | tee logs/turing/discover_${MODEL}.log
set -e

echo ""
echo "--- [${MODEL}] VAL-split probing ---"
python -u scripts/week1_quant_qual_probe.py \
    --model ${MODEL} \
    --cache-dir cache/week1_turing \
    --output-dir results/week1_turing \
    --log-dir logs/turing \
    2>&1 | tee logs/turing/probe_val_${MODEL}.log

PROBE_RC=${PIPESTATUS[0]}
[ ${PROBE_RC} -ne 0 ] && { echo "FAIL [${MODEL}] rc=${PROBE_RC}"; exit ${PROBE_RC}; }

PROBE_JSON="results/week1_turing/${MODEL}_quant_qual_probe.json"
[ ! -f "${PROBE_JSON}" ] && { echo "FAIL no JSON"; exit 3; }
read TOTAL PROCESSED SKIPPED ERRORS <<<"$(python -c "
import json
d = json.load(open('${PROBE_JSON}'))
s = d.get('extraction_stats', {})
p = s.get('processed', 0); k = s.get('skipped', 0); e = s.get('errors', 0)
print(p + k, p, k, e)
")"
[ "${TOTAL:-0}" -lt 50 ] && { echo "FAIL ${TOTAL}/200 samples"; exit 4; }

echo ""
echo "[$(date)] OK [${MODEL}] usable=${TOTAL}/200 (new=${PROCESSED} cached=${SKIPPED} err=${ERRORS})"
