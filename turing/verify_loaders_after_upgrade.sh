#!/bin/bash
# =============================================================================
# Post-upgrade verification: load (NOT probe) all 7 model loaders.
# =============================================================================
# After running turing/upgrade_env_for_2025.sh, run this to confirm the
# transformers 4.49 upgrade didn't break any of the 6 working models.
#
#     sbatch turing/verify_loaders_after_upgrade.sh
#
# What it does:
#   - For each of {qwen3-vl-8b, qwen2.5-vl-7b, internvl3-8b, gemma4-e4b,
#     llava-onevision-7b, phi3.5-vision, granite-vision-3.2-2b}:
#       try load_model(...) — just instantiate, no forward pass
#   - Report PASS/FAIL per model
#   - Exit 0 only if ALL 7 load successfully
#
# This catches the most-likely failure mode (transformers minor bump
# breaks a model loader) without spending 30+ min per model on full probing.
# Each load takes ~30s for small models, ~60s for 12B+. Total ~5-7 min.
# =============================================================================

#SBATCH -J verify-7
#SBATCH -p quick
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=64G
#SBATCH -t 0:30:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

set -uo pipefail
mkdir -p jobs

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

echo "================================================================"
echo "POST-UPGRADE LOADER VERIFICATION"
echo "  date:         $(date)"
echo "  node:         $(hostname)"
echo "  transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo "  GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "================================================================"

python -u - <<'PY'
import sys, traceback, gc, torch
sys.path.insert(0, '/home/ssboyane/VLAs')
from scripts.week1_quant_qual_probe import load_model

MODELS = [
    "qwen3-vl-8b",
    "qwen2.5-vl-7b",
    "internvl3-8b",
    "gemma4-e4b",
    "llava-onevision-7b",
    "phi3.5-vision",
    "granite-vision-3.2-2b",
]

passed, failed = [], []
for m in MODELS:
    print(f"\n--- {m} ---", flush=True)
    try:
        model, processor = load_model(m)
        # Tiny smoke: confirm model has parameters and processor exists
        n_params = sum(p.numel() for p in model.parameters())
        proc_class = type(processor).__name__
        print(f"  PASS — {n_params/1e9:.1f}B params, processor={proc_class}")
        passed.append(m)
        # Free VRAM before next load
        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  FAIL — {type(e).__name__}: {e}")
        traceback.print_exc()
        failed.append(m)

print()
print("=" * 64)
print(f"PASSED: {len(passed)}/{len(MODELS)}")
for m in passed: print(f"  ✓ {m}")
if failed:
    print(f"FAILED: {len(failed)}/{len(MODELS)}")
    for m in failed: print(f"  ✗ {m}")
    sys.exit(1)
print()
print("ALL LOADERS PASS — safe to run probing jobs.")
PY

VERIFY_RC=$?
echo ""
echo "================================================================"
if [ ${VERIFY_RC} -eq 0 ]; then
    echo "VERIFY: PASS — all 7 loaders work after transformers upgrade"
    echo ""
    echo "Submit Granite-Vision now:"
    echo "  sbatch turing/14_probe_granite_vision.sh"
    echo ""
    echo "Then resubmit aggregator:"
    echo "  sbatch --dependency=afterany:<granite_jobid> turing/09_weekb_aggregate.sh"
    exit 0
else
    echo "VERIFY: FAIL — at least one loader broke after upgrade"
    echo ""
    echo "Rollback transformers:"
    echo "  python -m pip install -r ~/.vla_setup_state/pip_pre_4.49.txt"
    echo ""
    echo "Then fall back to Idefics3 (Aug 2024, no upgrade needed):"
    echo "  sbatch turing/13_probe_idefics3.sh"
    exit ${VERIFY_RC}
fi
