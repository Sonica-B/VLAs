#!/bin/bash
# =============================================================================
# One-time env upgrade: transformers 4.46 → 4.49 for 2025 model support
# =============================================================================
# Run on the LOGIN NODE (not in srun):
#
#     bash turing/upgrade_env_for_2025.sh
#
# What it does:
#   1. Backs up current pip freeze list to ~/.vla_setup_state/pip_pre_4.49.txt
#   2. Installs transformers 4.49.x (smallest bump that supports
#      Granite-Vision-3.2 — IBM Feb 2025 release).
#   3. Bumps accelerate to a compat range.
#   4. Prints next-step verification command.
#
# Risk profile:
#   - Phi-3.5-Vision uses trust_remote_code; transformers minor bumps
#     CAN break custom-modeling files. Verify after upgrade.
#   - Other models (Qwen3-VL, Qwen2.5-VL, InternVL3, Gemma4, LLaVA-OV)
#     should be unaffected — these all use first-class HF classes.
#
# Rollback (if anything breaks):
#     python -m pip install -r ~/.vla_setup_state/pip_pre_4.49.txt
# =============================================================================

set -uo pipefail

readonly STATE_DIR="${HOME}/.vla_setup_state"
mkdir -p "${STATE_DIR}"
readonly BACKUP_FREEZE="${STATE_DIR}/pip_pre_4.49.txt"

# Refuse to run inside SLURM
if [ -n "${SLURM_JOB_ID:-}" ]; then
    echo "ERROR: Inside SLURM job ${SLURM_JOB_ID}. Run on login node." >&2
    exit 1
fi

# Activate env (with set -u guard for older conda)
set +u
if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "${HOME}/miniconda3/etc/profile.d/conda.sh"
fi
conda activate vla_physics_v2 2>/dev/null \
    || source activate vla_physics_v2 2>/dev/null \
    || { echo "FATAL: cannot activate vla_physics_v2" >&2; exit 1; }
set -u

# Confirm we're in the right env
PY_PATH=$(command -v python)
if [[ "${PY_PATH}" != *"vla_physics_v2"* ]]; then
    echo "FATAL: python is ${PY_PATH} (not in vla_physics_v2)" >&2
    exit 1
fi
echo "python: ${PY_PATH}"

CURRENT_TFM=$(python -c 'import transformers; print(transformers.__version__)')
echo "Current transformers: ${CURRENT_TFM}"

# 1. Backup current state
echo ""
echo "[1/4] Backing up current pip state to ${BACKUP_FREEZE}"
python -m pip freeze > "${BACKUP_FREEZE}"
echo "  $(wc -l < ${BACKUP_FREEZE}) packages backed up"

# 2. Use $HOME for tmp (not /tmp tmpfs) to avoid OOM on big wheel downloads
export TMPDIR="${HOME}/.tmp_pip"
mkdir -p "${TMPDIR}"

# 3. Upgrade transformers to 4.49.x — smallest bump supporting Granite-Vision-3.2
echo ""
echo "[2/4] Installing transformers >=4.49,<4.50"
if ! python -m pip install --no-cache-dir 'transformers>=4.49,<4.50'; then
    echo "FATAL: transformers upgrade failed" >&2
    echo "  Rollback: python -m pip install -r ${BACKUP_FREEZE}" >&2
    exit 1
fi

# 4. Bump accelerate to compat range
echo ""
echo "[3/4] Bumping accelerate to compatible range"
python -m pip install --no-cache-dir 'accelerate>=0.34,<1.1' || true

# 5. Verify the upgrade
echo ""
echo "[4/4] Verifying upgrade"
NEW_TFM=$(python -c 'import transformers; print(transformers.__version__)')
NEW_ACC=$(python -c 'import accelerate; print(accelerate.__version__)')
echo "  transformers: ${CURRENT_TFM} → ${NEW_TFM}"
echo "  accelerate:   ${NEW_ACC}"

# Quick sanity check: Granite-Vision config should now be importable
if python -c "from transformers.models.llava_next.configuration_llava_next import LlavaNextConfig; print('LlavaNextConfig OK')" 2>&1; then
    echo "  LlavaNextForConditionalGeneration available ✓"
else
    echo "  WARNING: LlavaNextConfig not importable — check upgrade"
fi

echo ""
echo "================================================================"
echo "UPGRADE COMPLETE — transformers ${NEW_TFM}"
echo "================================================================"
echo ""
echo "NEXT: verify ALL 7 model loaders still work:"
echo "  sbatch turing/verify_loaders_after_upgrade.sh"
echo ""
echo "If any loader breaks, ROLLBACK:"
echo "  python -m pip install -r ${BACKUP_FREEZE}"
echo ""
echo "If verification passes, submit Granite-Vision:"
echo "  sbatch turing/14_probe_granite_vision.sh"
