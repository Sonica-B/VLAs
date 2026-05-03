#!/bin/bash
# =============================================================================
# AUTONOMOUS FRESH-ENV SETUP — vla_physics_v2 for A100 driver 12.8 max
# =============================================================================
# Run on the LOGIN NODE (not in srun — needs lots of RAM for installs).
#
#     bash turing/setup_v2_env.sh
#
# What it does (idempotent — safe to re-run):
#   Phase 0: Pre-flight (login node, conda available, disk space)
#   Phase 1: Create vla_physics_v2 with python 3.11
#   Phase 2: Install torch 2.4.1+cu124 from PyTorch index (with retries)
#   Phase 3: Install bitsandbytes 0.44.1
#   Phase 4: Install all project deps from requirements_v2.txt (with retries)
#   Phase 5: CPU-only smoke test (imports, versions)
#   Phase 6: Submit GPU verification SLURM job + print monitor command
#
# After this completes successfully, run:
#     bash turing/migrate_to_v2.sh    # updates SLURM scripts to use v2
#     bash turing/submit_weekb_parallel.sh
#
# Logs go to: ~/vla_setup_v2_<timestamp>.log
# Re-running picks up from last successful phase.
# =============================================================================

set -uo pipefail   # NOT -e: we handle errors ourselves so we can retry

# ----- Configuration -----
readonly ENV_NAME="vla_physics_v2"
readonly PYTHON_VERSION="3.11"
readonly TIMESTAMP=$(date +%Y%m%d_%H%M%S)
readonly LOG_FILE="${HOME}/vla_setup_v2_${TIMESTAMP}.log"
readonly STATE_DIR="${HOME}/.vla_setup_state"
readonly PROJECT_ROOT="/home/ssboyane/VLAs"
readonly REQ_FILE="${PROJECT_ROOT}/turing/requirements_v2.txt"

# Pinned torch stack — DO NOT change without testing
readonly TORCH_VERSION="2.4.1"
readonly TV_VERSION="0.19.1"
readonly TA_VERSION="2.4.1"
readonly BNB_VERSION="0.44.1"
readonly CUDA_TAG="cu124"
readonly TORCH_INDEX="https://download.pytorch.org/whl/${CUDA_TAG}"

# Use $HOME for tmp (not /tmp tmpfs) — avoids OOM on big wheel downloads
readonly TMPDIR_PIP="${HOME}/.tmp_pip"
mkdir -p "${TMPDIR_PIP}" "${STATE_DIR}"
export TMPDIR="${TMPDIR_PIP}"

# ----- Logging helpers -----
log()    { echo "[$(date '+%H:%M:%S')] $*" | tee -a "${LOG_FILE}"; }
warn()   { echo "[$(date '+%H:%M:%S')] WARN: $*" | tee -a "${LOG_FILE}" >&2; }
err()    { echo "[$(date '+%H:%M:%S')] ERROR: $*" | tee -a "${LOG_FILE}" >&2; }
phase()  {
    log ""
    log "================================================================"
    log "PHASE: $*"
    log "================================================================"
}
mark_phase_done() { touch "${STATE_DIR}/$1.done"; }
phase_done()      { [ -f "${STATE_DIR}/$1.done" ]; }

# ----- Retry wrapper for network operations -----
# Usage: retry <max_attempts> <command...>
retry() {
    local max=$1; shift
    local delay=15
    local attempt=1
    while [ $attempt -le $max ]; do
        log "  attempt ${attempt}/${max}: $*"
        if "$@"; then
            return 0
        fi
        local rc=$?
        warn "  attempt ${attempt} failed (rc=${rc})"
        if [ $attempt -lt $max ]; then
            log "  sleeping ${delay}s before retry..."
            sleep $delay
            delay=$((delay * 2))
        fi
        attempt=$((attempt + 1))
    done
    err "all ${max} attempts failed: $*"
    return 1
}

# ----- Conda activation helper (works with old + new conda) -----
# IMPORTANT: conda's profile.d/conda.sh references unset vars on older versions.
# We disable `set -u` around the source/activate to avoid spurious errors.
activate_env() {
    set +u
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "${HOME}/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        # shellcheck disable=SC1091
        source "/opt/conda/etc/profile.d/conda.sh"
    fi
    if ! conda activate "${ENV_NAME}" 2>/dev/null; then
        if ! source activate "${ENV_NAME}" 2>/dev/null; then
            set -u
            err "Could not activate ${ENV_NAME}"
            return 1
        fi
    fi
    set -u

    # Verify python is from the env we just activated
    local py_path
    py_path=$(command -v python)
    if [[ "${py_path}" != *"${ENV_NAME}"* ]]; then
        err "After activate, python is ${py_path} (expected to contain ${ENV_NAME})"
        return 1
    fi
    log "  activated: ${py_path}"
    return 0
}

# =============================================================================
# PHASE 0: Pre-flight
# =============================================================================
phase "0/6 Pre-flight checks"
log "Logging to ${LOG_FILE}"
log "Project root: ${PROJECT_ROOT}"
log "Env name:     ${ENV_NAME}"

# 0.1 — Refuse to run inside SLURM (we want login-node resources)
if [ -n "${SLURM_JOB_ID:-}" ]; then
    err "Running inside SLURM job ${SLURM_JOB_ID}. This script needs login-node memory."
    err "Exit your srun first, then re-run this from the login node."
    exit 1
fi
log "  not inside SLURM ✓"

# 0.2 — Conda must be available
if ! command -v conda >/dev/null 2>&1; then
    err "conda not found in PATH. Source your conda init first."
    exit 1
fi
log "  conda: $(conda --version)"

# 0.3 — Project root + requirements file present
if [ ! -d "${PROJECT_ROOT}" ]; then
    err "Project root not found: ${PROJECT_ROOT}"
    exit 1
fi
if [ ! -f "${REQ_FILE}" ]; then
    err "Requirements file not found: ${REQ_FILE}"
    err "Did you git pull the latest physics-steering branch?"
    exit 1
fi
log "  requirements file: ${REQ_FILE}"

# 0.4 — Disk space (need ~30GB for env + caches)
AVAIL_GB=$(df -BG "${HOME}" | awk 'NR==2 {gsub("G","",$4); print $4}')
log "  ${HOME} available: ${AVAIL_GB}GB"
if [ "${AVAIL_GB:-0}" -lt 30 ]; then
    err "Need at least 30GB free in ${HOME}, have ${AVAIL_GB}GB"
    err "Run: du -sh ~/* | sort -h    to find what to clean"
    exit 1
fi

# 0.5 — Network reachability (PyPI + PyTorch index)
log "  testing network..."
if ! curl -sI --max-time 10 https://pypi.org >/dev/null; then
    err "Cannot reach pypi.org. Check network."
    exit 1
fi
if ! curl -sI --max-time 10 "${TORCH_INDEX}" >/dev/null; then
    err "Cannot reach ${TORCH_INDEX}. Check network."
    exit 1
fi
log "  network ✓"

# =============================================================================
# PHASE 1: Create env
# =============================================================================
phase "1/6 Create ${ENV_NAME} (python ${PYTHON_VERSION})"

if conda info --envs 2>/dev/null | grep -qE "^${ENV_NAME}\s"; then
    log "  env ${ENV_NAME} already exists"
    if phase_done "phase1_create"; then
        log "  Phase 1 marked done previously — skipping creation"
    else
        log "  but phase 1 not marked done — will use existing env"
    fi
else
    log "  creating fresh env..."
    if ! retry 2 conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip -y; then
        err "Failed to create env ${ENV_NAME}"
        exit 1
    fi
fi

if ! activate_env; then
    err "Cannot activate ${ENV_NAME} — aborting"
    exit 1
fi

# Sanity: python version matches
PY_VER=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if [ "${PY_VER}" != "${PYTHON_VERSION}" ]; then
    err "Python version mismatch: got ${PY_VER}, expected ${PYTHON_VERSION}"
    exit 1
fi
log "  python ${PY_VER} ✓"

# Upgrade pip (small but worthwhile for reliable resolves)
log "  upgrading pip..."
retry 3 python -m pip install --upgrade pip setuptools wheel >> "${LOG_FILE}" 2>&1 \
    || warn "pip upgrade failed (non-fatal)"

mark_phase_done "phase1_create"
log "Phase 1 done."

# =============================================================================
# PHASE 2: Install torch (separate from other deps because of cu124 index)
# =============================================================================
phase "2/6 Install torch ${TORCH_VERSION}+${CUDA_TAG}"

if phase_done "phase2_torch"; then
    # Verify it's actually installed and right version
    INSTALLED=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "missing")
    if [[ "${INSTALLED}" == "${TORCH_VERSION}+${CUDA_TAG}" ]]; then
        log "  torch ${INSTALLED} already installed ✓ — skipping"
    else
        warn "  phase 2 marked done but torch is '${INSTALLED}' (expected ${TORCH_VERSION}+${CUDA_TAG})"
        warn "  forcing reinstall"
        rm -f "${STATE_DIR}/phase2_torch.done"
    fi
fi

if ! phase_done "phase2_torch"; then
    log "  installing torch from ${TORCH_INDEX}..."
    if ! retry 3 python -m pip install \
            "torch==${TORCH_VERSION}+${CUDA_TAG}" \
            "torchvision==${TV_VERSION}+${CUDA_TAG}" \
            "torchaudio==${TA_VERSION}+${CUDA_TAG}" \
            --index-url "${TORCH_INDEX}"; then
        err "torch install failed after 3 attempts"
        exit 1
    fi

    # Verify version installed (CUDA check needs GPU — deferred to phase 6)
    INSTALLED=$(python -c "import torch; print(torch.__version__)")
    if [[ "${INSTALLED}" != "${TORCH_VERSION}+${CUDA_TAG}" ]]; then
        err "torch installed as '${INSTALLED}', expected '${TORCH_VERSION}+${CUDA_TAG}'"
        exit 1
    fi
    TORCH_FILE=$(python -c "import torch; print(torch.__file__)")
    if [[ "${TORCH_FILE}" != *"${ENV_NAME}"* ]]; then
        err "torch loaded from ${TORCH_FILE} — not in ${ENV_NAME}!"
        err "This indicates pip/python disconnect. Aborting."
        exit 1
    fi
    log "  torch ${INSTALLED} installed at ${TORCH_FILE} ✓"
    mark_phase_done "phase2_torch"
fi

# =============================================================================
# PHASE 3: Install bitsandbytes (must come AFTER torch — links against it)
# =============================================================================
phase "3/6 Install bitsandbytes ${BNB_VERSION}"

if phase_done "phase3_bnb"; then
    INSTALLED=$(python -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null || echo "missing")
    if [[ "${INSTALLED}" == "${BNB_VERSION}" ]]; then
        log "  bnb ${INSTALLED} already installed ✓ — skipping"
    else
        warn "  forcing reinstall (got '${INSTALLED}')"
        rm -f "${STATE_DIR}/phase3_bnb.done"
    fi
fi

if ! phase_done "phase3_bnb"; then
    log "  installing bitsandbytes..."
    if ! retry 3 python -m pip install --no-cache-dir "bitsandbytes==${BNB_VERSION}"; then
        # Fallback: try latest 0.44.x
        warn "  exact pin failed, trying bitsandbytes>=0.44.0,<0.45"
        if ! retry 2 python -m pip install --no-cache-dir 'bitsandbytes>=0.44.0,<0.45'; then
            err "bnb install failed after fallback"
            exit 1
        fi
    fi
    INSTALLED=$(python -c "import bitsandbytes; print(bitsandbytes.__version__)")
    log "  bitsandbytes ${INSTALLED} installed ✓"
    mark_phase_done "phase3_bnb"
fi

# =============================================================================
# PHASE 4: Install all project deps from requirements_v2.txt
# =============================================================================
phase "4/6 Install project deps from requirements_v2.txt"

if phase_done "phase4_reqs"; then
    log "  Phase 4 marked done — skipping"
else
    log "  installing from ${REQ_FILE}..."
    # Try once with full requirements file
    if retry 3 python -m pip install -r "${REQ_FILE}"; then
        log "  bulk install succeeded ✓"
    else
        warn "  bulk install failed — falling back to per-package install"
        warn "  this is slower but isolates which package is the problem"

        # Read requirements line-by-line, skip comments/blanks, install each
        FAILED_PKGS=()
        while IFS= read -r line; do
            # Strip comments + whitespace
            pkg=$(echo "${line}" | sed 's/#.*//' | xargs)
            [ -z "${pkg}" ] && continue
            log "  installing: ${pkg}"
            if ! retry 2 python -m pip install "${pkg}"; then
                warn "    FAILED: ${pkg}"
                FAILED_PKGS+=("${pkg}")
            fi
        done < "${REQ_FILE}"

        if [ ${#FAILED_PKGS[@]} -gt 0 ]; then
            err "Failed packages (${#FAILED_PKGS[@]}):"
            for p in "${FAILED_PKGS[@]}"; do
                err "  - ${p}"
            done
            err "Continuing — some failures may be non-essential. Verify in phase 5."
        fi
    fi
    mark_phase_done "phase4_reqs"
fi

# =============================================================================
# PHASE 5: CPU-only smoke test (catches missing imports without needing GPU)
# =============================================================================
phase "5/6 CPU smoke test"

python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import sys
import importlib

# (module_name, friendly_name, required)
PACKAGES = [
    ('torch', 'torch', True),
    ('torchvision', 'torchvision', True),
    ('bitsandbytes', 'bitsandbytes', True),
    ('transformers', 'transformers', True),
    ('accelerate', 'accelerate', True),
    ('peft', 'peft', True),
    ('datasets', 'datasets', True),
    ('huggingface_hub', 'huggingface-hub', True),
    ('safetensors', 'safetensors', True),
    ('numpy', 'numpy', True),
    ('scipy', 'scipy', True),
    ('sklearn', 'scikit-learn', True),
    ('pandas', 'pandas', True),
    ('PIL', 'pillow', True),
    ('einops', 'einops', True),
    ('timm', 'timm', True),
    ('matplotlib', 'matplotlib', True),
    ('tqdm', 'tqdm', True),
    ('qwen_vl_utils', 'qwen-vl-utils', True),
    ('sentencepiece', 'sentencepiece', True),
    ('h5py', 'h5py', False),
    ('cv2', 'opencv', False),
    ('decord', 'decord', False),
    ('seaborn', 'seaborn', False),
    ('plotly', 'plotly', False),
    ('hydra', 'hydra-core', False),
    ('wandb', 'wandb', False),
]

print(f"Python: {sys.version}")
print(f"Executable: {sys.executable}")
print()
print(f"{'package':<25} {'version':<20} {'status'}")
print('-' * 60)

failed_required = []
failed_optional = []
for modname, friendly, required in PACKAGES:
    try:
        mod = importlib.import_module(modname)
        ver = getattr(mod, '__version__', '?')
        print(f"{friendly:<25} {ver:<20} OK")
    except ImportError as e:
        msg = str(e)[:30]
        if required:
            print(f"{friendly:<25} {'--':<20} MISSING (required) — {msg}")
            failed_required.append(friendly)
        else:
            print(f"{friendly:<25} {'--':<20} MISSING (optional)")
            failed_optional.append(friendly)

print()
print("=" * 60)
if failed_required:
    print(f"FAIL: {len(failed_required)} required packages missing:")
    for p in failed_required:
        print(f"  - {p}")
    sys.exit(1)
else:
    print(f"PASS: all {sum(1 for _,_,r in PACKAGES if r)} required packages OK")
    if failed_optional:
        print(f"NOTE: {len(failed_optional)} optional packages missing (non-fatal):")
        for p in failed_optional:
            print(f"  - {p}")
PY

SMOKE_RC=$?
if [ ${SMOKE_RC} -ne 0 ]; then
    err "CPU smoke test failed — see log above"
    err "Fix missing required packages and re-run this script (it'll skip completed phases)"
    exit 1
fi
mark_phase_done "phase5_smoke"
log "Phase 5 done."

# Sanity check: project's own scripts are importable
log "  testing project script imports..."
cd "${PROJECT_ROOT}"
python - <<'PY' 2>&1 | tee -a "${LOG_FILE}"
import sys, os
sys.path.insert(0, os.getcwd())
errors = []
for modpath in [
    'scripts.week1_quant_qual_probe',
    'scripts.discover_probe_sites',
    'scripts.extract_training_features',
    'scripts.phys_lens_predict',
]:
    try:
        __import__(modpath)
        print(f"  [OK] {modpath}")
    except Exception as e:
        print(f"  [FAIL] {modpath}: {type(e).__name__}: {e}")
        errors.append(modpath)
if errors:
    print(f"\n{len(errors)} project module(s) failed to import.")
    print("These may need additional deps or have bugs — check error messages.")
    sys.exit(2)
print("\nAll project modules import OK")
PY
PROJ_RC=$?
if [ ${PROJ_RC} -ne 0 ]; then
    warn "Some project scripts failed to import — see log"
    warn "This MAY be OK if they need GPU; verify in phase 6"
fi

# =============================================================================
# PHASE 6: Submit GPU verification SLURM job
# =============================================================================
phase "6/6 Submit GPU verification SLURM job"

if [ ! -f "${PROJECT_ROOT}/turing/test_v2_env.sh" ]; then
    err "GPU test script not found: ${PROJECT_ROOT}/turing/test_v2_env.sh"
    err "Did you git pull the latest scripts?"
    exit 1
fi

log "  submitting test_v2_env.sh..."
cd "${PROJECT_ROOT}"
TEST_JOB=$(sbatch --parsable turing/test_v2_env.sh 2>&1)
if [[ ! "${TEST_JOB}" =~ ^[0-9]+$ ]]; then
    err "sbatch failed: ${TEST_JOB}"
    exit 1
fi
log "  submitted GPU test as job ${TEST_JOB}"
log ""
log "================================================================"
log "SETUP COMPLETE — env created, CPU smoke test passed."
log "================================================================"
log ""
log "GPU verification is queued as job ${TEST_JOB}."
log ""
log "Monitor:"
log "  squeue -u \$USER"
log "  tail -f jobs/v2-test.${TEST_JOB}.out"
log ""
log "When test job completes, check:"
log "  cat jobs/v2-test.${TEST_JOB}.out"
log ""
log "Look for: '=== ALL GPU TESTS PASSED ==='"
log ""
log "If pass:"
log "  bash turing/migrate_to_v2.sh        # update SLURM scripts to use v2"
log "  bash turing/submit_weekb_parallel.sh"
log ""
log "If fail: check the test log, fix issues, re-run setup_v2_env.sh"
log "         (idempotent — only re-runs failed/missing phases)"
log ""
log "Full log: ${LOG_FILE}"
