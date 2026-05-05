#!/bin/bash
# =============================================================================
# Migrate SLURM scripts from vla_physics → vla_physics_v2
# =============================================================================
# Run AFTER setup_v2_env.sh + test_v2_env.sh both pass.
#
#     bash turing/migrate_to_v2.sh
#
# What it does:
#   1. Verify v2 env passed GPU test (looks for ~/.vla_setup_state/gpu_test.done)
#   2. Backup all turing/*.sh scripts to turing/.backup_pre_v2/
#   3. Replace `vla_physics` references with `vla_physics_v2` in SLURM scripts
#   4. Print diff summary so you can review
#
# Idempotent: re-running detects already-migrated scripts and skips them.
# Reversible: restore from turing/.backup_pre_v2/ if needed.
# =============================================================================

set -uo pipefail

readonly PROJECT_ROOT="${HOME}/VLAs"
readonly BACKUP_DIR="${PROJECT_ROOT}/turing/.backup_pre_v2"
readonly STATE_DIR="${HOME}/.vla_setup_state"
readonly OLD_NAME="vla_physics"
readonly NEW_NAME="vla_physics_v2"

# These scripts get migrated. Add new SLURM scripts here as they're added.
SCRIPTS_TO_MIGRATE=(
    "01_week1_full_extraction.sh"
    "02_week1_permutation.sh"
    "03_week1_aggregate.sh"
    "04_week2_lora_full.sh"
    "08_weekb_extract_new_models.sh"
    "08b_weekb_array.sh"
    "09_weekb_aggregate.sh"
    "download_data.sh"
    # NB: setup_turing.sh and 00_setup_env.sh intentionally NOT migrated —
    # those are old setup scripts, superseded by setup_v2_env.sh.
)

# Helper: count occurrences in file, handle no-matches and file-missing cleanly.
# grep -c: outputs count to stdout (even when 0), exits 1 if no matches, 2 if error.
# We use `|| true` to suppress non-zero exits and ${X:-0} to default empty → 0.
count_in() {
    local pattern=$1; local file=$2
    if [ ! -f "${file}" ]; then
        echo "0"; return
    fi
    local n
    n=$(grep -cE "${pattern}" "${file}" 2>/dev/null || true)
    echo "${n:-0}"
}

cd "${PROJECT_ROOT}"

# ----- Step 1: Verify v2 env passed GPU test -----
echo "================================================================"
echo "MIGRATE TO ${NEW_NAME}"
echo "================================================================"

if [ ! -f "${STATE_DIR}/gpu_test.done" ]; then
    echo "ERROR: GPU test marker not found: ${STATE_DIR}/gpu_test.done"
    echo ""
    echo "This means the v2 env hasn't passed verification yet."
    echo "Don't migrate scripts before the env is verified working."
    echo ""
    echo "Run:"
    echo "  bash turing/setup_v2_env.sh    # if env not yet built"
    echo "  sbatch turing/test_v2_env.sh   # if just need to re-test"
    echo ""
    echo "If you want to migrate anyway (DANGEROUS), use --force:"
    echo "  bash turing/migrate_to_v2.sh --force"
    if [ "${1:-}" != "--force" ]; then
        exit 1
    fi
    echo ""
    echo "WARNING: --force used. Migrating without verified env."
fi
echo "  v2 env GPU test: PASSED ✓"

# ----- Step 2: Backup scripts -----
mkdir -p "${BACKUP_DIR}"
echo ""
echo "Backing up to ${BACKUP_DIR}/"
for script in "${SCRIPTS_TO_MIGRATE[@]}"; do
    src="${PROJECT_ROOT}/turing/${script}"
    if [ ! -f "${src}" ]; then
        echo "  [skip] ${script} (not present)"
        continue
    fi
    dst="${BACKUP_DIR}/${script}"
    if [ -f "${dst}" ]; then
        echo "  [exists] ${script} (backup already present, not overwriting)"
    else
        cp "${src}" "${dst}"
        echo "  [backup] ${script}"
    fi
done

# ----- Step 3: Migrate (replace vla_physics references) -----
echo ""
echo "Migrating ${OLD_NAME} → ${NEW_NAME}"
MIGRATED_COUNT=0
SKIPPED_COUNT=0
for script in "${SCRIPTS_TO_MIGRATE[@]}"; do
    src="${PROJECT_ROOT}/turing/${script}"
    if [ ! -f "${src}" ]; then
        continue
    fi

    # Check if already migrated (any `vla_physics_v2` references AND no bare `vla_physics`)
    HAS_V2=$(count_in "${NEW_NAME}" "${src}")
    HAS_OLD=$(count_in "(^|[^A-Za-z0-9_])${OLD_NAME}([^A-Za-z0-9_]|$)" "${src}")

    if [ "${HAS_V2}" -gt 0 ] && [ "${HAS_OLD}" -eq 0 ]; then
        echo "  [done]    ${script} (already on ${NEW_NAME})"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        continue
    fi

    # Replace whole-word `vla_physics` with `vla_physics_v2`
    # Word boundaries prevent touching `vla_physics_v2` already present.
    # Pattern: vla_physics not followed by _ or alphanumeric → replace with vla_physics_v2
    # GNU sed (Linux): -i without backup extension. The capture groups \1 and \2
    # preserve the surrounding boundary character (space, newline, end-of-string).
    sed -i -E "s/(^|[^A-Za-z0-9_])${OLD_NAME}([^A-Za-z0-9_]|$)/\1${NEW_NAME}\2/g" "${src}"

    # Verify the migration
    NEW_HAS_V2=$(count_in "${NEW_NAME}" "${src}")
    NEW_HAS_OLD=$(count_in "(^|[^A-Za-z0-9_])${OLD_NAME}([^A-Za-z0-9_]|$)" "${src}")
    if [ "${NEW_HAS_V2}" -gt 0 ] && [ "${NEW_HAS_OLD}" -eq 0 ]; then
        echo "  [migrate] ${script}  (${NEW_HAS_V2} ${NEW_NAME} refs)"
        MIGRATED_COUNT=$((MIGRATED_COUNT + 1))
    else
        echo "  [WARN]    ${script}  (after migrate: v2=${NEW_HAS_V2}, old=${NEW_HAS_OLD})"
    fi
done

# ----- Step 4: Summary -----
echo ""
echo "================================================================"
echo "MIGRATION SUMMARY"
echo "================================================================"
echo "  Migrated this run:  ${MIGRATED_COUNT}"
echo "  Already migrated:   ${SKIPPED_COUNT}"
echo "  Backups in:         ${BACKUP_DIR}/"
echo ""
echo "Verify with:"
echo "  grep -nE '(vla_physics)([^_]|\$)' turing/*.sh"
echo "  (should return nothing — only vla_physics_v2 refs remain)"
echo ""
echo "Next:"
echo "  bash turing/submit_weekb_parallel.sh"
echo ""
echo "Rollback (restore originals):"
echo "  cp ${BACKUP_DIR}/*.sh ${PROJECT_ROOT}/turing/"
