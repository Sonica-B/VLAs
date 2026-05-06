#!/bin/bash
# =============================================================================
# Create a clean submission branch with ONLY paper-reproduction essentials.
# =============================================================================
# Usage:
#     bash turing/cleanup_for_submission.sh
#
# What it does:
#   1. Creates branch `paper-submission-clean` off current physics-steering HEAD
#   2. Removes large/incidental files (jobs/, cache/, logs/, .backup_pre_v2/,
#      old setup scripts, exploratory notebooks)
#   3. Keeps the MINIMUM tree needed to re-run the probing pipeline:
#        scripts/        (probing, predictor, helpers, compute_h3_hits)
#        src/            (FeatureCache, splits, hooks, lora, etc.)
#        turing/         (only essential SLURM scripts)
#        data/physbench/ (val.json + image refs)
#        results/week1/ + results/week1_turing/ (per-model JSONs only)
#        tests/          (regression suite)
#        Root artifacts: PRE_REGISTRATION.md, DATASHEET_PHYSBENCH_DIAG.md,
#                        CROISSANT_METADATA.md, CROISSANT_METADATA.json,
#                        README.md, LICENSE, LICENSE-DATA, requirements.txt
#   4. Runs `python tests/regression.py` to confirm structural tests pass
#   5. Prints a manifest of what was kept and what was removed
#
# THIS DOES NOT auto-commit. Review changes via `git status` before committing.
# THIS DOES NOT delete uncommitted work — it operates on a NEW branch.
#
# Reversible: `git checkout physics-steering` to return to current state.
# =============================================================================

set -uo pipefail

readonly BRANCH_NAME="paper-submission-clean"
readonly PROJECT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "${PROJECT_ROOT}"

# Refuse to run if uncommitted changes exist
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: uncommitted changes present. Commit or stash before running this." >&2
    git status --short >&2
    exit 1
fi

# Refuse to overwrite an existing branch unless --force
if git show-ref --quiet "refs/heads/${BRANCH_NAME}"; then
    if [ "${1:-}" != "--force" ]; then
        echo "ERROR: branch ${BRANCH_NAME} already exists." >&2
        echo "Use: bash turing/cleanup_for_submission.sh --force  to recreate." >&2
        exit 1
    fi
    git branch -D "${BRANCH_NAME}"
fi

echo "==============================================================="
echo "Cleanup for paper submission"
echo "  Source branch: $(git branch --show-current)"
echo "  Target branch: ${BRANCH_NAME}"
echo "==============================================================="

# Create + check out the new branch
git checkout -b "${BRANCH_NAME}"

# ----- Files to REMOVE (not needed for reproduction) -----
TO_REMOVE=(
    # Job logs (regenerated on every SLURM submission)
    "jobs"
    "logs"
    # Caches (intermediate features — large, regeneratable)
    "cache"
    # Temporary backup directory from migrate_to_v2
    "turing/.backup_pre_v2"
    # Deprecated setup scripts (superseded by setup_v2_env.sh)
    "turing/00_setup_env.sh"
    "turing/setup_turing.sh"
    # Exploratory notebooks (results all in results/)
    "notebooks"
    # Old single-execution scripts no longer in pipeline
    "turing/04_week2_lora_full.sh"
    "turing/05_scas_amplify_all.sh"
    "turing/06_pem_train.sh"
    "turing/07_final_comparison.sh"
    "turing/run_all_conditions.sh"
    "turing/transfer_results.sh"
    "turing/download_data.sh"
    "turing/upgrade_env_for_2025.sh"  # one-time op already done
    "turing/verify_loaders_after_upgrade.sh"  # one-time op already done
    # Pixtral-specific scripts (model dropped from panel)
    "turing/11_probe_pixtral.sh"
    "turing/13_probe_idefics3.sh"  # backup option, not in final panel
    # Idefics3 in registry but optional
    # Old extraction script (Week 2 LORA path, not RQ-B)
    "scripts/extract_training_features.py"
)

# ----- Files to KEEP (HARD requirements) -----
# (Not deleting these — listed for clarity)
TO_KEEP=(
    # Core scripts
    "scripts/week1_quant_qual_probe.py"
    "scripts/week1_permutation_check.py"
    "scripts/discover_probe_sites.py"
    "scripts/phys_lens_predict.py"
    "scripts/compute_h3_hits.py"
    # Source modules
    "src/optim/features.py"
    "src/optim/physbench_split.py"
    "src/optim/permutation_baseline.py"
    "src/probing/"  # all probes
    "src/models/activation_extractor.py"
    # Data
    "data/physbench/val.json"
    "data/physbench/test.json"
    # All probe + permutation result JSONs
    "results/week1/*.json"
    "results/week1_turing/*.json"
    # Pinned env spec
    "turing/requirements_v2.txt"
    "turing/setup_v2_env.sh"
    "turing/test_v2_env.sh"
    # Active SLURM probe scripts (final panel only)
    "turing/10_probe_llava_ov.sh"
    "turing/12_probe_phi35v.sh"
    "turing/14_probe_granite_vision.sh"
    "turing/16_probe_idefics2.sh"
    "turing/17_probe_blip2.sh"
    "turing/15_permutation_active.sh"
    "turing/09_weekb_aggregate.sh"
    # Tests
    "tests/regression.py"
    "tests/run_local.sh"
    "tests/run_local.bat"
    # Docs
    "PRE_REGISTRATION.md"
    "DATASHEET_PHYSBENCH_DIAG.md"
    "CROISSANT_METADATA.md"
    "CROISSANT_METADATA.json"
    "README.md"
    "LICENSE"
    "LICENSE-DATA"
    "requirements.txt"
)

# ----- Execute removal -----
echo ""
echo "Removing non-essential files..."
removed=0
for path in "${TO_REMOVE[@]}"; do
    if [ -e "${path}" ]; then
        git rm -rf "${path}" 2>&1 | head -3
        removed=$((removed + 1))
    fi
done
echo "  Removed ${removed} paths"

# ----- Sanity check: regression tests still pass -----
echo ""
echo "Running regression tests on cleaned branch..."
if python tests/regression.py 2>&1 | tail -5; then
    echo ""
    echo "Tests pass on cleaned branch."
else
    echo ""
    echo "WARNING: tests failed -- review changes before committing."
fi

# ----- Manifest -----
echo ""
echo "==============================================================="
echo "MANIFEST: files in cleaned branch"
echo "==============================================================="
git ls-files | head -100
echo ""
echo "Total tracked files: $(git ls-files | wc -l)"
echo "Total tracked size:  $(git ls-files | xargs du -ch 2>/dev/null | tail -1 | cut -f1)"

echo ""
echo "==============================================================="
echo "DONE -- branch ${BRANCH_NAME} ready for review."
echo "==============================================================="
echo ""
echo "Next steps:"
echo "  1. Review removed/kept files: git status, git diff HEAD"
echo "  2. If good: git commit -m 'Clean tree for NeurIPS submission'"
echo "             git push origin ${BRANCH_NAME}"
echo "  3. To return to dev: git checkout physics-steering"
echo "  4. To restart cleanup: bash turing/cleanup_for_submission.sh --force"
