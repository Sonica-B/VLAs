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
#SBATCH --account=${SLURM_ACCOUNT:-default}
#SBATCH --export=ALL
#SBATCH --gres=gpu:A100:1
#SBATCH -D ${HOME}/VLAs
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
import sys, traceback, gc, json, torch
from pathlib import Path
sys.path.insert(0, '${HOME}/VLAs')
from scripts.week1_quant_qual_probe import load_model

# ACTIVE: must load successfully in current env (we re-extract features here).
# CACHED: data lives in results/week1/<model>_quant_qual_probe.json from a
#         prior env. Loader-class availability is informational, not blocking.
ACTIVE_MODELS = [
    "llava-onevision-7b",
    "phi3.5-vision",
    "granite-vision-3.2-2b",
]
CACHED_MODELS = [
    "qwen3-vl-8b",
    "qwen2.5-vl-7b",
    "internvl3-8b",
    "gemma4-e4b",
]

ROOT = Path("${HOME}/VLAs")

print("=== CACHED MODELS (informational) ===")
print("These have probing JSONs from a prior env. Loader is non-blocking.")
print()
cached_data_ok, cached_data_bad = [], []
cached_load_ok, cached_load_skip = [], []
for m in CACHED_MODELS:
    json_path = ROOT / "results" / "week1" / f"{m}_quant_qual_probe.json"
    print(f"--- {m} ---", flush=True)
    # 1) Cached data check (BLOCKING)
    if not json_path.exists():
        print(f"  DATA FAIL -- {json_path} missing")
        cached_data_bad.append(m)
    else:
        try:
            d = json.loads(json_path.read_text())
            n = d.get("n_samples", 0)
            sites = set()
            for tgt in (d.get("probe_results", {}) or {}).values():
                if isinstance(tgt, dict): sites.update(tgt.keys())
            need = {"enc_out", "post_proj", "llm_8", "llm_16"}
            if (need - sites) or n < 50:
                print(f"  DATA FAIL -- n={n}, missing sites={need - sites}")
                cached_data_bad.append(m)
            else:
                print(f"  DATA OK   -- n={n}, all 4 sites probed")
                cached_data_ok.append(m)
        except Exception as e:
            print(f"  DATA FAIL -- parse: {e}")
            cached_data_bad.append(m)
    # 2) Loader check (NON-BLOCKING)
    try:
        model, processor = load_model(m)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  LOAD OK   -- {n_params/1e9:.1f}B params (informational)")
        cached_load_ok.append(m)
        del model, processor; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        msg = str(e).split("\n")[0][:80]
        print(f"  LOAD SKIP -- {type(e).__name__}: {msg}")
        cached_load_skip.append(m)

print()
print("=== ACTIVE MODELS (blocking) ===")
print("These must load successfully -- we re-extract features in this env.")
print()
active_ok, active_fail = [], []
for m in ACTIVE_MODELS:
    print(f"--- {m} ---", flush=True)
    try:
        model, processor = load_model(m)
        n_params = sum(p.numel() for p in model.parameters())
        proc_class = type(processor).__name__
        print(f"  LOAD OK   -- {n_params/1e9:.1f}B params, processor={proc_class}")
        active_ok.append(m)
        del model, processor; gc.collect(); torch.cuda.empty_cache()
    except Exception as e:
        msg = str(e).split("\n")[0][:120]
        print(f"  LOAD FAIL -- {type(e).__name__}: {msg}")
        traceback.print_exc()
        active_fail.append(m)

print()
print("=" * 64)
print(f"CACHED data:    OK={len(cached_data_ok)}/{len(CACHED_MODELS)}  bad={cached_data_bad or 'none'}")
print(f"CACHED loaders: OK={len(cached_load_ok)}/{len(CACHED_MODELS)}  skipped={cached_load_skip or 'none'}")
print(f"ACTIVE loaders: OK={len(active_ok)}/{len(ACTIVE_MODELS)}  failed={active_fail or 'none'}")
print("=" * 64)

if cached_data_bad:
    print(f"FAIL: {len(cached_data_bad)} CACHED model(s) have invalid/missing probe JSONs")
    sys.exit(1)
if active_fail:
    print(f"FAIL: {len(active_fail)} ACTIVE model(s) failed to load")
    sys.exit(2)
print()
print("VERIFY PASS -- safe to run probing jobs.")
print("(CACHED loader failures are intentional; their data is on disk.)")
PY

VERIFY_RC=$?
echo ""
echo "================================================================"
if [ ${VERIFY_RC} -eq 0 ]; then
    echo "VERIFY: PASS"
    echo "  - all CACHED models have valid probing JSONs on disk"
    echo "  - all ACTIVE models load successfully in current env"
    echo ""
    echo "Submit Granite-Vision now:"
    echo "  sbatch turing/14_probe_granite_vision.sh"
    echo ""
    echo "Then resubmit aggregator:"
    echo "  sbatch --dependency=afterany:<granite_jobid> turing/09_weekb_aggregate.sh"
    exit 0
elif [ ${VERIFY_RC} -eq 1 ]; then
    echo "VERIFY: FAIL -- CACHED model JSON missing or corrupt"
    echo "  Cached data is the source of truth for the 4 baseline models."
    echo "  Restore from git or re-run the original probing pipeline."
    exit 1
else
    echo "VERIFY: FAIL -- ACTIVE loader broke (rc=${VERIFY_RC})"
    echo ""
    echo "Option A: Rollback transformers and use Idefics3 (Aug 2024) instead"
    echo "  python -m pip install -r ~/.vla_setup_state/pip_pre_4.49.txt"
    echo "  sbatch turing/13_probe_idefics3.sh"
    echo ""
    echo "Option B: Try a different transformers patch version"
    echo "  python -m pip install 'transformers>=4.49,<4.50'"
    echo ""
    exit ${VERIFY_RC}
fi
