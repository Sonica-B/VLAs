#!/bin/bash
# ============================================================================
# JOB 9: Week B Aggregator — runs PhysLens-Predict LOO regression
# ============================================================================
# Runs after the array job (08b_weekb_array.sh) completes. Aggregates whichever
# per-model probe JSONs landed in results/week1_turing/ and runs the LOO
# regression to compute the Gate 5 verdict.
#
# Submit (after the array job, with dependency):
#     ARRAY_JOB=$(sbatch --parsable turing/08b_weekb_array.sh)
#     sbatch --dependency=afterany:$ARRAY_JOB turing/09_weekb_aggregate.sh
#
# Note: --dependency=afterany means this runs after ALL array tasks finish
# regardless of success — we want the predictor LOO on whichever models
# DID succeed, even if some failed.
# ============================================================================

#SBATCH -J weekb-agg
#SBATCH -p quick
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 12:00:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out
# (no --gres=gpu — predictor regression is CPU-only)

set -e
mkdir -p jobs results/week1_turing logs/turing

source activate /home/ssboyane/VLAs/vla_physics 2>/dev/null \
    || conda activate vla_physics 2>/dev/null \
    || source activate vla_physics

echo "================================================================"
echo "[$(date)] WEEK B AGGREGATOR"
echo "  branch: $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse --short HEAD)"
echo "================================================================"

# ----- Inventory: which models actually have probe results? -----
echo ""
echo "--- Per-model probe JSONs present in results/week1_turing/ ---"
ls -la results/week1_turing/*_quant_qual_probe.json 2>/dev/null \
    || echo "(no probe JSONs found)"

# Brief status table from each JSON
echo ""
echo "--- Per-model extraction stats ---"
python << 'PY'
import json, glob, os
files = sorted(glob.glob('results/week1_turing/*_quant_qual_probe.json'))
if not files:
    print("(none)")
else:
    print(f"{'model':<22} {'processed':>10} {'errors':>7}")
    for f in files:
        try:
            d = json.load(open(f))
            stats = d.get('extraction_stats', {})
            model = d.get('model', os.path.basename(f).replace('_quant_qual_probe.json', ''))
            print(f"{model:<22} {stats.get('processed', '?'):>10} {stats.get('errors', '?'):>7}")
        except Exception as e:
            print(f"{f}: parse error: {e}")
PY

# ----- Run PhysLens-Predict LOO -----
echo ""
echo "================================================================"
echo "Running PhysLens-Predict LOO regression"
echo "================================================================"
python -u scripts/phys_lens_predict.py \
    --week1-dir results/week1_turing \
    --output results/week1_turing/phys_lens_predict_weekb.json \
    2>&1 | tee logs/turing/phys_lens_predict_weekb.log

echo ""
echo "[$(date)] AGGREGATOR COMPLETE"
echo ""
echo "--- Gate 5 decision ---"
echo "Open results/week1_turing/phys_lens_predict_weekb.json:"
echo "  PASS if loo_regression.median_abs_error < 0.20"
echo "  PASS if loo_regression.spearman_rho     > 0.5"
echo "  PASS if loo_regression.kill_gate_fired  == false"

# Print summary lines from the JSON
python << 'PY'
import json
try:
    d = json.load(open('results/week1_turing/phys_lens_predict_weekb.json'))
    loo = d.get('loo_regression') or {}
    print()
    print("=== Summary ===")
    print(f"  n_models with data:    {d.get('n_models_with_data')}")
    print(f"  median_abs_error:      {loo.get('median_abs_error')}")
    print(f"  spearman_rho:          {loo.get('spearman_rho')}")
    print(f"  spearman_p:            {loo.get('spearman_p')}")
    print(f"  spearman_bootstrap_ci: {loo.get('spearman_bootstrap_ci95')}")
    print(f"  kill_gate_fired:       {loo.get('kill_gate_fired')}")
    print(f"  verdict:               {loo.get('verdict')}")
except Exception as e:
    print(f"(could not parse predictor output: {e})")
PY
