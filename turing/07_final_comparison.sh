#!/bin/bash
# ============================================================================
# JOB 7/7: Final Cross-Method Comparison
# ============================================================================
# Aggregates ALL results from Jobs 1-6 into a single comparison table:
#
#   Baseline vs LoRA (B, C) vs SCAS (amplify, best alpha) vs PEM
#
# For each model × method, reports:
#   acc_quant, acc_qual, delta_quant, delta_qual
#
# Also compares Turing full-fidelity results against laptop partial results
# to quantify the impact of the resolution/sample-count compromises.
#
# Submit AFTER all previous jobs: sbatch turing/07_final_comparison.sh
# ============================================================================

#SBATCH -J final-comparison
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH --account=cngan
#SBATCH --export=ALL
#SBATCH -D /home/ssboyane/VLAs
#SBATCH -o jobs/%x.%j.out

set -e
mkdir -p jobs results/final_comparison

# Load environment (modules + pip packages).
source /home/ssboyane/VLAs/.turing_env


echo "=== JOB 7/7: Final Cross-Method Comparison ($(date)) ==="

python -u -c "
import json, sys
from pathlib import Path

print('='*80)
print(' FINAL COMPARISON: Laptop (partial) vs Turing (full-fidelity)')
print('='*80)

# Helper to load a JSON safely.
def load(path):
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text())
    return None

# --- Week 1: Laptop vs Turing probe results ---
print()
print('--- WEEK 1: Probing (task_type target, post_proj site) ---')
print(f'{\"model\":<20} {\"source\":<10} {\"quant_acc\":>10} {\"qual_acc\":>10} {\"d(q-l)\":>10}')
print('-'*64)

for model in ['qwen3-vl-8b', 'qwen2.5-vl-7b', 'internvl3-8b', 'gemma4-e4b']:
    for source, dir_name in [('laptop', 'results/week1'), ('turing', 'results/week1_turing')]:
        data = load(f'{dir_name}/{model}_quant_qual_probe.json')
        if data is None:
            print(f'{model:<20} {source:<10} {\"n/a\":>10} {\"n/a\":>10} {\"n/a\":>10}')
            continue
        try:
            pr = data['probe_results']['task_type']['post_proj']
            q = pr.get('quantitative', {}).get('mean_acc')
            l = pr.get('qualitative', {}).get('mean_acc')
            q_s = f'{q:.4f}' if q else 'n/a'
            l_s = f'{l:.4f}' if l else 'n/a'
            d = f'{q-l:+.4f}' if (q and l) else 'n/a'
            print(f'{model:<20} {source:<10} {q_s:>10} {l_s:>10} {d:>10}')
        except (KeyError, TypeError):
            print(f'{model:<20} {source:<10} {\"err\":>10} {\"err\":>10} {\"err\":>10}')

# --- Week 2: Laptop vs Turing LoRA results ---
print()
print('--- WEEK 2: LoRA Intervention (Qwen3-VL-8B) ---')
print(f'{\"condition\":<20} {\"source\":<10} {\"acc_quant\":>10} {\"acc_qual\":>10} {\"d_quant\":>10} {\"d_qual\":>10}')
print('-'*72)

for source, dir_name in [('laptop', 'results/week2'), ('turing', 'results/week2_turing')]:
    base = load(f'{dir_name}/baseline_eval.json')
    for cond in ['B', 'C']:
        data = load(f'{dir_name}/condition_{cond}_eval.json')
        if data is None or base is None:
            print(f'Cond {cond:<17} {source:<10} {\"n/a\":>10} {\"n/a\":>10} {\"n/a\":>10} {\"n/a\":>10}')
            continue
        aq = data.get('acc_quant')
        al = data.get('acc_qual')
        dq = aq - base.get('acc_quant', 0) if aq else None
        dl = al - base.get('acc_qual', 0) if al else None
        print(f'Cond {cond:<17} {source:<10} {aq:>10.4f} {al:>10.4f} {dq:>+10.4f} {dl:>+10.4f}')

# --- SCAS: Laptop vs Turing ---
print()
print('--- SCAS AMPLIFY (best alpha per model) ---')
print(f'{\"model\":<20} {\"source\":<10} {\"best_a\":>8} {\"acc_quant\":>10} {\"acc_qual\":>10} {\"d_quant\":>10} {\"d_qual\":>10}')
print('-'*76)

for model in ['qwen3-vl-8b', 'qwen2.5-vl-7b', 'internvl3-8b', 'gemma4-e4b']:
    for source, dir_name in [('laptop', 'results/week3'), ('turing', 'results/week3_turing')]:
        data = load(f'{dir_name}/scas_sweep_{model}.json')
        if data is None:
            print(f'{model:<20} {source:<10} {\"n/a\":>8} {\"n/a\":>10} {\"n/a\":>10} {\"n/a\":>10} {\"n/a\":>10}')
            continue
        sweep = data.get('sweep', [])
        baseline = next((s for s in sweep if s['alpha'] == 0), None)
        if not baseline:
            print(f'{model:<20} {source:<10} {\"err\":>8}')
            continue
        best = max(sweep, key=lambda s: (s.get('acc_quant') or 0) - (baseline.get('acc_quant') or 0))
        dq = best['acc_quant'] - baseline['acc_quant'] if best.get('acc_quant') and baseline.get('acc_quant') else None
        dl = best['acc_qual'] - baseline['acc_qual'] if best.get('acc_qual') and baseline.get('acc_qual') else None
        print(f'{model:<20} {source:<10} {best[\"alpha\"]:>8.1f} {best[\"acc_quant\"]:>10.4f} {best[\"acc_qual\"]:>10.4f} {dq:>+10.4f} {dl:>+10.4f}')

print()
print('='*80)
" 2>&1

# Save the comparison as a JSON.
echo ""
echo "Saving final comparison to results/final_comparison/"
python -u -c "
import json
from pathlib import Path

out = {}
for subdir in ['results/week1', 'results/week1_turing', 'results/week2', 'results/week2_turing', 'results/week3', 'results/week3_turing']:
    p = Path(subdir)
    if p.exists():
        out[subdir] = {f.name: json.loads(f.read_text()) for f in p.glob('*.json')}

Path('results/final_comparison').mkdir(parents=True, exist_ok=True)
Path('results/final_comparison/all_results.json').write_text(json.dumps(out, indent=2))
print('Written: results/final_comparison/all_results.json')
" 2>&1

echo ""
echo "=== JOB 7/7 Complete ($(date)) ==="
echo "ALL TURING JOBS COMPLETE."
