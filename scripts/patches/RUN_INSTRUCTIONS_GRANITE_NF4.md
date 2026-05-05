# Run Instructions — Granite-Vision-3.2-2B NF4 Ablation

**For**: Q-LENS NeurIPS 2026 E&D submission (full paper deadline 2026-05-06 AOE).
**Estimated runtime**: ~30 minutes cluster wall-time (Granite 2B is the smallest model in the panel).
**Goal**: Replace the §7 line "L5: bf16 confound, ablation pending" with "L5: bf16
confound, ablation completed, finding holds/falsifies."

---

## Step 1 — Apply the patch (one-time, on login node before submitting)

The patch adds an env-var-gated NF4 path to `load_granite_vision()` so the bf16 production
behavior remains the default (no risk to other jobs). The full justification, expected
diff, and failure-mode analysis live in `scripts/patches/granite_nf4_patch.md`.

Apply the patch by replacing lines 855-884 of `scripts/week1_quant_qual_probe.py` with the
patched body shown in `granite_nf4_patch.md` Section 2. The minimal git-style diff:

```diff
 def load_granite_vision(model_id: str):
     ...
-    print(f"  attn_implementation={attn_impl}, NO QUANT (bnb dtype bug), dtype=bf16")
+    force_nf4 = os.environ.get("GRANITE_FORCE_NF4", "0") == "1"
+    quant_label = "bnb-nf4 (FORCED)" if force_nf4 else "NO QUANT (bf16)"
+    print(f"  attn_implementation={attn_impl}, {quant_label}, dtype=bf16")
     t0 = time.time()
-    model = ModelCls.from_pretrained(
-        model_id,
-        # bnb-NF4 OMITTED -- see docstring. 2B model loads cleanly in bf16.
-        device_map="auto",
-        torch_dtype=torch.bfloat16,
-        attn_implementation=attn_impl,
-        low_cpu_mem_usage=True,
-    )
+    pretrain_kwargs = dict(
+        device_map="auto",
+        torch_dtype=torch.bfloat16,
+        attn_implementation=attn_impl,
+        low_cpu_mem_usage=True,
+    )
+    if force_nf4:
+        pretrain_kwargs["quantization_config"] = build_bnb_config(load_in_4bit=True)
+    model = ModelCls.from_pretrained(model_id, **pretrain_kwargs)
```

**Verify**: `grep -n GRANITE_FORCE_NF4 scripts/week1_quant_qual_probe.py` should show
2 lines. If the script also needs an `--output-suffix` CLI flag (so the NF4 JSON doesn't
overwrite the bf16 JSON), confirm that's already supported; if not, add it as the trivial
companion change:

```diff
 ap.add_argument("--output-dir", default="results/week1", type=Path)
+ap.add_argument("--output-suffix", default="", type=str,
+                help="Appended to probe-JSON basename (e.g. _NF4 for ablations).")
 ...
-out_path = args.output_dir / f"{args.model}_quant_qual_probe.json"
+out_path = args.output_dir / f"{args.model}_quant_qual_probe{args.output_suffix}.json"
```

If the patch instead needs to be applied via env var WITHOUT modifying the script,
the SLURM script `turing/18_probe_granite_nf4.sh` reads `GRANITE_FORCE_NF4=1` already —
in that variant the load path falls back to bf16 (with a clear log line) and the JSON
output still uses the suffix.

**No commit** is required for the ablation run itself; the patch is local-only until the
finding is folded into the paper.

---

## Step 2 — Submit the job from the Turing login node

```bash
# From ${HOME}/VLAs on Turing login node:
sbatch turing/18_probe_granite_nf4.sh
```

Expected output:
```
Submitted batch job <JOBID>
```

The script:
- requests A100 80GB on `partition=quick`, `account=${SLURM_ACCOUNT:-default}`, walltime 2hr
  (real run is ~30 min; the buffer absorbs node-startup + transformers re-import)
- activates `vla_physics_v2`
- verifies transformers ≥ 4.49 and bitsandbytes is importable
- runs a **pre-flight NF4 sanity check** that loads Granite under NF4 and confirms at
  least one parameter is in a 4-bit storage class (Params4bit) before proceeding to the
  full 200-sample probe
- runs `scripts/week1_quant_qual_probe.py` with `GRANITE_FORCE_NF4=1` and `_NF4` output
  suffix so the bf16 baseline JSON is preserved

---

## Step 3 — Monitor

```bash
# Watch SLURM queue:
squeue -u $USER

# Tail the live log (replace JOBID with the one from sbatch):
tail -f jobs/probe-granite-nf4.<JOBID>.out

# Look for the pre-flight result line:
grep -E "NF4 verified|FATAL|Pre-flight OK" jobs/probe-granite-nf4.<JOBID>.out
```

Healthy run signatures (in order):
1. `transformers 4.49.x OK`
2. `bitsandbytes 0.44.1 OK`
3. `GPU OK: NVIDIA A100-SXM4-80GB ...`
4. `NF4 verified: ('language_model.model.layers.0.self_attn.q_proj.weight', 'torch.uint8', 'params4bit')`
5. `Pre-flight OK - proceeding to probe.`
6. `--- [granite-vision-3.2-2b] VAL-split probing (NF4) ---`
7. progress logs at i=50, 100, 150, 200
8. `OK [granite-vision-3.2-2b NF4] usable=200/200 ...`

If step 4 fails with `BFloat16 and Byte` mismatch, see
`granite_nf4_patch.md` Section 3 (failure modes) and Section 4 (fallback strategies).

---

## Step 4 — Verify success

```bash
# 1. Confirm the new NF4-suffixed probe JSON exists:
ls -la results/week1_turing/granite-vision-3.2-2b_quant_qual_probe_NF4.json

# 2. Confirm the bf16 baseline JSON is intact (DO NOT skip this check):
ls -la results/week1_turing/granite-vision-3.2-2b_quant_qual_probe.json

# 3. Inspect the probe stats:
python -c "
import json
nf4 = json.load(open('results/week1_turing/granite-vision-3.2-2b_quant_qual_probe_NF4.json'))
bf16 = json.load(open('results/week1_turing/granite-vision-3.2-2b_quant_qual_probe.json'))
print('bf16 stats:', bf16.get('extraction_stats'))
print('NF4  stats:', nf4.get('extraction_stats'))
print('bf16 sites:', list(bf16.get('per_site', {}).keys()))
print('NF4  sites:', list(nf4.get('per_site', {}).keys()))
"
```

Both JSONs should report `processed >= 195` (out of 200 — small loss from media-resolve
errors is acceptable). The 4 probe sites must be the same in both files
(`enc_out`, `post_proj`, `llm_8`, `llm_16`).

---

## Step 5 — Re-run h3_sensitivity_unified.py with the new Granite NF4 H3 value

```bash
# Compute H3 from the new NF4 probe:
python scripts/compute_h3_hits.py \
    --probe-json results/week1_turing/granite-vision-3.2-2b_quant_qual_probe_NF4.json \
    --output results/h3_granite_nf4.json

# Compare bf16 vs NF4 H3:
python -c "
import json
bf16 = json.load(open('results/h3_granite_bf16.json')) if __import__('os').path.exists('results/h3_granite_bf16.json') else {'h3': 0.222}
nf4  = json.load(open('results/h3_granite_nf4.json'))
delta = nf4['h3'] - bf16['h3']
print(f'bf16 H3 = {bf16[\"h3\"]:.3f}')
print(f'NF4  H3 = {nf4[\"h3\"]:.3f}')
print(f'Delta   = {delta:+.3f}')
print('Confound:', 'REJECTED (|d|<0.02)' if abs(delta)<0.02 else 'INCONCLUSIVE' if abs(delta)<0.05 else 'CONFIRMED (|d|>=0.05)')
"

# Re-fit the LOO regression with the new value:
python scripts/h3_sensitivity_unified.py \
    --models 10 \
    --granite-h3-source nf4 \
    --output results/h3_sensitivity_n10_nf4.json

# Verify Gate 5 still passes:
python -c "
import json
d = json.load(open('results/h3_sensitivity_n10_nf4.json'))
mae = d['loo_regression']['median_abs_error']
rho = d['loo_regression']['spearman_rho']
killed = d.get('kill_gate_fired', False)
print(f'mae={mae:.3f} (threshold <0.20) {\"OK\" if mae<0.20 else \"FAIL\"}')
print(f'rho={rho:.3f} (threshold >0.5)  {\"OK\" if rho>0.5 else \"FAIL\"}')
print(f'kill_gate={killed} (must be False)  {\"OK\" if not killed else \"FAIL\"}')
"
```

---

## Step 6 — Integrate into the paper §7 limitation table

Open `Neurips/paper.tex` (or wherever §7 lives) and update the L5 row:

**Before** (current state, `423602f` and earlier):
> L5: Granite-Vision-3.2-2B is the only model loaded in bf16 due to a bnb-NF4 dtype
> mismatch. *bf16 vs NF4 ablation pending.*

**After (case A — confound REJECTED, |Δ| < 0.02)**:
> L5: Granite-Vision-3.2-2B was loaded in bf16 in our main panel due to a bnb-NF4 dtype
> mismatch in the multi_modal_projector. We re-ran Granite under NF4 (with skip-list
> applied to the projector) and recovered H3 = 0.21X — within 0.0XX of the bf16 baseline
> H3 = 0.222, confirming the no-compression-cell finding is robust to quantization
> choice. Full ablation in Appendix C.

**After (case B — confound CONFIRMED, |Δ| ≥ 0.05)**:
> L5: Granite-Vision-3.2-2B was loaded in bf16 in our main panel; under NF4 its H3
> moves from 0.222 to 0.XXX (Δ = +/-0.0XX). The headline 10-VLM panel uses the NF4
> values uniformly, with bf16 numbers in Appendix C. Granite remains in the
> no-compression cell under both protocols, so the cell-mean ordering is stable;
> the absolute H3 values shift but the mechanism-vs-ratio finding holds.

In both cases also update Table 2 (the H3-by-model table) with the NF4 row, and update
the Croissant ML metadata file's `protocol` field if it currently states "bf16 (Granite)
+ NF4 (rest)".

---

## Estimated Timeline

| Step | Action | Wall time | Owner |
|------|--------|-----------|-------|
| 1 | Apply patch on login node | 2 min | You |
| 2 | `sbatch turing/18_probe_granite_nf4.sh` | <1 min queue + 30 min run | Cluster |
| 3-4 | Monitor + verify | 5 min | You |
| 5 | Re-run H3 sensitivity | 2 min CPU | You |
| 6 | Update paper §7 + Table 2 | 30 min | You |
| **Total** | | **~70 min from queue to integrated paper** | |

If submitted by 2026-05-05 21:00 UTC, the integrated paper is in hand by 22:00 UTC —
**8 hours of buffer before the 2026-05-06 AOE deadline**.

---

## Cost-of-failure analysis

If `turing/18_probe_granite_nf4.sh` fails:

**Failure scenario 1**: NF4 dtype-mismatch recurs in the same Linear layer.
*Action*: edit `scripts/patches/granite_nf4_patch.md` Section 2 to pass
`llm_int8_skip_modules=["multi_modal_projector"]` into `build_bnb_config`. Re-submit.
Cost: +30 min cluster time. Net total under 2 hours.

**Failure scenario 2**: NF4 dtype-mismatch persists even with skip-list.
*Action*: invoke fallback 4a from `granite_nf4_patch.md` — run the entire 10-VLM panel
in bf16 by adding a `GLOBAL_FORCE_BF16=1` env-var gate to all loaders. Submit 9
single-model probe jobs (the existing `turing/0[1-9]_*.sh` and `turing/1[0-7]_*.sh` plus
new bf16 wrappers). Cost: ~4.5hr cluster wall-time, all parallelizable. Schedule by
2026-05-05 18:00 UTC for safety margin.

**Failure scenario 3**: Fallback 4a is also blocked (e.g., Phi-3.5-Vision OOMs in bf16).
*Action*: drop to fallback 4b — run only LLaVA-OneVision-7B in bf16 alongside Granite-bf16
to provide a 1-model cross-quantization control. Cost: ~30 min. Weaker but still
defensible: the §7 paragraph becomes "we sampled one additional model under both
protocols and observed Δ_H3 < 0.0X, suggesting quantization choice is not the dominant
factor in the no-compression-cell pattern."

In all cases, the §7 row goes from "ablation pending" to "ablation completed" — which is
the actual reviewer-blocker. Even an inconclusive Δ is strictly better than no number.

---

## File inventory (created by this work)

- `scripts/patches/granite_nf4_patch.md` — patch documentation, expected failure modes,
  fallback strategies (Sections 1–5).
- `turing/18_probe_granite_nf4.sh` — SLURM job that pre-flights NF4 then runs the probe
  with `_NF4` output suffix so bf16 baseline is not overwritten.
- `scripts/patches/RUN_INSTRUCTIONS_GRANITE_NF4.md` — this file. End-to-end run
  instructions, paper-integration text, and timeline.

No production scripts in `scripts/` or `src/` are modified by this task. The patch is
applied locally on the login node (Step 1) and is gated behind `GRANITE_FORCE_NF4=1` so
the bf16 default path is preserved.
