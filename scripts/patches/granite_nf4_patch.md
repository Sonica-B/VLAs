# Granite-Vision-3.2-2B NF4 Re-Run Patch

**Purpose**: Re-run Granite-Vision-3.2-2B in NF4 4-bit quantization (instead of bf16) to
remove the bf16-only confound from the Q-LENS 10-VLM panel. Granite is the **only** model
loaded in bf16 (the other 9 use bnb-NF4), the **only** 2025-released model, AND the only
member of the no-compression cell with mean H3 = 0.222. Reviewers will read this as a
quantization artifact unless we run the apples-to-apples ablation.

**Risk level**: MEDIUM. The original NF4 attempt failed with a `RuntimeError: self and
mat2 must have the same dtype, but got BFloat16 and Byte` (documented at
`scripts/week1_quant_qual_probe.py:841-851`). If the same failure recurs we fall back to
the bf16-for-all ablation (Section 4a).

**Target**: NeurIPS 2026 E&D submission (full paper deadline 2026-05-06 AOE). This patch
must produce a usable Granite-NF4 H3 value within ~30 minutes of cluster compute so the
finding can be folded into §7 (Limitations) before the deadline.

---

## Section 1 — Current state (where Granite is excluded from NF4)

The NF4-exclusion is implemented at three call sites. All three need to be neutralized to
re-run Granite under NF4.

**1a. `scripts/week1_quant_qual_probe.py:829-884` — `load_granite_vision()`**

This loader explicitly omits `quantization_config=`. The decisive lines:

```
829: def load_granite_vision(model_id: str):
...
841:     QUANTIZATION NOTE: bnb-NF4 triggers a dtype mismatch
842:         `RuntimeError: self and mat2 must have the same dtype, but got
843:          BFloat16 and Byte`
...
863:     print(f"  attn_implementation={attn_impl}, NO QUANT (bnb dtype bug), dtype=bf16")
864:     t0 = time.time()
865:     model = ModelCls.from_pretrained(
866:         model_id,
867:         # bnb-NF4 OMITTED -- see docstring. 2B model loads cleanly in bf16.
868:         device_map="auto",
869:         torch_dtype=torch.bfloat16,
870:         attn_implementation=attn_impl,
871:         low_cpu_mem_usage=True,
872:     )
```

Every other loader in this file passes `quantization_config=build_bnb_config(load_in_4bit=True)`
(see grep: lines 422, 448, 478, 504, 544, 575, 624, 657, 706, 758, 811). Granite's loader
is the only outlier.

**1b. `scripts/extract_training_features.py:309-327` — `load_model()`**

```
309:     eager_families = {"gemma", "phi35v", "molmo", "pixtral",
310:                       "idefics3", "idefics2", "blip2"}
...
315:     no_quant_families = {"granite_vision"}
316:     attn = pick_attn_impl(allow_sdpa=True)
317:     use_quant = family not in no_quant_families
...
326:     if use_quant:
327:         pretrain_kwargs["quantization_config"] = build_bnb_config(load_in_4bit=True)
```

`granite_vision` is the sole entry in `no_quant_families`.

**1c. `turing/14_probe_granite_vision.sh`**

The production SLURM script does not pass any flags forcing or forbidding NF4 — the
NF4-exclusion happens inside `load_granite_vision()`. Probe results land in
`results/week1_turing/granite-vision-3.2-2b_quant_qual_probe.json` (the bf16 baseline we
must NOT overwrite).

**Original error context** (from `scripts/week1_quant_qual_probe.py:841-851` docstring):

> Cause: a Linear-equivalent in the multi_modal_projector or vision_tower is not being
> properly wrapped by Linear4bit, so the raw uint8 (Byte) quantized weight reaches matmul
> without dequantization.

No `jobs/14_probe_granite*.out` log file is checked in (the `jobs/` directory only has
weekb-*.out and v2-test.out files). The error is documented inline in the loader
docstring rather than from a log artifact.

---

## Section 2 — Proposed patch (exact file:line changes)

The patch is **non-invasive** and gated by an environment variable so the bf16 production
path stays unchanged. Modify only `scripts/week1_quant_qual_probe.py` — `extract_training_features.py`
is not used by this ablation (we are re-running the val probe, not retraining SCAS PCA).

**Patched `load_granite_vision()`** (replaces lines 855-884):

```python
def load_granite_vision(model_id: str):
    from transformers import AutoProcessor
    try:
        from transformers import LlavaNextForConditionalGeneration as ModelCls
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelCls

    attn_impl = pick_attn_impl(allow_sdpa=True)
    # NEW: env-gated NF4 ablation for Q-LENS reviewer-blocker (May 2026).
    force_nf4 = os.environ.get("GRANITE_FORCE_NF4", "0") == "1"
    quant_label = "bnb-nf4 (FORCED)" if force_nf4 else "NO QUANT (bf16)"
    print(f"Loading {model_id}")
    print(f"  attn_implementation={attn_impl}, {quant_label}, dtype=bf16")
    t0 = time.time()
    pretrain_kwargs = dict(
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    if force_nf4:
        pretrain_kwargs["quantization_config"] = build_bnb_config(load_in_4bit=True)
    model = ModelCls.from_pretrained(model_id, **pretrain_kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
            print(f"  (set tokenizer.pad_token = eos_token)")
    except Exception:
        pass
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    return model, processor
```

**Diff summary**:
- Add `force_nf4 = os.environ.get("GRANITE_FORCE_NF4", "0") == "1"` gate
- Conditional `quantization_config=build_bnb_config(load_in_4bit=True)`
- Default `GRANITE_FORCE_NF4=0` preserves existing bf16 behavior — no production breakage

`os` is already imported at module level (line 46), no new imports needed.

---

## Section 3 — Expected failure modes if NF4 still doesn't work

The original error traceback (docstring lines 842-844):
```
RuntimeError: self and mat2 must have the same dtype, but got BFloat16 and Byte
```

Likely failure modes for the re-run, in decreasing order of probability:

1. **Same dtype-mismatch in `multi_modal_projector`** — Linear4bit replacement skips the
   2-layer MLP because Granite's projector module name doesn't match the bnb default
   `llm_int8_skip_modules` regex. *Mitigation*: pass
   `llm_int8_skip_modules=["multi_modal_projector"]` to `BitsAndBytesConfig` so the
   projector stays in bf16 while the LLM runs NF4. If this stabilizes, document as a
   "partial-quantization" condition (still much closer to apples-to-apples than full bf16).

2. **Same error in vision_tower (SigLIP)** — bnb skips the SigLIP encoder. *Mitigation*:
   add `vision_tower` to `llm_int8_skip_modules`. SigLIP-only-bf16 + LLM-NF4 is the same
   pattern as InternVL3 in our panel, so this is reviewer-defensible.

3. **OOM** — extremely unlikely on A100 80GB. Granite 2B in NF4 is ~1.2GB.

4. **Probe-site discovery breaks under NF4** — bnb sometimes wraps Linear inside `.weight`
   tensors which can confuse hook registration. *Mitigation*: hooks fire on the parent
   `nn.Linear` / `nn.Module`, not on weights. Should be transparent. If discovery fails,
   the existing fallback paths in `MODEL_REGISTRY["granite-vision-3.2-2b"]["probe_candidates"]`
   (3 paths, lines 382-399) provide redundancy.

5. **Numerically degenerate features** — NF4 quantization could collapse Granite's
   post_proj features (low rank → probe accuracy near chance). This would *itself* be a
   finding worth reporting: "the bf16 H3 = 0.222 is not robust to NF4 quantization."

---

## Section 4 — Fallback ablation strategies

### 4a. Force bf16 for ALL 10 models (apples-to-apples, all-bf16)

If NF4 cannot be made to work for Granite, the cleanest defensible alternative is to
re-run the **other 9 models in bf16** so the panel is uniformly bf16. Cost: ~9× the
single-model probe runtime (~30 min × 9 = 4.5 hr cluster wall-time, well within 12hr
quick-partition budget). VRAM for 8B-class models in bf16 is ~16GB, all fit on A100-80GB.

**Implementation sketch**: add a `GLOBAL_FORCE_BF16=1` env var that short-circuits every
loader's `quantization_config=build_bnb_config(load_in_4bit=True)`. Then run the existing
`turing/0[1-9]_*.sh` and `turing/1[0-7]_*.sh` probe scripts with that env var.

This is **strictly more conservative** than re-running just Granite under NF4 and would be
defensible against any reviewer challenge ("you removed a confound by introducing
uniformity"). The cost is one paragraph in §7 explaining the panel-wide re-run and a
header row in the H3 table noting "all bf16 panel — bnb-NF4 ablation in Appendix C".

### 4b. Cross-model bf16/NF4 comparison (LLaVA-OneVision-7B as second bf16 anchor)

Run **LLaVA-OneVision-7B in bf16** (it currently runs NF4) and compare its bf16 vs NF4
H3 values. If the delta is < 0.02, we have evidence that quantization choice does not
matter for H3, which retroactively justifies the existing Granite bf16 number. This is
weaker than 4a but cheaper (~30 min for one extra model) and produces a Δ_NF4-bf16 stat
that strengthens the limitation paragraph.

**Implementation sketch**: add `LLAVA_FORCE_BF16=1` env-var gate to `load_llava_ov()`,
modeled after the patch in §2.

### Recommendation

Run §2 (Granite NF4) first — 30 min job. If it succeeds, this resolves the confound
directly. If it fails with one of the failure modes in §3, run §4a (panel-wide bf16) as
a 4.5hr overnight job before the 2026-05-06 deadline. §4b is only worth doing if both
§2 and §4a fail (extremely unlikely).

---

## Section 5 — How to verify the patch worked

**5a. Sanity check at load time** (the SLURM script `turing/18_probe_granite_nf4.sh`
includes this):

```python
import torch
from transformers import LlavaNextForConditionalGeneration, BitsAndBytesConfig
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                         bnb_4bit_compute_dtype=torch.bfloat16,
                         bnb_4bit_use_double_quant=True)
m = LlavaNextForConditionalGeneration.from_pretrained(
    "ibm-granite/granite-vision-3.2-2b",
    device_map="auto", torch_dtype=torch.bfloat16,
    quantization_config=bnb, low_cpu_mem_usage=True)
# Assert NF4 weights present:
saw_4bit = any("4bit" in type(p).__name__.lower() for p in m.parameters())
print("Has 4-bit param:", saw_4bit)  # MUST be True under the patch
```

If `saw_4bit == False`, the patch is silently inactive (env var not exported, wrong
loader called, etc.).

**5b. H3 comparison** (the actual scientific check):

Once the probe job finishes, the new JSON is at
`results/week1_turing/granite-vision-3.2-2b_quant_qual_probe_NF4.json`. Run:

```bash
python scripts/compute_h3_hits.py \
    --probe-json results/week1_turing/granite-vision-3.2-2b_quant_qual_probe_NF4.json \
    --output results/h3_granite_nf4.json
```

Compare:
- `bf16  H3 = 0.222` (existing, from `granite-vision-3.2-2b_quant_qual_probe.json`)
- `NF4   H3 = ???`  (new)

If |Δ| < 0.02, the bf16 confound is **falsified** — finding holds, paragraph in §7 says
"bf16 vs NF4 H3 = 0.222 vs 0.21X (Δ = 0.0XX), confound rejected."

If |Δ| > 0.05, the confound is **confirmed** — Granite's no-compression-cell behavior
under NF4 is materially different and should be reported as the headline number with
bf16 in the appendix. The paper's mechanism-vs-ratio framing (commit `2155d3b`) still
holds because both Granite-bf16 and Granite-NF4 sit in the no-compression cell — the
H3 value moves but the cell membership doesn't.

**5c. LOO regression re-fit**:

Re-run the H3 sensitivity unifier with the new value:
```bash
python scripts/h3_sensitivity_unified.py \
    --models 10 \
    --granite-h3-source nf4 \
    --output results/h3_sensitivity_n10_nf4.json
```

Median absolute error should still be < 0.20 (Gate 5 threshold). Spearman ρ should still
be > 0.5. If Gate 5 trips after the swap, re-evaluate fallback 4a.
