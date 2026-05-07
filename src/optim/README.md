# src.optim — Inference Optimization Stack

Canonical implementation of every optimization the Option C pivot depends on.
Every script in `scripts/` that runs inference or probes should import from
here rather than reimplementing these helpers.

## Why

The 2-week NeurIPS 2026 critical path (diagnosis → intervention → PhysBench
delta) has to run full-scale experiments on a single 16 GB laptop GPU. This
module is what makes that feasible:

- **VRAM under 8 GB for every 8B model** via bitsandbytes 4-bit nf4 double-quant + bf16 compute.
- **No wasted forward passes** — a single forward through the VLM populates all 4 probe sites via hooks.
- **No wasted tokenization** — prompt cache means every script after the first eats tokenization cost once.
- **Zero-loss crashes** — JSONL append + resume means a crash at sample 847/1000 loses at most 1 sample.
- **Full tracebacks** — every run writes to `logs/{run}_{timestamp}.log`; no more scrollback hunting.

## Submodules

| Module            | Purpose                                                              |
|-------------------|----------------------------------------------------------------------|
| `vram.py`         | bnb config, VRAM snapshots, `hard_cleanup`, `assert_vram_below`      |
| `compute.py`      | `pick_attn_impl`, safe `try_compile`, `fast_gen_config`, `inference_ctx` |
| `features.py`     | `ProbeSites`, `FeatureCache` (mmap), `extract_features_one_pass`     |
| `cache.py`        | `PromptCache` — on-disk tokenized-prompt cache                       |
| `resilience.py`   | `JsonlAppender`, `resume_completed_ids`, `configure_traceback_logging` |
| `physbench_split.py` | `classify_quantitative`, `split_physbench` — core pivot utility   |

## Smoke test

```bash
python scripts/optim_smoke_test.py
```

Exit 0 on this machine means Week 1 experiments will have the full
optimization stack available. Exit non-zero means something is broken on
this box and needs fixing before Week 1 starts.

## Usage pattern for a new eval script

```python
from src.optim import (
    build_bnb_config, hard_cleanup, snapshot_vram, format_vram_delta,
    pick_attn_impl, try_compile, fast_gen_config, inference_ctx,
    JsonlAppender, resume_completed_ids, configure_traceback_logging,
    PromptCache,
    ProbeSites, FeatureCache, extract_features_one_pass,
    classify_quantitative, split_physbench,
)

logger = configure_traceback_logging("logs", "my_eval")

# 1. Resume
done = resume_completed_ids("results/my_eval.jsonl")
logger.info(f"resuming, {len(done)} already complete")

# 2. Load model under VRAM guard
before = snapshot_vram()
model = AutoModel.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    quantization_config=build_bnb_config(load_in_4bit=True),
    torch_dtype=torch.bfloat16,
    attn_implementation=pick_attn_impl(),
    device_map="auto",
    low_cpu_mem_usage=True,
)
model = try_compile(model, "qwen3-vl-8b")
logger.info(format_vram_delta(before, snapshot_vram()))

# 3. Run eval with prompt cache + resilient writer
prompts = PromptCache("cache/prompts", "qwen3-vl-8b")
sites = ProbeSites.for_model("qwen3-vl-8b")
features = FeatureCache("cache/features", "qwen3-vl-8b", "val")

with JsonlAppender("results/my_eval.jsonl") as out:
    for sample in samples:
        if sample["id"] in done:
            continue
        # ... tokenize with prompts cache hit, forward with inference_ctx,
        #     extract features with extract_features_one_pass, score, write.
        out.write({"sample_id": sample["id"], "score": ...})

# 4. Clean shutdown
hard_cleanup(model)
```

## Non-goals

- **Not a framework.** These are utility functions, not a class hierarchy.
  Scripts stay in control of the loop.
- **No distributed / multi-GPU.** The whole project is single-GPU. If that
  changes we revisit.
- **No speculative decoding.** Added complexity, marginal win for batch=1 MC.
- **No custom CUDA kernels.** Everything is stock PyTorch + HF + bnb.
