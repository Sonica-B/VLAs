#!/usr/bin/env python3
"""
Smoke test for src.optim — verifies every helper imports and runs on the
current machine without a GPU or a loaded model.

Run:
    python scripts/optim_smoke_test.py

Exit code 0 means all submodules are usable; non-zero means something on the
inference optimization stack is broken on this box and the Week 1 experiments
would hit the same failure.
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Apply env vars BEFORE importing torch-dependent modules.
from src.optim.vram import set_cuda_alloc_env
set_cuda_alloc_env()

import torch  # noqa: E402

from src.optim import (  # noqa: E402
    # vram
    build_bnb_config, snapshot_vram, hard_cleanup, format_vram_delta, assert_vram_below,
    # compute
    pick_attn_impl, try_compile, fast_gen_config, inference_ctx,
    # features
    ProbeSites, FeatureCache, register_probe_hooks, extract_features_one_pass,
    # cache
    PromptCache,
    # resilience
    JsonlAppender, configure_traceback_logging, resume_completed_ids,
    # physbench split
    classify_quantitative, split_physbench,
)


def section(name: str) -> None:
    print(f"\n--- {name} ---")


def test_vram():
    section("vram")
    before = snapshot_vram()
    print(f"  before: {before}")
    # bnb config should build regardless of GPU presence (transformers has it).
    bnb = build_bnb_config(load_in_4bit=True)
    print(f"  bnb_config: {type(bnb).__name__ if bnb else 'None (transformers missing)'}")
    hard_cleanup()
    after = snapshot_vram()
    print(f"  after cleanup: {after}")
    print(f"  {format_vram_delta(before, after)}")
    # Assert-below with a very high limit should pass even on CPU.
    assert_vram_below(1000.0, label="smoke")
    print("  assert_vram_below OK")


def test_compute():
    section("compute")
    impl = pick_attn_impl(prefer_flash=True, allow_sdpa=True)
    print(f"  pick_attn_impl: {impl}")
    # try_compile on a trivial nn.Module — should no-op if triton missing.
    m = torch.nn.Linear(4, 4)
    m = try_compile(m, model_name="dummy_linear")
    gen = fast_gen_config(max_new_tokens=16)
    print(f"  fast_gen_config keys: {sorted(gen.keys())}")
    # inference_ctx on CPU — should not raise.
    with inference_ctx(device_type="cpu"):
        y = m(torch.randn(2, 4))
    print(f"  inference_ctx cpu forward OK: {tuple(y.shape)}")


def test_features():
    section("features")
    # Build a tiny fake VLM-ish module with 4 hookable sites.
    class FakeVLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(16, 16)
            self.visual.merger = torch.nn.Linear(16, 32)  # attached as attr
            self.model = torch.nn.ModuleDict({
                "layers": torch.nn.ModuleList([torch.nn.Linear(32, 32) for _ in range(20)]),
            })

        def forward(self, x):
            v = self.visual(x)
            m = self.visual.merger(v)
            # Thread through layers 0..16.
            h = m
            for i in range(17):
                h = self.model["layers"][i](h)
            return h

    model = FakeVLM()

    # Hand-built sites for the fake model (not using for_model — that's for real VLMs).
    sites = ProbeSites(
        model_name="fake",
        paths={
            "enc_out":  "visual",
            "post_proj": "visual.merger",
            "llm_8":    "model.layers.8",
            "llm_16":   "model.layers.16",
        },
    )

    # ignore_cleanup_errors=True: Windows holds mmap file locks until GC runs.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cache = FeatureCache(Path(tmp), "fake", "smoke")

        batch = torch.randn(3, 16)
        captured = extract_features_one_pass(
            model=model,
            sites=sites,
            forward_fn=lambda: model(batch),
            sample_ids=["s0", "s1", "s2"],
            cache=cache,
        )
        for site, arr in captured.items():
            print(f"  {site}: {arr.shape} dtype={arr.dtype}")
        assert set(captured.keys()) == {"enc_out", "post_proj", "llm_8", "llm_16"}
        assert captured["post_proj"].shape == (3, 32)
        # Re-open cache and verify mmap read.
        cache2 = FeatureCache(Path(tmp), "fake", "smoke")
        assert cache2.completed_ids() == {"s0", "s1", "s2"}
        mm = cache2.load_site("post_proj")
        assert mm.shape == (3, 32)
        print(f"  mmap reload OK: completed_ids={len(cache2.completed_ids())}")
        # Explicitly release the memmap so Windows can delete the file.
        del mm
        del cache, cache2
        import gc as _gc
        _gc.collect()


def test_prompt_cache():
    section("cache")
    with tempfile.TemporaryDirectory() as tmp:
        pc = PromptCache(Path(tmp), "fake-model")
        assert pc.get("hello") is None
        pc.put("hello", {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]})
        got = pc.get("hello")
        assert got == {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
        assert len(pc) == 1
        # Second PromptCache reads the persisted index.
        pc2 = PromptCache(Path(tmp), "fake-model")
        assert pc2.get("hello") is not None
        print("  put/get/reload OK")


def test_resilience():
    section("resilience")
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        log_dir = Path(tmp) / "logs"
        logger = configure_traceback_logging(log_dir, "smoke", also_stdout=False)
        logger.info("logger works")
        # JsonlAppender + resume.
        jsonl = Path(tmp) / "results.jsonl"
        with JsonlAppender(jsonl) as app:
            app.write({"sample_id": "a", "score": 1})
            app.write({"sample_id": "b", "score": 0})
        done = resume_completed_ids(jsonl)
        assert done == {"a", "b"}
        # Simulate a crash: append a truncated line.
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write('{"sample_id": "c", "sco')  # no newline, no closing
        done2 = resume_completed_ids(jsonl)
        assert done2 == {"a", "b"}, f"expected resume to skip truncated line, got {done2}"
        print(f"  jsonl append+resume OK, completed={sorted(done2)}")

        # Status-filter regression test: error entries should NOT block retry.
        jsonl_status = Path(tmp) / "results_status.jsonl"
        with JsonlAppender(jsonl_status) as app:
            app.write({"sample_id": "ok1", "status": "ok", "score": 1})
            app.write({"sample_id": "err1", "status": "error", "error": "boom"})
            app.write({"sample_id": "ok2", "status": "ok", "score": 1})
            app.write({"sample_id": "legacy", "score": 1})  # no status = legacy ok
        done_filtered = resume_completed_ids(jsonl_status)
        assert done_filtered == {"ok1", "ok2", "legacy"}, (
            f"status filter failed: expected {{ok1, ok2, legacy}}, got {done_filtered}"
        )
        print(f"  status filter OK: err1 correctly excluded, {sorted(done_filtered)}")
        print(f"  log files in {log_dir}: {[p.name for p in log_dir.iterdir()]}")
        # Close all logger handlers so Windows can delete the log file on cleanup.
        for h in list(logger.handlers):
            try:
                h.close()
            except Exception:
                pass
            logger.removeHandler(h)


def test_physbench_split():
    section("physbench_split")
    samples = [
        {"question": "Which object is heavier?", "options": ["A) ball", "B) box"], "answer": "A"},
        {"question": "What is the mass of the red object?", "options": ["A) 2 kg", "B) 5 kg", "C) 10 kg"], "answer": "B"},
        {"question": "Will the cup fall?", "options": ["yes", "no"], "answer": "yes"},
        {"question": "How many blocks are stacked?", "options": ["2", "3", "4", "5"], "answer": "3"},
        {"question": "Is the surface slippery?", "options": ["yes", "no"], "answer": "no"},
    ]
    labels = [classify_quantitative(s) for s in samples]
    print(f"  labels: {labels}")
    quant, qual = split_physbench(samples)
    print(f"  quant n={len(quant)} qual n={len(qual)}")
    # Expected: samples 1 (mass + units), 3 (how many) → quantitative.
    assert len(quant) == 2, f"expected 2 quantitative samples, got {len(quant)}"
    assert len(qual) == 3, f"expected 3 qualitative samples, got {len(qual)}"
    print("  classifier OK on 5-sample sanity set")


def main():
    print("=" * 60)
    print("src.optim smoke test")
    print("=" * 60)
    print(f"python: {sys.version.split()[0]}")
    print(f"torch:  {torch.__version__}")
    print(f"cuda:   {'available' if torch.cuda.is_available() else 'NOT available (CPU only)'}")

    test_vram()
    test_compute()
    test_features()
    test_prompt_cache()
    test_resilience()
    test_physbench_split()

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
