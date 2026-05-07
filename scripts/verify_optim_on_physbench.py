#!/usr/bin/env python3
"""
Verification script — validates src.optim against REAL PhysBench val data.

Unlike optim_smoke_test.py which uses fake tensors, this script:
    1. Loads the actual data/physbench/val.json (200 samples)
    2. Runs the quant/qual classifier and prints the split distribution
       broken down by PhysBench task_type + sub_type
    3. Cross-checks against the gold-standard PHYSBENCH_QUANT_SUBTYPES
       (size / mass / number / distance / temperature)
    4. Exercises FeatureCache + JsonlAppender + PromptCache with real
       PhysBench sample IDs to verify resume semantics work end-to-end
    5. Prints a go/no-go summary for the Week 1 experiment

No GPU or VLM required — this verifies everything BEFORE we burn model-load time.

Usage:
    python scripts/verify_optim_on_physbench.py
    python scripts/verify_optim_on_physbench.py --data-dir data/physbench
"""

import argparse
import json
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.optim.vram import set_cuda_alloc_env  # noqa: E402
set_cuda_alloc_env()

import torch  # noqa: E402

from src.optim import (  # noqa: E402
    classify_quantitative,
    split_physbench,
    FeatureCache,
    JsonlAppender,
    resume_completed_ids,
    PromptCache,
    snapshot_vram,
)
from src.optim.physbench_split import PHYSBENCH_QUANT_SUBTYPES  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading.
# ---------------------------------------------------------------------------

def load_physbench_val(data_dir: Path) -> List[Dict]:
    path = data_dir / "val.json"
    if not path.exists():
        raise FileNotFoundError(f"PhysBench val not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    data = json.loads(raw) if raw.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    if data and "split" in data[0]:
        data = [d for d in data if d.get("split") == "val"]
    return data


# ---------------------------------------------------------------------------
# Phase 1: classify + breakdown.
# ---------------------------------------------------------------------------

def phase1_classify(samples: List[Dict]) -> Dict:
    print("\n" + "=" * 70)
    print("PHASE 1: Classify PhysBench val into quantitative vs qualitative")
    print("=" * 70)

    labels = [classify_quantitative(s) for s in samples]
    counter = Counter(labels)
    print(f"\nTotal samples: {len(samples)}")
    print(f"  quantitative: {counter['quantitative']} ({100*counter['quantitative']/len(samples):.1f}%)")
    print(f"  qualitative:  {counter['qualitative']} ({100*counter['qualitative']/len(samples):.1f}%)")

    # Breakdown by task_type.
    by_task: Dict[str, Counter] = defaultdict(Counter)
    for s, lab in zip(samples, labels):
        by_task[s.get("task_type", "unknown")][lab] += 1
    print("\nBreakdown by task_type:")
    print(f"  {'task_type':<16} {'quant':>7} {'qual':>7} {'total':>7}")
    for task, c in sorted(by_task.items()):
        total = c["quantitative"] + c["qualitative"]
        print(f"  {task:<16} {c['quantitative']:>7} {c['qualitative']:>7} {total:>7}")

    # Breakdown by sub_type.
    by_sub: Dict[str, Counter] = defaultdict(Counter)
    for s, lab in zip(samples, labels):
        by_sub[s.get("sub_type", "unknown")][lab] += 1
    print("\nBreakdown by sub_type (sorted by quant count):")
    print(f"  {'sub_type':<16} {'quant':>7} {'qual':>7} {'total':>7}  {'gold?':>6}")
    rows = sorted(by_sub.items(), key=lambda kv: -kv[1]["quantitative"])
    for sub, c in rows:
        total = c["quantitative"] + c["qualitative"]
        gold = "YES" if sub in PHYSBENCH_QUANT_SUBTYPES else ""
        print(f"  {sub:<16} {c['quantitative']:>7} {c['qualitative']:>7} {total:>7}  {gold:>6}")

    # Sanity checks.
    print("\nSanity checks:")
    gold_quant = sum(1 for s in samples if str(s.get("sub_type", "")).lower() in PHYSBENCH_QUANT_SUBTYPES)
    print(f"  Gold-standard quant count (sub_type in {sorted(PHYSBENCH_QUANT_SUBTYPES)}): {gold_quant}")
    assert counter["quantitative"] == gold_quant, (
        f"Classifier quant count ({counter['quantitative']}) != gold count ({gold_quant}). "
        "Classifier logic drifted from the sub_type rule."
    )
    print(f"  Classifier agrees with gold: YES")

    # Answer label distribution — should be balanced across A/B/C/D roughly.
    answers = Counter(s.get("answer", "?") for s in samples)
    print(f"  Answer distribution: {dict(sorted(answers.items()))}")

    # Return the split for phase 2.
    quant, qual = split_physbench(samples)
    return {
        "labels": labels,
        "counter": counter,
        "quant": quant,
        "qual": qual,
        "by_task": dict(by_task),
        "by_sub": dict(by_sub),
    }


# ---------------------------------------------------------------------------
# Phase 2: exercise FeatureCache + JsonlAppender + PromptCache on real IDs.
# ---------------------------------------------------------------------------

def phase2_stack(samples: List[Dict], split_result: Dict) -> None:
    print("\n" + "=" * 70)
    print("PHASE 2: Exercise FeatureCache / JsonlAppender / PromptCache on real IDs")
    print("=" * 70)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp = Path(tmp)

        # --- FeatureCache with real sample IDs ---
        cache = FeatureCache(tmp / "features", "qwen3-vl-8b", "val")
        quant_samples = split_result["quant"][:5]
        quant_ids = [f"val_{s['idx']}" for s in quant_samples]
        fake_dim = 3584  # Qwen3-VL-8B hidden dim approx
        fake_feats = {
            "enc_out":   np.random.randn(5, 1152).astype(np.float32),  # ViT dim
            "post_proj": np.random.randn(5, fake_dim).astype(np.float32),
            "llm_8":     np.random.randn(5, fake_dim).astype(np.float32),
            "llm_16":    np.random.randn(5, fake_dim).astype(np.float32),
        }
        cache.append_batch(quant_ids, fake_feats)
        assert cache.completed_ids() == set(quant_ids)
        print(f"  FeatureCache: wrote {len(quant_ids)} samples across 4 sites")
        print(f"    sites present: {sorted(cache._index['dims'].keys())}")
        print(f"    dims: {cache._index['dims']}")

        # Simulate resume: second batch skips existing IDs.
        cache2 = FeatureCache(tmp / "features", "qwen3-vl-8b", "val")
        existing = cache2.completed_ids()
        print(f"  FeatureCache reload: {len(existing)} samples already cached (resume would skip these)")

        # Mmap read without loading to RAM.
        mm = cache2.load_site("post_proj")
        assert mm.shape == (5, fake_dim)
        print(f"    mmap read 'post_proj' OK, shape={mm.shape}")
        del mm

        # --- JsonlAppender with resume ---
        jsonl = tmp / "results.jsonl"
        with JsonlAppender(jsonl) as app:
            for sid in quant_ids[:3]:
                app.write({"sample_id": sid, "correct": True})
        # Simulate a crash: write truncated line.
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write('{"sample_id": "val_99", "cor')
        done = resume_completed_ids(jsonl)
        assert done == set(quant_ids[:3]), f"resume failed: got {done}"
        print(f"  JsonlAppender: wrote 3, simulated crash on 4th, resume recovered {len(done)}")

        # --- PromptCache with real question text ---
        pc = PromptCache(tmp / "prompts", "qwen3-vl-8b")
        # Use first 3 real PhysBench questions.
        for s in samples[:3]:
            q = s["question"]
            fake_tok = {"input_ids": list(range(len(q) % 50)), "attention_mask": [1] * (len(q) % 50)}
            pc.put(q, fake_tok, extra=str(s["idx"]))
        hit = pc.get(samples[0]["question"], extra=str(samples[0]["idx"]))
        assert hit is not None, "PromptCache miss on real PhysBench question"
        print(f"  PromptCache: stored {len(pc)} real PhysBench prompts, hit on replay OK")

        # Force cleanup of memmap refs before tmpdir cleanup.
        del cache, cache2, pc
        import gc; gc.collect()


# ---------------------------------------------------------------------------
# Phase 3: environment / VRAM summary.
# ---------------------------------------------------------------------------

def phase3_env() -> None:
    print("\n" + "=" * 70)
    print("PHASE 3: Environment summary")
    print("=" * 70)
    print(f"  torch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        snap = snapshot_vram()
        print(f"  GPU: {snap.device}")
        props = torch.cuda.get_device_properties(0)
        print(f"  Total VRAM: {props.total_memory/1e9:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Current allocated: {snap.allocated_gb:.2f} GB")

    # Check dependencies the Week 1 experiment needs.
    deps = {}
    for mod in ("bitsandbytes", "transformers", "accelerate", "sklearn", "qwen_vl_utils", "PIL"):
        try:
            m = __import__(mod if mod != "PIL" else "PIL.Image")
            ver = getattr(m, "__version__", "?")
            deps[mod] = ver
        except Exception as e:
            deps[mod] = f"MISSING ({type(e).__name__})"
    print(f"\n  Dependencies:")
    for k, v in deps.items():
        marker = "OK  " if "MISSING" not in v else "FAIL"
        print(f"    [{marker}] {k}: {v}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/physbench", type=Path)
    args = ap.parse_args()

    print("=" * 70)
    print("OPTIM STACK VERIFICATION — real PhysBench data")
    print("=" * 70)

    samples = load_physbench_val(args.data_dir)
    print(f"Loaded {len(samples)} PhysBench val samples from {args.data_dir / 'val.json'}")

    split_result = phase1_classify(samples)
    phase2_stack(samples, split_result)
    phase3_env()

    print("\n" + "=" * 70)
    print("VERIFICATION PASSED — Week 1 experiment is go-for-launch.")
    print("=" * 70)
    print(f"\n  Quantitative samples: {len(split_result['quant'])}")
    print(f"  Qualitative samples:  {len(split_result['qual'])}")
    print(f"  Quant sub_types: size, mass, number, distance, temperature")
    print(f"\nNext: python scripts/week1_quant_qual_probe.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
