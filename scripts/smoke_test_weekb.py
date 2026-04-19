#!/usr/bin/env python3
"""
Pre-flight smoke test for Week B models (5070 Ti local verification).

Run this BEFORE submitting the Turing SLURM job. It verifies for each new
model:
  1. The model class imports successfully
  2. The model loads in 4-bit under ~16GB VRAM
  3. The processor loads
  4. The PROBE_CANDIDATES paths resolve to real modules
  5. A single-image forward pass completes without error
  6. Compression ratio is measurable

Runs per model in ~2-4 minutes on a 5070 Ti. Runs sequentially with hard
cleanup between models to free VRAM.

Usage:
    # Test all 4 Week B models:
    python scripts/smoke_test_weekb.py

    # Test a specific model:
    python scripts/smoke_test_weekb.py --model llava-onevision-7b

    # Quick smoke (model load + paths only, no forward pass):
    python scripts/smoke_test_weekb.py --no-forward
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.extract_training_features import (
    MODEL_REGISTRY, PROBE_CANDIDATES, load_model,
)
from scripts.discover_probe_sites import measure_compression, _extract_tensor
from src.optim.vram import snapshot_vram, hard_cleanup
from src.optim.features import _resolve_module


WEEK_B_MODELS = [
    "llava-onevision-7b",
    "phi3.5-vision",
    "pixtral-12b",
    "molmo-7b",
]


def smoke_test_one(model_key: str, do_forward: bool = True) -> Dict:
    """Run full smoke test on one model. Returns result dict."""
    result = {
        "model": model_key,
        "load_ok": False,
        "processor_ok": False,
        "paths_resolved": False,
        "forward_ok": False,
        "compression_ratio": None,
        "vram_gb": None,
        "errors": [],
        "working_candidate_idx": None,
    }

    print(f"\n{'='*70}")
    print(f"SMOKE TEST: {model_key}")
    print(f"{'='*70}")

    # --- Step 1: load model + processor ---
    t0 = time.time()
    try:
        model, processor, family = load_model(model_key)
        result["load_ok"] = True
        result["processor_ok"] = True
        result["vram_gb"] = snapshot_vram().allocated_gb
        print(f"  load_ok: {time.time()-t0:.1f}s, VRAM {result['vram_gb']:.1f}GB")
    except Exception as e:
        result["errors"].append(f"LOAD FAILED: {type(e).__name__}: {e}")
        print(f"  LOAD FAILED: {type(e).__name__}: {e}")
        traceback.print_exc()
        return result

    # --- Step 2: resolve PROBE_CANDIDATES ---
    print(f"  probing {len(PROBE_CANDIDATES.get(model_key, []))} candidate path sets...")
    paths_resolved = None
    for idx, paths in enumerate(PROBE_CANDIDATES.get(model_key, [])):
        try:
            for site_name, path in paths.items():
                _resolve_module(model, path)
            paths_resolved = paths
            result["paths_resolved"] = True
            result["working_candidate_idx"] = idx
            print(f"  paths_resolved: candidate {idx} works")
            for site, path in paths.items():
                print(f"    {site}: {path}")
            break
        except Exception as e:
            print(f"    candidate {idx} failed: {type(e).__name__}: {e}")
            continue

    if not result["paths_resolved"]:
        # Dump top-level structure for manual inspection
        print(f"  WARNING: no PROBE_CANDIDATES resolved for {model_key}")
        print(f"  Top-level module children (for manual PROBE_CANDIDATES fix):")
        try:
            for name, mod in list(model.named_children())[:20]:
                print(f"    .{name}: {type(mod).__name__}")
        except Exception:
            pass
        result["errors"].append("PROBE_CANDIDATES did not resolve — see top-level dump above")

    # --- Step 3: forward pass + compression measurement ---
    if do_forward and result["paths_resolved"]:
        print(f"  testing forward pass...")
        t0 = time.time()
        try:
            ratio = measure_compression(
                model, processor,
                paths_resolved["enc_out"],
                paths_resolved["post_proj"],
                family,
            )
            result["compression_ratio"] = ratio
            result["forward_ok"] = True
            print(f"  forward_ok: {time.time()-t0:.1f}s, compression_ratio={ratio}")
        except Exception as e:
            result["errors"].append(f"FORWARD FAILED: {type(e).__name__}: {e}")
            print(f"  FORWARD FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    # --- Cleanup ---
    try:
        hard_cleanup(model, processor)
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=None,
                    choices=WEEK_B_MODELS,
                    help="Test a specific model (default: all 4 Week B models)")
    ap.add_argument("--no-forward", action="store_true",
                    help="Skip forward-pass test (only verify load + paths)")
    args = ap.parse_args()

    models_to_test = [args.model] if args.model else WEEK_B_MODELS

    results = []
    for m in models_to_test:
        try:
            r = smoke_test_one(m, do_forward=not args.no_forward)
        except Exception as e:
            r = {"model": m, "errors": [f"UNCAUGHT: {type(e).__name__}: {e}"]}
            traceback.print_exc()
        results.append(r)

    # Summary
    print(f"\n{'='*70}")
    print(f"SMOKE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"  {'model':<22} {'load':>6} {'paths':>7} {'fwd':>6} {'C':>8} {'VRAM':>7}")
    print(f"  {'-'*62}")
    for r in results:
        load = "ok" if r.get("load_ok") else "FAIL"
        paths = "ok" if r.get("paths_resolved") else "FAIL"
        fwd = ("ok" if r.get("forward_ok")
               else ("n/a" if args.no_forward else "FAIL"))
        C = (f"{r['compression_ratio']:.1f}" if r.get("compression_ratio")
             else "n/a")
        vram = (f"{r['vram_gb']:.1f}GB" if r.get("vram_gb") else "n/a")
        print(f"  {r['model']:<22} {load:>6} {paths:>7} {fwd:>6} {C:>8} {vram:>7}")

    # Errors block
    any_failures = False
    for r in results:
        if r.get("errors"):
            any_failures = True
            print(f"\n  {r['model']} ERRORS:")
            for e in r["errors"]:
                print(f"    - {e}")

    print(f"\n{'='*70}")
    num_ok = sum(1 for r in results if r.get("forward_ok" if not args.no_forward else "paths_resolved"))
    print(f"  {num_ok}/{len(results)} models ready for Turing Week B run")
    print(f"{'='*70}")

    if num_ok == 0:
        print("\nABORT: no Week B models passed smoke test. Fix errors before submitting SLURM job.")
        return 2
    if num_ok < len(results):
        print(f"\nWARN: {len(results) - num_ok} model(s) failed. SLURM job will handle failures gracefully but those models will not contribute to n=8.")
        return 1
    print("\nOK: all Week B models ready. Submit Turing SLURM job.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
