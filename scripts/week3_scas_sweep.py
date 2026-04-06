#!/usr/bin/env python3
"""
Week 3 Phase 3a: SCAS alpha-sweep on PhysBench val across models.

Tests the Subspace-Contrast Activation Steering (SCAS) intervention at
multiple amplification factors (alpha) on each model. The core hypothesis:
amplifying the low-variance physics subspace at the merger output improves
quantitative physics accuracy while preserving qualitative accuracy.

Predictions from the mathematical proof (Theorem 3):
  - Δ_quant > 0 for positive alpha (physics signal amplified)
  - Δ_qual ≈ 0 (qualitative signal in V_high is untouched)
  - Δ_quant scales with sqrt(compression_ratio) across models
  - SCAS outperforms LoRA Condition B from Week 2 (-3.6pp on quant)

Usage:
    # Single model, default alpha sweep
    python scripts/week3_scas_sweep.py --model qwen3-vl-8b

    # Specific alphas
    python scripts/week3_scas_sweep.py --model qwen3-vl-8b --alphas 0 1 3 5 10

    # All models
    python scripts/week3_scas_sweep.py --all

    # Quick smoke test
    python scripts/week3_scas_sweep.py --model qwen3-vl-8b --alphas 0 3 --max-samples 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.optim.vram import (
    build_bnb_config, snapshot_vram, format_vram_delta, hard_cleanup,
)
from src.optim.compute import pick_attn_impl
from src.optim.resilience import configure_traceback_logging
from src.optim.physbench_split import classify_quantitative
from src.optim.steering import compute_steering_vector, make_steering_hook
from src.optim.features import FeatureCache

from scripts.run_physbench_eval import (
    load_physbench_data, resolve_media_paths, format_question_for_vlm, extract_answer,
)


# ---------------------------------------------------------------------------
# Model registry (same as week1_quant_qual_probe.py).
# ---------------------------------------------------------------------------

MODEL_LOADERS = {
    "qwen3-vl-8b": ("Qwen/Qwen3-VL-8B-Instruct", "qwen3"),
    "qwen2.5-vl-7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen25"),
    "internvl3-8b": ("OpenGVLab/InternVL3-8B-hf", "internvl3"),
    "gemma4-e4b": ("google/gemma-3-4b-it", "gemma"),
}

# Module path for the merger hook (where SCAS is injected).
MERGER_PATHS = {
    "qwen3-vl-8b": "model.visual.merger",
    "qwen2.5-vl-7b": "model.visual.merger",
    "internvl3-8b": "model.multi_modal_projector",
    "gemma4-e4b": "model.multi_modal_projector",
}


def load_model(model_key: str):
    """Load a VLM for inference (same logic as week1/week2 scripts)."""
    hf_id, family = MODEL_LOADERS[model_key]
    from transformers import AutoProcessor

    if family in ("qwen3", "qwen25"):
        try:
            if family == "qwen3":
                from transformers import Qwen3VLForConditionalGeneration as Cls
            else:
                from transformers import Qwen2_5_VLForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForVision2Seq as Cls
    elif family == "internvl3":
        try:
            from transformers import InternVLForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as Cls
    elif family == "gemma":
        try:
            from transformers import Gemma3ForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as Cls
    else:
        from transformers import AutoModelForVision2Seq as Cls

    attn = pick_attn_impl(allow_sdpa=True)
    print(f"Loading {hf_id} ({attn}, bnb-nf4, bf16)")
    t0 = time.time()
    model = Cls.from_pretrained(
        hf_id,
        quantization_config=build_bnb_config(load_in_4bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn if family != "gemma" else "eager",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    return model, processor


def resolve_merger_module(model, model_key: str):
    """Get the actual nn.Module for the merger/projector."""
    path = MERGER_PATHS[model_key]
    obj = model
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


# ---------------------------------------------------------------------------
# PhysBench eval with optional steering hook.
# ---------------------------------------------------------------------------

def evaluate_with_steering(
    model, processor, model_key: str, data_dir: Path,
    steering_info: Optional[Dict], alpha: float,
    logger, max_samples: Optional[int] = None,
) -> Dict:
    """Evaluate PhysBench val with optional SCAS steering.

    Returns:
        Dict with acc_all, acc_quant, acc_qual, n_total, n_quant, n_qual,
        correct_quant, correct_qual, per_sample results.
    """
    from qwen_vl_utils import process_vision_info

    samples = load_physbench_data(str(data_dir), split="val", max_samples=max_samples)
    for s in samples:
        s.setdefault("sample_id", f"val_{s.get('idx', '?')}")

    # Register steering hook if provided.
    handle = None
    if steering_info is not None and alpha != 0.0:
        merger_module = resolve_merger_module(model, model_key)
        hook_fn = make_steering_hook(steering_info, alpha=alpha)
        handle = merger_module.register_forward_hook(hook_fn)
        logger.info(f"  SCAS hook registered on {MERGER_PATHS[model_key]} with alpha={alpha}")

    device = next(model.parameters()).device
    correct = {"quantitative": 0, "qualitative": 0}
    total = {"quantitative": 0, "qualitative": 0}

    t0 = time.time()
    for i, sample in enumerate(samples):
        try:
            media_paths = resolve_media_paths(sample, str(data_dir))
            if not any(p for p in media_paths):
                continue
            messages, has_media = format_question_for_vlm(sample, media_paths)
            if not has_media:
                continue

            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[prompt], images=image_inputs, videos=video_inputs,
                padding=True, return_tensors="pt",
            ).to(device)

            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs, max_new_tokens=10, do_sample=False,
                    num_beams=1, use_cache=True,
                )
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
            pred = extract_answer(response) or ""
            gt = str(sample.get("answer", "")).strip().upper()

            slice_name = classify_quantitative(sample)
            total[slice_name] += 1
            if pred == gt and pred in {"A", "B", "C", "D"}:
                correct[slice_name] += 1

            del inputs, output_ids
        except Exception as e:
            logger.debug(f"  {sample.get('sample_id')}: {type(e).__name__}: {e}")

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(samples) - i - 1) / rate
            logger.info(
                f"  [alpha={alpha}] {i+1}/{len(samples)} "
                f"q={correct['quantitative']}/{total['quantitative']} "
                f"l={correct['qualitative']}/{total['qualitative']} "
                f"ETA {eta/60:.1f}min"
            )

    # Remove hook.
    if handle is not None:
        handle.remove()

    n_q = total["quantitative"]
    n_l = total["qualitative"]
    acc_q = correct["quantitative"] / n_q if n_q else None
    acc_l = correct["qualitative"] / n_l if n_l else None
    acc_all = (correct["quantitative"] + correct["qualitative"]) / (n_q + n_l) if (n_q + n_l) else None

    return {
        "alpha": alpha,
        "acc_all": acc_all,
        "acc_quant": acc_q,
        "acc_qual": acc_l,
        "n_quant": n_q,
        "n_qual": n_l,
        "correct_quant": correct["quantitative"],
        "correct_qual": correct["qualitative"],
    }


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b", choices=list(MODEL_LOADERS.keys()))
    ap.add_argument("--all", action="store_true", help="Run all models sequentially")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 1.0, 3.0, 5.0, 10.0],
                    help="Alpha values to sweep")
    ap.add_argument("--method", default="contrast", choices=["contrast", "amplify"])
    ap.add_argument("--low-var-k", type=int, default=64)
    ap.add_argument("--data-dir", default="data/physbench", type=Path)
    ap.add_argument("--cache-dir", default="cache/week1", type=Path)
    ap.add_argument("--output-dir", default="results/week3", type=Path)
    ap.add_argument("--log-dir", default="logs", type=Path)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    models = list(MODEL_LOADERS.keys()) if args.all else [args.model]

    for model_key in models:
        logger = configure_traceback_logging(args.log_dir, f"week3_scas_{model_key}")
        logger.info("=" * 70)
        logger.info(f"SCAS alpha-sweep: model={model_key}, alphas={args.alphas}")
        logger.info(f"  method={args.method}, low_var_k={args.low_var_k}")
        logger.info("=" * 70)

        # Check if cache exists for this model.
        cache = FeatureCache(args.cache_dir / "features", model_key, "val")
        if not cache.completed_ids():
            logger.warning(f"No cached features for {model_key}; skipping")
            continue

        # Compute steering vector (offline, from cache).
        logger.info("Computing steering vector from cached features...")
        t0 = time.time()
        sv_info = compute_steering_vector(
            cache_dir=str(args.cache_dir),
            model_name=model_key,
            method=args.method,
            low_var_k=args.low_var_k,
            site="post_proj",
        )
        logger.info(f"  method={sv_info['method']}, K={sv_info['low_var_k']}, "
                    f"dim={sv_info['feature_dim']}, "
                    f"V_low explains {sv_info['explained_variance_low']:.4f} of total variance")
        if "projection_retention" in sv_info:
            logger.info(f"  contrast projection retention: {sv_info['projection_retention']:.4f}")
        logger.info(f"  computed in {time.time()-t0:.1f}s")

        # Load model.
        before = snapshot_vram()
        model, processor = load_model(model_key)
        logger.info(format_vram_delta(before, snapshot_vram()))

        # Alpha sweep.
        sweep_results: List[Dict] = []
        for alpha in args.alphas:
            logger.info(f"\n--- alpha={alpha} ---")
            t_eval = time.time()
            result = evaluate_with_steering(
                model, processor, model_key, args.data_dir,
                steering_info=sv_info if alpha != 0 else None,
                alpha=alpha, logger=logger,
                max_samples=args.max_samples,
            )
            result["elapsed_s"] = time.time() - t_eval
            sweep_results.append(result)
            logger.info(
                f"  alpha={alpha}: acc_quant={result['acc_quant']:.4f} "
                f"acc_qual={result['acc_qual']:.4f} "
                f"acc_all={result['acc_all']:.4f}"
            )

        # Save results.
        out_path = args.output_dir / f"scas_sweep_{model_key}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_key,
                "method": args.method,
                "low_var_k": args.low_var_k,
                "steering_info": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in sv_info.items()
                    if k != "components"  # skip the full PCA matrix (large)
                },
                "sweep": sweep_results,
            }, f, indent=2)
        logger.info(f"Results written: {out_path}")

        # Print summary table.
        baseline = sweep_results[0] if sweep_results[0]["alpha"] == 0 else None
        print(f"\n{'='*70}")
        print(f" SCAS SWEEP: {model_key}")
        print(f"{'='*70}")
        print(f"  {'alpha':>8} {'acc_quant':>10} {'acc_qual':>10} {'d_quant':>10} {'d_qual':>10}")
        print(f"  {'-'*52}")
        for r in sweep_results:
            dq = (r["acc_quant"] - baseline["acc_quant"]) if baseline else None
            dl = (r["acc_qual"] - baseline["acc_qual"]) if baseline else None
            dq_s = f"{dq:+.4f}" if dq is not None else "n/a"
            dl_s = f"{dl:+.4f}" if dl is not None else "n/a"
            print(f"  {r['alpha']:>8.1f} {r['acc_quant']:>10.4f} {r['acc_qual']:>10.4f} "
                  f"{dq_s:>10} {dl_s:>10}")
        print(f"{'='*70}\n")

        # Cleanup before next model.
        hard_cleanup(model, processor)

    return 0


if __name__ == "__main__":
    sys.exit(main())
