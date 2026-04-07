#!/usr/bin/env python3
"""
Extract post_proj features from PhysBench TRAINING samples for leakage-free PCA.

WHY THIS EXISTS (critical for paper validity):
    The current SCAS steering vector is computed from PCA on VAL features,
    then EVALUATED on the same VAL features. This is train/test leakage
    that invalidates the +3.64pp result (a hostile reviewer would reject
    on this basis alone).

    Fix: Compute PCA on TRAINING features (1793 clean samples from
    PhysBench test split, disjoint from val). Then evaluate SCAS on
    val/test with the training-derived steering vector.

    The 1793 training samples are from `cache/week2/training_data/
    lora_train_clean.jsonl` — these were extracted from PhysBench test
    split by `week2_prepare_training_data.py` with an explicit
    `assert not (train_ids & val_ids)` guarantee of zero val overlap.

DATA FLOW:
    1. Load 1793 training samples from lora_train_clean.jsonl
    2. For each sample: load model, run forward pass, capture post_proj
       features via hook (same pipeline as Week 1)
    3. Save to cache/week1/features/{model}_train/ (FeatureCache)
    4. Later: compute_steering_vector(split="train") uses these features

Usage:
    # On local laptop (slow, ~30 min per model)
    python scripts/extract_training_features.py --model qwen3-vl-8b --max-samples 500

    # On Turing (fast, full 1793 samples)
    FULL_RESOLUTION=1 python scripts/extract_training_features.py --model qwen3-vl-8b
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.optim.vram import build_bnb_config, snapshot_vram, format_vram_delta, hard_cleanup
from src.optim.compute import pick_attn_impl, inference_ctx
from src.optim.features import FeatureCache, ProbeSites, register_probe_hooks, _resolve_module
from src.optim.resilience import configure_traceback_logging
from src.optim.physbench_split import classify_quantitative
from scripts.run_physbench_eval import resolve_media_paths, format_question_for_vlm


# Reuse model loaders from week1.
MODEL_REGISTRY = {
    "qwen3-vl-8b": ("Qwen/Qwen3-VL-8B-Instruct", "qwen3"),
    "qwen2.5-vl-7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen25"),
    "internvl3-8b": ("OpenGVLab/InternVL3-8B-hf", "internvl3"),
    "gemma4-e4b": ("google/gemma-3-4b-it", "gemma"),
}

# Probe site candidates (same as week1, verified on each model).
PROBE_CANDIDATES = {
    "qwen3-vl-8b": [
        {"enc_out": "model.visual.blocks.26", "post_proj": "model.visual.merger",
         "llm_8": "model.language_model.layers.8", "llm_16": "model.language_model.layers.16"},
    ],
    "qwen2.5-vl-7b": [
        {"enc_out": "model.visual.blocks.31", "post_proj": "model.visual.merger",
         "llm_8": "model.language_model.layers.8", "llm_16": "model.language_model.layers.16"},
    ],
    "internvl3-8b": [
        {"enc_out": "model.vision_tower.encoder.layer.23", "post_proj": "model.multi_modal_projector",
         "llm_8": "model.language_model.layers.8", "llm_16": "model.language_model.layers.16"},
    ],
    "gemma4-e4b": [
        {"enc_out": "model.vision_tower.vision_model.encoder.layers.26",
         "post_proj": "model.multi_modal_projector",
         "llm_8": "model.language_model.layers.8", "llm_16": "model.language_model.layers.16"},
    ],
}


def load_model(model_key: str):
    """Load a VLM in 4-bit for feature extraction."""
    hf_id, family = MODEL_REGISTRY[model_key]
    from transformers import AutoProcessor

    if family == "qwen3":
        from transformers import Qwen3VLForConditionalGeneration as Cls
    elif family == "qwen25":
        from transformers import Qwen2_5_VLForConditionalGeneration as Cls
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
        device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation=attn if family != "gemma" else "eager",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")
    return model, processor, family


def discover_sites(model, model_key: str) -> ProbeSites:
    """Try probe site candidates for this model."""
    candidates = PROBE_CANDIDATES.get(model_key, [])
    for paths in candidates:
        try:
            for _, path in paths.items():
                _resolve_module(model, path)
            return ProbeSites(model_name=model_key, paths=paths)
        except KeyError:
            continue
    raise RuntimeError(f"No probe sites resolved for {model_key}")


def build_inputs(processor, messages, family):
    """Build model inputs from Qwen-format messages."""
    if family in ("qwen3", "qwen25"):
        from qwen_vl_utils import process_vision_info
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img, vid = process_vision_info(messages)
        return processor(text=[text], images=img, videos=vid, padding=True, return_tensors="pt")
    else:
        # PIL path for InternVL/Gemma (same as week1).
        from PIL import Image
        pil_images = []
        text_parts = []
        for msg in messages:
            for part in msg.get("content", []):
                t = part.get("type")
                if t == "text":
                    text_parts.append(part.get("text", ""))
                elif t == "image":
                    p = part.get("image")
                    if p and Path(p).exists():
                        try:
                            img = Image.open(p).convert("RGB")
                            full_res = os.environ.get("FULL_RESOLUTION", "0") == "1"
                            if not full_res:
                                img = img.resize((448, 448))
                            pil_images.append(img)
                        except Exception:
                            pass
                elif t == "video":
                    vp = part.get("video")
                    if vp and Path(vp).exists():
                        try:
                            import decord
                            vr = decord.VideoReader(vp, num_threads=1)
                            frame = vr[0].asnumpy()
                            pil_images.append(Image.fromarray(frame).convert("RGB"))
                        except Exception:
                            pass
        if not pil_images:
            raise RuntimeError("no images")
        pil_images = pil_images[:1] if os.environ.get("FULL_RESOLUTION", "0") != "1" else pil_images[:5]
        prompt = "\n".join(p for p in text_parts if p.strip())
        chat = [{"role": "user", "content": [{"type": "image", "image": img} for img in pil_images] + [{"type": "text", "text": prompt}]}]
        try:
            prompt_text = processor.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt_text = prompt
        return processor(images=pil_images, text=prompt_text, return_tensors="pt", padding=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b", choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--data-dir", default="data/physbench", type=Path)
    ap.add_argument("--training-data", default="cache/week2/training_data/lora_train_clean.jsonl", type=Path)
    ap.add_argument("--cache-dir", default="cache/week1", type=Path)
    ap.add_argument("--log-dir", default="logs", type=Path)
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    logger = configure_traceback_logging(args.log_dir, f"extract_train_{args.model}")
    logger.info("=" * 70)
    logger.info(f"Extracting TRAINING features for leakage-free PCA [{args.model}]")
    logger.info("=" * 70)

    # Load training samples.
    if not args.training_data.exists():
        logger.error(f"Training data not found: {args.training_data}")
        logger.error("Run: python scripts/week2_prepare_training_data.py")
        return 2
    with open(args.training_data) as f:
        train_samples = [json.loads(l) for l in f if l.strip()]
    if args.max_samples:
        train_samples = train_samples[:args.max_samples]
    logger.info(f"Training samples: {len(train_samples)}")

    # Verify disjointness from val.
    val_samples = json.loads((args.data_dir / "val.json").read_text())
    val_ids = {f"val_{s.get('idx', '?')}" for s in val_samples}
    train_ids = {s.get("sample_id", "") for s in train_samples}
    overlap = val_ids & train_ids
    assert len(overlap) == 0, f"LEAKAGE: {len(overlap)} val IDs in training data!"
    logger.info(f"Val/train overlap check: 0 (verified)")

    # Load model.
    before = snapshot_vram()
    model, processor, family = load_model(args.model)
    logger.info(format_vram_delta(before, snapshot_vram()))

    # Discover probe sites.
    sites = discover_sites(model, args.model)
    logger.info(f"Probe sites: {sites.paths}")

    # Register hooks.
    captured, handles = register_probe_hooks(model, sites)

    # Feature cache (training split).
    cache = FeatureCache(args.cache_dir / "features", args.model, "train")
    cached_ids = cache.completed_ids()
    logger.info(f"Already cached: {len(cached_ids)} training samples")

    device = next(model.parameters()).device
    processed = 0
    skipped = 0
    errors = 0
    t_start = time.time()

    for i, sample in enumerate(train_samples):
        sid = sample.get("sample_id", f"train_{i}")
        if sid in cache.completed_ids():
            skipped += 1
            continue

        try:
            media_paths = resolve_media_paths(sample, str(args.data_dir))
            if not any(p for p in media_paths):
                errors += 1
                continue
            messages, has_media = format_question_for_vlm(sample, media_paths)
            if not has_media:
                errors += 1
                continue

            inputs = build_inputs(processor, messages, family)
            inputs_dev = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}
            captured.clear()
            with inference_ctx():
                _ = model(**inputs_dev)
            del inputs_dev

            # Pool and save features.
            np_feats = {}
            for site in sites.paths:
                if site not in captured:
                    raise RuntimeError(f"hook {site} did not fire")
                t = captured[site]
                if t.ndim == 0:
                    raise RuntimeError(f"{site}: scalar")
                if t.ndim >= 2:
                    reduce_dims = tuple(range(t.ndim - 1))
                    t = t.mean(dim=reduce_dims)
                np_feats[site] = t.unsqueeze(0).numpy().astype(np.float32)

            cache.append_batch([sid], np_feats)
            processed += 1

            if processed % 50 == 0 or processed == 1:
                elapsed = time.time() - t_start
                rate = processed / max(elapsed, 1e-6)
                remaining = (len(train_samples) - i - 1) / max(rate, 1e-6)
                vram = snapshot_vram()
                logger.info(
                    f"  [{i+1}/{len(train_samples)}] ok={processed} skip={skipped} "
                    f"err={errors} | {rate:.2f} q/s | ETA {remaining/60:.1f}min "
                    f"| VRAM {vram.allocated_gb:.1f}GB"
                )

        except Exception as e:
            errors += 1
            if errors <= 5:
                logger.debug(f"  {sid}: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    for h in handles:
        h.remove()

    elapsed = time.time() - t_start
    logger.info(f"Complete: processed={processed} skipped={skipped} errors={errors} "
                f"in {elapsed/60:.1f} min")

    # Verify cache.
    final_cache = FeatureCache(args.cache_dir / "features", args.model, "train")
    n_cached = len(final_cache.completed_ids())
    logger.info(f"Cache: {n_cached} training samples in {args.cache_dir}/features/{args.model}_train/")

    for site in sites.paths:
        arr = final_cache.load_site(site)
        logger.info(f"  {site}: shape={arr.shape} mean={arr.mean():.4f} std={arr.std():.4f}")

    hard_cleanup(model, processor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
