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


# Reuse model loaders from week1. Week B additions (2026-04-19): only models
# whose processors follow the standard HF API (processor(images=..., text=...,
# return_tensors="pt")) — avoids custom-processor integration work.
#
# Dropped from initial Week B plan (custom-processor complications):
#   - MiniCPM-V-2.6 (uses model.chat() with custom msgs= format)
#   - GLM-4.5V (MoE + custom processor)
#   - DeepSeek-VL2 (custom tokenization)
#
# Included (safe, standard HF API):
MODEL_REGISTRY = {
    "qwen3-vl-8b": ("Qwen/Qwen3-VL-8B-Instruct", "qwen3"),
    "qwen2.5-vl-7b": ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen25"),
    "internvl3-8b": ("OpenGVLab/InternVL3-8B-hf", "internvl3"),
    "gemma4-e4b": ("google/gemma-3-4b-it", "gemma"),
    # --- Week B additions (2026-04-19) ---
    "llava-onevision-7b": ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "llava_ov"),
    "phi3.5-vision":      ("microsoft/Phi-3.5-vision-instruct",         "phi35v"),
    "pixtral-12b":        ("mistral-community/pixtral-12b",              "pixtral"),
    "molmo-7b":           ("allenai/Molmo-7B-D-0924",                    "molmo"),
    # --- Pixtral replacement (2026-05-03) ---
    "idefics3-8b":        ("HuggingFaceM4/Idefics3-8B-Llama3",           "idefics3"),
}

# Probe site candidates. Multiple per model so discover_sites() can fall back
# when transformers-version module-tree shapes vary. Paths ending with a numeric
# index (e.g., ".26") are resolved via __getitem__ on ModuleList.
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
    # --- Week B additions (2026-04-19) ---
    # LLaVA-OneVision-7B (llava-hf/llava-onevision-qwen2-7b-ov-hf):
    # SigLIP (26 layers, indexed 0-25) + MLP projector + Qwen2-7B (28 layers).
    # CONFIRMED by Turing discovery 2026-05-02 on transformers 4.46.3:
    # vision_tower / multi_modal_projector / language_model are TOP-LEVEL.
    # language_model wraps Qwen2 → decoder layers at language_model.model.layers.N
    "llava-onevision-7b": [
        # PRIMARY (Turing-verified): top-level + inner .model.
        {"enc_out": "vision_tower.vision_model.encoder.layers.25",
         "post_proj": "multi_modal_projector",
         "llm_8": "language_model.model.layers.8",
         "llm_16": "language_model.model.layers.16"},
        # Fallback: flat (no inner .model.)
        {"enc_out": "vision_tower.vision_model.encoder.layers.25",
         "post_proj": "multi_modal_projector",
         "llm_8": "language_model.layers.8",
         "llm_16": "language_model.layers.16"},
        # Fallback: with `model.` prefix (very-old transformers)
        {"enc_out": "model.vision_tower.vision_model.encoder.layers.25",
         "post_proj": "model.multi_modal_projector",
         "llm_8": "model.language_model.model.layers.8",
         "llm_16": "model.language_model.model.layers.16"},
    ],
    # Phi-3.5-Vision: CLIP ViT-L (23 layers) + img_projection + Phi-3.5-mini (32 layers)
    # NOTE: identical paths to scripts/week1_quant_qual_probe.py for consistency.
    "phi3.5-vision": [
        {"enc_out":   "model.vision_embed_tokens.img_processor.vision_model.encoder.layers.23",
         "post_proj": "model.vision_embed_tokens.img_projection",
         "llm_8":     "model.layers.8",
         "llm_16":    "model.layers.16"},
        {"enc_out":   "vision_embed_tokens.img_processor.vision_model.encoder.layers.23",
         "post_proj": "vision_embed_tokens.img_projection",
         "llm_8":     "model.layers.8",
         "llm_16":    "model.layers.16"},
    ],
    # Pixtral-12B: CLIP-ViT (24 layers) + pixtral-style image_proj + Mistral-12B-Nemo (40 layers)
    # Same top-level layout as LLaVA-OV (LlavaForConditionalGeneration base).
    # PixtralVisionModel does NOT support SDPA — eager required (see eager_families below).
    "pixtral-12b": [
        # PRIMARY: top-level + inner .model. (analogous to LLaVA-OV verified)
        {"enc_out":   "vision_tower.transformer.layers.23",
         "post_proj": "multi_modal_projector",
         "llm_8":     "language_model.model.layers.8",
         "llm_16":    "language_model.model.layers.16"},
        # Fallback: flat (no inner .model.)
        {"enc_out":   "vision_tower.transformer.layers.23",
         "post_proj": "multi_modal_projector",
         "llm_8":     "language_model.layers.8",
         "llm_16":    "language_model.layers.16"},
        # Fallback: with `model.` prefix
        {"enc_out":   "model.vision_tower.transformer.layers.23",
         "post_proj": "model.multi_modal_projector",
         "llm_8":     "model.language_model.model.layers.8",
         "llm_16":    "model.language_model.model.layers.16"},
    ],
    # Idefics3-8B-Llama3 (Pixtral replacement, 2026-05-03):
    # SigLIP-SO400M (26 enc layers) + pixel-shuffle r=2 connector + Llama 3.1 8B (32 layers).
    # Pixel-shuffle r=2 → 4x token compression (676 → 169 per tile).
    # Mid-compression data point fills 2.4x → 114x gap in LOO regression.
    "idefics3-8b": [
        # PRIMARY: HF Idefics3ForConditionalGeneration layout
        {"enc_out":   "model.vision_model.encoder.layers.25",
         "post_proj": "model.connector",
         "llm_8":     "model.text_model.layers.8",
         "llm_16":    "model.text_model.layers.16"},
        # Fallback: top-level (no `model.` wrapper)
        {"enc_out":   "vision_model.encoder.layers.25",
         "post_proj": "connector",
         "llm_8":     "text_model.layers.8",
         "llm_16":    "text_model.layers.16"},
        # Fallback: nested connector path
        {"enc_out":   "model.vision_model.encoder.layers.25",
         "post_proj": "model.connector.modality_projection",
         "llm_8":     "model.text_model.layers.8",
         "llm_16":    "model.text_model.layers.16"},
    ],
    # Molmo-7B-D: custom vision adapter + Qwen2-7B. Paths are best-guess; run
    # scripts/discover_probe_sites.py first to verify or auto-detect.
    "molmo-7b": [
        {"enc_out":   "model.vision_backbone.image_vit.transformer.resblocks.22",
         "post_proj": "model.vision_backbone.image_projector",
         "llm_8":     "model.transformer.blocks.8",
         "llm_16":    "model.transformer.blocks.16"},
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
    # --- Week B families (2026-04-19) ---
    elif family == "llava_ov":
        try:
            from transformers import LlavaOnevisionForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as Cls
    elif family == "phi35v":
        # Phi-3.5-Vision: trust_remote_code + eager attn (model's custom
        # modeling file hardcodes flash-attn as default; forcing eager avoids
        # flash-attn dependency issues).
        from transformers import AutoModelForCausalLM as Cls
    elif family == "pixtral":
        try:
            from transformers import LlavaForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForImageTextToText as Cls
    elif family == "molmo":
        # Molmo uses custom modeling code in its HF repo
        from transformers import AutoModelForCausalLM as Cls
    elif family == "idefics3":
        # Idefics3-8B-Llama3 (Pixtral replacement, 2026-05-03)
        try:
            from transformers import Idefics3ForConditionalGeneration as Cls
        except ImportError:
            from transformers import AutoModelForVision2Seq as Cls
    else:
        from transformers import AutoModelForVision2Seq as Cls

    # Attention implementation: eager for families with custom attention OR
    # families whose vision tower lacks SDPA support.
    # - gemma/phi35v/molmo: custom attention modules
    # - pixtral: PixtralVisionModel doesn't implement SDPA in transformers 4.46.x
    eager_families = {"gemma", "phi35v", "molmo", "pixtral"}
    attn = pick_attn_impl(allow_sdpa=True)
    print(f"Loading {hf_id} ({attn}, bnb-nf4, bf16)")
    t0 = time.time()
    # Pre-patch config for Phi-3.5-Vision: its custom modeling file doesn't
    # forward attn_implementation to super().__init__, so the parent's FA2
    # dispatch check fires before our kwarg takes effect. Pre-setting on
    # config makes the parent's _check_and_adjust_attn_implementation read
    # "eager" rather than defaulting to FA2.
    pretrain_kwargs = dict(
        quantization_config=build_bnb_config(load_in_4bit=True),
        device_map="auto", torch_dtype=torch.bfloat16,
        attn_implementation=attn if family not in eager_families else "eager",
        trust_remote_code=True, low_cpu_mem_usage=True,
    )
    if family == "phi35v":
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        for attr in ("_attn_implementation", "_attn_implementation_internal",
                     "attn_implementation"):
            setattr(config, attr, "eager")
        pretrain_kwargs["config"] = config
    elif family == "pixtral":
        # Pixtral: LlavaModel.__init__'s inner AutoModel.from_config(vision_config)
        # call doesn't receive our outer attn_implementation kwarg, so we must
        # pre-patch the nested vision_config too. PixtralVisionModel raises
        # ValueError on SDPA in transformers 4.46.x (HF issue #28005).
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(hf_id, trust_remote_code=True)
        for cfg in (config, getattr(config, "vision_config", None)):
            if cfg is None:
                continue
            for attr in ("_attn_implementation", "_attn_implementation_internal",
                         "attn_implementation"):
                setattr(cfg, attr, "eager")
        pretrain_kwargs["config"] = config

    model = Cls.from_pretrained(hf_id, **pretrain_kwargs)
    # Per-family processor kwargs
    if family == "phi35v":
        processor = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=True, num_crops=4,
        )
    else:
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=True)
    # Pixtral / Llava family: tokenizer often ships without pad_token; set so
    # processor(..., padding=True) doesn't raise.
    try:
        tok = getattr(processor, "tokenizer", None)
        if tok is not None and getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
    except Exception:
        pass
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
                # Pixtral / variable-resolution models: projector may emit
                # list[Tensor] (one per image) when image_sizes vary. Coerce
                # to a single pooled vector.
                if isinstance(t, list):
                    if len(t) == 0:
                        raise RuntimeError(f"{site}: empty list captured")
                    pooled_per_img = [
                        x.mean(dim=tuple(range(x.ndim - 1))) if x.ndim >= 2 else x
                        for x in t
                    ]
                    t = torch.stack(pooled_per_img, dim=0).mean(dim=0)
                elif t.ndim == 0:
                    raise RuntimeError(f"{site}: scalar")
                elif t.ndim >= 2:
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
