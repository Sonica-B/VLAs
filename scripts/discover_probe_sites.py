#!/usr/bin/env python3
"""
Auto-discover probe sites (enc_out, post_proj, llm_8, llm_16) for a new VLM.

Walks model.named_modules() and uses heuristics to identify:
  - enc_out: last vision-tower encoder block
  - post_proj: the projector/merger/resampler module
  - llm_8 / llm_16: LLM decoder layer 8 and 16

Prints suggested PROBE_CANDIDATES entries for the model registry, AND reports
the measured compression ratio C = enc_seq_len / post_proj_seq_len using a
single dummy image forward pass.

Usage (run once per new model before committing its PROBE_CANDIDATES):

    python scripts/discover_probe_sites.py --model llava-onevision-7b
    python scripts/discover_probe_sites.py --model phi3.5-vision
    python scripts/discover_probe_sites.py --model pixtral-12b
    python scripts/discover_probe_sites.py --model molmo-7b

Output format can be copy-pasted into PROBE_CANDIDATES in
scripts/extract_training_features.py.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from scripts.extract_training_features import MODEL_REGISTRY, load_model


# Heuristic patterns — ordered most-specific to least-specific.
# Each entry: (regex on module name, bucket).
DISCOVERY_RULES: List[Tuple[str, str]] = [
    # Vision encoder last block. Added `transformer.layers` for Pixtral and
    # `vision_backbone.image_vit.transformer.resblocks` for Molmo-style backbones.
    (r"(visual|vision_tower|vision|vpm)\.(blocks|encoder\.layers|encoder\.layer|vision_model\.encoder\.layers|trunk\.blocks|transformer\.layers|transformer\.resblocks|image_vit\.transformer\.resblocks)\.(\d+)$", "enc_block"),
    # Projector / merger / resampler. Added `image_projector` for Molmo.
    (r"(multi_modal_projector|merger|resampler|linear_proj|projector|image_projector|mlp_connector)$", "projector"),
    # LLM decoder layers
    (r"(language_model|llm|transformer|language)(?:\.model)?\.(layers|blocks)\.(\d+)$", "llm_layer"),
]


def walk_modules(model) -> Dict[str, List[Tuple[str, int]]]:
    """Return a dict of buckets: enc_block, projector, llm_layer → list of (name, idx)."""
    buckets = {"enc_block": [], "projector": [], "llm_layer": []}
    for name, _mod in model.named_modules():
        for pat, bucket in DISCOVERY_RULES:
            m = re.search(pat, name)
            if m:
                idx = None
                # extract trailing digit group if present
                digits = re.findall(r"\.(\d+)$", name)
                if digits:
                    idx = int(digits[-1])
                buckets[bucket].append((name, idx))
                break
    return buckets


def best_enc_block(buckets) -> Optional[str]:
    """Pick the last (highest-index) vision-encoder block."""
    if not buckets["enc_block"]:
        return None
    # Sort by idx descending
    sorted_blocks = sorted(
        buckets["enc_block"], key=lambda x: (x[1] if x[1] is not None else -1), reverse=True
    )
    return sorted_blocks[0][0]


def best_projector(buckets) -> Optional[str]:
    """Pick the shortest projector name (most-specific, outermost wrapper)."""
    if not buckets["projector"]:
        return None
    return min(buckets["projector"], key=lambda x: len(x[0]))[0]


def best_llm_layer(buckets, target_idx: int) -> Optional[str]:
    """Pick the LLM layer at target_idx (prefer shorter dotted-path)."""
    if not buckets["llm_layer"]:
        return None
    matches = [b for b in buckets["llm_layer"] if b[1] == target_idx]
    if not matches:
        return None
    return min(matches, key=lambda x: len(x[0]))[0]


def _extract_tensor(out) -> Optional[torch.Tensor]:
    """Extract the main tensor from whatever a module returns (tuple / ModelOutput / tensor).

    Mirrors register_probe_hooks's handling in src/optim/features.py so the
    discovery hook sees the same tensor the probe hook would see.
    """
    if isinstance(out, tuple):
        candidate = out[0]
        return candidate if isinstance(candidate, torch.Tensor) else None
    if hasattr(out, "last_hidden_state"):
        t = out.last_hidden_state
        return t if isinstance(t, torch.Tensor) else None
    if hasattr(out, "hidden_states") and out.hidden_states is not None:
        t = out.hidden_states[-1]
        return t if isinstance(t, torch.Tensor) else None
    if isinstance(out, torch.Tensor):
        return out
    return None


def measure_compression(model, processor, enc_path: str, post_path: str,
                         family: str) -> Optional[float]:
    """Run a dummy forward pass and measure enc_seq_len / post_proj_seq_len."""
    try:
        from src.optim.features import _resolve_module
    except Exception as e:
        print(f"  compression-measure skipped: {e}")
        return None

    # Register hooks on enc_out and post_proj to capture shapes.
    enc_out_shape: dict = {}
    post_proj_shape: dict = {}

    try:
        enc_mod = _resolve_module(model, enc_path)
        post_mod = _resolve_module(model, post_path)
    except Exception as e:
        print(f"  compression-measure: resolve failed: {type(e).__name__}: {e}")
        return None

    def _capture(store, tag):
        def hook(_m, _inp, out):
            t = _extract_tensor(out)
            if t is not None:
                try:
                    store[tag] = tuple(t.shape)
                except Exception:
                    pass
        return hook

    h1 = enc_mod.register_forward_hook(_capture(enc_out_shape, "shape"))
    h2 = post_mod.register_forward_hook(_capture(post_proj_shape, "shape"))

    try:
        from PIL import Image
        import numpy as np
        dummy = Image.fromarray(
            (np.random.rand(448, 448, 3) * 255).astype(np.uint8)
        )
        prompt = "Describe this image."
        # Family-specific input build (minimal)
        if family in ("qwen3", "qwen25"):
            # Qwen family requires qwen_vl_utils for vision processing
            try:
                from qwen_vl_utils import process_vision_info
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": dummy}, {"type": "text", "text": prompt}]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                img_inputs, vid_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text], images=img_inputs, videos=vid_inputs,
                    padding=True, return_tensors="pt",
                )
            except ImportError:
                # Fallback without qwen_vl_utils
                inputs = processor(text=[prompt], images=[dummy], return_tensors="pt", padding=True)
        else:
            # Universal PIL + text path — build chat template if possible
            try:
                chat = [{"role": "user", "content": [
                    {"type": "image", "image": dummy}, {"type": "text", "text": prompt}]}]
                text = processor.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                text = prompt
            inputs = processor(images=[dummy], text=text, return_tensors="pt", padding=True)
        # Move to model device. BatchFeature supports .items() and tensor values.
        try:
            inputs_on_dev = {
                k: (v.to(model.device) if torch.is_tensor(v) else v)
                for k, v in inputs.items()
            }
        except Exception:
            inputs_on_dev = inputs
        with torch.inference_mode():
            _ = model(**inputs_on_dev)
    except Exception as e:
        print(f"  compression-measure forward failed: {type(e).__name__}: {e}")
        return None
    finally:
        h1.remove()
        h2.remove()

    enc_shape = enc_out_shape.get("shape")
    post_shape = post_proj_shape.get("shape")
    if enc_shape is None or post_shape is None:
        print(f"  compression-measure: hook did not fire "
              f"(enc_shape={enc_shape}, post_shape={post_shape})")
        return None
    # seq-len axis is typically -2 (batch, seq, dim) for transformer blocks,
    # but for some projectors the output is [batch, dim] (ndim=2, no seq axis).
    try:
        if len(enc_shape) >= 3:
            enc_seq = enc_shape[-2]
        elif len(enc_shape) == 2:
            enc_seq = enc_shape[0]  # might be a flat per-token embedding
        else:
            return None
        if len(post_shape) >= 3:
            post_seq = post_shape[-2]
        elif len(post_shape) == 2:
            post_seq = post_shape[0]
        else:
            return None
        if post_seq == 0:
            return None
        return float(enc_seq) / float(post_seq)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--llm-layers", nargs=2, type=int, default=[8, 16],
                    help="LLM decoder layer indices to probe (default: 8 16)")
    ap.add_argument("--no-compression", action="store_true",
                    help="Skip compression-ratio measurement (no dummy forward pass)")
    args = ap.parse_args()

    print(f"Loading {args.model} to discover probe sites...")
    t0 = time.time()
    model, processor, family = load_model(args.model)
    print(f"  loaded in {time.time() - t0:.1f}s")

    buckets = walk_modules(model)
    print(f"\nDiscovered modules:")
    print(f"  encoder blocks: {len(buckets['enc_block'])} found")
    print(f"  projector/merger candidates: {[p[0] for p in buckets['projector']]}")
    print(f"  LLM layers: {len(buckets['llm_layer'])} found")

    enc = best_enc_block(buckets)
    proj = best_projector(buckets)
    llm8 = best_llm_layer(buckets, args.llm_layers[0])
    llm16 = best_llm_layer(buckets, args.llm_layers[1])

    print(f"\nSuggested PROBE_CANDIDATES entry for {args.model}:")
    print(f'    "{args.model}": [')
    print(f'        {{"enc_out": "{enc}",')
    print(f'         "post_proj": "{proj}",')
    print(f'         "llm_{args.llm_layers[0]}": "{llm8}",')
    print(f'         "llm_{args.llm_layers[1]}": "{llm16}"}},')
    print(f'    ],')

    if enc is None or proj is None or llm8 is None or llm16 is None:
        print("\nWARNING: One or more sites not discovered. You may need custom paths.")
        # Print a few adjacent module names for manual inspection
        print("\nTop-level modules (for manual inspection):")
        top_level = sorted(set(n.split(".")[0] for n, _ in model.named_modules() if n))
        for n in top_level[:20]:
            print(f"  {n}")
        return 1

    # Optional compression-ratio probe
    if not args.no_compression:
        print(f"\nMeasuring compression ratio (dummy 448x448 image)...")
        ratio = measure_compression(model, processor, enc, proj, family)
        if ratio is not None:
            print(f"  enc_seq / post_seq = {ratio:.1f}x")
            print(f"  (add to MODEL_COMPRESSION in scripts/phys_lens_predict.py)")
        else:
            print(f"  compression-ratio measurement failed; compute manually from architecture.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
