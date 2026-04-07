#!/usr/bin/env python3
"""
Week 3 Phase 3b: Train the Physics Expert Module (PEM) on Qwen3-VL-8B.

The PEM is a lightweight ~2M param bypass module that:
  1. Extracts physics features from the low-variance PCA subspace (FIXED)
  2. Transforms them into LLM space via a learned MLP
  3. Injects them into the LLM input via a learned gate (init~0)
  4. The base VLM is 100% FROZEN

This should outperform SCAS (zero-shot +3.64pp) because:
  - The transform LEARNS a richer mapping than uniform amplification
  - The gate enables domain-conditional activation (physics ON, non-physics OFF)
  - The PCA extractor provides the same physics subspace but the MLP
    can learn nonlinear transformations the linear SCAS can't do

Usage:
    # Train PEM on Qwen3-VL-8B (laptop, ~30 min)
    python scripts/week3_pem_train.py --max-samples 500

    # Full training (Turing, ~2 hr)
    python scripts/week3_pem_train.py
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.optim.vram import build_bnb_config, snapshot_vram, format_vram_delta, hard_cleanup
from src.optim.compute import pick_attn_impl, inference_ctx
from src.optim.resilience import configure_traceback_logging
from src.optim.physbench_split import classify_quantitative
from src.optim.pem import PhysicsExpertModule
from src.optim.features import _resolve_module

from scripts.run_physbench_eval import (
    load_physbench_data, resolve_media_paths, format_question_for_vlm, extract_answer,
)


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


def load_model_frozen():
    """Load Qwen3-VL-8B fully frozen (no grad on any base param)."""
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    attn = pick_attn_impl()
    print(f"Loading {MODEL_ID} (frozen, {attn}, bnb-nf4)")
    t0 = time.time()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=build_bnb_config(load_in_4bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Freeze EVERYTHING.
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    print(f"  loaded + frozen in {time.time()-t0:.1f}s")
    return model, processor


def build_training_batch(processor, sample, data_dir):
    """Build a single training batch from a PhysBench sample."""
    from qwen_vl_utils import process_vision_info

    media_paths = resolve_media_paths(sample, str(data_dir))
    messages, has_media = format_question_for_vlm(sample, media_paths)
    if not has_media:
        raise RuntimeError("no media")
    answer = str(sample.get("answer", "")).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise RuntimeError(f"invalid answer: {answer}")

    # VRAM-adaptive: on laptop, convert videos to single-frame JPEGs
    # (qwen_vl_utils ignores nframes and always uses fps=24, generating
    # hundreds of frames per video — 30s/sample instead of 2s).
    full_res = os.environ.get("FULL_RESOLUTION", "0") == "1"
    for msg in messages:
        content = msg.get("content", [])
        new_content = []
        for part in content:
            t = part.get("type")
            if t == "image":
                if not full_res:
                    part["max_pixels"] = 256 * 256
                    part["min_pixels"] = 28 * 28
                new_content.append(part)
            elif t == "video":
                if full_res:
                    part["nframes"] = 4
                    new_content.append(part)
                else:
                    # Convert video to single-frame JPEG (bypass fps=24 default).
                    vid_path = part.get("video")
                    if vid_path and Path(vid_path).exists():
                        try:
                            import decord
                            vr = decord.VideoReader(vid_path, num_threads=1)
                            frame = vr[0].asnumpy()
                            from PIL import Image as _PILImg
                            pil = _PILImg.fromarray(frame)
                            import hashlib
                            h = hashlib.md5(vid_path.encode()).hexdigest()[:12]
                            tmp_dir = Path("cache/week2/video_frames")
                            tmp_dir.mkdir(parents=True, exist_ok=True)
                            fp = tmp_dir / f"{h}.jpg"
                            if not fp.exists():
                                pil.save(str(fp), quality=80)
                            new_content.append({
                                "type": "image", "image": str(fp),
                                "max_pixels": 256 * 256, "min_pixels": 28 * 28,
                            })
                        except Exception:
                            pass
            else:
                new_content.append(part)
        msg["content"] = new_content

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    full_text = prompt_text + answer
    image_inputs, video_inputs = process_vision_info(messages)

    full = processor(
        text=[full_text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    )
    prompt_ids = processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    labels = full["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full["labels"] = labels
    return full


def evaluate_pem(model, processor, pem, data_dir, logger, max_samples=None):
    """Evaluate model with PEM hook on PhysBench val."""
    from qwen_vl_utils import process_vision_info

    samples = load_physbench_data(str(data_dir), split="val", max_samples=max_samples)
    for s in samples:
        s.setdefault("sample_id", f"val_{s.get('idx', '?')}")

    # Register PEM hooks.
    enc_capture, enc_hook = pem.make_enc_capture_hook()
    pem_hook = pem.make_injection_hook(enc_capture)

    enc_module = _resolve_module(model, "model.visual.blocks.26")
    merger_module = _resolve_module(model, "model.visual.merger")
    h1 = enc_module.register_forward_hook(enc_hook)
    h2 = merger_module.register_forward_hook(pem_hook)

    pem.eval()
    device = next(model.parameters()).device
    correct = {"quantitative": 0, "qualitative": 0}
    total = {"quantitative": 0, "qualitative": 0}

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
            img, vid = process_vision_info(messages)
            inputs = processor(
                text=[prompt], images=img, videos=vid,
                padding=True, return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                out = model.generate(
                    **inputs, max_new_tokens=10, do_sample=False,
                    num_beams=1, use_cache=True,
                )
            resp = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            pred = extract_answer(resp) or ""
            gt = str(sample.get("answer", "")).strip().upper()
            sl = classify_quantitative(sample)
            total[sl] += 1
            if pred == gt and pred in {"A", "B", "C", "D"}:
                correct[sl] += 1
            del inputs, out
        except Exception:
            pass

        if (i + 1) % 50 == 0:
            logger.info(f"  [PEM eval] {i+1}/{len(samples)} "
                        f"q={correct['quantitative']}/{total['quantitative']} "
                        f"l={correct['qualitative']}/{total['qualitative']}")

    h1.remove()
    h2.remove()

    nq, nl = total["quantitative"], total["qualitative"]
    return {
        "acc_quant": correct["quantitative"] / nq if nq else None,
        "acc_qual": correct["qualitative"] / nl if nl else None,
        "acc_all": (correct["quantitative"] + correct["qualitative"]) / (nq + nl) if (nq + nl) else None,
        "n_quant": nq, "n_qual": nl,
        "correct_quant": correct["quantitative"],
        "correct_qual": correct["qualitative"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/physbench", type=Path)
    ap.add_argument("--cache-dir", default="cache/week1", type=Path)
    ap.add_argument("--training-dir", default="cache/week2/training_data", type=Path)
    ap.add_argument("--output-dir", default="results/week3", type=Path)
    ap.add_argument("--log-dir", default="logs", type=Path)
    ap.add_argument("--low-var-k", type=int, default=64)
    ap.add_argument("--hidden-dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--adapter-path", type=Path, default=None,
                    help="Path to a saved PEM checkpoint to evaluate")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_traceback_logging(args.log_dir, "week3_pem")

    logger.info("=" * 70)
    logger.info("Week 3: Physics Expert Module (PEM) Training")
    logger.info("=" * 70)

    # Create PEM from cached PCA basis.
    logger.info("Creating PEM from Week 1 PCA cache...")
    pem = PhysicsExpertModule.from_pca_cache(
        cache_dir=str(args.cache_dir),
        model_name="qwen3-vl-8b",
        low_var_k=args.low_var_k,
        llm_dim=4096,
        hidden_dim=args.hidden_dim,
    )
    n_params = sum(p.numel() for p in pem.parameters() if p.requires_grad)
    logger.info(f"PEM: {n_params:,} trainable params")

    # Load frozen base model.
    before = snapshot_vram()
    model, processor = load_model_frozen()
    logger.info(format_vram_delta(before, snapshot_vram()))

    # Move PEM to device.
    device = next(model.parameters()).device
    pem = pem.to(device)

    if args.eval_only:
        if args.adapter_path:
            logger.info(f"Loading PEM from {args.adapter_path}")
            state = torch.load(args.adapter_path / "pem_state.pt", map_location=device)
            pem.load_state_dict(state)
        result = evaluate_pem(model, processor, pem, args.data_dir, logger)
        logger.info(f"PEM eval: {result}")
        return 0

    # Load training data.
    train_path = args.training_dir / "lora_train.jsonl"
    if not train_path.exists():
        logger.error(f"Training data not found. Run: python scripts/week2_prepare_training_data.py")
        return 2
    with open(train_path) as f:
        train_samples = [json.loads(l) for l in f if l.strip()]
    if args.max_samples:
        train_samples = train_samples[:args.max_samples]
    logger.info(f"Training samples: {len(train_samples)}")

    # Register hooks: enc_capture feeds enc_out to PEM, PEM injects into merger output.
    enc_capture, enc_hook = pem.make_enc_capture_hook()
    pem_hook = pem.make_injection_hook(enc_capture)

    enc_module = _resolve_module(model, "model.visual.blocks.26")
    merger_module = _resolve_module(model, "model.visual.merger")
    h1 = enc_module.register_forward_hook(enc_hook)
    h2 = merger_module.register_forward_hook(pem_hook)

    # Optimizer — ONLY PEM params, not base model.
    optimizer = torch.optim.AdamW(pem.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = (len(train_samples) // 16) * args.epochs

    logger.info(f"Epochs: {args.epochs}, lr: {args.lr}, total_steps: ~{total_steps}")
    logger.info(f"Base model: FROZEN, only PEM trains")

    # Training loop.
    import random
    best_loss = float("inf")
    ckpt_dir = args.output_dir / "pem_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    accum_steps = 16
    t_start = time.time()

    for epoch in range(args.epochs):
        rng = random.Random(42 + epoch)
        rng.shuffle(train_samples)
        pem.train()
        epoch_losses = []
        accum = 0
        skipped = 0

        pbar = tqdm(enumerate(train_samples), total=len(train_samples),
                     desc=f"PEM Epoch {epoch+1}/{args.epochs}")

        for i, sample in pbar:
            try:
                batch = build_training_batch(processor, sample, args.data_dir)
            except Exception:
                skipped += 1
                continue

            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

            # Forward through frozen model + PEM hooks.
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / accum_steps

            # NaN guard: skip sample if loss is NaN (prevents poisoning the optimizer).
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                skipped += 1
                pbar.set_postfix_str(f"NaN loss, skipped", refresh=True)
                continue

            loss.backward()
            accum += 1
            epoch_losses.append(out.loss.item())

            if accum >= accum_steps:
                torch.nn.utils.clip_grad_norm_(pem.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0

                avg_loss = sum(epoch_losses[-50:]) / len(epoch_losses[-50:])
                # Check gate value.
                with torch.no_grad():
                    gate_val = pem.gate.linear.bias.item()
                    gate_sigmoid = torch.sigmoid(torch.tensor(gate_val)).item()

                pbar.set_postfix_str(
                    f"loss={out.loss.item():.3f} avg={avg_loss:.3f} "
                    f"gate_bias={gate_val:.2f} gate_open={gate_sigmoid:.3f} "
                    f"skip={skipped}"
                )

        pbar.close()

        avg_epoch = sum(epoch_losses) / max(len(epoch_losses), 1)
        logger.info(f"Epoch {epoch+1}: avg_loss={avg_epoch:.4f} "
                    f"samples={len(epoch_losses)} skipped={skipped} "
                    f"gate_sigmoid={gate_sigmoid:.4f}")

        if avg_epoch < best_loss:
            best_loss = avg_epoch
            torch.save(pem.state_dict(), ckpt_dir / "pem_state.pt")
            logger.info(f"  saved best checkpoint (loss={best_loss:.4f})")

    # Remove hooks.
    h1.remove()
    h2.remove()

    elapsed = time.time() - t_start
    logger.info(f"Training complete: {elapsed/60:.1f} min")

    # Evaluate with best PEM.
    logger.info("Loading best PEM checkpoint for evaluation...")
    pem.load_state_dict(torch.load(ckpt_dir / "pem_state.pt", map_location=device))

    result = evaluate_pem(model, processor, pem, args.data_dir, logger)
    logger.info(f"PEM eval: acc_quant={result['acc_quant']:.4f} "
                f"acc_qual={result['acc_qual']:.4f}")

    # Save results.
    out_path = args.output_dir / "pem_eval_qwen3-vl-8b.json"
    with open(out_path, "w") as f:
        json.dump({
            "model": "qwen3-vl-8b",
            "pem_params": n_params,
            "low_var_k": args.low_var_k,
            "epochs": args.epochs,
            "training_time_min": elapsed / 60,
            "best_train_loss": best_loss,
            "eval": result,
        }, f, indent=2)
    logger.info(f"Results: {out_path}")

    # Print comparison.
    print(f"\n{'='*60}")
    print(f" PEM RESULT vs BASELINES (Qwen3-VL-8B)")
    print(f"{'='*60}")
    print(f"  {'method':<25} {'acc_quant':>10} {'acc_qual':>10}")
    print(f"  {'-'*48}")
    print(f"  {'Baseline':<25} {'0.7091':>10} {'0.6207':>10}")
    print(f"  {'LoRA B (merger)':<25} {'0.6727':>10} {'0.5655':>10}")
    print(f"  {'LoRA C (LLM)':<25} {'0.7273':>10} {'0.6069':>10}")
    print(f"  {'SCAS amplify a=3':<25} {'0.7455':>10} {'0.6207':>10}")
    if result['acc_quant'] is not None:
        print(f"  {'PEM (this run)':<25} {result['acc_quant']:>10.4f} {result['acc_qual']:>10.4f}")
    print(f"{'='*60}")

    hard_cleanup(model, processor, pem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
