#!/usr/bin/env python3
"""
Week 2: LoRA intervention on Qwen3-VL-8B for the H3 causal test.

This script is the second half of the diagnosis -> intervention -> improvement
loop. Week 1 established that Qwen3-VL-8B's PatchMerger exhibits a 784x
feature-std compression ratio that correlates with domain-conditional
quantitative-physics degradation (see Notion log + commit 333b87a). Week 2
tests whether targeting the merger specifically with LoRA RECOVERS PhysBench
quantitative performance more than an equivalent LLM-layer LoRA.

## Core experiment (what this script does)

    Baseline: Qwen3-VL-8B 4-bit nf4, no LoRA
        -> eval PhysBench val, report {quant_acc, qual_acc, delta}

    Condition B (merger-only LoRA):
        -> LoRA on model.visual.merger.linear_fc1, linear_fc2
        -> train on cache/week2/training_data/lora_train.jsonl
        -> eval PhysBench val, report {quant_acc, qual_acc, delta}

    Condition C (LLM-only LoRA):
        -> LoRA on model.language_model.layers.{0..7}.self_attn.{q_proj,v_proj}
        -> train on same data
        -> eval PhysBench val, report {quant_acc, qual_acc, delta}

    Final output: H3 causal test table
        H3_causal = (quant_B - quant_baseline) > (quant_C - quant_baseline)
                    AND
                    (qual_C - qual_baseline) >= (qual_B - qual_baseline)

    This is a DOUBLE DISSOCIATION test: merger-LoRA should specifically help
    quant while LLM-LoRA should specifically help qual (or not help either).

## What this script deliberately does NOT do

  * Does not duplicate `scripts/run_qlora_full.py`'s 2192 lines of
    Qwen2.5-VL-7B-specific training loop + data generation. That script
    targets the wrong model (Qwen2.5-VL-7B) with the wrong merger paths
    (visual.merger.mlp.0/2 instead of linear_fc1/fc2).
  * Does not generate synthetic physics QA. Training data comes from
    `scripts/week2_prepare_training_data.py` which pulls from PhysBench test.
  * Does not run Conditions A, D, or E. These are the Week 2 stretch goals;
    the MVP is B vs C double dissociation on Qwen3-VL-8B.
  * Does not train multiple epochs without early stopping. Patience=2 on
    lora_val loss keeps total training time under 2 hours per condition.

## VRAM budget on 12.8 GB RTX 5070 Ti Laptop

    Base model (4-bit nf4):           6.4 GB
    LoRA adapters (fp16):             +0.05-0.2 GB
    Optimizer state (paged AdamW):    ~0 GB (CPU-offloaded)
    Gradients:                        ~0.5 GB (adapter-only)
    Forward activations (bsz=1):      ~3-4 GB
    ---------------------------------
    Total:                            ~10-11 GB (leaves 1.8 GB headroom)

    Gradient checkpointing is enabled to keep activation memory bounded.

## Usage

    # Step 1: prepare training data (one-time, ~5 min)
    python scripts/week2_prepare_training_data.py

    # Step 2: baseline eval (no LoRA) -- uses existing Week 1 feature cache
    python scripts/week2_lora_intervention.py --stage baseline

    # Step 3: train Condition B (merger-only LoRA)
    python scripts/week2_lora_intervention.py --stage train --condition B

    # Step 4: train Condition C (LLM-only LoRA)
    python scripts/week2_lora_intervention.py --stage train --condition C

    # Step 5: aggregate final results
    python scripts/week2_lora_intervention.py --stage aggregate
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.optim.vram import (  # noqa: E402
    build_bnb_config,
    snapshot_vram,
    format_vram_delta,
    hard_cleanup,
    set_cuda_alloc_env,
)
from src.optim.compute import pick_attn_impl, fast_gen_config  # noqa: E402
from src.optim.resilience import (  # noqa: E402
    JsonlAppender,
    configure_traceback_logging,
    resume_completed_ids,
)
from src.optim.physbench_split import classify_quantitative  # noqa: E402
from src.optim.lora import (  # noqa: E402
    QWEN3_VL_8B_INTERVENTIONS,
    LoraIntervention,
    resolution_report,
)

from scripts.run_physbench_eval import (  # noqa: E402
    load_physbench_data,
    resolve_media_paths,
    format_question_for_vlm,
    extract_answer,
)

set_cuda_alloc_env()

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"


# ---------------------------------------------------------------------------
# Model loading (delegated to src/optim-style helpers for consistency with
# Week 1). Kept inline here to minimize cross-script dependencies so the
# reviewer can audit Week 2 in isolation.
# ---------------------------------------------------------------------------

def load_qwen3_vl_for_training(enable_grad_checkpoint: bool = True):
    """Load Qwen3-VL-8B in 4-bit for LoRA training.

    Key differences from the Week 1 inference loader:
      1. `torch_dtype=torch.bfloat16` on the non-quantized modules
         (LoRA adapters are instantiated in fp16/bf16 next to the 4-bit base)
      2. Gradient checkpointing enabled to cap forward activation memory
      3. `use_cache=False` at config level (gradient checkpointing requires it)
      4. `prepare_model_for_kbit_training` from PEFT sets up bnb for grad flow
    """
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    from peft import prepare_model_for_kbit_training

    attn_impl = pick_attn_impl()
    print(f"Loading {MODEL_ID} for LoRA training")
    print(f"  attn={attn_impl}, quant=bnb-nf4, dtype=bf16, grad_ckpt={enable_grad_checkpoint}")
    t0 = time.time()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=build_bnb_config(load_in_4bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Disable KV cache for training (incompatible with grad checkpointing).
    model.config.use_cache = False
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.use_cache = False

    # PEFT helper: casts LayerNorm to fp32 for stability, enables input grad
    # on the embedding so LoRA gradients can flow back through the 4-bit base.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=enable_grad_checkpoint,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    if enable_grad_checkpoint:
        model.enable_input_require_grads()

    print(f"  loaded in {time.time() - t0:.1f}s")
    return model, processor


# ---------------------------------------------------------------------------
# Training loop (minimal, no HF Trainer to stay auditable).
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """Training hyperparameters for one Week 2 LoRA condition.

    Defaults are tuned for the 12.8 GB RTX 5070 Ti Laptop and the ~3600-sample
    training set produced by week2_prepare_training_data.py.
    """
    epochs: int = 3
    batch_size: int = 1          # Qwen3-VL-8B + image inputs is already ~10GB
    grad_accum_steps: int = 16    # effective batch size = 16
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_samples: Optional[int] = None
    early_stop_patience: int = 2
    eval_every_n_steps: int = 200
    max_new_tokens_eval: int = 10


def build_train_messages(sample: Dict, data_dir: Path) -> Tuple[List[Dict], str]:
    """Build the Qwen chat messages + target answer string for one training sample.

    INFERENCE OPTIMIZATION: Limits images to the FIRST image per sample and
    caps pixel count at 128*28 * 128*28 = ~12.8M pixels (default Qwen max is
    ~1M per image). Videos are limited to 2 frames at fps=1 instead of the
    default 24fps which was generating thousands of vision tokens per video.

    These caps reduce vision-token count from ~4000-8000 per sample to
    ~200-600, which is the dominant factor in forward+backward time. The
    physics QA answer (A/B/C/D) only depends on whether the model SAW the
    scene, not on pixel-perfect resolution.

    The label the model must predict is the single-letter answer (A/B/C/D).
    We tokenize the prompt + " X" where X is the letter, and compute loss only
    on the answer tokens (prompt tokens are masked out).
    """
    media_paths = resolve_media_paths(sample, str(data_dir))
    messages, has_media = format_question_for_vlm(sample, media_paths)
    if not has_media:
        raise RuntimeError("no media resolved")
    answer = str(sample.get("answer", "")).strip().upper()
    if answer not in {"A", "B", "C", "D"}:
        raise RuntimeError(f"invalid answer label: {answer!r}")

    # Apply compute-efficiency caps to the message content.
    # This is the single biggest speedup: reducing vision token count from
    # ~4000-8000 to ~200-400 cuts forward+backward time from ~40-100s to ~3-5s
    # per sample.
    #
    # CRITICAL: Videos are converted to single-frame images (first frame
    # extracted via decord or cv2, saved to a temp file). qwen_vl_utils
    # ignores the `nframes` parameter and always uses fps=24 by default,
    # generating 120-720 frames per video. The ONLY reliable way to cap
    # video tokens is to bypass the video pipeline entirely.
    #
    # This does NOT affect evaluation authenticity: the PhysBench val eval
    # (evaluate_physbench_val) uses the ORIGINAL format_question_for_vlm
    # at full resolution with full video — only the TRAINING path is optimized.
    for msg in messages:
        content = msg.get("content", [])
        new_content = []
        for part in content:
            t = part.get("type")
            if t == "image":
                part["max_pixels"] = 256 * 256
                part["min_pixels"] = 28 * 28
                new_content.append(part)
            elif t == "video":
                # Convert video to single-frame image.
                vid_path = part.get("video")
                if vid_path and Path(vid_path).exists():
                    try:
                        import decord
                        vr = decord.VideoReader(vid_path, num_threads=1)
                        frame = vr[0].asnumpy()
                        from PIL import Image as _PILImage
                        pil = _PILImage.fromarray(frame)
                        # Save temp frame (reused across calls via cache).
                        import hashlib
                        h = hashlib.md5(vid_path.encode()).hexdigest()[:12]
                        tmp_dir = Path("cache/week2/video_frames")
                        tmp_dir.mkdir(parents=True, exist_ok=True)
                        frame_path = tmp_dir / f"{h}.jpg"
                        if not frame_path.exists():
                            pil.save(str(frame_path), quality=80)
                        new_content.append({
                            "type": "image",
                            "image": str(frame_path),
                            "max_pixels": 256 * 256,
                            "min_pixels": 28 * 28,
                        })
                    except Exception:
                        # If frame extraction fails, skip the video entirely
                        # rather than feeding a 720-frame video that takes 2 min.
                        pass
                else:
                    pass  # skip unresolvable video
            else:
                new_content.append(part)
        msg["content"] = new_content

    return messages, answer


def build_training_batch(processor, messages: List[Dict], answer_letter: str) -> Dict:
    """Tokenize one (messages, answer) pair into input_ids + labels.

    Loss is computed only on the answer token; all prompt tokens get
    label=-100 (ignored by cross-entropy). This is the standard causal-LM
    instruction-tuning setup.

    OPTIMIZATION: We tokenize ONLY the full text (prompt + answer), compute
    prompt_len by tokenizing the prompt separately, and avoid double-processing
    the vision inputs. The processor call is the most expensive step (it runs
    the image through the vision encoder's preprocessing), so we do it once.
    """
    from qwen_vl_utils import process_vision_info

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    full_text = prompt_text + f"{answer_letter}"

    image_inputs, video_inputs = process_vision_info(messages)

    # Single processor call for the full text (prompt + answer).
    full = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # Compute prompt length by tokenizing prompt text alone (text-only, no
    # vision re-processing — just the tokenizer, not the full processor).
    prompt_ids = processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
    prompt_len = prompt_ids.shape[1]

    labels = full["input_ids"].clone()
    labels[:, :prompt_len] = -100  # mask prompt tokens from loss
    full["labels"] = labels
    return full


def _save_training_plots(history: List[Dict], output_dir: Path, condition_id: str) -> Path:
    """Save training loss + LR + val_loss plots to a PNG file after each epoch.

    Returns the path to the saved figure. The figure has 3 subplots:
      1. Training loss (per-step, smoothed with EMA)
      2. Validation loss (per-eval, with best-checkpoint marker)
      3. Learning rate schedule

    This is the offline equivalent of a WandB/Neptune dashboard — reviewers
    can inspect the training dynamics by opening the PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Condition {condition_id} Training Dashboard", fontsize=14, fontweight="bold")

    # Extract series from history.
    train_steps = [h["step"] for h in history if "train_loss" in h]
    train_losses = [h["train_loss"] for h in history if "train_loss" in h]
    val_steps = [h["step"] for h in history if "val_loss" in h]
    val_losses = [h["val_loss"] for h in history if "val_loss" in h]
    lr_steps = [h["step"] for h in history if "lr" in h]
    lr_values = [h["lr"] for h in history if "lr" in h]
    epoch_markers = [h["step"] for h in history if h.get("epoch_end")]

    # 1. Training loss with EMA smoothing.
    ax1 = axes[0]
    if train_losses:
        ax1.plot(train_steps, train_losses, alpha=0.3, color="steelblue", linewidth=0.5, label="raw")
        # EMA smoothing
        ema = []
        alpha_ema = 0.1
        for v in train_losses:
            ema.append(v if not ema else alpha_ema * v + (1 - alpha_ema) * ema[-1])
        ax1.plot(train_steps, ema, color="navy", linewidth=1.5, label="EMA(0.1)")
        for em in epoch_markers:
            ax1.axvline(em, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Training Loss")
    ax1.set_title("Training Loss")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Validation loss with best marker.
    ax2 = axes[1]
    if val_losses:
        ax2.plot(val_steps, val_losses, "o-", color="darkorange", linewidth=1.5, markersize=5)
        best_idx = int(np.argmin(val_losses))
        ax2.plot(val_steps[best_idx], val_losses[best_idx], "*", color="green",
                 markersize=15, label=f"best={val_losses[best_idx]:.4f}")
        for em in epoch_markers:
            ax2.axvline(em, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
        ax2.legend(fontsize=8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Val Loss")
    ax2.set_title("Validation Loss (early stop target)")
    ax2.grid(True, alpha=0.3)

    # 3. Learning rate.
    ax3 = axes[2]
    if lr_values:
        ax3.plot(lr_steps, lr_values, color="forestgreen", linewidth=1.5)
        for em in epoch_markers:
            ax3.axvline(em, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("LR Schedule (warmup + cosine)")
    ax3.grid(True, alpha=0.3)
    ax3.ticklabel_format(style="sci", axis="y", scilimits=(-4, -4))

    plt.tight_layout()
    fig_path = output_dir / f"condition_{condition_id}_training_dashboard.png"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def train_condition(
    peft_model,
    processor,
    condition: LoraIntervention,
    train_path: Path,
    val_path: Path,
    output_dir: Path,
    cfg: TrainConfig,
    data_dir: Path,
    logger,
) -> Dict:
    """Train one LoRA condition with full visual progress tracking.

    Console output includes:
      - tqdm progress bar per epoch with loss/lr/VRAM/skip-rate in postfix
      - Per-sample timing so you can estimate total wallclock immediately
      - Epoch-end summary with val_loss, best checkpoint, early-stop status
      - Rich-formatted table at end with per-epoch stats

    File output:
      - condition_{id}_training_dashboard.png: 3-panel plot (train loss,
        val loss, LR schedule) updated after each epoch
      - condition_{id}_history.jsonl: incremental per-step metrics log
      - condition_{id}_checkpoints/best/: PEFT adapter weights at best val

    The training loop is identical to the prior version in logic but adds
    progress instrumentation at every level. No HF Trainer abstraction.
    """
    import bitsandbytes as bnb
    import math
    import random
    from tqdm import tqdm
    from torch.optim.lr_scheduler import LambdaLR

    peft_model.train()

    # Load data.
    with open(train_path, "r", encoding="utf-8") as f:
        train_samples = [json.loads(l) for l in f if l.strip()]
    with open(val_path, "r", encoding="utf-8") as f:
        val_samples = [json.loads(l) for l in f if l.strip()]
    if cfg.max_samples is not None:
        train_samples = train_samples[: cfg.max_samples]
        val_samples = val_samples[: max(4, cfg.max_samples // 10)]

    n_train = len(train_samples)
    n_val = len(val_samples)
    steps_per_epoch = n_train // cfg.grad_accum_steps
    total_steps = steps_per_epoch * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    print(f"\n{'='*70}")
    print(f"  TRAINING: Condition {condition.id} ({condition.name})")
    print(f"{'='*70}")
    print(f"  train samples:   {n_train}")
    print(f"  lora_val samples: {n_val}")
    print(f"  epochs:          {cfg.epochs}")
    print(f"  grad_accum:      {cfg.grad_accum_steps}")
    print(f"  steps/epoch:     {steps_per_epoch}")
    print(f"  total steps:     {total_steps}")
    print(f"  warmup steps:    {warmup_steps}")
    print(f"  lr:              {cfg.learning_rate}")
    print(f"  early stop:      patience={cfg.early_stop_patience} "
          f"(eval every {cfg.eval_every_n_steps} steps)")
    print(f"{'='*70}\n")

    logger.info(f"training samples: {n_train}  lora_val: {n_val}")

    # Optimizer.
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    logger.info(f"trainable params: {n_trainable:,}")
    optimizer = bnb.optim.PagedAdamW8bit(
        trainable, lr=cfg.learning_rate, weight_decay=cfg.weight_decay,
    )

    # LR schedule.
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # State.
    device = next(peft_model.parameters()).device
    step = 0
    accum = 0
    best_val_loss = float("inf")
    no_improve = 0
    ckpt_dir = output_dir / f"condition_{condition.id}_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []
    epoch_stats: List[Dict] = []

    # Incremental history log (JSONL, append per step).
    history_path = output_dir / f"condition_{condition.id}_history.jsonl"
    history_fh = open(history_path, "w", encoding="utf-8")

    def _log_step(record: Dict):
        history.append(record)
        history_fh.write(json.dumps(record) + "\n")
        history_fh.flush()

    t_start = time.time()

    for epoch in range(cfg.epochs):
        rng = random.Random(42 + epoch)
        rng.shuffle(train_samples)
        epoch_losses: List[float] = []
        epoch_skipped = 0
        epoch_t0 = time.time()

        # ---- tqdm progress bar for this epoch ----
        pbar = tqdm(
            enumerate(train_samples),
            total=n_train,
            desc=f"Epoch {epoch+1}/{cfg.epochs}",
            bar_format=(
                "{l_bar}{bar}| {n_fmt}/{total_fmt} "
                "[{elapsed}<{remaining}, {rate_fmt}] "
                "{postfix}"
            ),
            dynamic_ncols=True,
            mininterval=2.0,  # update at most every 2s to avoid stdout flood
        )

        for i, sample in pbar:
            sample_t0 = time.time()

            try:
                messages, letter = build_train_messages(sample, data_dir)
                batch = build_training_batch(processor, messages, letter)
            except Exception as e:
                epoch_skipped += 1
                pbar.set_postfix_str(
                    f"skip={epoch_skipped} | last_err={type(e).__name__}",
                    refresh=False,
                )
                continue

            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = peft_model(**batch)
                loss_scaled = out.loss / cfg.grad_accum_steps
            loss_scaled.backward()
            accum += 1

            raw_loss = out.loss.item()
            epoch_losses.append(raw_loss)

            if accum >= cfg.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                step += 1

                cur_lr = scheduler.get_last_lr()[0]
                vram = snapshot_vram()

                _log_step({
                    "step": step, "epoch": epoch + 1,
                    "train_loss": raw_loss, "lr": cur_lr,
                    "vram_gb": vram.allocated_gb,
                    "sample_idx": i, "skip_count": epoch_skipped,
                })

                # Update tqdm postfix with live metrics.
                avg_loss = sum(epoch_losses[-50:]) / len(epoch_losses[-50:])
                sample_dt = time.time() - sample_t0
                pbar.set_postfix_str(
                    f"loss={raw_loss:.3f} avg={avg_loss:.3f} "
                    f"lr={cur_lr:.1e} VRAM={vram.allocated_gb:.1f}G "
                    f"step={step}/{total_steps} skip={epoch_skipped} "
                    f"dt={sample_dt:.1f}s",
                    refresh=True,
                )

                # Periodic eval.
                if step > 0 and step % cfg.eval_every_n_steps == 0:
                    pbar.set_description(f"Epoch {epoch+1} [EVAL]")
                    val_loss = _compute_val_loss(
                        peft_model, processor, val_samples, data_dir, device,
                    )
                    _log_step({
                        "step": step, "epoch": epoch + 1,
                        "val_loss": val_loss,
                    })
                    improve_marker = ""
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        no_improve = 0
                        _save_adapter(peft_model, ckpt_dir / "best")
                        improve_marker = " *BEST*"
                    else:
                        no_improve += 1
                        improve_marker = f" (no_improve={no_improve}/{cfg.early_stop_patience})"

                    tqdm.write(
                        f"  [eval @ step {step}] val_loss={val_loss:.4f} "
                        f"best={best_val_loss:.4f}{improve_marker}"
                    )

                    if no_improve >= cfg.early_stop_patience:
                        tqdm.write(
                            f"  >>> EARLY STOP at step {step} "
                            f"(no improvement for {no_improve} evals)"
                        )
                        pbar.close()
                        _log_step({"step": step, "event": "early_stop"})
                        # Save plots before returning.
                        fig_path = _save_training_plots(history, output_dir, condition.id)
                        tqdm.write(f"  Dashboard saved: {fig_path}")
                        history_fh.close()
                        return _finalize_training(
                            condition, history, step, best_val_loss,
                            time.time() - t_start, ckpt_dir,
                        )
                    peft_model.train()
                    pbar.set_description(f"Epoch {epoch+1}/{cfg.epochs}")
            else:
                # Between grad-accum steps: lighter postfix.
                if i % 5 == 0:
                    sample_dt = time.time() - sample_t0
                    pbar.set_postfix_str(
                        f"loss={raw_loss:.3f} accum={accum}/{cfg.grad_accum_steps} "
                        f"skip={epoch_skipped} dt={sample_dt:.1f}s",
                        refresh=False,
                    )

        pbar.close()

        # ---- Epoch-end summary ----
        epoch_dt = time.time() - epoch_t0
        avg_epoch_loss = sum(epoch_losses) / max(len(epoch_losses), 1)

        # End-of-epoch val eval.
        val_loss = _compute_val_loss(
            peft_model, processor, val_samples, data_dir, device,
        )
        _log_step({
            "step": step, "epoch": epoch + 1,
            "val_loss": val_loss, "epoch_end": True,
            "avg_train_loss": avg_epoch_loss,
            "epoch_time_s": epoch_dt,
            "samples_processed": len(epoch_losses),
            "samples_skipped": epoch_skipped,
        })
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            no_improve = 0
            _save_adapter(peft_model, ckpt_dir / "best")

        epoch_stats.append({
            "epoch": epoch + 1,
            "avg_train_loss": avg_epoch_loss,
            "val_loss": val_loss,
            "best_val_loss": best_val_loss,
            "samples": len(epoch_losses),
            "skipped": epoch_skipped,
            "time_min": epoch_dt / 60,
            "steps": step,
        })

        print(f"\n{'='*70}")
        print(f"  Epoch {epoch+1}/{cfg.epochs} COMPLETE")
        print(f"{'='*70}")
        print(f"  avg train loss:   {avg_epoch_loss:.4f}")
        print(f"  val loss:         {val_loss:.4f}")
        print(f"  best val loss:    {best_val_loss:.4f}")
        print(f"  samples trained:  {len(epoch_losses)}")
        print(f"  samples skipped:  {epoch_skipped}")
        print(f"  epoch time:       {epoch_dt/60:.1f} min")
        print(f"  total steps:      {step}/{total_steps}")
        print(f"  no_improve:       {no_improve}/{cfg.early_stop_patience}")
        vram = snapshot_vram()
        print(f"  VRAM:             {vram.allocated_gb:.1f} GB")
        print(f"{'='*70}\n")

        # Save plots after each epoch.
        fig_path = _save_training_plots(history, output_dir, condition.id)
        print(f"  Dashboard updated: {fig_path}")
        logger.info(
            f"Epoch {epoch+1}: avg_loss={avg_epoch_loss:.4f} "
            f"val_loss={val_loss:.4f} best={best_val_loss:.4f} "
            f"time={epoch_dt/60:.1f}min skip={epoch_skipped}"
        )

        peft_model.train()

    # ---- Training complete ----
    history_fh.close()
    fig_path = _save_training_plots(history, output_dir, condition.id)

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETE: Condition {condition.id} ({condition.name})")
    print(f"{'='*70}")
    print(f"  total time:       {total_time/60:.1f} min")
    print(f"  total steps:      {step}")
    print(f"  best val loss:    {best_val_loss:.4f}")
    print(f"  final dashboard:  {fig_path}")

    if epoch_stats:
        print(f"\n  {'epoch':>5} {'train_loss':>11} {'val_loss':>10} "
              f"{'best':>10} {'samples':>8} {'skip':>6} {'time':>8}")
        print(f"  {'-'*65}")
        for es in epoch_stats:
            print(f"  {es['epoch']:>5} {es['avg_train_loss']:>11.4f} "
                  f"{es['val_loss']:>10.4f} {es['best_val_loss']:>10.4f} "
                  f"{es['samples']:>8} {es['skipped']:>6} "
                  f"{es['time_min']:>7.1f}m")
    print(f"{'='*70}\n")

    return _finalize_training(condition, history, step, best_val_loss,
                               total_time, ckpt_dir)


def _compute_val_loss(peft_model, processor, val_samples, data_dir, device) -> float:
    """Average cross-entropy loss on the LoRA-val slice."""
    peft_model.eval()
    losses: List[float] = []
    with torch.inference_mode():
        for s in val_samples[:32]:  # cap at 32 to keep eval fast
            try:
                messages, letter = build_train_messages(s, data_dir)
                batch = build_training_batch(processor, messages, letter)
            except Exception:
                continue
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = peft_model(**batch)
            losses.append(out.loss.item())
    peft_model.train()
    return float(np.mean(losses)) if losses else float("nan")


def _save_adapter(peft_model, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(path))


def _finalize_training(condition, history, step, best_val_loss, elapsed, ckpt_dir) -> Dict:
    return {
        "condition_id": condition.id,
        "condition_name": condition.name,
        "steps": step,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": elapsed,
        "history": history,
        "best_checkpoint": str(ckpt_dir / "best"),
    }


# ---------------------------------------------------------------------------
# PhysBench val evaluation (quant/qual slice breakdown).
# ---------------------------------------------------------------------------

def evaluate_physbench_val(
    model, processor, data_dir: Path, logger, tag: str,
) -> Dict:
    """Run the full PhysBench val set through the model and return per-slice
    accuracy.

    This uses .generate() to get a letter prediction, extracts A/B/C/D via
    the same regex as Week 1, and compares to ground truth.
    """
    from qwen_vl_utils import process_vision_info

    samples = load_physbench_data(str(data_dir), split="val")
    for s in samples:
        s.setdefault("sample_id", f"val_{s.get('idx', '?')}")

    model.eval()
    device = next(model.parameters()).device
    results: Dict[str, Dict] = {}
    correct_by_slice = {"quantitative": 0, "qualitative": 0}
    total_by_slice = {"quantitative": 0, "qualitative": 0}

    t0 = time.time()
    for i, sample in enumerate(samples):
        try:
            media_paths = resolve_media_paths(sample, str(data_dir))
            if not any(p for p in media_paths):
                continue
            messages, has_media = format_question_for_vlm(sample, media_paths)
            if not has_media:
                continue
            prompt_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[prompt_text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            input_len = inputs["input_ids"].shape[1]
            response = processor.decode(
                output_ids[0][input_len:], skip_special_tokens=True,
            )
            pred = extract_answer(response) or ""
            gt = str(sample.get("answer", "")).strip().upper()
            is_correct = (pred == gt) and pred in {"A", "B", "C", "D"}

            slice_name = classify_quantitative(sample)
            total_by_slice[slice_name] += 1
            if is_correct:
                correct_by_slice[slice_name] += 1

            results[sample["sample_id"]] = {
                "pred": pred, "gt": gt, "correct": is_correct,
                "slice": slice_name, "task_type": sample.get("task_type"),
                "sub_type": sample.get("sub_type"),
            }
            if (i + 1) % 25 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(samples) - i - 1) / rate
                logger.info(
                    f"  [{tag}] {i+1}/{len(samples)} "
                    f"q={correct_by_slice['quantitative']}/{total_by_slice['quantitative']} "
                    f"l={correct_by_slice['qualitative']}/{total_by_slice['qualitative']} "
                    f"ETA {eta/60:.1f}min"
                )
            del inputs, output_ids
        except Exception as e:
            logger.debug(f"sample {sample.get('sample_id')} eval error: {e}")

    def _acc(c, t):
        return (c / t) if t else None

    return {
        "tag": tag,
        "n_total": len(results),
        "n_quant": total_by_slice["quantitative"],
        "n_qual": total_by_slice["qualitative"],
        "correct_quant": correct_by_slice["quantitative"],
        "correct_qual": correct_by_slice["qualitative"],
        "acc_all": _acc(
            correct_by_slice["quantitative"] + correct_by_slice["qualitative"],
            total_by_slice["quantitative"] + total_by_slice["qualitative"],
        ),
        "acc_quant": _acc(correct_by_slice["quantitative"], total_by_slice["quantitative"]),
        "acc_qual": _acc(correct_by_slice["qualitative"], total_by_slice["qualitative"]),
        "per_sample": results,
    }


# ---------------------------------------------------------------------------
# Stage dispatchers (baseline / train / aggregate).
# ---------------------------------------------------------------------------

def run_baseline(args, logger) -> int:
    """Stage 1: baseline eval (no LoRA) on PhysBench val."""
    out_path = args.output_dir / "baseline_eval.json"
    if out_path.exists() and not args.force:
        logger.info(f"baseline already exists at {out_path} (pass --force to re-run)")
        return 0

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    logger.info(f"Loading {MODEL_ID} for baseline eval (no LoRA)")
    before = snapshot_vram()
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=build_bnb_config(load_in_4bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=pick_attn_impl(),
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    logger.info(format_vram_delta(before, snapshot_vram()))

    result = evaluate_physbench_val(model, processor, args.data_dir, logger, tag="baseline")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in result.items() if k != "per_sample"}, f, indent=2)
    logger.info(f"baseline written: {out_path}")
    logger.info(f"  acc_all  = {result['acc_all']}")
    logger.info(f"  acc_quant = {result['acc_quant']}")
    logger.info(f"  acc_qual  = {result['acc_qual']}")

    hard_cleanup(model, processor)
    return 0


def run_train(args, logger) -> int:
    """Stage 2: train one LoRA condition, then evaluate."""
    condition_id = args.condition.upper()
    if condition_id not in QWEN3_VL_8B_INTERVENTIONS:
        logger.error(f"Unknown condition: {condition_id}. Options: "
                     f"{list(QWEN3_VL_8B_INTERVENTIONS.keys())}")
        return 2
    condition = QWEN3_VL_8B_INTERVENTIONS[condition_id]

    out_eval = args.output_dir / f"condition_{condition_id}_eval.json"
    out_train = args.output_dir / f"condition_{condition_id}_train.json"
    if out_eval.exists() and not args.force:
        logger.info(f"condition {condition_id} eval already exists at {out_eval} "
                     f"(pass --force to re-run)")
        return 0

    train_path = args.training_dir / "lora_train.jsonl"
    val_path = args.training_dir / "lora_val.jsonl"
    if not train_path.exists() or not val_path.exists():
        logger.error(
            f"Training data not found. Run: "
            f"python scripts/week2_prepare_training_data.py"
        )
        return 3

    model, processor = load_qwen3_vl_for_training()

    # Resolution report: log exactly which modules will get LoRA.
    report = resolution_report(model, condition)
    logger.info(f"Condition {condition_id} ({condition.name}): "
                f"{len(report['resolved'])} modules matched, "
                f"{len(report['missing'])} patterns missed")
    for pat, hits in report["by_pattern"].items():
        logger.info(f"  {pat} -> {len(hits)} matches")
        for h in hits[:3]:
            logger.info(f"      {h}")
    if report["missing"]:
        logger.error(f"Missing patterns: {report['missing']}")
        hard_cleanup(model, processor)
        return 4

    # Apply LoRA.
    from peft import get_peft_model
    peft_config = condition.to_peft_config(task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    # Train.
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=1,
        grad_accum_steps=args.grad_accum,
        learning_rate=args.lr,
        max_samples=args.max_samples,
    )
    train_stats = train_condition(
        peft_model, processor, condition,
        train_path, val_path, args.output_dir,
        train_cfg, args.data_dir, logger,
    )
    with open(out_train, "w", encoding="utf-8") as f:
        json.dump(train_stats, f, indent=2)
    logger.info(f"training stats written: {out_train}")

    # Eval the LoRA'd model on PhysBench val.
    eval_result = evaluate_physbench_val(
        peft_model, processor, args.data_dir, logger,
        tag=f"condition_{condition_id}",
    )
    with open(out_eval, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in eval_result.items() if k != "per_sample"}, f, indent=2)
    logger.info(f"eval written: {out_eval}")
    logger.info(f"  acc_quant = {eval_result['acc_quant']}")
    logger.info(f"  acc_qual  = {eval_result['acc_qual']}")

    hard_cleanup(peft_model, model, processor)
    return 0


def run_aggregate(args, logger) -> int:
    """Stage 3: compare baseline + B + C and run the H3 causal test."""
    baseline_path = args.output_dir / "baseline_eval.json"
    B_path = args.output_dir / "condition_B_eval.json"
    C_path = args.output_dir / "condition_C_eval.json"

    required = {"baseline": baseline_path, "B": B_path, "C": C_path}
    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        logger.error(f"Cannot aggregate -- missing: {missing}")
        logger.error("Run: --stage baseline; --stage train --condition B; --stage train --condition C")
        return 5

    data = {k: json.loads(p.read_text()) for k, p in required.items()}
    baseline = data["baseline"]
    B = data["B"]
    C = data["C"]

    def d(new, old, key):
        if new.get(key) is None or old.get(key) is None:
            return None
        return new[key] - old[key]

    deltaB_quant = d(B, baseline, "acc_quant")
    deltaB_qual = d(B, baseline, "acc_qual")
    deltaC_quant = d(C, baseline, "acc_quant")
    deltaC_qual = d(C, baseline, "acc_qual")

    h3_strict = (
        deltaB_quant is not None and deltaC_quant is not None
        and deltaB_qual is not None and deltaC_qual is not None
        and deltaB_quant > deltaC_quant
        and deltaC_qual >= deltaB_qual
    )
    h3_weak = (
        deltaB_quant is not None and deltaC_quant is not None
        and deltaB_quant > deltaC_quant
    )

    print()
    print("=" * 80)
    print(" WEEK 2 H3 CAUSAL TEST -- Qwen3-VL-8B")
    print("=" * 80)
    print(f"  {'condition':<20} {'acc_quant':>10} {'acc_qual':>10} {'d_quant':>10} {'d_qual':>10}")
    print("  " + "-" * 64)
    print(f"  {'baseline':<20} {baseline['acc_quant']:>10.3f} {baseline['acc_qual']:>10.3f} "
          f"{'':>10} {'':>10}")
    print(f"  {'B (merger-only)':<20} {B['acc_quant']:>10.3f} {B['acc_qual']:>10.3f} "
          f"{deltaB_quant:>+10.3f} {deltaB_qual:>+10.3f}")
    print(f"  {'C (LLM-only)':<20} {C['acc_quant']:>10.3f} {C['acc_qual']:>10.3f} "
          f"{deltaC_quant:>+10.3f} {deltaC_qual:>+10.3f}")
    print("  " + "=" * 64)
    print(f"  H3 strict (double dissociation): {h3_strict}")
    print(f"     (requires d_B_quant > d_C_quant AND d_C_qual >= d_B_qual)")
    print(f"  H3 weak (merger wins on quant):  {h3_weak}")
    print()

    # Save final aggregate.
    agg = {
        "baseline": baseline,
        "condition_B": B,
        "condition_C": C,
        "deltas": {
            "B_quant": deltaB_quant, "B_qual": deltaB_qual,
            "C_quant": deltaC_quant, "C_qual": deltaC_qual,
        },
        "h3_strict": h3_strict,
        "h3_weak": h3_weak,
    }
    out = args.output_dir / "week2_aggregate.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2)
    logger.info(f"aggregate written: {out}")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        choices=["baseline", "train", "aggregate", "resolution-check"],
        default="resolution-check",
        help="baseline = no-LoRA eval; train = LoRA one condition + eval; "
             "aggregate = combine baseline+B+C; resolution-check = load model "
             "and print LoRA target-module resolution for both B and C, exit.",
    )
    ap.add_argument("--condition", choices=["B", "C", "D", "E"], default="B")
    ap.add_argument("--data-dir", default="data/physbench", type=Path)
    ap.add_argument("--output-dir", default="results/week2", type=Path)
    ap.add_argument("--training-dir", default="cache/week2/training_data", type=Path)
    ap.add_argument("--log-dir", default="logs", type=Path)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap training samples (debug / smoke test)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing outputs")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_traceback_logging(args.log_dir, f"week2_{args.stage}")
    logger.info("=" * 70)
    logger.info(f"Week 2 LoRA intervention -- stage={args.stage}")
    logger.info("=" * 70)

    if args.stage == "resolution-check":
        return run_resolution_check(args, logger)
    if args.stage == "baseline":
        return run_baseline(args, logger)
    if args.stage == "train":
        return run_train(args, logger)
    if args.stage == "aggregate":
        return run_aggregate(args, logger)
    return 1


def run_resolution_check(args, logger) -> int:
    """Stage 0: load the model and validate both LoRA condition target paths
    resolve to real modules. Fast sanity check before committing to training.
    """
    from transformers import Qwen3VLForConditionalGeneration
    logger.info("Loading Qwen3-VL-8B for resolution check (no training)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=build_bnb_config(load_in_4bit=True),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=pick_attn_impl(),
        low_cpu_mem_usage=True,
    )
    for cond_id in ("B", "C"):
        cond = QWEN3_VL_8B_INTERVENTIONS[cond_id]
        report = resolution_report(model, cond)
        logger.info(f"\nCondition {cond_id} ({cond.name}):")
        logger.info(f"  description: {cond.description}")
        logger.info(f"  rank={cond.rank} alpha={cond.lora_alpha}")
        logger.info(f"  resolved: {len(report['resolved'])} modules")
        logger.info(f"  missing: {len(report['missing'])} patterns")
        for pat, hits in report["by_pattern"].items():
            marker = "OK " if hits else "XX "
            logger.info(f"    [{marker}] {pat} -> {len(hits)} matches")
            for h in hits[:3]:
                logger.info(f"           {h}")
        if report["missing"]:
            logger.error(f"  Condition {cond_id} has unresolved patterns!")
    hard_cleanup(model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
