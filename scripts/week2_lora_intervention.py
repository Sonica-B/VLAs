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
    return messages, answer


def build_training_batch(processor, messages: List[Dict], answer_letter: str) -> Dict:
    """Tokenize one (messages, answer) pair into input_ids + labels.

    Loss is computed only on the answer token; all prompt tokens get
    label=-100 (ignored by cross-entropy). This is the standard causal-LM
    instruction-tuning setup.
    """
    from qwen_vl_utils import process_vision_info

    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    # Append the target letter as the response the model must produce.
    full_text = prompt_text + f"{answer_letter}"

    image_inputs, video_inputs = process_vision_info(messages)

    # Tokenize prompt alone to find its length (for label masking).
    prompt_only = processor(
        text=[prompt_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    full = processor(
        text=[full_text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt_len = prompt_only["input_ids"].shape[1]

    labels = full["input_ids"].clone()
    labels[:, :prompt_len] = -100  # mask prompt tokens from loss
    full["labels"] = labels
    return full


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
    """Train one LoRA condition. Returns a dict of training stats.

    This is a minimal loop (no HF Trainer) so the reviewer can see exactly
    what happens on every step. Key properties:

      * Single-sample effective-batch via grad accumulation (memory-safe on 12 GB)
      * Paged AdamW 8-bit optimizer (optimizer state lives on CPU)
      * Cosine LR schedule with linear warmup (cfg.warmup_ratio of total steps)
      * LoRA-val loss computed every cfg.eval_every_n_steps
      * Early stopping on LoRA-val loss plateau (patience=cfg.early_stop_patience)
      * Checkpoint saved at every eval step; best_loss tracked for restore
    """
    import bitsandbytes as bnb
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

    logger.info(f"training samples: {len(train_samples)}  lora_val: {len(val_samples)}")

    # Optimizer: PagedAdamW8bit. Optimizer state lives in CPU RAM, gradients
    # live on GPU only transiently. ~0 GB VRAM overhead.
    trainable = [p for p in peft_model.parameters() if p.requires_grad]
    logger.info(f"trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer = bnb.optim.PagedAdamW8bit(
        trainable,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # LR schedule: linear warmup then cosine decay.
    total_steps = (len(train_samples) // cfg.grad_accum_steps) * cfg.epochs
    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        # Cosine decay from 1.0 to 0.1
        import math
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = LambdaLR(optimizer, lr_lambda)

    # Training loop.
    device = next(peft_model.parameters()).device
    step = 0
    accum = 0
    best_val_loss = float("inf")
    no_improve = 0
    ckpt_dir = output_dir / f"condition_{condition.id}_checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict] = []

    t_start = time.time()
    for epoch in range(cfg.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{cfg.epochs} ===")
        import random
        rng = random.Random(42 + epoch)
        rng.shuffle(train_samples)

        for i, sample in enumerate(train_samples):
            try:
                messages, letter = build_train_messages(sample, data_dir)
                batch = build_training_batch(processor, messages, letter)
            except Exception as e:
                logger.debug(f"sample {sample.get('sample_id')} skipped: {e}")
                continue

            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = peft_model(**batch)
                loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            accum += 1

            if accum >= cfg.grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum = 0
                step += 1

                if step % 10 == 0:
                    vram = snapshot_vram()
                    logger.info(
                        f"  step {step}/{total_steps} "
                        f"loss={out.loss.item():.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} "
                        f"VRAM {vram.allocated_gb:.1f}GB"
                    )

                if step > 0 and step % cfg.eval_every_n_steps == 0:
                    val_loss = _compute_val_loss(peft_model, processor, val_samples, data_dir, device)
                    history.append({"step": step, "val_loss": val_loss})
                    logger.info(f"  [eval] step={step} val_loss={val_loss:.4f}")
                    if val_loss < best_val_loss - 1e-4:
                        best_val_loss = val_loss
                        no_improve = 0
                        _save_adapter(peft_model, ckpt_dir / "best")
                    else:
                        no_improve += 1
                        if no_improve >= cfg.early_stop_patience:
                            logger.info(f"  early stop at step {step} "
                                         f"(no improvement for {no_improve} evals)")
                            return _finalize_training(
                                condition, history, step, best_val_loss,
                                time.time() - t_start, ckpt_dir,
                            )
                    peft_model.train()

    # Final eval at end of training.
    val_loss = _compute_val_loss(peft_model, processor, val_samples, data_dir, device)
    history.append({"step": step, "val_loss": val_loss, "final": True})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        _save_adapter(peft_model, ckpt_dir / "best")

    return _finalize_training(condition, history, step, best_val_loss,
                               time.time() - t_start, ckpt_dir)


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
