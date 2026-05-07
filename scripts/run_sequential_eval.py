#!/usr/bin/env python3
"""
Sequential multi-model PhysBench evaluation with aggressive memory management.
Designed for consumer GPUs (16GB VRAM) running 4-bit quantized models.

Evaluates 3 VLMs sequentially on PhysBench, fully unloading each model before
loading the next to stay within 16GB VRAM:
  - Qwen3-VL-8B   (~5-6 GB in 4-bit)
  - InternVL3-8B   (~5-6 GB in 4-bit)
  - Gemma 3 12B    (~6.6 GB in 4-bit)

GLM-4.5V (106B MoE) is skipped locally — requires A100 (48GB fp16).

Usage:
    # Run all 3 models sequentially on val set
    python scripts/run_sequential_eval.py --split val --output-dir results/multi_model

    # Run specific model only
    python scripts/run_sequential_eval.py --model qwen3-vl-8b --split val

    # Quick smoke test (5 samples)
    python scripts/run_sequential_eval.py --split val --max-samples 5

    # Run on test set (hours per model)
    python scripts/run_sequential_eval.py --split test --output-dir results/multi_model
"""

import argparse
import gc
import json
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

# Aggressive CUDA memory management
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_physbench_eval import (
    load_physbench_data,
    resolve_media_paths,
    extract_answer,
)

# ---------------------------------------------------------------------------
# Version checks
# ---------------------------------------------------------------------------

def check_dependencies():
    """Check that required packages are available and at correct versions."""
    import transformers
    tf_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if tf_version < (4, 49):
        print(f"WARNING: transformers {transformers.__version__} detected.")
        print("  Qwen3VLForConditionalGeneration requires transformers >= 4.49")
        print("  Install from source: pip install git+https://github.com/huggingface/transformers.git")
        print("  Or: pip install transformers>=4.49")

    try:
        import bitsandbytes  # noqa: F401
    except ImportError:
        print("ERROR: bitsandbytes is required for 4-bit quantization.")
        print("  Install: pip install bitsandbytes>=0.43.0")
        sys.exit(1)


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------

def get_gpu_memory_info() -> tuple[float, float]:
    """Return (free_gb, total_gb) for the current GPU."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9


def unload_model(model, processor):
    """Completely free GPU memory between model runs.

    Deletes model and processor, runs garbage collection, clears CUDA cache,
    and verifies that memory was actually freed.
    """
    model_device = None
    if hasattr(model, "device"):
        model_device = model.device

    # Move model to CPU first if possible (helps with some quantized models)
    try:
        if hasattr(model, "cpu"):
            model.cpu()
    except Exception:
        pass

    del model
    del processor

    # Aggressive garbage collection
    gc.collect()
    gc.collect()  # Second pass catches cyclic references

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Reset peak memory stats for next model
        torch.cuda.reset_peak_memory_stats()

    # Brief pause to let memory settle
    time.sleep(2)

    # Verify GPU is actually free
    if torch.cuda.is_available():
        free_mem, total_mem = get_gpu_memory_info()
        usage_pct = (total_mem - free_mem) / total_mem * 100 if total_mem > 0 else 0
        print(f"  GPU memory after unload: {free_mem:.1f} / {total_mem:.1f} GB free ({usage_pct:.0f}% used)")
        if free_mem < total_mem * 0.75:
            print(f"  WARNING: GPU not fully freed — {free_mem:.1f}GB free of {total_mem:.1f}GB")
            print(f"  Attempting forceful cleanup...")
            # Force another round
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(3)
            free_mem, total_mem = get_gpu_memory_info()
            print(f"  After second cleanup: {free_mem:.1f} / {total_mem:.1f} GB free")


def print_gpu_status(label: str = ""):
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        print(f"  {label}No CUDA GPU available")
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    free, total = get_gpu_memory_info()
    print(f"  {label}VRAM: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, "
          f"{free:.1f}/{total:.1f}GB free")


# ---------------------------------------------------------------------------
# 4-bit quantization config
# ---------------------------------------------------------------------------

def make_bnb_4bit_config():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_qwen3_vl(model_id: str) -> tuple:
    """Load Qwen3-VL-8B in 4-bit."""
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        print("  WARNING: Qwen3VLForConditionalGeneration not found.")
        print("  Requires transformers >= 4.49. Trying AutoModelForVision2Seq fallback...")
        from transformers import AutoModelForVision2Seq as Qwen3VLForConditionalGeneration

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=make_bnb_4bit_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, processor


def load_internvl3(model_id: str) -> tuple:
    """Load InternVL3-8B in 4-bit. Requires trust_remote_code=True."""
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        model_id,
        quantization_config=make_bnb_4bit_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def load_gemma3(model_id: str) -> tuple:
    """Load Gemma 3 12B in 4-bit."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=make_bnb_4bit_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Per-model inference
# ---------------------------------------------------------------------------

def infer_qwen3_vl(model, processor, messages: list, media_paths: list,
                    max_new_tokens: int = 32) -> str:
    """Inference for Qwen3-VL (uses qwen_vl_utils like Qwen2.5-VL)."""
    from qwen_vl_utils import process_vision_info

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    del inputs, output_ids
    torch.cuda.empty_cache()
    return response


def infer_internvl3(model, tokenizer, messages: list, media_paths: list,
                    max_new_tokens: int = 32) -> str:
    """Inference for InternVL3 — uses model.chat() API."""
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    images = []
    for p in media_paths:
        if p and os.path.exists(p) and any(p.lower().endswith(ext) for ext in
                                            [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
            images.append(Image.open(p).convert("RGB"))

    # Build text prompt — InternVL uses <image> tokens
    question_text = ""
    for msg in messages:
        if msg["role"] == "user":
            for part in msg.get("content", []):
                if isinstance(part, dict) and part.get("type") == "text":
                    question_text += part["text"]

    image_prefix = "".join("<image>\n" for _ in images)
    prompt = image_prefix + question_text

    if hasattr(model, "chat"):
        pixel_values = None
        if images:
            pixel_values = _internvl_process_images(images, model)
        generation_config = {"max_new_tokens": max_new_tokens, "do_sample": False}
        response = model.chat(tokenizer, pixel_values=pixel_values, question=prompt,
                              generation_config=generation_config)
        torch.cuda.empty_cache()
        return response

    # Fallback: manual tokenization
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    response = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    del inputs, output_ids
    torch.cuda.empty_cache()
    return response


def _internvl_process_images(images, model):
    """Process images for InternVL3 using its expected preprocessing."""
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = T.Compose([
        T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values.to(model.device, dtype=torch.bfloat16)


def infer_gemma3(model, processor, messages: list, media_paths: list,
                 max_new_tokens: int = 32) -> str:
    """Inference for Gemma 3 multimodal."""
    from PIL import Image

    images = []
    for p in media_paths:
        if p and os.path.exists(p) and any(p.lower().endswith(ext) for ext in
                                            [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
            images.append(Image.open(p).convert("RGB"))

    # Build Gemma3 chat format with actual PIL images
    gemma_messages = []
    for msg in messages:
        content_parts = []
        if msg["role"] == "user":
            img_idx = 0
            for part in msg.get("content", []):
                if isinstance(part, dict):
                    if part.get("type") == "image" and img_idx < len(images):
                        content_parts.append({"type": "image", "image": images[img_idx]})
                        img_idx += 1
                    elif part.get("type") == "text":
                        content_parts.append({"type": "text", "text": part["text"]})
        gemma_messages.append({
            "role": msg["role"],
            "content": content_parts or msg.get("content", ""),
        })

    inputs = processor.apply_chat_template(
        gemma_messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    del inputs, output_ids
    torch.cuda.empty_cache()
    return response


# ---------------------------------------------------------------------------
# Question formatting (per model architecture)
# ---------------------------------------------------------------------------

def format_question_qwen3(item: dict, media_paths: list) -> tuple:
    """Format for Qwen3-VL (same template as Qwen2.5-VL)."""
    question_text = item.get("question", "")
    content = []
    media_idx = 0
    parts = re.split(r"(<image>|<video>)", question_text)

    has_media = False
    for part in parts:
        if part == "<image>" and media_idx < len(media_paths):
            path = media_paths[media_idx]
            media_idx += 1
            if path and os.path.exists(path):
                content.append({
                    "type": "image", "image": path,
                    "max_pixels": 256 * 256, "min_pixels": 28 * 28,
                })
                has_media = True
            else:
                content.append({"type": "text", "text": "[image unavailable]"})
        elif part == "<video>" and media_idx < len(media_paths):
            path = media_paths[media_idx]
            media_idx += 1
            if path and os.path.exists(path):
                content.append({"type": "video", "video": path, "nframes": 4})
                has_media = True
            else:
                content.append({"type": "text", "text": "[video unavailable]"})
        elif part.strip():
            content.append({"type": "text", "text": part})

    content.append({
        "type": "text",
        "text": "\nAnswer with ONLY the letter (A, B, C, or D) of the correct option.",
    })
    messages = [{"role": "user", "content": content}]
    return messages, has_media


def format_question_generic(item: dict, media_paths: list) -> tuple:
    """Generic formatter for InternVL3, Gemma3 — builds content list."""
    question_text = item.get("question", "")
    content = []
    media_idx = 0
    parts = re.split(r"(<image>|<video>)", question_text)

    has_media = False
    for part in parts:
        if part == "<image>" and media_idx < len(media_paths):
            path = media_paths[media_idx]
            media_idx += 1
            if path and os.path.exists(path):
                content.append({"type": "image", "image": path})
                has_media = True
            else:
                content.append({"type": "text", "text": "[image unavailable]"})
        elif part == "<video>" and media_idx < len(media_paths):
            path = media_paths[media_idx]
            media_idx += 1
            content.append({"type": "text", "text": "[video input - not supported by this model]"})
        elif part.strip():
            content.append({"type": "text", "text": part})

    content.append({
        "type": "text",
        "text": "\nAnswer with ONLY the letter (A, B, C, or D) of the correct option.",
    })
    messages = [{"role": "user", "content": content}]
    return messages, has_media


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS_TO_RUN = [
    {
        "key": "qwen3-vl-8b",
        "name": "Qwen3-VL-8B",
        "id": "Qwen/Qwen3-VL-8B-Instruct",
        "load_fn": load_qwen3_vl,
        "infer_fn": infer_qwen3_vl,
        "format_fn": format_question_qwen3,
        "vram_4bit": 6,
    },
    {
        "key": "internvl3-8b",
        "name": "InternVL3-8B",
        "id": "OpenGVLab/InternVL3-8B",
        "load_fn": load_internvl3,
        "infer_fn": infer_internvl3,
        "format_fn": format_question_generic,
        "vram_4bit": 6,
    },
    {
        "key": "gemma3-12b",
        "name": "Gemma3-12B",
        "id": "google/gemma-3-12b-it",
        "load_fn": load_gemma3,
        "infer_fn": infer_gemma3,
        "format_fn": format_question_generic,
        "vram_4bit": 7,
    },
]

# Baseline results for comparison table (Qwen2.5-VL-7B from prior run)
BASELINE_RESULTS = {
    "model": "Qwen2.5-VL-7B",
    "overall_accuracy": 44.0,
    "by_task_type": {
        "property": {"accuracy": 51.1},
        "dynamics": {"accuracy": 41.1},
        "relationships": {"accuracy": 47.4},
        "scene": {"accuracy": 36.6},
    },
}

TASK_TYPE_ORDER = ["property", "dynamics", "relationships", "scene"]


# ---------------------------------------------------------------------------
# Evaluation loop for a single model
# ---------------------------------------------------------------------------

def evaluate_model(
    model_config: dict,
    model,
    processor,
    data: list,
    data_dir: str,
    output_dir: str,
) -> dict:
    """Run PhysBench evaluation for a single model and save results."""
    model_key = model_config["key"]
    infer_fn = model_config["infer_fn"]
    format_fn = model_config["format_fn"]
    model_output_dir = os.path.join(output_dir, f"{model_key}_4bit")
    os.makedirs(model_output_dir, exist_ok=True)

    results = []
    correct = 0
    total = 0
    skipped = 0
    errors = 0

    by_task_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_ability = defaultdict(lambda: {"correct": 0, "total": 0})
    by_mode = defaultdict(lambda: {"correct": 0, "total": 0})

    start_time = time.time()

    for i, item in enumerate(data):
        media_paths = resolve_media_paths(item, data_dir)
        messages, has_media = format_fn(item, media_paths)

        try:
            response = infer_fn(model, processor, messages, media_paths)
            predicted = extract_answer(response)
            gt_answer = item.get("answer", "").strip().upper()
            is_correct = predicted == gt_answer

            if is_correct:
                correct += 1
            total += 1

            task_type = item.get("task_type", "unknown")
            ability = item.get("ability_type", "unknown")
            mode = item.get("mode", "unknown")

            by_task_type[task_type]["total"] += 1
            by_ability[ability]["total"] += 1
            by_mode[mode]["total"] += 1
            if is_correct:
                by_task_type[task_type]["correct"] += 1
                by_ability[ability]["correct"] += 1
                by_mode[mode]["correct"] += 1

            results.append({
                "idx": item.get("idx", i),
                "gt_answer": gt_answer,
                "predicted": predicted,
                "raw_response": response[:200],
                "correct": is_correct,
                "task_type": task_type,
                "ability_type": ability,
                "mode": mode,
            })

            # Progress every 50 samples or at end
            if (i + 1) % 50 == 0 or (i + 1) == len(data):
                elapsed = time.time() - start_time
                acc = correct / total * 100 if total > 0 else 0
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(data) - i - 1) / rate if rate > 0 else 0
                print(f"    [{i+1}/{len(data)}] Acc: {acc:.1f}% ({correct}/{total}) | "
                      f"Err: {errors} | {rate:.2f} q/s | ETA: {eta/60:.1f}min")

        except Exception as e:
            errors += 1
            results.append({"idx": item.get("idx", i), "error": str(e)[:300]})
            if errors <= 5:
                print(f"    ERROR item {i}: {e}")
            elif errors == 6:
                print(f"    (suppressing further error messages)")

        # Checkpoint every 200 samples
        if (i + 1) % 200 == 0:
            _save_checkpoint(results, model_output_dir, model_key, i + 1)

    elapsed = time.time() - start_time
    overall_acc = correct / total * 100 if total > 0 else 0

    def _cat_acc(tracker):
        return {k: {"accuracy": round(v["correct"] / v["total"] * 100, 2) if v["total"] > 0 else 0,
                     "correct": v["correct"], "total": v["total"]}
                for k, v in sorted(tracker.items())}

    summary = {
        "model": model_key,
        "model_id": model_config["id"],
        "model_name": model_config["name"],
        "quantization": "4bit-nf4",
        "overall_accuracy": round(overall_acc, 2),
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "questions_per_second": round(total / elapsed, 2) if elapsed > 0 else 0,
        "by_task_type": _cat_acc(by_task_type),
        "by_ability_type": _cat_acc(by_ability),
        "by_mode": _cat_acc(by_mode),
    }

    # Save results
    results_path = os.path.join(model_output_dir, f"{model_key}_results.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "predictions": results}, f, indent=2)

    summary_path = os.path.join(model_output_dir, f"{model_key}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"    Results saved: {model_output_dir}/")
    return summary


def _save_checkpoint(results, output_dir, model_key, n_done):
    """Save intermediate checkpoint so progress isn't lost if script crashes."""
    checkpoint_path = os.path.join(output_dir, f"{model_key}_checkpoint_{n_done}.json")
    with open(checkpoint_path, "w") as f:
        json.dump(results, f)
    print(f"    [Checkpoint saved: {n_done} items]")


# ---------------------------------------------------------------------------
# Comparison table and figure generation
# ---------------------------------------------------------------------------

def generate_comparison_table(all_summaries: dict, output_dir: str):
    """Print and save a comparison table across all models."""
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}")

    header = f"| {'Model':<20} | {'Overall':>8} |"
    for tt in TASK_TYPE_ORDER:
        header += f" {tt.capitalize():>12} |"
    header += f" {'Time':>8} |"

    sep = "|" + "-" * 22 + "|" + "-" * 10 + "|"
    for _ in TASK_TYPE_ORDER:
        sep += "-" * 14 + "|"
    sep += "-" * 10 + "|"

    print(header)
    print(sep)

    # Baseline row
    row = f"| {'Qwen2.5-VL-7B*':<20} | {BASELINE_RESULTS['overall_accuracy']:>7.1f}% |"
    for tt in TASK_TYPE_ORDER:
        acc = BASELINE_RESULTS["by_task_type"].get(tt, {}).get("accuracy", 0)
        row += f" {acc:>11.1f}% |"
    row += f" {'baseline':>8} |"
    print(row)

    # Model rows
    for key, summary in all_summaries.items():
        if "overall_accuracy" not in summary:
            status = summary.get("status", "failed")
            row = f"| {key:<20} | {'--':>8} |"
            for _ in TASK_TYPE_ORDER:
                row += f" {'--':>12} |"
            row += f" {status:>8} |"
            print(row)
            continue

        row = f"| {summary.get('model_name', key):<20} | {summary['overall_accuracy']:>7.1f}% |"
        for tt in TASK_TYPE_ORDER:
            acc = summary.get("by_task_type", {}).get(tt, {}).get("accuracy", 0)
            row += f" {acc:>11.1f}% |"
        elapsed = summary.get("elapsed_seconds", 0)
        if elapsed > 3600:
            time_str = f"{elapsed/3600:.1f}h"
        elif elapsed > 60:
            time_str = f"{elapsed/60:.0f}min"
        else:
            time_str = f"{elapsed:.0f}s"
        row += f" {time_str:>8} |"
        print(row)

    print(f"\n* Qwen2.5-VL-7B baseline from prior full test-set run")

    # Note about GLM
    print(f"\nNote: GLM-4.5V (106B MoE) skipped — requires A100 (48GB fp16)")

    # Save comparison JSON
    comparison = {
        "baseline": BASELINE_RESULTS,
        "models": all_summaries,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, "comparison_table.json")
    with open(comparison_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nSaved: {comparison_path}")


def generate_comparison_figure(all_summaries: dict, output_dir: str):
    """Generate a grouped bar chart comparing models across physics domains."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  matplotlib not available — skipping figure generation")
        return

    # Collect data
    model_names = ["Qwen2.5-VL-7B\n(baseline)"]
    model_accs = {tt: [BASELINE_RESULTS["by_task_type"].get(tt, {}).get("accuracy", 0)]
                  for tt in TASK_TYPE_ORDER}
    overall_accs = [BASELINE_RESULTS["overall_accuracy"]]

    for key, summary in all_summaries.items():
        if "overall_accuracy" not in summary:
            continue
        model_names.append(summary.get("model_name", key))
        overall_accs.append(summary["overall_accuracy"])
        for tt in TASK_TYPE_ORDER:
            model_accs[tt].append(
                summary.get("by_task_type", {}).get(tt, {}).get("accuracy", 0)
            )

    if len(model_names) < 2:
        print("  Not enough models with results to generate comparison figure")
        return

    n_models = len(model_names)
    n_categories = len(TASK_TYPE_ORDER) + 1  # +1 for overall
    x = np.arange(n_categories)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [overall_accs[i]] + [model_accs[tt][i] for tt in TASK_TYPE_ORDER]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, edgecolor="white")
        # Add value labels on bars
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Multi-Model PhysBench Comparison (4-bit quantized)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Overall"] + [tt.capitalize() for tt in TASK_TYPE_ORDER])
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "comparison_chart.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fig_path} (300 DPI)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sequential multi-model PhysBench evaluation (16GB GPU optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_sequential_eval.py --split val
  python scripts/run_sequential_eval.py --model qwen3-vl-8b --split val
  python scripts/run_sequential_eval.py --split test --output-dir results/multi_model
  python scripts/run_sequential_eval.py --split val --max-samples 5
        """,
    )
    parser.add_argument("--model", default="all",
                        choices=["all"] + [m["key"] for m in MODELS_TO_RUN],
                        help="Which model to evaluate (default: all)")
    parser.add_argument("--split", default="val", choices=["test", "val"],
                        help="PhysBench split to evaluate (default: val)")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "physbench"),
                        help="Path to PhysBench data directory")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "multi_model"),
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    # Pre-flight checks
    check_dependencies()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This script requires a GPU.")
        sys.exit(1)

    free_mem, total_mem = get_gpu_memory_info()
    gpu_name = torch.cuda.get_device_name(0)

    # Select models
    if args.model == "all":
        models = MODELS_TO_RUN
    else:
        models = [m for m in MODELS_TO_RUN if m["key"] == args.model]

    # Load data once (shared across all models)
    data = load_physbench_data(args.data_dir, split=args.split, max_samples=args.max_samples)
    if not data:
        print("ERROR: No data loaded. Check --data-dir and --split.")
        sys.exit(1)

    # Header
    print(f"\n{'='*60}")
    print(f"MULTI-MODEL PHYSBENCH EVALUATION (Sequential, 4-bit)")
    print(f"{'='*60}")
    print(f"  GPU: {gpu_name} ({total_mem:.0f} GB)")
    print(f"  Free VRAM: {free_mem:.1f} GB")
    print(f"  Split: {args.split} ({len(data)} samples)")
    print(f"  Models: {', '.join(m['name'] for m in models)}")
    print(f"  Output: {args.output_dir}")
    print(f"  GLM-4.5V: SKIPPED (106B MoE, requires A100 48GB fp16)")
    print(f"{'='*60}")

    all_summaries = {}
    pipeline_start = time.time()

    for idx, model_config in enumerate(models):
        model_num = idx + 1
        model_total = len(models)
        model_key = model_config["key"]
        model_name = model_config["name"]

        print(f"\n{'='*60}")
        print(f"Model {model_num}/{model_total}: {model_name}")
        print(f"  HuggingFace: {model_config['id']}")
        print(f"  Expected VRAM (4-bit): ~{model_config['vram_4bit']} GB")
        print(f"{'='*60}")

        # Check VRAM
        free_mem, total_mem = get_gpu_memory_info()
        if free_mem < model_config["vram_4bit"]:
            print(f"  WARNING: Only {free_mem:.1f} GB free, need ~{model_config['vram_4bit']} GB")
            print(f"  Attempting to proceed anyway...")

        try:
            # Load
            print(f"  Loading {model_name} (4-bit NF4)...")
            load_start = time.time()
            model, processor = model_config["load_fn"](model_config["id"])
            load_time = time.time() - load_start

            vram_used = torch.cuda.memory_allocated() / 1e9
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Loaded in {load_time:.0f}s | {param_count/1e9:.1f}B params | {vram_used:.1f} GB VRAM")

            # Evaluate
            print(f"  Evaluating {args.split} set ({len(data)} samples)...")
            summary = evaluate_model(
                model_config=model_config,
                model=model,
                processor=processor,
                data=data,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
            )
            all_summaries[model_key] = summary

            # Print per-domain results
            print(f"\n  {model_name} accuracy: {summary['overall_accuracy']:.1f}%")
            print(f"  Per-domain:", end="")
            for tt in TASK_TYPE_ORDER:
                acc = summary.get("by_task_type", {}).get(tt, {}).get("accuracy", 0)
                print(f" {tt.capitalize()} {acc:.1f}%", end="")
            print()

            # Unload
            print(f"  Unloading {model_name}...")
            unload_model(model, processor)
            print(f"  Model unloaded. Moving to next...\n")

        except Exception as e:
            print(f"\n  *** FAILED: {model_name} ***")
            print(f"  Error: {e}")
            traceback.print_exc()
            all_summaries[model_key] = {
                "status": "failed",
                "error": str(e),
                "model_name": model_name,
            }

            # Try to clean up GPU even on failure
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            time.sleep(3)

            # Suggest fixes for common errors
            error_str = str(e).lower()
            if "qwen3vl" in error_str or "no module" in error_str:
                print(f"\n  FIX: pip install git+https://github.com/huggingface/transformers.git")
            elif "bitsandbytes" in error_str:
                print(f"\n  FIX: pip install bitsandbytes>=0.43.0")
            elif "out of memory" in error_str or "oom" in error_str:
                print(f"\n  FIX: Reduce --max-samples or close other GPU processes")
            elif "trust_remote_code" in error_str:
                print(f"\n  FIX: This model requires trust_remote_code=True (already set)")

            print(f"\n  Continuing with remaining models...\n")

    pipeline_elapsed = time.time() - pipeline_start

    # Summary
    print(f"\n{'='*60}")
    print(f"PIPELINE COMPLETE — {pipeline_elapsed/60:.1f} minutes total")
    print(f"{'='*60}")

    # Comparison table
    generate_comparison_table(all_summaries, args.output_dir)

    # Comparison figure
    generate_comparison_figure(all_summaries, args.output_dir)

    # Save combined results
    combined = {
        "pipeline": {
            "total_elapsed_seconds": round(pipeline_elapsed, 1),
            "split": args.split,
            "n_samples": len(data),
            "gpu": gpu_name,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "baseline": BASELINE_RESULTS,
        "models": all_summaries,
    }
    combined_path = os.path.join(args.output_dir, "sequential_eval_results.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined results: {combined_path}")


if __name__ == "__main__":
    main()
