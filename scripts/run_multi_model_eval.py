#!/usr/bin/env python3
"""
Multi-model PhysBench evaluation for NeurIPS 2026 paper.
Evaluates 4 SOTA VLMs on physics understanding benchmark.

Supported models:
  - Qwen3-VL-8B   (Qwen/Qwen3-VL-8B-Instruct)
  - InternVL3-8B   (OpenGVLab/InternVL3-8B)
  - Gemma 3 12B    (google/gemma-3-12b-it)
  - GLM-4.5V       (zai-org/GLM-4.5V)

Usage:
    # Single model
    python scripts/run_multi_model_eval.py --model qwen3-vl-8b --split val

    # All models sequentially
    python scripts/run_multi_model_eval.py --model all --split test

    # Quick test
    python scripts/run_multi_model_eval.py --model qwen3-vl-8b --split val --max-samples 20
"""

import argparse
import gc
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

# ---------------------------------------------------------------------------
# Project imports (reuse data loading / answer extraction from existing eval)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_physbench_eval import (
    load_physbench_data,
    resolve_media_paths,
    extract_answer,
)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "qwen3-vl-8b": {
        "id": "Qwen/Qwen3-VL-8B-Instruct",
        "class": "Qwen3VLForConditionalGeneration",
        "processor": "AutoProcessor",
        "trust_remote_code": False,
        "vram_fp16": 16,
    },
    "internvl3-8b": {
        "id": "OpenGVLab/InternVL3-8B",
        "class": "AutoModel",
        "processor": "AutoTokenizer",
        "trust_remote_code": True,
        "vram_fp16": 16,
    },
    "gemma3-12b": {
        "id": "google/gemma-3-12b-it",
        "class": "AutoModelForImageTextToText",
        "processor": "AutoProcessor",
        "trust_remote_code": False,
        "vram_fp16": 24,
    },
    "glm-4.5v": {
        "id": "zai-org/GLM-4.5V",
        "class": "AutoModelForCausalLM",
        "processor": "AutoProcessor",
        "trust_remote_code": True,
        "vram_fp16": 48,
    },
}


def get_available_vram() -> float:
    """Return available GPU VRAM in GB."""
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    return props.total_mem / 1e9


def can_run_model(model_key: str, quantize: str) -> bool:
    """Check if the current GPU can run the given model."""
    vram = get_available_vram()
    info = MODELS[model_key]
    needed_fp16 = info["vram_fp16"]

    if quantize == "4bit":
        # ~40% of fp16 VRAM
        return vram >= needed_fp16 * 0.4
    elif quantize == "8bit":
        return vram >= needed_fp16 * 0.6
    else:
        return vram >= needed_fp16


def choose_quantization(model_key: str) -> str:
    """Auto-select quantization based on available VRAM."""
    vram = get_available_vram()
    needed = MODELS[model_key]["vram_fp16"]

    if vram >= needed * 1.2:
        return "none"  # bf16/fp16
    elif vram >= needed * 0.4:
        return "4bit"
    else:
        return "skip"


# ---------------------------------------------------------------------------
# Model loading — each model family has its own loader
# ---------------------------------------------------------------------------

def _make_bnb_config_4bit():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _make_bnb_config_8bit():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_8bit=True)


def _base_model_kwargs(quantize: str, trust_remote_code: bool):
    kwargs = {
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if quantize == "4bit":
        kwargs["quantization_config"] = _make_bnb_config_4bit()
    elif quantize == "8bit":
        kwargs["quantization_config"] = _make_bnb_config_8bit()
    return kwargs


def load_qwen3_vl(quantize: str):
    """Load Qwen3-VL-8B-Instruct."""
    from transformers import AutoProcessor
    # Qwen3VL uses its own class — try importing it
    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForVision2Seq as Qwen3VLForConditionalGeneration
        print("  WARNING: Qwen3VLForConditionalGeneration not found, falling back to AutoModelForVision2Seq")

    model_id = MODELS["qwen3-vl-8b"]["id"]
    kwargs = _base_model_kwargs(quantize, trust_remote_code=False)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, processor


def load_internvl3(quantize: str):
    """Load InternVL3-8B with trust_remote_code."""
    from transformers import AutoModel, AutoTokenizer

    model_id = MODELS["internvl3-8b"]["id"]
    kwargs = _base_model_kwargs(quantize, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, tokenizer


def load_gemma3(quantize: str):
    """Load Gemma 3 12B multimodal."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model_id = MODELS["gemma3-12b"]["id"]
    kwargs = _base_model_kwargs(quantize, trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def load_glm45v(quantize: str):
    """Load GLM-4.5V (MoE, needs trust_remote_code)."""
    from transformers import AutoModelForCausalLM, AutoProcessor

    model_id = MODELS["glm-4.5v"]["id"]
    kwargs = _base_model_kwargs(quantize, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    return model, processor


MODEL_LOADERS = {
    "qwen3-vl-8b": load_qwen3_vl,
    "internvl3-8b": load_internvl3,
    "gemma3-12b": load_gemma3,
    "glm-4.5v": load_glm45v,
}


def load_model(model_key: str, quantize: str):
    """Load model and processor/tokenizer by key."""
    info = MODELS[model_key]
    print(f"\n{'='*60}")
    print(f"Loading {model_key}: {info['id']}")
    print(f"  Quantization: {quantize}")
    print(f"  VRAM available: {get_available_vram():.1f} GB")
    print(f"{'='*60}")

    loader = MODEL_LOADERS[model_key]
    model, processor = loader(quantize)

    param_count = sum(p.numel() for p in model.parameters())
    vram_used = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  Parameters: {param_count / 1e9:.1f}B")
    print(f"  VRAM used: {vram_used:.1f} GB")

    return model, processor


# ---------------------------------------------------------------------------
# Per-model inference — each model has a different chat template / input format
# ---------------------------------------------------------------------------

def _load_image(path: str):
    """Load an image from path, return PIL Image."""
    from PIL import Image
    return Image.open(path).convert("RGB")


def run_inference_qwen3(model, processor, messages: list, media_paths: list, max_new_tokens: int = 32) -> str:
    """Inference for Qwen3-VL (similar to Qwen2.5-VL but with Qwen3VL class)."""
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


def run_inference_internvl3(model, tokenizer, messages: list, media_paths: list, max_new_tokens: int = 32) -> str:
    """Inference for InternVL3 — uses model.chat() API with tokenizer."""
    from PIL import Image
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    # InternVL3 expects images loaded and passed via pixel_values
    # Use the model's built-in chat method if available
    images = []
    for p in media_paths:
        if p and os.path.exists(p) and any(p.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
            images.append(Image.open(p).convert("RGB"))

    # Build the text prompt — InternVL uses <image> tokens
    question_text = ""
    for msg in messages:
        if msg["role"] == "user":
            for part in msg.get("content", []):
                if isinstance(part, dict) and part.get("type") == "text":
                    question_text += part["text"]

    # Prepend <image> tokens for each image
    image_prefix = "".join(f"<image>\n" for _ in images)
    prompt = image_prefix + question_text

    # Use model.chat() if it exists (InternVL custom method)
    if hasattr(model, "chat"):
        pixel_values = None
        if images:
            pixel_values = _internvl_process_images(images, model)
        response = model.chat(tokenizer, pixel_values=pixel_values, question=prompt,
                              generation_config={"max_new_tokens": max_new_tokens, "do_sample": False})
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


def run_inference_gemma3(model, processor, messages: list, media_paths: list, max_new_tokens: int = 32) -> str:
    """Inference for Gemma 3 multimodal."""
    from PIL import Image

    images = []
    for p in media_paths:
        if p and os.path.exists(p) and any(p.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
            images.append(Image.open(p).convert("RGB"))

    # Build Gemma3 chat format
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
        gemma_messages.append({"role": msg["role"], "content": content_parts or msg.get("content", "")})

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


def run_inference_glm45v(model, processor, messages: list, media_paths: list, max_new_tokens: int = 32) -> str:
    """Inference for GLM-4.5V."""
    from PIL import Image

    images = []
    for p in media_paths:
        if p and os.path.exists(p) and any(p.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
            images.append(Image.open(p).convert("RGB"))

    # Build GLM chat messages
    glm_content = []
    for msg in messages:
        if msg["role"] == "user":
            img_idx = 0
            for part in msg.get("content", []):
                if isinstance(part, dict):
                    if part.get("type") == "image" and img_idx < len(images):
                        glm_content.append({"type": "image", "image": images[img_idx]})
                        img_idx += 1
                    elif part.get("type") == "text":
                        glm_content.append({"type": "text", "text": part["text"]})

    glm_messages = [{"role": "user", "content": glm_content}]

    inputs = processor.apply_chat_template(
        glm_messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    del inputs, output_ids
    torch.cuda.empty_cache()
    return response


INFERENCE_FNS = {
    "qwen3-vl-8b": run_inference_qwen3,
    "internvl3-8b": run_inference_internvl3,
    "gemma3-12b": run_inference_gemma3,
    "glm-4.5v": run_inference_glm45v,
}


# ---------------------------------------------------------------------------
# Per-model message formatting
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
                content.append({"type": "image", "image": path, "max_pixels": 256 * 256, "min_pixels": 28 * 28})
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

    content.append({"type": "text", "text": "\nAnswer with ONLY the letter (A, B, C, or D) of the correct option."})
    messages = [{"role": "user", "content": content}]
    return messages, has_media


def format_question_generic(item: dict, media_paths: list) -> tuple:
    """Generic formatter for InternVL3, Gemma3, GLM-4.5V — builds content list."""
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
            # Most models don't natively support video — skip
            content.append({"type": "text", "text": "[video input — not supported by this model]"})
        elif part.strip():
            content.append({"type": "text", "text": part})

    content.append({"type": "text", "text": "\nAnswer with ONLY the letter (A, B, C, or D) of the correct option."})
    messages = [{"role": "user", "content": content}]
    return messages, has_media


FORMAT_FNS = {
    "qwen3-vl-8b": format_question_qwen3,
    "internvl3-8b": format_question_generic,
    "gemma3-12b": format_question_generic,
    "glm-4.5v": format_question_generic,
}


# ---------------------------------------------------------------------------
# Evaluation loop (shared across all models)
# ---------------------------------------------------------------------------

def evaluate_model(
    model_key: str,
    model,
    processor,
    data: list,
    data_dir: str,
    output_dir: str,
    skip_missing_media: bool = False,
):
    """Run PhysBench evaluation for a single model."""
    info = MODELS[model_key]
    os.makedirs(output_dir, exist_ok=True)

    format_fn = FORMAT_FNS[model_key]
    inference_fn = INFERENCE_FNS[model_key]

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

        if skip_missing_media and not has_media and item.get("mode") != "general":
            skipped += 1
            continue

        try:
            response = inference_fn(model, processor, messages, media_paths)
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

            if (i + 1) % 50 == 0 or (i + 1) == len(data):
                elapsed = time.time() - start_time
                acc = correct / total * 100 if total > 0 else 0
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(data) - i - 1) / rate if rate > 0 else 0
                print(f"  [{model_key}] [{i+1}/{len(data)}] "
                      f"Acc: {acc:.1f}% ({correct}/{total}) | "
                      f"Skip: {skipped} Err: {errors} | "
                      f"{rate:.1f} q/s | ETA: {eta/60:.1f}min")

        except Exception as e:
            errors += 1
            results.append({"idx": item.get("idx", i), "error": str(e)[:300]})
            if errors <= 5:
                print(f"  ERROR [{model_key}] item {i}: {e}")
            elif errors == 6:
                print(f"  [{model_key}] (suppressing further errors)")

    elapsed = time.time() - start_time
    overall_acc = correct / total * 100 if total > 0 else 0

    def _cat_acc(tracker):
        return {k: {"accuracy": round(v["correct"] / v["total"] * 100, 2) if v["total"] > 0 else 0,
                     "correct": v["correct"], "total": v["total"]}
                for k, v in sorted(tracker.items())}

    summary = {
        "model": model_key,
        "model_id": info["id"],
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

    # Print summary
    print(f"\n{'='*70}")
    print(f"RESULTS — {model_key} ({info['id']})")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print(f"Skipped: {skipped} | Errors: {errors} | Time: {elapsed:.0f}s")
    for k, v in summary["by_task_type"].items():
        print(f"  {k:30s}: {v['accuracy']:5.1f}% ({v['correct']}/{v['total']})")

    # Save results
    results_path = os.path.join(output_dir, f"{model_key}_results.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "predictions": results}, f, indent=2)

    summary_path = os.path.join(output_dir, f"{model_key}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Results saved to: {results_path}")
    return summary


# ---------------------------------------------------------------------------
# Cleanup between models
# ---------------------------------------------------------------------------

def unload_model(model, processor):
    """Aggressively free GPU memory between model evaluations.

    Moves model to CPU, deletes references, runs multi-pass GC, clears CUDA
    cache, and verifies memory was actually freed. Critical for sequential
    evaluation on 16GB consumer GPUs.
    """
    # Move model to CPU first (helps with some quantized models)
    try:
        if hasattr(model, "cpu"):
            model.cpu()
    except Exception:
        pass

    del model
    del processor

    # Multi-pass garbage collection to catch cyclic references
    gc.collect()
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Brief pause to let memory settle
    time.sleep(2)

    if torch.cuda.is_available():
        free_mem, total_mem = torch.cuda.mem_get_info()
        free_gb = free_mem / 1e9
        total_gb = total_mem / 1e9
        usage_pct = (total_gb - free_gb) / total_gb * 100 if total_gb > 0 else 0
        print(f"  GPU memory after unload: {free_gb:.1f} / {total_gb:.1f} GB free ({usage_pct:.0f}% used)")
        if free_gb < total_gb * 0.75:
            print(f"  WARNING: GPU not fully freed. Attempting forceful cleanup...")
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(3)
            free_mem, total_mem = torch.cuda.mem_get_info()
            print(f"  After second cleanup: {free_mem/1e9:.1f} / {total_mem/1e9:.1f} GB free")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-model PhysBench evaluation")
    parser.add_argument("--model", default="all",
                        choices=list(MODELS.keys()) + ["all"],
                        help="Which model to evaluate (or 'all')")
    parser.add_argument("--split", default="test", choices=["test", "val"])
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "physbench"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "multi_model"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--quantize", choices=["none", "4bit", "8bit", "auto"], default="auto",
                        help="Quantization mode ('auto' selects based on VRAM)")
    parser.add_argument("--skip-missing-media", action="store_true")
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    # Determine which models to run
    if args.model == "all":
        model_keys = list(MODELS.keys())
    else:
        model_keys = [args.model]

    # Load data once
    data = load_physbench_data(args.data_dir, split=args.split, max_samples=args.max_samples)
    if not data:
        print("No data loaded. Exiting.")
        sys.exit(1)

    print(f"\nPhysBench multi-model evaluation")
    print(f"  Models: {', '.join(model_keys)}")
    print(f"  Split: {args.split} ({len(data)} questions)")
    print(f"  VRAM: {get_available_vram():.1f} GB")

    all_summaries = {}

    for model_key in model_keys:
        # Determine quantization
        if args.quantize == "auto":
            quantize = choose_quantization(model_key)
        else:
            quantize = args.quantize

        if quantize == "skip":
            print(f"\n*** SKIPPING {model_key}: requires {MODELS[model_key]['vram_fp16']}GB VRAM, "
                  f"only {get_available_vram():.1f}GB available ***")
            all_summaries[model_key] = {"status": "skipped", "reason": "insufficient_vram"}
            continue

        if not can_run_model(model_key, quantize):
            print(f"\n*** SKIPPING {model_key}: insufficient VRAM even with {quantize} ***")
            all_summaries[model_key] = {"status": "skipped", "reason": "insufficient_vram"}
            continue

        try:
            model, processor = load_model(model_key, quantize)
            output_dir = os.path.join(args.output_dir, f"{model_key}_{quantize}")
            summary = evaluate_model(
                model_key=model_key,
                model=model,
                processor=processor,
                data=data,
                data_dir=args.data_dir,
                output_dir=output_dir,
                skip_missing_media=args.skip_missing_media,
            )
            all_summaries[model_key] = summary
            unload_model(model, processor)

        except Exception as e:
            print(f"\n*** FAILED {model_key}: {e} ***")
            import traceback
            traceback.print_exc()
            all_summaries[model_key] = {"status": "failed", "error": str(e)}
            # Try to clean up
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Print comparison table
    print(f"\n{'='*70}")
    print("MULTI-MODEL COMPARISON")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'Time':>8}")
    print("-" * 60)
    for key in model_keys:
        s = all_summaries.get(key, {})
        if "overall_accuracy" in s:
            print(f"{key:<20} {s['overall_accuracy']:>9.2f}% "
                  f"{s['correct']:>10} {s['total']:>8} {s.get('elapsed_seconds', 0):>7.0f}s")
        else:
            status = s.get("status", "unknown")
            reason = s.get("reason", s.get("error", ""))[:30]
            print(f"{key:<20} {'—':>10} {'—':>10} {'—':>8} {status}: {reason}")

    # Save combined summary
    combined_path = os.path.join(args.output_dir, "multi_model_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nCombined summary: {combined_path}")


if __name__ == "__main__":
    main()
