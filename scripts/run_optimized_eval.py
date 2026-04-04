#!/usr/bin/env python3
"""
Optimized multi-model PhysBench evaluation pipeline.

Final production script with every optimization available:
  - AWQ quantization (10x faster) with bitsandbytes 4-bit fallback
  - Flash Attention 2 with SDPA fallback
  - torch.compile with reduce-overhead mode
  - Aggressive GPU memory management between models
  - max_new_tokens=10 (PhysBench is multiple choice)
  - torch.inference_mode() over torch.no_grad()

Models (local, 16GB GPU):
  1. Qwen3-VL-8B     (Qwen/Qwen3-VL-8B-Instruct)    ~6GB 4-bit
  2. InternVL3-8B     (OpenGVLab/InternVL3-8B)         ~6GB 4-bit
  3. Gemma 4 E4B      (google/gemma-4-e4b-it)          ~5GB 4-bit

GLM-4.5V (zai-org/GLM-4.5V, ~48GB) is Turing-only — use slurm_multi_model_eval.sh.

Usage:
    # Run all 3 models on val set
    python scripts/run_optimized_eval.py --split val --data-dir data/physbench --output-dir results/multi_model

    # Run specific model
    python scripts/run_optimized_eval.py --model qwen3-vl-8b --split val

    # Run on test set
    python scripts/run_optimized_eval.py --split test

    # Quick smoke test
    python scripts/run_optimized_eval.py --split val --max-samples 5
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
# Optimization stack detection
# ---------------------------------------------------------------------------

HAS_FLASH_ATTN = False
HAS_AWQ = False

def check_dependencies():
    """Check and report optimization availability."""
    global HAS_FLASH_ATTN, HAS_AWQ

    print("=== Optimization Stack ===")

    # Flash Attention 2
    try:
        import flash_attn
        HAS_FLASH_ATTN = True
        print(f"  Flash Attention: v{flash_attn.__version__} (2x attention speedup)")
    except ImportError:
        print("  Flash Attention: NOT INSTALLED (using SDPA fallback)")

    # AWQ
    try:
        import awq  # noqa: F401
        HAS_AWQ = True
        print("  AWQ: available (10x quantization speedup)")
    except ImportError:
        print("  AWQ: NOT INSTALLED (using bitsandbytes 4-bit)")

    # bitsandbytes
    try:
        import bitsandbytes
        print(f"  bitsandbytes: v{bitsandbytes.__version__}")
    except ImportError:
        print("  bitsandbytes: NOT INSTALLED (required if AWQ unavailable)")

    # transformers version
    import transformers
    print(f"  transformers: v{transformers.__version__}")
    tf_version = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if tf_version < (4, 49):
        print("    WARNING: Qwen3VLForConditionalGeneration needs transformers >= 4.49")
        print("    Install: pip install git+https://github.com/huggingface/transformers.git")

    # torch.compile
    print(f"  torch.compile: {'available' if hasattr(torch, 'compile') else 'NOT available'}")
    print(f"  torch: v{torch.__version__}")

    # GPU
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        mem = props.total_memory / 1e9
        print(f"  GPU: {name} ({mem:.0f}GB)")
        # Check compute capability for bf16
        cc = (props.major, props.minor)
        if cc >= (8, 0):
            print(f"  Compute capability: {cc[0]}.{cc[1]} (bf16 + flash attn supported)")
        elif cc >= (7, 0):
            print(f"  Compute capability: {cc[0]}.{cc[1]} (fp16 ok, flash attn may work)")
        else:
            print(f"  Compute capability: {cc[0]}.{cc[1]} (older GPU, some optimizations unavailable)")
    else:
        print("  GPU: NO CUDA GPU DETECTED")

    print("=" * 30)
    return HAS_FLASH_ATTN, HAS_AWQ


# ---------------------------------------------------------------------------
# GPU memory management
# ---------------------------------------------------------------------------

def get_gpu_memory_info() -> tuple[float, float]:
    """Return (free_gb, total_gb) for the current GPU."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    free, total = torch.cuda.mem_get_info()
    return free / 1e9, total / 1e9


def full_gpu_cleanup():
    """Nuclear option: completely free GPU memory between models."""
    gc.collect()
    gc.collect()  # Second pass catches cyclic references

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # Force Python to release memory
    gc.collect()

    # Brief pause to let memory settle
    time.sleep(2)

    # Verify
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        used = (total - free) / 1e9
        print(f"  GPU after cleanup: {used:.1f}GB used / {total/1e9:.1f}GB total")
        if used > 1.0:
            print("  WARNING: GPU not fully freed. Residual memory may cause OOM on next model.")
            # Try one more aggressive round
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(3)
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1e9
            print(f"  After second cleanup: {used:.1f}GB used / {total/1e9:.1f}GB total")


def unload_model(model, processor):
    """Completely free model from GPU."""
    try:
        if hasattr(model, "cpu"):
            model.cpu()
    except Exception:
        pass
    del model
    del processor
    full_gpu_cleanup()


# ---------------------------------------------------------------------------
# Quantization config
# ---------------------------------------------------------------------------

def make_bnb_4bit_config():
    """Create optimized bitsandbytes 4-bit config."""
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,  # Double quant saves ~0.4GB
    )


def get_attn_implementation():
    """Return best available attention implementation."""
    if HAS_FLASH_ATTN:
        return "flash_attention_2"
    return "sdpa"


def try_compile(model, model_name: str):
    """Attempt torch.compile on model for faster inference."""
    if not hasattr(torch, "compile"):
        return model
    # torch.compile requires triton, which is not available on Windows
    try:
        import triton  # noqa: F401
    except ImportError:
        print(f"  torch.compile skipped for {model_name}: triton not available")
        return model
    try:
        model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=False)
        print(f"  torch.compile applied to {model_name} (reduce-overhead mode)")
    except Exception as e:
        print(f"  torch.compile skipped for {model_name}: {e}")
    return model


# ---------------------------------------------------------------------------
# Model loaders (with AWQ -> bnb fallback)
# ---------------------------------------------------------------------------

def load_qwen3_vl(model_id: str) -> tuple:
    """Load Qwen3-VL-8B with best available quantization."""
    from transformers import AutoProcessor

    try:
        from transformers import Qwen3VLForConditionalGeneration
    except ImportError:
        print("  WARNING: Qwen3VLForConditionalGeneration not found.")
        print("  Requires transformers >= 4.49. Trying AutoModelForVision2Seq fallback...")
        from transformers import AutoModelForVision2Seq as Qwen3VLForConditionalGeneration

    model = None

    # Try AWQ first (fastest)
    if HAS_AWQ:
        try:
            from awq import AutoAWQForCausalLM
            awq_id = f"{model_id}-AWQ"
            print(f"  Trying AWQ: {awq_id}")
            model = AutoAWQForCausalLM.from_quantized(
                awq_id, fuse_layers=True, trust_remote_code=True,
            )
            print(f"  Loaded with AWQ (fastest)")
        except Exception as e:
            print(f"  AWQ not available for {model_id}: {e}")
            model = None

    # Fall back to bitsandbytes 4-bit
    if model is None:
        attn_impl = get_attn_implementation()
        print(f"  Loading with bitsandbytes 4-bit, attention={attn_impl}")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=make_bnb_4bit_config(),
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    model = try_compile(model, "Qwen3-VL-8B")
    return model, processor


def load_internvl3(model_id: str) -> tuple:
    """Load InternVL3-8B with best available quantization."""
    from transformers import AutoModel, AutoTokenizer

    model = None

    # Try AWQ first
    if HAS_AWQ:
        try:
            from awq import AutoAWQForCausalLM
            awq_id = f"{model_id}-AWQ"
            print(f"  Trying AWQ: {awq_id}")
            model = AutoAWQForCausalLM.from_quantized(
                awq_id, fuse_layers=True, trust_remote_code=True,
            )
            print(f"  Loaded with AWQ (fastest)")
        except Exception as e:
            print(f"  AWQ not available for {model_id}: {e}")
            model = None

    # Fall back to bitsandbytes 4-bit
    if model is None:
        attn_impl = get_attn_implementation()
        print(f"  Loading with bitsandbytes 4-bit, attention={attn_impl}")
        try:
            model = AutoModel.from_pretrained(
                model_id,
                quantization_config=make_bnb_4bit_config(),
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation=attn_impl,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
        except (ValueError, RuntimeError) as e:
            err_str = str(e)
            # InternVL custom code calls .item() during init which is incompatible
            # with meta tensors (used by bnb quantization and device_map="auto").
            # Fall back to bf16 on CPU, then move to GPU.
            if ("meta tensors" in err_str or "scaled_dot_product_attention" in err_str):
                print(f"  bnb 4-bit failed ({type(e).__name__}), falling back to bf16 on CPU then GPU")
                model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="eager",
                    trust_remote_code=True,
                    low_cpu_mem_usage=False,
                )
                model = model.cuda()
            else:
                raise

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    model = try_compile(model, "InternVL3-8B")
    return model, tokenizer


def load_gemma4(model_id: str) -> tuple:
    """Load Gemma 4 E4B with bitsandbytes 4-bit (small enough, AWQ unnecessary)."""
    from transformers import AutoProcessor

    # Gemma 4 E4B is only 4B params — bnb 4-bit is fine
    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        from transformers import AutoModelForVision2Seq as AutoModelForImageTextToText

    attn_impl = get_attn_implementation()
    print(f"  Loading with bitsandbytes 4-bit, attention={attn_impl}")
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        quantization_config=make_bnb_4bit_config(),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model.eval()
    model = try_compile(model, "Gemma4-E4B")
    return model, processor


# ---------------------------------------------------------------------------
# Per-model inference (optimized: max_new_tokens=10, inference_mode)
# ---------------------------------------------------------------------------

GENERATION_CONFIG = {
    "max_new_tokens": 10,   # PhysBench is multiple choice — only need 1-2 tokens
    "do_sample": False,     # Greedy for reproducibility
    "temperature": 1.0,
    "num_beams": 1,         # No beam search needed
    "use_cache": True,      # Enable KV-cache
}


def infer_qwen3_vl(model, processor, messages: list, media_paths: list,
                    max_new_tokens: int = 10) -> str:
    """Inference for Qwen3-VL (uses qwen_vl_utils)."""
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

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    del inputs, output_ids
    torch.cuda.empty_cache()
    return response


def infer_internvl3(model, tokenizer, messages: list, media_paths: list,
                    max_new_tokens: int = 10) -> str:
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
        with torch.inference_mode():
            response = model.chat(tokenizer, pixel_values=pixel_values, question=prompt,
                                  generation_config=generation_config)
        torch.cuda.empty_cache()
        return response

    # Fallback: manual tokenization
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
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


def infer_gemma4(model, processor, messages: list, media_paths: list,
                 max_new_tokens: int = 10) -> str:
    """Inference for Gemma 4 E4B multimodal."""
    from PIL import Image

    images = []
    for p in media_paths:
        if p and os.path.exists(p) and any(p.lower().endswith(ext) for ext in
                                            [".jpg", ".jpeg", ".png", ".bmp", ".webp"]):
            images.append(Image.open(p).convert("RGB"))

    # Build Gemma4 chat format with actual PIL images
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

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            num_beams=1, use_cache=True,
        )

    input_len = inputs["input_ids"].shape[1]
    response = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    del inputs, output_ids
    torch.cuda.empty_cache()
    return response


# ---------------------------------------------------------------------------
# Question formatting
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
    """Generic formatter for InternVL3, Gemma4 — builds content list."""
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
        "key": "gemma4-e4b",
        "name": "Gemma4-E4B",
        "id": "google/gemma-4-e4b-it",
        "load_fn": load_gemma4,
        "infer_fn": infer_gemma4,
        "format_fn": format_question_generic,
        "vram_4bit": 5,
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
    model_output_dir = os.path.join(output_dir, f"{model_key}_optimized")
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

            # Progress every 50 samples
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
                print("    (suppressing further error messages)")

        # Checkpoint every 200 samples (crash protection)
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
        "quantization": "4bit-nf4-double-quant",
        "optimizations": {
            "attention": "flash_attention_2" if HAS_FLASH_ATTN else "sdpa",
            "quantization": "AWQ" if HAS_AWQ else "bitsandbytes-4bit-nf4",
            "torch_compile": hasattr(torch, "compile"),
            "max_new_tokens": 10,
            "inference_mode": True,
        },
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

    # Save results immediately (crash protection)
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
# Comparison table and figure
# ---------------------------------------------------------------------------

def print_comparison_table(all_summaries: dict):
    """Print formatted comparison table."""

    # Header
    print()
    print("+" + "=" * 68 + "+")
    print("|" + "MULTI-MODEL PHYSBENCH COMPARISON".center(68) + "|")
    print("+" + "=" * 20 + "+" + "=" * 8 + "+" + "=" * 9 + "+" + "=" * 9 + "+" + "=" * 9 + "+" + "=" * 9 + "+")
    print(f"| {'Model':<18} | {'Ovrl':>6} | {'Prop.':>7} | {'Dynam.':>7} | {'Relat.':>7} | {'Scene':>7} |")
    print("+" + "-" * 20 + "+" + "-" * 8 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+" + "-" * 9 + "+")

    # Baseline row
    b = BASELINE_RESULTS
    print(f"| {'Qwen2.5-VL-7B*':<18} | {b['overall_accuracy']:>5.1f}% "
          f"| {b['by_task_type']['property']['accuracy']:>6.1f}% "
          f"| {b['by_task_type']['dynamics']['accuracy']:>6.1f}% "
          f"| {b['by_task_type']['relationships']['accuracy']:>6.1f}% "
          f"| {b['by_task_type']['scene']['accuracy']:>6.1f}% |")

    # Model rows
    for key, summary in all_summaries.items():
        name = summary.get("model_name", key)
        if "overall_accuracy" not in summary:
            status = summary.get("status", "failed")
            print(f"| {name:<18} | {'--':>6} | {'--':>7} | {'--':>7} | {'--':>7} | {status:>7} |")
            continue

        oa = summary["overall_accuracy"]
        tt = summary.get("by_task_type", {})
        prop_acc = tt.get("property", {}).get("accuracy", 0)
        dyn_acc = tt.get("dynamics", {}).get("accuracy", 0)
        rel_acc = tt.get("relationships", {}).get("accuracy", 0)
        scn_acc = tt.get("scene", {}).get("accuracy", 0)
        print(f"| {name:<18} | {oa:>5.1f}% | {prop_acc:>6.1f}% | {dyn_acc:>6.1f}% | {rel_acc:>6.1f}% | {scn_acc:>6.1f}% |")

    print("+" + "=" * 20 + "+" + "=" * 8 + "+" + "=" * 9 + "+" + "=" * 9 + "+" + "=" * 9 + "+" + "=" * 9 + "+")
    print("  * Qwen2.5-VL-7B baseline from prior full test-set run")
    print("  Note: GLM-4.5V (48GB) is Turing-only — use slurm_multi_model_eval.sh")
    print()


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
    n_categories = len(TASK_TYPE_ORDER) + 1
    x = np.arange(n_categories)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.Set2(np.linspace(0, 1, max(n_models, 3)))

    for i, (name, color) in enumerate(zip(model_names, colors)):
        vals = [overall_accs[i]] + [model_accs[tt][i] for tt in TASK_TYPE_ORDER]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width, label=name, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Multi-Model PhysBench Comparison\n(Optimized: AWQ/bnb-4bit + FlashAttn/SDPA + torch.compile)")
    ax.set_xticks(x)
    ax.set_xticklabels(["Overall"] + [tt.capitalize() for tt in TASK_TYPE_ORDER])
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "optimized_comparison_chart.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fig_path} (300 DPI)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Optimized multi-model PhysBench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_optimized_eval.py --split val --data-dir data/physbench --output-dir results/multi_model
  python scripts/run_optimized_eval.py --model qwen3-vl-8b --split val
  python scripts/run_optimized_eval.py --split test
  python scripts/run_optimized_eval.py --split val --max-samples 5
        """,
    )
    parser.add_argument("--model", default=None, type=str,
                        help="Run specific model only (e.g. qwen3-vl-8b, internvl3-8b, gemma4-e4b)")
    parser.add_argument("--split", default="val", choices=["test", "val"],
                        help="PhysBench split to evaluate (default: val)")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "physbench"),
                        help="Path to PhysBench data directory")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "multi_model"),
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of samples (for testing)")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile (for debugging)")
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    # Pre-flight: check all optimizations
    check_dependencies()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU detected. This script requires a GPU.")
        sys.exit(1)

    free_mem, total_mem = get_gpu_memory_info()
    gpu_name = torch.cuda.get_device_name(0)

    # Select models
    if args.model:
        model_key = args.model.lower()
        models = [m for m in MODELS_TO_RUN if m["key"] == model_key]
        if not models:
            valid = ", ".join(m["key"] for m in MODELS_TO_RUN)
            print(f"ERROR: Unknown model '{args.model}'. Valid: {valid}")
            sys.exit(1)
    else:
        models = MODELS_TO_RUN

    # Load data once (shared across all models)
    data = load_physbench_data(args.data_dir, split=args.split, max_samples=args.max_samples)
    if not data:
        print("ERROR: No data loaded. Check --data-dir and --split.")
        sys.exit(1)

    # Header
    print(f"\n{'='*70}")
    print(f"  OPTIMIZED MULTI-MODEL PHYSBENCH EVALUATION")
    print(f"{'='*70}")
    print(f"  GPU: {gpu_name} ({total_mem:.0f}GB, {free_mem:.1f}GB free)")
    print(f"  Split: {args.split} ({len(data)} samples)")
    print(f"  Models: {', '.join(m['name'] for m in models)}")
    print(f"  Optimizations: {'AWQ' if HAS_AWQ else 'bnb-4bit'} + "
          f"{'FlashAttn2' if HAS_FLASH_ATTN else 'SDPA'} + "
          f"{'torch.compile' if hasattr(torch, 'compile') and not args.no_compile else 'no-compile'}")
    print(f"  max_new_tokens: 10 (MC-only optimization)")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}")

    all_summaries = {}
    pipeline_start = time.time()

    for idx, model_config in enumerate(models):
        model_num = idx + 1
        model_name = model_config["name"]

        print(f"\n{'='*70}")
        print(f"  MODEL {model_num}/{len(models)}: {model_name}")
        print(f"  HuggingFace: {model_config['id']}")
        print(f"  Expected VRAM (4-bit): ~{model_config['vram_4bit']}GB")
        print(f"{'='*70}")

        # Check VRAM
        free_mem, _ = get_gpu_memory_info()
        if free_mem < model_config["vram_4bit"]:
            print(f"  WARNING: Only {free_mem:.1f}GB free, need ~{model_config['vram_4bit']}GB")

        try:
            # Step 1: Load
            print(f"  Loading {model_name}...")
            load_start = time.time()
            model, processor = model_config["load_fn"](model_config["id"])
            load_time = time.time() - load_start

            vram_used = torch.cuda.memory_allocated() / 1e9
            param_count = sum(p.numel() for p in model.parameters())
            print(f"  Loaded in {load_time:.0f}s | {param_count/1e9:.1f}B params | {vram_used:.1f}GB VRAM")

            # Step 2: Evaluate
            print(f"  Evaluating {args.split} set ({len(data)} samples)...")
            summary = evaluate_model(
                model_config=model_config,
                model=model,
                processor=processor,
                data=data,
                data_dir=args.data_dir,
                output_dir=args.output_dir,
            )
            all_summaries[model_config["key"]] = summary

            # Per-model results
            print(f"\n  {model_name} accuracy: {summary['overall_accuracy']:.1f}%")
            print(f"  Per-domain:", end="")
            for tt in TASK_TYPE_ORDER:
                acc = summary.get("by_task_type", {}).get(tt, {}).get("accuracy", 0)
                print(f" {tt.capitalize()} {acc:.1f}%", end="")
            print()
            print(f"  Speed: {summary['questions_per_second']:.2f} q/s | "
                  f"Time: {summary['elapsed_seconds']:.0f}s")

            # Step 3: Unload
            print(f"  Unloading {model_name}...")
            unload_model(model, processor)
            print(f"  Model unloaded. GPU freed.\n")

        except Exception as e:
            print(f"\n  *** FAILED: {model_name} ***")
            print(f"  Error: {e}")
            traceback.print_exc()
            all_summaries[model_config["key"]] = {
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

            # Suggest fixes
            error_str = str(e).lower()
            if "qwen3vl" in error_str or "no module" in error_str:
                print("  FIX: pip install git+https://github.com/huggingface/transformers.git")
            elif "bitsandbytes" in error_str:
                print("  FIX: pip install bitsandbytes>=0.43.0")
            elif "out of memory" in error_str or "oom" in error_str:
                print("  FIX: Reduce --max-samples or close other GPU processes")
            elif "trust_remote_code" in error_str:
                print("  FIX: trust_remote_code=True already set — check model ID")
            elif "does not exist" in error_str or "404" in error_str:
                print(f"  FIX: Model ID '{model_config['id']}' may not exist yet on HuggingFace")

            print(f"  Continuing with remaining models...\n")

    pipeline_elapsed = time.time() - pipeline_start

    # Final summary
    print(f"\n{'='*70}")
    print(f"  PIPELINE COMPLETE — {pipeline_elapsed/60:.1f} minutes total")
    print(f"{'='*70}")

    # Comparison table
    print_comparison_table(all_summaries)

    # Comparison figure
    generate_comparison_figure(all_summaries, args.output_dir)

    # Save combined results
    combined = {
        "pipeline": {
            "total_elapsed_seconds": round(pipeline_elapsed, 1),
            "split": args.split,
            "n_samples": len(data),
            "gpu": gpu_name,
            "optimizations": {
                "quantization": "AWQ" if HAS_AWQ else "bitsandbytes-4bit-nf4-double-quant",
                "attention": "flash_attention_2" if HAS_FLASH_ATTN else "sdpa",
                "torch_compile": hasattr(torch, "compile") and not args.no_compile,
                "max_new_tokens": 10,
                "inference_mode": True,
            },
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "baseline": BASELINE_RESULTS,
        "models": all_summaries,
    }
    os.makedirs(args.output_dir, exist_ok=True)
    combined_path = os.path.join(args.output_dir, "optimized_eval_results.json")
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Combined results: {combined_path}")


if __name__ == "__main__":
    main()
