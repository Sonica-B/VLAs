#!/usr/bin/env python3
"""
PhysBench Evaluation Script for VLMs.

Evaluates a VLM on the PhysBench benchmark (10,002 multiple-choice physics
questions across 4 domains). Designed as our BASELINE measurement and for
post-training evaluation after QLoRA interventions.

Usage:
    # Baseline evaluation (Qwen2.5-VL-7B, 4-bit)
    python scripts/run_physbench_eval.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --quantize 4bit \
        --data-dir data/physbench \
        --output-dir results/physbench

    # Evaluate a fine-tuned adapter
    python scripts/run_physbench_eval.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --adapter-path results/qlora/condition_a/adapter \
        --quantize 4bit \
        --data-dir data/physbench

    # Quick test on validation split only
    python scripts/run_physbench_eval.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --quantize 4bit \
        --split val \
        --max-samples 50

Requirements:
    pip install torch transformers accelerate bitsandbytes qwen-vl-utils pillow
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch


def load_physbench_data(data_dir: str, split: str = "test", max_samples: int = None):
    """Load PhysBench questions from JSON or parquet."""
    data_dir = Path(data_dir)

    # Try test.json first
    json_path = data_dir / f"{split}.json"
    if not json_path.exists() and split == "test":
        json_path = data_dir / "test.json"

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        # Support both JSON array and JSONL formats
        if content.startswith("["):
            data = json.loads(content)
        else:
            data = [json.loads(line) for line in content.splitlines() if line.strip()]
        # Filter by split if the JSON contains mixed splits
        if data and isinstance(data[0], dict) and "split" in data[0]:
            data = [d for d in data if d.get("split") == split]
        print(f"Loaded {len(data)} questions from {json_path}")
    else:
        print(f"ERROR: Could not load PhysBench data from {data_dir}")
        print(f"  JSON path tried: {json_path}")
        print(f"\nPlease run: python scripts/download_physbench.py --data-dir {data_dir}")
        sys.exit(1)

    if max_samples:
        data = data[:max_samples]
        print(f"  (Limited to {max_samples} samples)")

    return data


def resolve_media_paths(item: dict, data_dir: str) -> list:
    """Resolve file_name entries to absolute paths, checking image/ and video/ subdirs."""
    data_dir = Path(data_dir)
    file_names = item.get("file_name", [])
    if isinstance(file_names, str):
        file_names = [file_names]

    resolved = []
    for fname in file_names:
        # Check multiple possible locations
        candidates = [
            data_dir / fname,
            data_dir / "image" / fname,
            data_dir / "video" / fname,
            data_dir / "images" / fname,
            data_dir / "videos" / fname,
        ]
        found = None
        for c in candidates:
            if c.exists():
                found = str(c)
                break
        resolved.append(found)  # None if not found

    return resolved


def format_question_for_vlm(item: dict, media_paths: list) -> tuple:
    """
    Format a PhysBench question into VLM input.

    Returns (messages, has_media) for the Qwen2.5-VL chat template.
    The question text contains <image> and <video> placeholders that map
    sequentially to entries in file_name.
    """
    question_text = item.get("question", "")

    # Build content list: interleave media and text
    content = []
    media_idx = 0
    file_names = item.get("file_name", [])
    if isinstance(file_names, str):
        file_names = [file_names]

    # Split question on <image> and <video> placeholders
    parts = re.split(r"(<image>|<video>)", question_text)

    has_media = False
    for part in parts:
        if part == "<image>" and media_idx < len(media_paths):
            path = media_paths[media_idx]
            media_idx += 1
            if path and os.path.exists(path):
                content.append({"type": "image", "image": f"file://{path}"})
                has_media = True
            else:
                content.append({"type": "text", "text": "[image unavailable]"})
        elif part == "<video>" and media_idx < len(media_paths):
            path = media_paths[media_idx]
            media_idx += 1
            if path and os.path.exists(path):
                content.append({"type": "video", "video": f"file://{path}"})
                has_media = True
            else:
                content.append({"type": "text", "text": "[video unavailable]"})
        elif part.strip():
            content.append({"type": "text", "text": part})

    # Add instruction suffix
    content.append({
        "type": "text",
        "text": "\nAnswer with ONLY the letter (A, B, C, or D) of the correct option."
    })

    messages = [{"role": "user", "content": content}]
    return messages, has_media


def extract_answer(response: str) -> str:
    """Extract A/B/C/D answer from model response."""
    response = response.strip()

    # Direct single letter
    if response in ("A", "B", "C", "D"):
        return response

    # "The answer is X" pattern
    match = re.search(r"(?:answer|option)\s*(?:is|:)\s*([A-D])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # First letter if it's A-D
    match = re.search(r"\b([A-D])\b", response)
    if match:
        return match.group(1).upper()

    # Fallback: first character
    if response and response[0].upper() in "ABCD":
        return response[0].upper()

    return "X"  # Could not parse


def load_model(model_name: str, quantize: str = "4bit", adapter_path: str = None):
    """
    Load a VLM model with optional quantization and LoRA adapter.

    Returns (model, processor).
    """
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    print(f"\nLoading model: {model_name}")
    print(f"  Quantization: {quantize}")
    if adapter_path:
        print(f"  Adapter: {adapter_path}")

    # Quantization config
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
        "trust_remote_code": True,
    }

    if quantize == "4bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif quantize == "8bit":
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    # Detect model family
    model_name_lower = model_name.lower()
    if "qwen2.5-vl" in model_name_lower or "qwen2-vl" in model_name_lower:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name, **model_kwargs
        )
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load LoRA adapter if provided
    if adapter_path:
        from peft import PeftModel
        print(f"  Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("  Adapter merged.")

    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  Parameters: {param_count / 1e9:.1f}B")
    print(f"  VRAM used: {vram:.1f} GB")

    return model, processor


def run_inference(model, processor, messages: list, max_new_tokens: int = 32) -> str:
    """Run single inference on Qwen2.5-VL style model."""
    from qwen_vl_utils import process_vision_info

    # Apply chat template
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Process vision inputs
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    # Decode only the generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated = output_ids[0][input_len:]
    response = processor.decode(generated, skip_special_tokens=True)

    return response


def evaluate(
    model,
    processor,
    data: list,
    data_dir: str,
    output_dir: str,
    model_name: str,
    skip_missing_media: bool = False,
):
    """Run evaluation on PhysBench data."""
    os.makedirs(output_dir, exist_ok=True)

    results = []
    correct = 0
    total = 0
    skipped = 0
    errors = 0

    # Per-category tracking
    by_task_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_sub_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_ability = defaultdict(lambda: {"correct": 0, "total": 0})
    by_mode = defaultdict(lambda: {"correct": 0, "total": 0})

    start_time = time.time()

    for i, item in enumerate(data):
        media_paths = resolve_media_paths(item, data_dir)
        messages, has_media = format_question_for_vlm(item, media_paths)

        # Skip if media is required but missing
        if skip_missing_media and not has_media and item.get("mode") != "general":
            skipped += 1
            continue

        try:
            response = run_inference(model, processor, messages)
            predicted = extract_answer(response)
            gt_answer = item.get("answer", "").strip().upper()
            is_correct = predicted == gt_answer

            if is_correct:
                correct += 1
            total += 1

            # Track per-category
            task_type = item.get("task_type", "unknown")
            sub_type = item.get("sub_type", "unknown")
            ability = item.get("ability_type", "unknown")
            mode = item.get("mode", "unknown")

            by_task_type[task_type]["total"] += 1
            by_sub_type[sub_type]["total"] += 1
            by_ability[ability]["total"] += 1
            by_mode[mode]["total"] += 1
            if is_correct:
                by_task_type[task_type]["correct"] += 1
                by_sub_type[sub_type]["correct"] += 1
                by_ability[ability]["correct"] += 1
                by_mode[mode]["correct"] += 1

            result_entry = {
                "idx": item.get("idx", i),
                "question": item.get("question", "")[:200],
                "gt_answer": gt_answer,
                "predicted": predicted,
                "raw_response": response[:200],
                "correct": is_correct,
                "task_type": task_type,
                "sub_type": sub_type,
                "ability_type": ability,
                "mode": mode,
                "has_media": has_media,
            }
            results.append(result_entry)

            # Progress
            if (i + 1) % 50 == 0 or (i + 1) == len(data):
                elapsed = time.time() - start_time
                acc = correct / total * 100 if total > 0 else 0
                rate = total / elapsed if elapsed > 0 else 0
                eta = (len(data) - i - 1) / rate if rate > 0 else 0
                print(
                    f"  [{i + 1}/{len(data)}] "
                    f"Acc: {acc:.1f}% ({correct}/{total}) | "
                    f"Skip: {skipped} Err: {errors} | "
                    f"{rate:.1f} q/s | ETA: {eta / 60:.1f}min"
                )

        except Exception as e:
            errors += 1
            results.append({
                "idx": item.get("idx", i),
                "error": str(e)[:200],
                "task_type": item.get("task_type", "unknown"),
            })
            if errors <= 5:
                print(f"  ERROR on item {i}: {e}")
            elif errors == 6:
                print("  (suppressing further error messages)")

    # Compute final metrics
    elapsed = time.time() - start_time
    overall_acc = correct / total * 100 if total > 0 else 0

    def compute_category_acc(tracker):
        return {
            k: {
                "accuracy": v["correct"] / v["total"] * 100 if v["total"] > 0 else 0,
                "correct": v["correct"],
                "total": v["total"],
            }
            for k, v in sorted(tracker.items())
        }

    summary = {
        "model": model_name,
        "overall_accuracy": round(overall_acc, 2),
        "correct": correct,
        "total": total,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": round(elapsed, 1),
        "questions_per_second": round(total / elapsed, 2) if elapsed > 0 else 0,
        "by_task_type": compute_category_acc(by_task_type),
        "by_sub_type": compute_category_acc(by_sub_type),
        "by_ability_type": compute_category_acc(by_ability),
        "by_mode": compute_category_acc(by_mode),
    }

    # Print summary
    print("\n" + "=" * 70)
    print(f"PHYSBENCH EVALUATION RESULTS - {model_name}")
    print("=" * 70)
    print(f"Overall Accuracy: {overall_acc:.2f}% ({correct}/{total})")
    print(f"Skipped: {skipped} | Errors: {errors}")
    print(f"Time: {elapsed:.0f}s ({total / elapsed:.1f} q/s)")

    print(f"\n--- By Task Type (Domain) ---")
    for k, v in summary["by_task_type"].items():
        print(f"  {k:30s}: {v['accuracy']:5.1f}% ({v['correct']}/{v['total']})")

    print(f"\n--- By Sub Type ---")
    for k, v in summary["by_sub_type"].items():
        print(f"  {k:30s}: {v['accuracy']:5.1f}% ({v['correct']}/{v['total']})")

    print(f"\n--- By Ability ---")
    for k, v in summary["by_ability_type"].items():
        print(f"  {k:30s}: {v['accuracy']:5.1f}% ({v['correct']}/{v['total']})")

    print(f"\n--- Published Baselines (from PhysBench paper) ---")
    print(f"  GPT-4o:                        49.5%")
    print(f"  Gemini-1.5-Pro:                43.2%")
    print(f"  Qwen2-VL-72B:                  46.8%")
    print(f"  InternVL2-76B:                 42.2%")
    print(f"  Qwen2-VL-7B:                   37.0%")
    print(f"  {'Our result':30s}: {overall_acc:.1f}%")

    # Save results
    results_path = os.path.join(output_dir, "physbench_results.json")
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "predictions": results}, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    summary_path = os.path.join(output_dir, "physbench_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM on PhysBench")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--adapter-path",
        default=None,
        help="Path to LoRA adapter (for post-training evaluation)",
    )
    parser.add_argument(
        "--quantize",
        choices=["none", "4bit", "8bit"],
        default="4bit",
        help="Quantization mode",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "physbench"),
        help="PhysBench data directory",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "results", "physbench"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for debugging)",
    )
    parser.add_argument(
        "--skip-missing-media",
        action="store_true",
        help="Skip questions with missing images/videos instead of using placeholders",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Evaluate text-only (no images/videos) to measure text-only baseline",
    )

    args = parser.parse_args()
    args.data_dir = os.path.abspath(args.data_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    # Append model info and condition to output dir
    model_short = args.model.split("/")[-1]
    condition = "baseline"
    if args.adapter_path:
        condition = Path(args.adapter_path).parent.name
    args.output_dir = os.path.join(args.output_dir, f"{model_short}_{condition}")

    print("PhysBench Evaluation")
    print(f"  Model: {args.model}")
    print(f"  Split: {args.split}")
    print(f"  Data dir: {args.data_dir}")
    print(f"  Output dir: {args.output_dir}")

    # Load data
    data = load_physbench_data(args.data_dir, split=args.split, max_samples=args.max_samples)
    if not data:
        print("No data loaded. Exiting.")
        sys.exit(1)

    # Print dataset stats
    task_types = defaultdict(int)
    for item in data:
        task_types[item.get("task_type", "unknown")] += 1
    print(f"\nDataset distribution:")
    for k, v in sorted(task_types.items()):
        print(f"  {k}: {v}")

    # Load model
    model, processor = load_model(
        args.model,
        quantize=args.quantize,
        adapter_path=args.adapter_path,
    )

    # Run evaluation
    summary = evaluate(
        model=model,
        processor=processor,
        data=data,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=f"{model_short} ({args.quantize})",
        skip_missing_media=args.skip_missing_media,
    )

    return summary


if __name__ == "__main__":
    main()
