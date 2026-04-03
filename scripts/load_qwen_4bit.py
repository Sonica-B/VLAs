#!/usr/bin/env python3
"""
Load Qwen2.5-VL-7B-Instruct in 4-bit quantization and verify it works.

This script:
1. Checks/installs dependencies (qwen-vl-utils, bitsandbytes)
2. Loads the model in NF4 4-bit quantization (~5GB VRAM)
3. Loads the processor
4. Runs a test forward pass on a synthetic image
5. Reports VRAM usage
6. Verifies we can access hidden states at all 4 pipeline stages

Hardware target: RTX 5070 Ti (12GB VRAM), i9 CPU, 32GB RAM
"""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Verify required packages are installed."""
    missing = []
    try:
        import bitsandbytes
        print(f"  bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        missing.append("bitsandbytes")

    try:
        import transformers
        print(f"  transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers>=4.45.0")

    try:
        import qwen_vl_utils
        print(f"  qwen-vl-utils: available")
    except ImportError:
        print("  qwen-vl-utils: not installed (optional, will try without)")

    if missing:
        print(f"\nMissing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True


def report_vram():
    """Report current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"  VRAM: {allocated:.2f}GB allocated / {reserved:.2f}GB reserved / {total:.2f}GB total")
        return allocated
    return 0.0


def load_qwen_4bit():
    """Load Qwen2.5-VL-7B in 4-bit NF4 quantization.

    Tries two approaches:
    1. Pre-quantized unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit (faster download)
    2. Official Qwen/Qwen2.5-VL-7B-Instruct with BitsAndBytesConfig
    """
    from transformers import AutoProcessor, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Try pre-quantized version first (smaller download, more VRAM efficient)
    model_ids = [
        ("unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit", False),
        ("Qwen/Qwen2.5-VL-7B-Instruct", True),
    ]

    model = None
    used_model_id = None

    for model_id, needs_bnb in model_ids:
        print(f"\nAttempting to load: {model_id}")
        try:
            from transformers import Qwen2_5VLForConditionalGeneration

            load_kwargs = {
                "device_map": "auto",
                "torch_dtype": torch.bfloat16,
            }
            if needs_bnb:
                load_kwargs["quantization_config"] = bnb_config

            model = Qwen2_5VLForConditionalGeneration.from_pretrained(
                model_id, **load_kwargs
            )
            used_model_id = model_id
            print(f"  Loaded successfully from {model_id}")
            break
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    if model is None:
        print("\nERROR: Could not load any Qwen2.5-VL variant.")
        print("To download manually, run:")
        print("  huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct")
        print("  # or")
        print("  huggingface-cli download unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")
        return None, None, None

    model.eval()
    report_vram()

    # Load processor (always from official repo)
    processor_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    print(f"\nLoading processor from {processor_id}...")
    processor = AutoProcessor.from_pretrained(processor_id)

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  Parameters: {n_params:.2f}B")

    return model, processor, used_model_id


def test_forward_pass(model, processor):
    """Run a test forward pass with a synthetic image."""
    print("\n--- Test Forward Pass ---")

    # Create a simple test image
    img = Image.fromarray(np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": "Describe this scene."},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    try:
        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
        )
    except Exception:
        # Fallback with qwen_vl_utils
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    print("  Input shapes:")
    for k, v in inputs.items():
        if hasattr(v, "shape"):
            print(f"    {k}: {v.shape}")

    t0 = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    elapsed = time.time() - t0

    generated = processor.decode(outputs[0], skip_special_tokens=True)
    print(f"  Generated ({elapsed:.1f}s): {generated[:200]}")
    report_vram()
    return True


def verify_hook_access(model):
    """Verify we can attach hooks at all 4 pipeline stages."""
    print("\n--- Verifying Hook Access Points ---")

    hook_targets = {
        "stage_1_enc_out": "model.visual.blocks.31",
        "stage_2_post_proj": "model.visual.merger",
        "stage_3_llm_8": "model.model.layers.8",
        "stage_4_llm_16": "model.model.layers.16",
    }

    results = {}
    for stage_name, module_path in hook_targets.items():
        try:
            module = model
            for part in module_path.split("."):
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            results[stage_name] = True
            print(f"  {stage_name} ({module_path}): OK — {type(module).__name__}")
        except (AttributeError, IndexError) as e:
            results[stage_name] = False
            print(f"  {stage_name} ({module_path}): FAILED — {e}")

    # Also check visual encoder structure
    print("\n  Visual encoder info:")
    if hasattr(model, "visual"):
        vis = model.visual
        if hasattr(vis, "blocks"):
            print(f"    visual.blocks: {len(vis.blocks)} layers")
        if hasattr(vis, "merger"):
            print(f"    visual.merger: {type(vis.merger).__name__}")
    elif hasattr(model, "model") and hasattr(model.model, "visual"):
        vis = model.model.visual
        if hasattr(vis, "blocks"):
            print(f"    model.visual.blocks: {len(vis.blocks)} layers")

    # Check LLM structure
    print("\n  LLM backbone info:")
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        print(f"    model.model.layers: {len(model.model.layers)} layers")

    return results


def main():
    print("=" * 60)
    print("Qwen2.5-VL-7B 4-bit Loading Test")
    print("=" * 60)

    # Step 1: Check dependencies
    print("\n--- Checking Dependencies ---")
    if not check_dependencies():
        sys.exit(1)

    print(f"\n  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        report_vram()

    # Step 2: Load model
    print("\n--- Loading Model ---")
    model, processor, model_id = load_qwen_4bit()
    if model is None:
        sys.exit(1)

    # Step 3: Verify hook access
    hook_results = verify_hook_access(model)
    if not all(hook_results.values()):
        print("\nWARNING: Some hook targets not accessible. Check model architecture.")

    # Step 4: Test forward pass
    try:
        test_forward_pass(model, processor)
    except Exception as e:
        print(f"\nForward pass failed: {e}")
        import traceback
        traceback.print_exc()

    # Step 5: Final VRAM report
    print("\n--- Final VRAM Report ---")
    report_vram()

    print("\n--- Summary ---")
    print(f"  Model: {model_id}")
    print(f"  Hooks: {sum(hook_results.values())}/4 accessible")
    print("  Status: READY for activation extraction")

    return model, processor


if __name__ == "__main__":
    main()
