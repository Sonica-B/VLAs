"""
VLM model loader for Qwen2.5-VL-7B, InternVL 2.5-8B, and LLaVA-OneVision-7B.

Provides a unified `load_vlm()` interface that handles model-specific
loading quirks (trust_remote_code, processor type, dtype, device placement).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

logger = logging.getLogger(__name__)

# Mapping from short model name → HuggingFace model ID
MODEL_HF_IDS: Dict[str, str] = {
    "qwen2_5_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "internvl2_5_8b": "OpenGVLab/InternVL2_5-8B",
    "llava_onevision_7b": "lmms-lab/llava-onevision-qwen2-7b-ov",
}

# Model families that require trust_remote_code=True
TRUST_REMOTE_CODE_MODELS = {"internvl2_5_8b"}

# Model families that use a specialized AutoModel class
SPECIALIZED_MODEL_CLASSES: Dict[str, str] = {
    "qwen2_5_vl_7b": "Qwen2_5VLForConditionalGeneration",
    "llava_onevision_7b": "LlavaOnevisionForConditionalGeneration",
}


def _get_bnb_config(load_in_4bit: bool, load_in_8bit: bool) -> Optional[BitsAndBytesConfig]:
    """Build BitsAndBytesConfig for quantized loading."""
    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


def load_vlm(
    model_name: str,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
    attn_implementation: str = "flash_attention_2",
) -> Tuple[Any, Any]:
    """Load a VLM model and its processor.

    Args:
        model_name: One of "qwen2_5_vl_7b", "internvl2_5_8b", "llava_onevision_7b".
        load_in_4bit: Load model in 4-bit NF4 quantization (reduces VRAM ~4×).
        load_in_8bit: Load model in 8-bit quantization (reduces VRAM ~2×).
        torch_dtype: Float type for model weights. bfloat16 recommended for A100.
        device_map: HuggingFace device_map strategy. "auto" for multi-GPU.
        attn_implementation: Attention kernel. "flash_attention_2" recommended.

    Returns:
        (model, processor): The loaded model and its associated processor/tokenizer.

    Raises:
        ValueError: If model_name is not recognized.

    Example:
        >>> model, processor = load_vlm("qwen2_5_vl_7b", load_in_4bit=True)
        >>> print(model.device)
    """
    if model_name not in MODEL_HF_IDS:
        raise ValueError(
            f"Unknown model: {model_name!r}. "
            f"Available: {list(MODEL_HF_IDS.keys())}"
        )

    hf_id = MODEL_HF_IDS[model_name]
    trust_remote_code = model_name in TRUST_REMOTE_CODE_MODELS
    bnb_config = _get_bnb_config(load_in_4bit, load_in_8bit)

    logger.info(f"Loading model: {hf_id}")
    logger.info(f"  dtype={torch_dtype}, 4bit={load_in_4bit}, 8bit={load_in_8bit}")

    load_kwargs: Dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }

    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
    else:
        load_kwargs["torch_dtype"] = torch_dtype

    # Flash attention only applies to non-quantized loads (4bit uses its own kernel)
    if not load_in_4bit and not load_in_8bit:
        try:
            load_kwargs["attn_implementation"] = attn_implementation
        except Exception:
            logger.warning("Flash attention not available; falling back to eager attention.")

    # Load model
    model = _load_model_by_family(model_name, hf_id, load_kwargs)
    model.eval()

    # Load processor
    processor = _load_processor(model_name, hf_id, trust_remote_code)

    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"Loaded {model_name}: {n_params:.2f}B parameters")
    return model, processor


def _load_model_by_family(
    model_name: str, hf_id: str, load_kwargs: Dict[str, Any]
) -> Any:
    """Load model using the appropriate AutoModel class for this VLM family."""
    if model_name == "qwen2_5_vl_7b":
        # TODO: Replace with direct import once transformers supports Qwen2.5-VL natively
        try:
            from transformers import Qwen2_5VLForConditionalGeneration
            return Qwen2_5VLForConditionalGeneration.from_pretrained(hf_id, **load_kwargs)
        except ImportError:
            logger.warning("Qwen2_5VLForConditionalGeneration not available in current transformers; "
                           "falling back to AutoModelForCausalLM")
            return AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

    elif model_name == "internvl2_5_8b":
        # InternVL requires trust_remote_code=True and uses AutoModel
        return AutoModel.from_pretrained(hf_id, **load_kwargs)

    elif model_name == "llava_onevision_7b":
        try:
            from transformers import LlavaOnevisionForConditionalGeneration
            return LlavaOnevisionForConditionalGeneration.from_pretrained(hf_id, **load_kwargs)
        except ImportError:
            return AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)

    else:
        return AutoModelForCausalLM.from_pretrained(hf_id, **load_kwargs)


def _load_processor(model_name: str, hf_id: str, trust_remote_code: bool) -> Any:
    """Load the model-appropriate processor (tokenizer + image processor)."""
    # TODO: For InternVL, may need to build a custom processor wrapper
    # because it uses a different tokenizer protocol.
    try:
        processor = AutoProcessor.from_pretrained(
            hf_id, trust_remote_code=trust_remote_code
        )
    except Exception:
        # Fallback: tokenizer only (for models without AutoProcessor support)
        processor = AutoTokenizer.from_pretrained(
            hf_id, trust_remote_code=trust_remote_code
        )
    return processor


def get_model_hidden_dims(model_name: str) -> Dict[str, int]:
    """Return expected hidden dimensions at each pipeline stage for a model.

    Useful for initializing probe input dimensions without loading the model.
    """
    dims: Dict[str, Dict[str, int]] = {
        "qwen2_5_vl_7b": {
            "stage_1_enc_out": 1280,
            "stage_2_post_proj": 3584,
            "stage_3_llm_8": 3584,
            "stage_4_llm_16": 3584,
        },
        "internvl2_5_8b": {
            "stage_1_enc_out": 1024,
            "stage_2_post_proj": 4096,
            "stage_3_llm_8": 4096,
            "stage_4_llm_16": 4096,
        },
        "llava_onevision_7b": {
            "stage_1_enc_out": 1152,
            "stage_2_post_proj": 3584,
            "stage_3_llm_8": 3584,
            "stage_4_llm_16": 3584,
        },
    }
    if model_name not in dims:
        raise ValueError(f"Unknown model: {model_name!r}")
    return dims[model_name]
