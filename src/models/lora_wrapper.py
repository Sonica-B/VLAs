"""
LoRA wrapper for applying targeted fine-tuning to VLM components.

Supports 5 ablation conditions:
  A: Encoder-only LoRA (last 6 ViT blocks, Q/V)
  B: Projection-only LoRA (MLP projector layers)
  C: LLM-only LoRA (first 8 LLM layers, Q/V)
  D: Encoder + Projection LoRA (A + B combined)
  E: Full model LoRA (all Q/V across the entire model)

Uses PEFT library's LoraConfig and get_peft_model().
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)

# Ablation condition IDs
ABLATION_CONDITIONS = ["A", "B", "C", "D", "E"]


@dataclass
class LoRAConditionConfig:
    """Configuration for a single LoRA ablation condition.

    Attributes:
        condition_id: One of "A", "B", "C", "D", "E".
        rank: LoRA rank r.
        lora_alpha: LoRA scaling alpha (typically 2 * rank).
        lora_dropout: Dropout applied to LoRA layers.
        target_modules: List of regex patterns or module names to apply LoRA to.
        bias: How to handle bias — "none", "all", or "lora_only".
    """

    condition_id: str
    rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=list)
    bias: str = "none"


# Default module patterns per condition per model family
# These patterns are passed to PEFT's LoraConfig as target_modules (regex supported)
CONDITION_MODULE_PATTERNS: Dict[str, Dict[str, List[str]]] = {
    "A": {  # Encoder-only: last 6 ViT blocks Q/V
        "qwen2_5_vl_7b": [
            r"visual\.blocks\.(2[6-9]|3[01])\.attn\.(q|v)_proj"
        ],
        "internvl2_5_8b": [
            r"vision_model\.encoder\.layers\.(1[89]|2[0-3])\.self_attn\.(q|v)_proj"
        ],
        "llava_onevision_7b": [
            r"vision_tower\.vision_model\.encoder\.layers\.(2[1-6])\.self_attn\.(q|v)_proj"
        ],
    },
    "B": {  # Projection-only: all MLP projector linear layers
        "qwen2_5_vl_7b": [r"visual\.merger\.mlp\.\d+"],
        "internvl2_5_8b": [r"mlp1\.\d+"],
        "llava_onevision_7b": [r"mm_projector\.\d+"],
    },
    "C": {  # LLM-only: first 8 layers Q/V
        "qwen2_5_vl_7b": [r"model\.layers\.[0-7]\.self_attn\.(q|v)_proj"],
        "internvl2_5_8b": [
            r"language_model\.model\.layers\.[0-7]\.self_attn\.(q|v)_proj"
        ],
        "llava_onevision_7b": [
            r"model\.language_model\.model\.layers\.[0-7]\.self_attn\.(q|v)_proj"
        ],
    },
    "D": {  # Encoder + Projection (A ∪ B)
        "qwen2_5_vl_7b": [
            r"visual\.blocks\.(2[6-9]|3[01])\.attn\.(q|v)_proj",
            r"visual\.merger\.mlp\.\d+",
        ],
        "internvl2_5_8b": [
            r"vision_model\.encoder\.layers\.(1[89]|2[0-3])\.self_attn\.(q|v)_proj",
            r"mlp1\.\d+",
        ],
        "llava_onevision_7b": [
            r"vision_tower\.vision_model\.encoder\.layers\.(2[1-6])\.self_attn\.(q|v)_proj",
            r"mm_projector\.\d+",
        ],
    },
    "E": {  # Full model: all Q/V + projection
        "qwen2_5_vl_7b": ["q_proj", "v_proj", r"visual\.merger\.mlp\.\d+"],
        "internvl2_5_8b": ["q_proj", "v_proj", r"mlp1\.\d+"],
        "llava_onevision_7b": ["q_proj", "v_proj", r"mm_projector\.\d+"],
    },
}


class LoRAWrapper:
    """Applies LoRA fine-tuning configuration to a VLM model.

    Args:
        model: Base VLM model (output of vlm_loader.load_vlm).
        model_name: Short model name for pattern lookup.
        condition: Ablation condition ("A", "B", "C", "D", or "E").
        rank: LoRA rank. Default 16.
        lora_alpha: LoRA alpha. Default 32 (= 2 × rank).
        lora_dropout: LoRA dropout probability. Default 0.05.

    Example:
        >>> wrapper = LoRAWrapper(model, "qwen2_5_vl_7b", condition="D")
        >>> peft_model = wrapper.apply()
        >>> print(wrapper.count_trainable_params())
    """

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        condition: str,
        rank: int = 16,
        lora_alpha: Optional[int] = None,
        lora_dropout: float = 0.05,
    ) -> None:
        if condition not in ABLATION_CONDITIONS:
            raise ValueError(f"condition must be one of {ABLATION_CONDITIONS}")
        if model_name not in next(iter(CONDITION_MODULE_PATTERNS.values())):
            raise ValueError(f"Unknown model_name: {model_name!r}")

        self.model = model
        self.model_name = model_name
        self.condition = condition
        self.rank = rank
        self.lora_alpha = lora_alpha if lora_alpha is not None else 2 * rank
        self.lora_dropout = lora_dropout
        self._peft_model: Optional[Any] = None

    def _build_lora_config(self) -> LoraConfig:
        """Build PEFT LoraConfig for the given condition and model."""
        target_modules = CONDITION_MODULE_PATTERNS[self.condition][self.model_name]

        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            target_modules=target_modules,
            # Use modules_to_save to preserve original weights for non-LoRA components
            modules_to_save=None,
        )

    def apply(self) -> Any:
        """Wrap the model with LoRA and return the PEFT model.

        Returns:
            peft_model: PEFT model with LoRA adapters applied.
        """
        config = self._build_lora_config()
        self._peft_model = get_peft_model(self.model, config)

        n_trainable = self.count_trainable_params()
        n_total = sum(p.numel() for p in self._peft_model.parameters())
        logger.info(
            f"LoRA condition {self.condition} applied to {self.model_name}: "
            f"{n_trainable / 1e6:.1f}M / {n_total / 1e9:.2f}B trainable "
            f"({100 * n_trainable / n_total:.2f}%)"
        )
        return self._peft_model

    def count_trainable_params(self) -> int:
        """Return the number of trainable parameters in the PEFT model."""
        if self._peft_model is None:
            raise RuntimeError("Call apply() first.")
        return sum(p.numel() for p in self._peft_model.parameters() if p.requires_grad)

    def save_adapter(self, output_dir: str) -> None:
        """Save LoRA adapter weights to disk (not the full model).

        Args:
            output_dir: Directory to save adapter_config.json and adapter_model.safetensors.
        """
        if self._peft_model is None:
            raise RuntimeError("Call apply() first.")
        self._peft_model.save_pretrained(output_dir)
        logger.info(f"LoRA adapter saved to: {output_dir}")

    @staticmethod
    def load_adapter(base_model: nn.Module, adapter_dir: str) -> Any:
        """Load a saved LoRA adapter onto a base model.

        Args:
            base_model: The base VLM model (same architecture as when adapter was saved).
            adapter_dir: Directory containing adapter_config.json.

        Returns:
            PEFT model with adapter loaded.
        """
        from peft import PeftModel

        peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
        logger.info(f"LoRA adapter loaded from: {adapter_dir}")
        return peft_model

    def print_trainable_modules(self) -> None:
        """Print names of all trainable modules for debugging."""
        if self._peft_model is None:
            raise RuntimeError("Call apply() first.")
        print(f"\nTrainable modules for Condition {self.condition} ({self.model_name}):")
        for name, param in self._peft_model.named_parameters():
            if param.requires_grad:
                print(f"  {name:80s}  [{param.shape}]")
