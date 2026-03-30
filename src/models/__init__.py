"""VLM loading, activation extraction, and LoRA wrapping utilities."""

from src.models.activation_extractor import ActivationExtractor

try:
    from src.models.vlm_loader import load_vlm
    from src.models.lora_wrapper import LoRAWrapper
    __all__ = ["load_vlm", "ActivationExtractor", "LoRAWrapper"]
except ImportError:
    # peft / full VLM deps not installed; lightweight extractor still works
    __all__ = ["ActivationExtractor"]
