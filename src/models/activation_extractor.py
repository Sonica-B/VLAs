"""
Per-patch activation extractor for VLMs.

Uses PyTorch forward hooks to capture intermediate representations at
4 pipeline stages per model:
  - Stage 1: Vision encoder output (pre-projection)
  - Stage 2: Post-projection (cross-modal token space)
  - Stage 3: LLM layer 8 hidden state
  - Stage 4: LLM layer 16 hidden state

Activations are returned as float32 tensors with shape [N_patches, D]
where N_patches is the number of visual tokens and D is the hidden dimension.

HDF5 Storage Schema:
    {scenario_id}.h5
    ├── stage_1_enc_out      [N_patches, D_enc]    float32
    ├── stage_2_post_proj    [N_patches, D_llm]    float32
    ├── stage_3_llm_8        [N_patches, D_llm]    float32
    ├── stage_4_llm_16       [N_patches, D_llm]    float32
    └── attrs:
        ├── scenario_id      str
        ├── model_name       str
        ├── patch_grid_size  int
        └── physics_labels   JSON string
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

logger = logging.getLogger(__name__)

# Stage names in order
STAGE_NAMES = [
    "stage_1_enc_out",
    "stage_2_post_proj",
    "stage_3_llm_8",
    "stage_4_llm_16",
]


class ActivationExtractor:
    """Extracts per-patch activations from a VLM at 4 pipeline stages.

    Args:
        model: The loaded VLM model (output of vlm_loader.load_vlm).
        model_name: Short model name, e.g., "qwen2_5_vl_7b".
        patch_grid_size: Expected spatial patch grid size (e.g., 14 for 14×14).
        device: Device to run inference on.

    Example:
        >>> extractor = ActivationExtractor(model, "qwen2_5_vl_7b")
        >>> acts = extractor.extract(image=pil_image, processor=processor)
        >>> print(acts["stage_1_enc_out"].shape)  # [196, 1280]
    """

    # Module path templates for each model family and stage
    # These are resolved via get_attr() walks on the model tree
    HOOK_CONFIGS: Dict[str, Dict[str, str]] = {
        "qwen2_5_vl_7b": {
            "stage_1_enc_out": "model.visual.blocks.31",        # Last ViT block
            "stage_2_post_proj": "model.visual.merger",          # MLP merger
            "stage_3_llm_8": "model.model.layers.8",
            "stage_4_llm_16": "model.model.layers.16",
        },
        "internvl2_5_8b": {
            "stage_1_enc_out": "vision_model.encoder.layers.23",
            "stage_2_post_proj": "mlp1",
            "stage_3_llm_8": "language_model.model.layers.8",
            "stage_4_llm_16": "language_model.model.layers.16",
        },
        "llava_onevision_7b": {
            "stage_1_enc_out": "model.vision_tower.vision_model.encoder.layers.26",
            "stage_2_post_proj": "model.mm_projector",
            "stage_3_llm_8": "model.language_model.model.layers.8",
            "stage_4_llm_16": "model.language_model.model.layers.16",
        },
    }

    def __init__(
        self,
        model: nn.Module,
        model_name: str,
        patch_grid_size: int = 14,
        device: str = "cuda",
    ) -> None:
        if model_name not in self.HOOK_CONFIGS:
            raise ValueError(
                f"Unknown model: {model_name!r}. "
                f"Available: {list(self.HOOK_CONFIGS.keys())}"
            )
        self.model = model
        self.model_name = model_name
        self.patch_grid_size = patch_grid_size
        self.device = device
        self.n_patches = patch_grid_size * patch_grid_size

        self._hooks: List[Any] = []
        self._captured: Dict[str, Optional[torch.Tensor]] = {s: None for s in STAGE_NAMES}

    def _get_module_by_path(self, path: str) -> nn.Module:
        """Resolve a dotted module path to the actual nn.Module."""
        parts = path.split(".")
        module = self.model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _make_hook(self, stage_name: str) -> Callable:
        """Create a forward hook that captures the output tensor for a stage."""
        def hook(module: nn.Module, input: Any, output: Any) -> None:
            # Output can be a tuple (hidden_state, ...) or a plain tensor
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            # Extract visual token positions
            # Shape at encoder stage: [B, N, D] — we want [N, D]
            if tensor.dim() == 3:
                # Take first batch element; clone to avoid holding graph
                self._captured[stage_name] = tensor[0].detach().float().cpu()
            elif tensor.dim() == 2:
                self._captured[stage_name] = tensor.detach().float().cpu()
            else:
                logger.warning(
                    f"Unexpected tensor shape at {stage_name}: {tensor.shape}"
                )

        return hook

    def register_hooks(self) -> None:
        """Register forward hooks on all 4 pipeline stages."""
        if self._hooks:
            raise RuntimeError("Hooks already registered. Call clear_hooks() first.")

        hook_config = self.HOOK_CONFIGS[self.model_name]
        for stage_name, module_path in hook_config.items():
            try:
                module = self._get_module_by_path(module_path)
                handle = module.register_forward_hook(self._make_hook(stage_name))
                self._hooks.append(handle)
                logger.debug(f"Registered hook: {stage_name} → {module_path}")
            except AttributeError as e:
                logger.error(
                    f"Could not find module '{module_path}' for stage {stage_name}. "
                    f"Model architecture may differ from config. Error: {e}"
                )
                raise

    def clear_hooks(self) -> None:
        """Remove all registered forward hooks and free captured tensors."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        for stage in STAGE_NAMES:
            self._captured[stage] = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def extract(
        self,
        image: Image.Image,
        processor: Any,
        text_prompt: str = "Describe the physics of this scene.",
    ) -> Dict[str, torch.Tensor]:
        """Run a forward pass and return per-patch activations at all 4 stages.

        Args:
            image: PIL Image to process.
            processor: Model processor (from vlm_loader.load_vlm).
            text_prompt: Text prompt to use for the forward pass. The prompt
                influences how the LLM processes visual tokens.

        Returns:
            Dict mapping stage name → float32 tensor of shape [N_visual_tokens, D].
            Note: N_visual_tokens may differ from patch_grid_size² if the model
            uses dynamic resolution or tiling.
        """
        self.register_hooks()
        try:
            # Prepare inputs — model-specific preprocessing
            inputs = self._prepare_inputs(image, processor, text_prompt)
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

            # Forward pass (generation mode with max_new_tokens=1 just to trigger full forward)
            _ = self.model.generate(**inputs, max_new_tokens=1, do_sample=False)

        finally:
            self.clear_hooks()

        # Validate all stages were captured
        result: Dict[str, torch.Tensor] = {}
        for stage in STAGE_NAMES:
            captured = self._captured[stage]
            if captured is None:
                logger.warning(f"Stage {stage} was not captured — hook may not have fired.")
                continue
            # Slice to first N_patches tokens if we got more (e.g., after tiling)
            result[stage] = self._slice_visual_tokens(captured, stage)

        return result

    def _prepare_inputs(
        self, image: Image.Image, processor: Any, text_prompt: str
    ) -> Dict[str, torch.Tensor]:
        """Prepare model inputs. Model-specific preprocessing handled here.

        TODO: Implement model-specific chat templates (Qwen/InternVL/LLaVA have
        different conversation formats). For now uses a generic format.
        """
        # TODO: Add model-specific chat template formatting
        # Qwen2.5-VL: uses apply_chat_template with role=user, content=[image, text]
        # InternVL: uses build_conversation_input_ids
        # LLaVA: uses apply_chat_template with DEFAULT_IMAGE_TOKEN
        inputs = processor(
            text=text_prompt,
            images=image,
            return_tensors="pt",
        )
        return inputs

    def _slice_visual_tokens(
        self, tensor: torch.Tensor, stage: str
    ) -> torch.Tensor:
        """Extract visual token positions from a mixed visual+text sequence.

        After projection, visual tokens are interleaved with text tokens in the LLM.
        This method attempts to isolate only the visual token positions.

        For simplicity in Stage 1 (encoder output), all tokens are visual.
        For Stages 2-4, we use the first N_visual_tokens positions.

        TODO: Use proper visual token position masks from the processor output
        rather than slicing by position.
        """
        # At encoder stage, all tokens are visual patches
        if stage == "stage_1_enc_out":
            return tensor

        # At later stages, the sequence starts with visual tokens
        # (BOS + visual tokens + separator + text tokens)
        # Use first N_patches as approximate visual token positions
        if tensor.shape[0] >= self.n_patches:
            return tensor[: self.n_patches]
        return tensor

    def extract_batch(
        self,
        images: List[Image.Image],
        processor: Any,
        text_prompt: str = "Describe the physics of this scene.",
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract activations for a batch of images (one forward pass per image).

        Args:
            images: List of PIL Images.
            processor: Model processor.
            text_prompt: Shared text prompt for all images.

        Returns:
            List of activation dicts, one per image.
        """
        # TODO: Implement true batched extraction for efficiency.
        # Currently runs one image at a time to avoid sequence length issues.
        results = []
        for image in images:
            acts = self.extract(image, processor, text_prompt)
            results.append(acts)
        return results

    def save_to_hdf5(
        self,
        activations: Dict[str, torch.Tensor],
        output_path: str | Path,
        scenario_id: str = "",
        physics_labels: Optional[Dict] = None,
    ) -> None:
        """Save extracted activations to an HDF5 file.

        Args:
            activations: Dict from extract() — stage_name → tensor.
            output_path: Output .h5 file path.
            scenario_id: Identifier for this sample (stored as attribute).
            physics_labels: Optional physics labels to store as JSON attribute.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with h5py.File(output_path, "w") as f:
            for stage_name, tensor in activations.items():
                arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                f.create_dataset(stage_name, data=arr, compression="gzip", compression_opts=4)

            f.attrs["scenario_id"] = scenario_id
            f.attrs["model_name"] = self.model_name
            f.attrs["patch_grid_size"] = self.patch_grid_size
            if physics_labels is not None:
                f.attrs["physics_labels"] = json.dumps(
                    {k: v.tolist() if hasattr(v, "tolist") else v for k, v in physics_labels.items()}
                )

    @staticmethod
    def load_from_hdf5(path: str | Path) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Load activations and metadata from an HDF5 file.

        Returns:
            (activations, attrs): Dict of stage → ndarray, and HDF5 attributes.
        """
        path = Path(path)
        activations = {}
        attrs = {}
        with h5py.File(path, "r") as f:
            for key in f.keys():
                activations[key] = f[key][:]
            attrs = dict(f.attrs)
        return activations, attrs
