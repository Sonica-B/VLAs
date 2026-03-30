"""
Per-patch activation extractor for VLMs.

Two classes are provided:

1. ``ActivationExtractor`` — full VLM extractor (Qwen2.5-VL / InternVL2.5 / LLaVA).
   Requires a GPU and the real VLM loaded via vlm_loader.load_vlm().
   Uses PyTorch forward hooks to capture intermediate representations at
   4 pipeline stages per model.

2. ``LightweightViTExtractor`` — CPU-friendly test extractor using
   ``google/vit-base-patch16-224``. No GPU needed. Used to validate the
   full probing pipeline (synthetic data → probe → saliency map) without
   loading a 7B model. Activations are extracted at ViT layers 3, 6, 9, 12
   and reported under the same 4 stage names so downstream code is unchanged.

Stage naming (both classes use the same keys):
    stage_1_enc_out    — early / vision encoder output
    stage_2_post_proj  — mid / post-projection (cross-modal space)
    stage_3_llm_8      — LLM layer 8 equivalent
    stage_4_llm_16     — LLM layer 16 / final

HDF5 Storage Schema (ActivationExtractor.save_to_hdf5):
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

# Stage names in order — used by both extractors
STAGE_NAMES = [
    "stage_1_enc_out",
    "stage_2_post_proj",
    "stage_3_llm_8",
    "stage_4_llm_16",
]


# ---------------------------------------------------------------------------
# Lightweight ViT extractor (CPU, no VLM required)
# ---------------------------------------------------------------------------

class LightweightViTExtractor:
    """CPU-compatible activation extractor using google/vit-base-patch16-224.

    Validates the full Phase 1 pipeline (synthetic data → activations → probe
    → saliency map) without requiring a GPU or large VLM weights.

    Activations are reported under the same 4 stage names as ActivationExtractor
    so all downstream code (PatchLabelAssigner, LinearProbe, PhysicsSaliencyMap)
    works without modification.

    Stage → ViT layer mapping:
        stage_1_enc_out   → hidden state after layer 3  (early features)
        stage_2_post_proj → hidden state after layer 6  (mid features)
        stage_3_llm_8     → hidden state after layer 9  (late features)
        stage_4_llm_16    → hidden state after layer 12 (final encoder output)

    Args:
        device: Torch device string. Default "cpu".
        model_id: HuggingFace model ID. Default "google/vit-base-patch16-224".

    Example:
        >>> extractor = LightweightViTExtractor()
        >>> extractor.load()
        >>> acts = extractor.extract(pil_image)
        >>> acts["stage_1_enc_out"].shape  # torch.Size([196, 768])
    """

    MODEL_ID = "google/vit-base-patch16-224"
    N_PATCHES = 196         # 14 × 14 patches for 224×224 input
    HIDDEN_DIM = 768
    PATCH_GRID_SIZE = 14
    model_name = "vit_base_patch16_224"  # for downstream labelling

    # hidden_states index (0 = patch embeddings, 1-12 = transformer layers)
    _STAGE_LAYER_IDX: Dict[str, int] = {
        "stage_1_enc_out": 3,
        "stage_2_post_proj": 6,
        "stage_3_llm_8": 9,
        "stage_4_llm_16": 12,
    }

    def __init__(
        self,
        device: str = "cpu",
        model_id: str = "google/vit-base-patch16-224",
    ) -> None:
        self.device = device
        self.MODEL_ID = model_id
        self._model: Optional[nn.Module] = None
        self._processor = None

    def load(self) -> None:
        """Download (first time) and load the ViT-base model into memory."""
        from transformers import ViTModel, ViTImageProcessor

        logger.info(f"Loading {self.MODEL_ID} on {self.device}...")
        self._processor = ViTImageProcessor.from_pretrained(self.MODEL_ID)
        self._model = ViTModel.from_pretrained(self.MODEL_ID)
        self._model.to(self.device)
        self._model.eval()
        logger.info(
            f"Loaded {self.MODEL_ID}: "
            f"{sum(p.numel() for p in self._model.parameters()) / 1e6:.1f}M params"
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @torch.no_grad()
    def extract(self, image: Image.Image) -> Dict[str, torch.Tensor]:
        """Extract activations at 4 pipeline stages for a single image.

        Args:
            image: PIL Image. Will be resized to 224×224 by the processor.

        Returns:
            Dict mapping stage name → float32 tensor of shape [196, 768].
            CLS token is excluded; only patch tokens are returned.
        """
        if not self.is_loaded:
            self.load()

        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self._model(**inputs, output_hidden_states=True)
        # hidden_states: tuple of length 13 (embedding + 12 layers), each [1, 197, 768]

        result: Dict[str, torch.Tensor] = {}
        for stage_name, layer_idx in self._STAGE_LAYER_IDX.items():
            hidden = outputs.hidden_states[layer_idx]   # [1, 197, 768]
            # Index 0 is CLS token; patch tokens are indices 1..196
            patches = hidden[0, 1:, :].float().cpu()    # [196, 768]
            result[stage_name] = patches

        return result

    def extract_batch(
        self, images: List[Image.Image]
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract activations for a list of images (sequential)."""
        return [self.extract(img) for img in images]

    def save_to_hdf5(
        self,
        activations: Dict[str, torch.Tensor],
        output_path: str | Path,
        scenario_id: str = "",
        physics_labels: Optional[Dict] = None,
    ) -> None:
        """Save activations to HDF5 (same format as ActivationExtractor)."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "w") as f:
            for stage_name, tensor in activations.items():
                arr = tensor.numpy() if isinstance(tensor, torch.Tensor) else tensor
                f.create_dataset(stage_name, data=arr, compression="gzip", compression_opts=4)
            f.attrs["scenario_id"] = scenario_id
            f.attrs["model_name"] = self.model_name
            f.attrs["patch_grid_size"] = self.PATCH_GRID_SIZE
            if physics_labels is not None:
                f.attrs["physics_labels"] = json.dumps(
                    {k: v.tolist() if hasattr(v, "tolist") else v for k, v in physics_labels.items()}
                )


# ---------------------------------------------------------------------------
# Full VLM extractor (GPU, real models)
# ---------------------------------------------------------------------------

class ActivationExtractor:
    """Extracts per-patch activations from a VLM at 4 pipeline stages.

    Uses PyTorch forward hooks to capture intermediate representations.
    Implements model-specific input preparation (chat templates) for each
    of the 3 supported VLM families.

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
    HOOK_CONFIGS: Dict[str, Dict[str, str]] = {
        "qwen2_5_vl_7b": {
            "stage_1_enc_out": "model.visual.blocks.31",
            "stage_2_post_proj": "model.visual.merger",
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
            if isinstance(output, tuple):
                tensor = output[0]
            else:
                tensor = output

            if tensor.dim() == 3:
                self._captured[stage_name] = tensor[0].detach().float().cpu()
            elif tensor.dim() == 2:
                self._captured[stage_name] = tensor.detach().float().cpu()
            else:
                logger.warning(f"Unexpected tensor shape at {stage_name}: {tensor.shape}")

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
                    f"Error: {e}"
                )
                raise

    def clear_hooks(self) -> None:
        """Remove all registered forward hooks and free captured tensors."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        for stage in STAGE_NAMES:
            self._captured[stage] = None
        if torch.cuda.is_available():
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
            text_prompt: Text prompt for the forward pass.

        Returns:
            Dict mapping stage name → float32 tensor [N_visual_tokens, D].
        """
        self.register_hooks()
        try:
            inputs = self._prepare_inputs(image, processor, text_prompt)
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            _ = self.model.generate(**inputs, max_new_tokens=1, do_sample=False)
        finally:
            self.clear_hooks()

        result: Dict[str, torch.Tensor] = {}
        for stage in STAGE_NAMES:
            captured = self._captured[stage]
            if captured is None:
                logger.warning(f"Stage {stage} was not captured — hook may not have fired.")
                continue
            result[stage] = self._slice_visual_tokens(captured, stage)

        return result

    # ------------------------------------------------------------------
    # Model-specific input preparation
    # ------------------------------------------------------------------

    def _prepare_inputs(
        self, image: Image.Image, processor: Any, text_prompt: str
    ) -> Dict[str, torch.Tensor]:
        """Prepare model inputs using the correct chat template per VLM family.

        Each VLM has its own expected input format:
        - Qwen2.5-VL: messages list with role/content dicts, apply_chat_template
        - InternVL 2.5-8B: <image>\\n{text} prompt, torchvision preprocessing
        - LLaVA-OneVision: conversation format with DEFAULT_IMAGE_TOKEN placeholder
        """
        if self.model_name == "qwen2_5_vl_7b":
            return self._prepare_qwen_inputs(image, processor, text_prompt)
        elif self.model_name == "internvl2_5_8b":
            return self._prepare_internvl_inputs(image, processor, text_prompt)
        elif self.model_name == "llava_onevision_7b":
            return self._prepare_llava_inputs(image, processor, text_prompt)
        else:
            # Fallback: try generic processor call
            return self._prepare_generic_inputs(image, processor, text_prompt)

    def _prepare_qwen_inputs(
        self, image: Image.Image, processor: Any, text_prompt: str
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for Qwen2.5-VL-7B-Instruct.

        Uses the multimodal messages format with apply_chat_template.
        Qwen2.5-VL processor expects:
            text: formatted chat string from apply_chat_template
            images: list of PIL images
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]
        try:
            # Newer transformers (>=4.49) support direct apply_chat_template
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
        except Exception:
            # Fallback: try qwen_vl_utils process_vision_info if available
            try:
                from qwen_vl_utils import process_vision_info
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
            except ImportError:
                logger.warning("qwen_vl_utils not available; using generic input prep")
                inputs = self._prepare_generic_inputs(image, processor, text_prompt)
        return inputs

    def _prepare_internvl_inputs(
        self, image: Image.Image, processor: Any, text_prompt: str
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for InternVL 2.5-8B.

        InternVL uses a custom preprocessing pipeline:
        - Images are normalized with ImageNet stats and resized to 448×448
        - Text uses <image>\\n prefix, and special <IMG_CONTEXT> tokens
          are inserted by the tokenizer during generation
        - processor here is actually a tokenizer (AutoTokenizer)

        Reference: https://huggingface.co/OpenGVLab/InternVL2_5-8B
        """
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        pixel_values = transform(image).unsqueeze(0)  # [1, 3, 448, 448]

        # InternVL uses <image> as image placeholder in the prompt
        question = f"<image>\n{text_prompt}"
        tokenized = processor(question, return_tensors="pt")

        inputs = {
            "pixel_values": pixel_values,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        return inputs

    def _prepare_llava_inputs(
        self, image: Image.Image, processor: Any, text_prompt: str
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for LLaVA-OneVision-7B.

        LLaVA expects:
        - A conversation with DEFAULT_IMAGE_TOKEN (<image>) in the text
        - apply_chat_template to format the conversation
        - processor handles both tokenization and image preprocessing

        Reference: https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov
        """
        DEFAULT_IMAGE_TOKEN = "<image>"

        conversation = [
            {
                "role": "user",
                "content": f"{DEFAULT_IMAGE_TOKEN}\n{text_prompt}",
            }
        ]

        try:
            text = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
        except Exception:
            # Fallback for older processor versions
            inputs = self._prepare_generic_inputs(image, processor, text_prompt)

        return inputs

    def _prepare_generic_inputs(
        self, image: Image.Image, processor: Any, text_prompt: str
    ) -> Dict[str, torch.Tensor]:
        """Generic fallback: call processor with text + image directly."""
        try:
            inputs = processor(
                text=text_prompt,
                images=image,
                return_tensors="pt",
            )
        except TypeError:
            # Some processors don't accept both text and images simultaneously
            inputs = processor(images=image, return_tensors="pt")
        return inputs

    # ------------------------------------------------------------------
    # Token slicing
    # ------------------------------------------------------------------

    def _slice_visual_tokens(
        self, tensor: torch.Tensor, stage: str
    ) -> torch.Tensor:
        """Extract visual token positions from a mixed visual+text sequence.

        At encoder stages (stage_1) all tokens are visual patches.
        At LLM stages (2-4), visual tokens precede text tokens; we use the
        first n_patches positions as an approximation.

        For production use, replace with actual visual_token_mask from processor
        outputs (e.g., image_token_mask in Qwen, pixel_values indices in LLaVA).
        """
        if stage == "stage_1_enc_out":
            return tensor

        if tensor.shape[0] >= self.n_patches:
            return tensor[: self.n_patches]
        return tensor

    # ------------------------------------------------------------------
    # Batch extraction & I/O
    # ------------------------------------------------------------------

    def extract_batch(
        self,
        images: List[Image.Image],
        processor: Any,
        text_prompt: str = "Describe the physics of this scene.",
    ) -> List[Dict[str, torch.Tensor]]:
        """Extract activations for a list of images (sequential forward passes)."""
        return [self.extract(img, processor, text_prompt) for img in images]

    def save_to_hdf5(
        self,
        activations: Dict[str, torch.Tensor],
        output_path: str | Path,
        scenario_id: str = "",
        physics_labels: Optional[Dict] = None,
    ) -> None:
        """Save extracted activations to an HDF5 file."""
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
