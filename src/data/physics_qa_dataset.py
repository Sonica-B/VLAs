"""
PyTorch Dataset for physics QA fine-tuning with HuggingFace Trainer.

Loads physics QA samples from JSONL files (produced by PhysicsQAGenerator)
and formats each sample using the appropriate VLM chat template. Returns
tokenized tensors with labels masked on the question tokens (standard SFT
/ causal-LM convention: only compute loss on the answer tokens).

JSONL format (one object per line):
    {
        "id": "q_000001",
        "image_path": "data/physion/dominoes/trial_003/video/frame_0015.png",
        "question": "Which object will fall first?",
        "answer": "The red sphere (higher mass, same height).",
        "qa_type": "stability",
        "physics_property": "mass",
        "ground_truth_values": {"mass": [1.2, 3.4], "friction": [0.5, 0.5]}
    }

Usage:
    from src.data.physics_qa_dataset import PhysicsQADataset, get_collator
    dataset = PhysicsQADataset("data/physion/qa_train.jsonl", processor, "qwen2_5_vl_7b")
    collator = get_collator(processor.tokenizer)
    trainer = Trainer(model=peft_model, train_dataset=dataset, data_collator=collator, ...)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Label mask value: positions with this id are ignored in cross-entropy loss
IGNORE_INDEX = -100


class PhysicsQADataset(Dataset):
    """Dataset for physics QA SFT training.

    Loads from JSONL, applies the model-specific chat template, tokenizes,
    and builds per-token labels (question tokens masked with IGNORE_INDEX).

    Args:
        jsonl_path: Path to the QA JSONL file.
        processor: Model processor (handles both tokenization and image preprocessing).
        model_name: Short model name for selecting the correct chat template.
        max_seq_len: Maximum total sequence length. Longer samples are truncated.
        image_root: Root directory for resolving relative image paths.
            If None, uses the JSONL file's parent directory.
        pad_to_max: If True, pad all sequences to max_seq_len. Recommended False
            for variable-length batches (use data_collator instead).
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        processor: Any,
        model_name: str,
        max_seq_len: int = 2048,
        image_root: Optional[str | Path] = None,
        pad_to_max: bool = False,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.processor = processor
        self.model_name = model_name
        self.max_seq_len = max_seq_len
        self.image_root = Path(image_root) if image_root else self.jsonl_path.parent
        self.pad_to_max = pad_to_max

        self._samples: List[Dict[str, Any]] = self._load_jsonl()
        logger.info(
            f"PhysicsQADataset: loaded {len(self._samples)} samples "
            f"from {self.jsonl_path}"
        )

    def _load_jsonl(self) -> List[Dict[str, Any]]:
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"QA JSONL not found: {self.jsonl_path}")
        samples = []
        with open(self.jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self._samples[idx]
        image = self._load_image(sample["image_path"])
        question = sample["question"]
        answer = sample["answer"]

        if "qwen" in self.model_name:
            return self._format_qwen(image, question, answer)
        elif "internvl" in self.model_name:
            return self._format_internvl(image, question, answer)
        elif "llava" in self.model_name:
            return self._format_llava(image, question, answer)
        else:
            return self._format_generic(image, question, answer)

    def _load_image(self, image_path: str) -> Image.Image:
        p = Path(image_path)
        if not p.is_absolute():
            p = self.image_root / p
        return Image.open(p).convert("RGB")

    # ------------------------------------------------------------------
    # Model-specific formatting helpers
    # ------------------------------------------------------------------

    def _build_labels(
        self, input_ids: torch.Tensor, answer_start_token: int
    ) -> torch.Tensor:
        """Build labels tensor: IGNORE_INDEX for all tokens before the answer.

        Finds the position where the answer begins by locating the first
        occurrence of answer_start_token (e.g., "assistant" role token)
        and masks all tokens before that position.
        """
        labels = input_ids.clone()
        # Mask everything up to and including the answer start token
        answer_positions = (input_ids == answer_start_token).nonzero(as_tuple=True)[0]
        if len(answer_positions) > 0:
            mask_end = int(answer_positions[-1].item()) + 1
            labels[:mask_end] = IGNORE_INDEX
        else:
            # Fallback: mask first 60% of sequence as a rough question mask
            mask_end = int(len(input_ids) * 0.6)
            labels[:mask_end] = IGNORE_INDEX
        return labels

    def _truncate(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[-1] > self.max_seq_len:
            return tensor[..., : self.max_seq_len]
        return tensor

    def _format_qwen(
        self, image: Image.Image, question: str, answer: str
    ) -> Dict[str, torch.Tensor]:
        """Format for Qwen2.5-VL SFT.

        The full conversation (question + answer) is tokenized together.
        The assistant turn start token separates prompt from answer labels.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            },
            {
                "role": "assistant",
                "content": answer,
            },
        ]

        # tokenize=True returns full input_ids for SFT
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=False,
        )

        input_ids = self._truncate(inputs["input_ids"][0])
        attention_mask = self._truncate(inputs["attention_mask"][0])
        pixel_values = inputs.get("pixel_values")

        # Find the answer start: look for token IDs that map to "assistant"
        # In Qwen2-tokenizer, the assistant marker is typically encoded as a
        # special token. Use a heuristic: find the last "\n" before the answer.
        # A more precise approach would use the tokenizer's chat template offsets.
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Re-tokenize just the question portion to find where answer starts
        q_messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]
        q_text = self.processor.apply_chat_template(
            q_messages, tokenize=False, add_generation_prompt=True
        )
        q_inputs = self.processor(text=[q_text], images=[image], return_tensors="pt", padding=False)
        q_len = min(q_inputs["input_ids"].shape[1], input_ids.shape[0])
        labels[q_len:] = input_ids[q_len:]

        result: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values[0] if pixel_values.dim() == 4 else pixel_values
        for key in ("image_grid_thw", "image_sizes"):
            if key in inputs:
                result[key] = inputs[key][0] if inputs[key].dim() > 1 else inputs[key]
        return result

    def _format_internvl(
        self, image: Image.Image, question: str, answer: str
    ) -> Dict[str, torch.Tensor]:
        """Format for InternVL2.5 SFT.

        Uses torchvision preprocessing and builds prompt with <image> token.
        """
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(image)  # [3, 448, 448]

        # InternVL SFT format: <image>\nQ: {question}\nA: {answer}
        full_text = f"<image>\nQuestion: {question}\nAnswer: {answer}"
        question_text = f"<image>\nQuestion: {question}\nAnswer:"

        full_ids = self.processor(full_text, return_tensors="pt")["input_ids"][0]
        q_ids = self.processor(question_text, return_tensors="pt")["input_ids"][0]

        full_ids = self._truncate(full_ids)
        q_len = min(len(q_ids), len(full_ids))
        attention_mask = torch.ones_like(full_ids)

        labels = full_ids.clone()
        labels[:q_len] = IGNORE_INDEX

        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

    def _format_llava(
        self, image: Image.Image, question: str, answer: str
    ) -> Dict[str, torch.Tensor]:
        """Format for LLaVA-OneVision SFT."""
        DEFAULT_IMAGE_TOKEN = "<image>"

        # Build full conversation (question + answer) for SFT
        full_conversation = [
            {"role": "user", "content": f"{DEFAULT_IMAGE_TOKEN}\n{question}"},
            {"role": "assistant", "content": answer},
        ]
        full_text = self.processor.apply_chat_template(
            full_conversation, tokenize=False, add_generation_prompt=False
        )

        # Build prompt-only for finding answer start position
        prompt_conversation = [
            {"role": "user", "content": f"{DEFAULT_IMAGE_TOKEN}\n{question}"},
        ]
        prompt_text = self.processor.apply_chat_template(
            prompt_conversation, tokenize=False, add_generation_prompt=True
        )

        full_inputs = self.processor(
            text=[full_text], images=[image], return_tensors="pt", padding=False
        )
        prompt_inputs = self.processor(
            text=[prompt_text], images=[image], return_tensors="pt", padding=False
        )

        input_ids = self._truncate(full_inputs["input_ids"][0])
        attention_mask = self._truncate(full_inputs["attention_mask"][0])
        q_len = min(prompt_inputs["input_ids"].shape[1], input_ids.shape[0])

        labels = input_ids.clone()
        labels[:q_len] = IGNORE_INDEX

        result: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if "pixel_values" in full_inputs:
            result["pixel_values"] = full_inputs["pixel_values"][0]
        if "image_sizes" in full_inputs:
            result["image_sizes"] = full_inputs["image_sizes"][0]
        return result

    def _format_generic(
        self, image: Image.Image, question: str, answer: str
    ) -> Dict[str, torch.Tensor]:
        """Generic fallback formatter for unknown model families."""
        full_text = f"Question: {question}\nAnswer: {answer}"
        prompt_text = f"Question: {question}\nAnswer:"

        full_inputs = self.processor(text=full_text, images=image, return_tensors="pt")
        prompt_inputs = self.processor(text=prompt_text, images=image, return_tensors="pt")

        input_ids = self._truncate(full_inputs["input_ids"][0])
        attention_mask = self._truncate(full_inputs["attention_mask"][0])
        q_len = min(prompt_inputs["input_ids"].shape[1], input_ids.shape[0])

        labels = input_ids.clone()
        labels[:q_len] = IGNORE_INDEX

        result: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        for key in ("pixel_values", "image_sizes"):
            if key in full_inputs:
                result[key] = full_inputs[key][0]
        return result


class DataCollatorForVLMSFT:
    """Data collator for variable-length VLM SFT batches.

    Pads input_ids, attention_mask, and labels to the longest sequence in
    the batch. Handles optional pixel_values (stacked or left as-is if
    variable size).

    Args:
        pad_token_id: Token ID used for padding input_ids. Default 0.
        label_pad_token_id: Value used for padding labels. Default IGNORE_INDEX.
    """

    def __init__(
        self,
        pad_token_id: int = 0,
        label_pad_token_id: int = IGNORE_INDEX,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Find max length in batch
        max_len = max(f["input_ids"].shape[0] for f in features)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for f in features:
            seq_len = f["input_ids"].shape[0]
            pad_len = max_len - seq_len

            batch_input_ids.append(
                torch.cat([f["input_ids"],
                           torch.full((pad_len,), self.pad_token_id, dtype=torch.long)])
            )
            batch_attention_mask.append(
                torch.cat([f["attention_mask"],
                           torch.zeros(pad_len, dtype=torch.long)])
            )
            batch_labels.append(
                torch.cat([f["labels"],
                           torch.full((pad_len,), self.label_pad_token_id, dtype=torch.long)])
            )

        result: Dict[str, torch.Tensor] = {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels),
        }

        # Stack pixel_values if all same shape (single-image, single-tile case)
        if "pixel_values" in features[0]:
            try:
                result["pixel_values"] = torch.stack([f["pixel_values"] for f in features])
            except RuntimeError:
                # Variable-size pixel_values (tiled inputs) — pass as list
                result["pixel_values"] = [f["pixel_values"] for f in features]

        # Pass through any other keys (image_grid_thw, image_sizes, etc.)
        for key in features[0]:
            if key not in result:
                try:
                    result[key] = torch.stack([f[key] for f in features])
                except (RuntimeError, TypeError):
                    result[key] = [f[key] for f in features]

        return result


def get_collator(processor_or_tokenizer: Any) -> DataCollatorForVLMSFT:
    """Create a DataCollatorForVLMSFT using the processor's pad token id."""
    pad_id = 0
    tokenizer = getattr(processor_or_tokenizer, "tokenizer", processor_or_tokenizer)
    if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
        pad_id = tokenizer.pad_token_id
    return DataCollatorForVLMSFT(pad_token_id=pad_id)
