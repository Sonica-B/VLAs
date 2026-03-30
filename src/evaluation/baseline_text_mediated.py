"""
Baseline 2: Physics-Context Baseline (PCB) — text-mediated physics reasoning.

Inspired by the Physics-Context Baseline (PCB) approach in prior work on
vision-language physics understanding. The key idea: rather than asking the
model to infer physics properties from the image alone, we provide the
ground-truth physics description as text context alongside the image.

This tests the hypothesis:
  "If the VLM is given explicit physics properties as text, can it correctly
   reason about outcomes — and how much does this help over image-only input?"

Comparing PCB accuracy with the image-only baseline isolates how well the
VLM's language understanding (rather than vision encoding) handles physics.

Prompt format:
  [CONTEXT] The red sphere has mass=2.3 kg, friction=0.4, elasticity=0.7.
             The blue cube has mass=0.8 kg, friction=0.9, elasticity=0.3.
  [IMAGE]
  [QUESTION] Which object will travel farther when given the same push?
  [ANSWER] The blue cube (lower mass, higher friction works less but lower mass
           dominates) ...
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _format_physics_context(physics_labels: Dict[str, np.ndarray]) -> str:
    """Format physics properties as a human-readable context string.

    Args:
        physics_labels: dict of property → float32 array [N_objects].

    Returns:
        Multi-line string describing each object's properties.

    Example output:
        "Object 1: mass=1.24 kg, friction=0.63, elasticity=0.41.
         Object 2: mass=3.87 kg, friction=0.22, elasticity=0.78."
    """
    n_objects = max(len(v) for v in physics_labels.values())
    lines = []
    for i in range(n_objects):
        parts = []
        if "mass" in physics_labels and i < len(physics_labels["mass"]):
            parts.append(f"mass={physics_labels['mass'][i]:.2f} kg")
        if "friction" in physics_labels and i < len(physics_labels["friction"]):
            parts.append(f"friction={physics_labels['friction'][i]:.2f}")
        if "elasticity" in physics_labels and i < len(physics_labels["elasticity"]):
            parts.append(f"elasticity={physics_labels['elasticity'][i]:.2f}")
        lines.append(f"Object {i + 1}: {', '.join(parts)}.")
    return " ".join(lines)


def build_pcb_prompt(
    physics_context: str,
    question: str,
    model_name: str = "generic",
) -> str:
    """Build the text-mediated (PCB) prompt for a given model family.

    Args:
        physics_context: Formatted string of physics properties (from
            _format_physics_context).
        question: The physics question to answer.
        model_name: VLM family for chat template formatting.

    Returns:
        Formatted prompt string.
    """
    system_context = (
        "You are analyzing a physical scene. "
        "The following object properties have been measured precisely:\n"
        f"{physics_context}\n"
        "Use these values to answer the question accurately."
    )

    if "qwen" in model_name or "llava" in model_name:
        # These use a user/assistant conversation format
        return f"{system_context}\n\nQuestion: {question}\nAnswer:"
    elif "internvl" in model_name:
        return f"<context>{system_context}</context>\n{question}"
    else:
        return f"{system_context}\n\n{question}"


class TextMediatedBaseline:
    """Physics-Context Baseline (PCB): evaluate VLM with explicit physics text context.

    Provides ground-truth physics descriptions as additional text context
    alongside the image, then measures whether this improves physics QA accuracy.

    Args:
        model: Loaded VLM model (from vlm_loader.load_vlm).
        processor: Model processor.
        model_name: Short model name for prompt formatting.
        max_new_tokens: Maximum tokens to generate per answer.
        device: Inference device.

    Usage:
        >>> baseline = TextMediatedBaseline(model, processor, "qwen2_5_vl_7b")
        >>> result = baseline.evaluate_sample(image, physics_labels, question, answer)
        >>> accuracy = baseline.evaluate_dataset(samples)
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        model_name: str,
        max_new_tokens: int = 64,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device

    def _generate(self, image: Image.Image, prompt: str) -> str:
        """Run VLM generation and return the decoded response string."""
        import torch

        if "qwen" in self.model_name:
            inputs = self._prepare_qwen(image, prompt)
        elif "internvl" in self.model_name:
            inputs = self._prepare_internvl(image, prompt)
        else:
            inputs = self._prepare_generic(image, prompt)

        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens
        n_input = inputs["input_ids"].shape[1]
        generated = output_ids[0, n_input:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def _prepare_qwen(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(text=[text], images=[image], return_tensors="pt", padding=True)

    def _prepare_internvl(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        import torchvision.transforms as T
        from torchvision.transforms.functional import InterpolationMode
        import torch

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = transform(image).unsqueeze(0)
        question = f"<image>\n{prompt}"
        tokenized = self.processor(question, return_tensors="pt")
        return {**tokenized, "pixel_values": pixel_values}

    def _prepare_generic(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        return self.processor(text=prompt, images=image, return_tensors="pt")

    def evaluate_sample(
        self,
        image: Image.Image,
        physics_labels: Dict[str, np.ndarray],
        question: str,
        ground_truth_answer: str,
    ) -> Dict[str, Any]:
        """Run PCB evaluation on a single sample.

        Returns dict with keys: prompt, response, ground_truth, correct.
        Correctness is determined by checking if the ground_truth answer
        string appears in the model's response (case-insensitive substring match).
        """
        context = _format_physics_context(physics_labels)
        prompt = build_pcb_prompt(context, question, self.model_name)
        response = self._generate(image, prompt)

        correct = ground_truth_answer.lower() in response.lower()

        return {
            "context": context,
            "prompt": prompt,
            "response": response,
            "ground_truth": ground_truth_answer,
            "correct": correct,
        }

    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]],
        save_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run PCB evaluation over a list of samples.

        Each sample dict must have keys:
            image (PIL.Image), physics_labels (dict), question (str), answer (str).

        Returns:
            summary dict with overall_accuracy and per_sample_results.
        """
        per_sample_results = []
        n_correct = 0

        for i, sample in enumerate(samples):
            result = self.evaluate_sample(
                image=sample["image"],
                physics_labels=sample["physics_labels"],
                question=sample["question"],
                ground_truth_answer=sample["answer"],
            )
            per_sample_results.append(result)
            n_correct += int(result["correct"])

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  PCB eval: {i + 1}/{len(samples)} — "
                    f"running accuracy={n_correct / (i + 1):.3f}"
                )

        accuracy = n_correct / max(len(samples), 1)
        summary = {
            "baseline": "text_mediated_pcb",
            "model": self.model_name,
            "n_samples": len(samples),
            "n_correct": n_correct,
            "overall_accuracy": accuracy,
            "per_sample_results": per_sample_results,
        }

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                # Don't serialize large prompt strings to keep file small
                compact = {k: v for k, v in summary.items() if k != "per_sample_results"}
                compact["per_sample_correct"] = [r["correct"] for r in per_sample_results]
                json.dump(compact, f, indent=2)
            logger.info(f"PCB results saved to {save_path}")

        logger.info(
            f"PCB Baseline — {self.model_name}: "
            f"accuracy={accuracy:.3f} ({n_correct}/{len(samples)})"
        )
        return summary


def generate_pcb_qa_pairs(
    physics_labels_list: List[Dict[str, np.ndarray]],
    question_templates: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Generate PCB evaluation QA pairs from a list of physics label dicts.

    Creates questions that require comparing objects by their physics properties.
    Used to build evaluation sets without the full Physion++ dataset.

    Args:
        physics_labels_list: List of physics label dicts, one per scene.
        question_templates: Optional custom question templates.

    Returns:
        List of QA dicts with keys: question, answer, physics_labels, qa_type.
    """
    if question_templates is None:
        question_templates = [
            ("Which object has greater mass?", "mass", "max"),
            ("Which object has less friction?", "friction", "min"),
            ("Which object is more elastic (higher elasticity)?", "elasticity", "max"),
            ("Which object will bounce higher when dropped from the same height?", "elasticity", "max"),
            ("Which object is harder to push along a surface?", "friction", "max"),
        ]

    qa_pairs = []

    for physics_labels in physics_labels_list:
        n_objects = len(next(iter(physics_labels.values())))
        if n_objects < 2:
            continue

        for question, prop, direction in question_templates:
            if prop not in physics_labels:
                continue
            values = physics_labels[prop]
            obj_idx = int(np.argmax(values) if direction == "max" else np.argmin(values))
            answer = f"Object {obj_idx + 1}"

            qa_pairs.append({
                "question": question,
                "answer": answer,
                "physics_labels": physics_labels,
                "qa_type": "pcb_comparison",
                "property": prop,
            })

    return qa_pairs
