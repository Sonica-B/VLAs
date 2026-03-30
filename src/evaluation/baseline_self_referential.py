"""
Baseline 3: Self-Referential Baseline — PhyCritic-style two-step reasoning.

Inspired by PhyCritic (self-critique with physical reasoning):
  Step 1 — Physics Elicitation: ask the VLM to describe/predict the physical
            properties of objects in the scene from the image alone.
  Step 2 — Conditioned Answering: feed the model's own Step-1 description
            back as context and ask the target physics question.

This tests whether VLMs can form a self-consistent physical world model:
  "Does the model's own physics predictions, when used as context,
   improve its downstream physical reasoning performance?"

Comparison against Baseline 2 (PCB with ground-truth context) isolates the
effect of prediction error in the elicited physics descriptions.

Two-step prompt structure:

  Step 1 prompt:
    "Look at this image and estimate the physical properties of each object:
     mass (in kg), surface friction (0–1), and elasticity (0–1).
     List each object with its estimated values."

  Step 2 prompt:
    [Step 1 response inserted as context]
    "Using your physics estimates above, answer:
     {question}"
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


_ELICITATION_PROMPT = (
    "Look at this image and carefully estimate the physical properties of each "
    "visible object. For each object, estimate:\n"
    "  - mass (kg): how heavy it appears relative to the others\n"
    "  - friction (0 to 1): how rough/sticky its surface looks\n"
    "  - elasticity (0 to 1): how bouncy/rigid the material appears\n\n"
    "Format your answer as:\n"
    "Object 1: mass=X, friction=Y, elasticity=Z\n"
    "Object 2: mass=X, friction=Y, elasticity=Z\n"
    "(and so on for all objects)\n\n"
    "Be specific with numerical estimates."
)

_CONDITIONED_ANSWER_TEMPLATE = (
    "Based on your physics estimates:\n{elicited_context}\n\n"
    "Now answer the following question using your estimates above:\n"
    "{question}\n\n"
    "Answer:"
)


def _parse_elicited_properties(elicited_text: str) -> Dict[str, np.ndarray]:
    """Parse elicited physics estimates from free-form model output.

    Extracts mass, friction, elasticity values for each object mentioned.
    Uses regex to find "Object N: mass=X, friction=Y, elasticity=Z" patterns.
    Falls back to NaN if parsing fails.

    Args:
        elicited_text: Raw text from the physics elicitation step.

    Returns:
        dict with keys mass, friction, elasticity, each a float32 array.
        Empty arrays if parsing completely fails.
    """
    obj_pattern = re.compile(
        r"[Oo]bject\s*(\d+)[^\n]*?mass\s*[=:]\s*([\d.]+)"
        r"[^\n]*?friction\s*[=:]\s*([\d.]+)"
        r"[^\n]*?elasticity\s*[=:]\s*([\d.]+)",
        re.IGNORECASE | re.DOTALL,
    )

    masses, frictions, elasticities = [], [], []
    for m in obj_pattern.finditer(elicited_text):
        try:
            masses.append(float(m.group(2)))
            frictions.append(float(m.group(3)))
            elasticities.append(float(m.group(4)))
        except ValueError:
            continue

    if not masses:
        logger.debug("Could not parse elicited physics from: %s", elicited_text[:200])
        return {
            "mass": np.array([], dtype=np.float32),
            "friction": np.array([], dtype=np.float32),
            "elasticity": np.array([], dtype=np.float32),
        }

    return {
        "mass": np.array(masses, dtype=np.float32),
        "friction": np.array(frictions, dtype=np.float32),
        "elasticity": np.array(elasticities, dtype=np.float32),
    }


def _compute_elicitation_error(
    elicited: Dict[str, np.ndarray],
    ground_truth: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute MAE between elicited and ground-truth physics properties.

    Properties are normalised to [0,1] range before comparison.

    Returns:
        dict of property → MAE (NaN if either is empty or shapes mismatch).
    """
    errors = {}
    for prop in ["mass", "friction", "elasticity"]:
        e = elicited.get(prop, np.array([]))
        g = ground_truth.get(prop, np.array([]))

        if len(e) == 0 or len(g) == 0 or len(e) != len(g):
            errors[prop] = float("nan")
            continue

        # Normalise both to [0,1] using ground-truth range for fair comparison
        g_range = g.max() - g.min()
        if g_range > 0:
            e_norm = np.clip((e - g.min()) / g_range, 0, 1)
            g_norm = (g - g.min()) / g_range
        else:
            e_norm = np.ones_like(e) * 0.5
            g_norm = np.ones_like(g) * 0.5

        errors[prop] = float(np.mean(np.abs(e_norm - g_norm)))

    return errors


class SelfReferentialBaseline:
    """PhyCritic-style two-step self-referential physics reasoning baseline.

    Step 1: Elicit physics property estimates from the VLM (image-only input).
    Step 2: Feed the model's own estimates as context and answer the question.

    Comparing this against Baseline 2 (PCB with ground-truth context) quantifies
    the gap between VLM-elicited and ground-truth physics descriptions.

    Args:
        model: Loaded VLM model (from vlm_loader.load_vlm).
        processor: Model processor.
        model_name: Short model name for chat template formatting.
        max_new_tokens_elicit: Tokens to generate for physics elicitation.
        max_new_tokens_answer: Tokens to generate for final answer.
        device: Inference device.
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        model_name: str,
        max_new_tokens_elicit: int = 256,
        max_new_tokens_answer: int = 128,
        device: str = "cuda",
    ) -> None:
        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.max_new_tokens_elicit = max_new_tokens_elicit
        self.max_new_tokens_answer = max_new_tokens_answer
        self.device = device

    def _generate(self, image: Image.Image, prompt: str, max_new_tokens: int) -> str:
        """Run VLM generation for a single image + prompt and return response."""
        import torch

        inputs = self._prepare_inputs(image, prompt)
        inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        n_input = inputs["input_ids"].shape[1]
        generated = output_ids[0, n_input:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def _prepare_inputs(self, image: Image.Image, prompt: str) -> Dict[str, Any]:
        """Model-specific input preparation (same pattern as activation_extractor)."""
        if "qwen" in self.model_name:
            messages = [{"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return self.processor(text=[text], images=[image], return_tensors="pt", padding=True)

        elif "internvl" in self.model_name:
            import torchvision.transforms as T
            from torchvision.transforms.functional import InterpolationMode
            transform = T.Compose([
                T.Lambda(lambda img: img.convert("RGB")),
                T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            pixel_values = transform(image).unsqueeze(0)
            tokenized = self.processor(f"<image>\n{prompt}", return_tensors="pt")
            return {**tokenized, "pixel_values": pixel_values}

        elif "llava" in self.model_name:
            conversation = [{"role": "user", "content": f"<image>\n{prompt}"}]
            text = self.processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            return self.processor(text=[text], images=[image], return_tensors="pt", padding=True)

        else:
            return self.processor(text=prompt, images=image, return_tensors="pt")

    def elicit_physics(self, image: Image.Image) -> Tuple[str, Dict[str, np.ndarray]]:
        """Step 1: Ask model to predict physics properties from image.

        Args:
            image: Scene image.

        Returns:
            (raw_response, parsed_labels): Raw generation and parsed dict.
        """
        raw_response = self._generate(image, _ELICITATION_PROMPT, self.max_new_tokens_elicit)
        parsed = _parse_elicited_properties(raw_response)
        return raw_response, parsed

    def answer_with_context(
        self,
        image: Image.Image,
        elicited_context: str,
        question: str,
    ) -> str:
        """Step 2: Answer physics question using elicited context.

        Args:
            image: Scene image (shown again in step 2).
            elicited_context: Raw text from step 1.
            question: The target physics question.

        Returns:
            Model response string.
        """
        prompt = _CONDITIONED_ANSWER_TEMPLATE.format(
            elicited_context=elicited_context.strip(),
            question=question,
        )
        return self._generate(image, prompt, self.max_new_tokens_answer)

    def evaluate_sample(
        self,
        image: Image.Image,
        physics_labels_gt: Dict[str, np.ndarray],
        question: str,
        ground_truth_answer: str,
    ) -> Dict[str, Any]:
        """Run full two-step evaluation on a single sample.

        Returns dict with:
            step1_response: Raw physics elicitation output
            step1_parsed:   Parsed physics estimates
            elicitation_error: MAE between estimates and ground truth
            step2_response: Final answer from conditioned generation
            ground_truth:   Expected answer
            correct:        Whether response contains ground truth (substring)
        """
        # Step 1
        step1_response, step1_parsed = self.elicit_physics(image)
        elicitation_error = _compute_elicitation_error(step1_parsed, physics_labels_gt)

        # Step 2
        step2_response = self.answer_with_context(image, step1_response, question)

        correct = ground_truth_answer.lower() in step2_response.lower()

        return {
            "step1_response": step1_response,
            "step1_parsed": {k: v.tolist() for k, v in step1_parsed.items()},
            "elicitation_error": elicitation_error,
            "step2_response": step2_response,
            "ground_truth": ground_truth_answer,
            "correct": correct,
        }

    def evaluate_dataset(
        self,
        samples: List[Dict[str, Any]],
        save_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run two-step evaluation over a list of samples.

        Each sample dict must have:
            image, physics_labels (ground truth), question, answer.

        Returns summary dict with overall_accuracy and elicitation quality stats.
        """
        per_sample_results = []
        n_correct = 0
        all_elicitation_errors: Dict[str, List[float]] = {
            "mass": [], "friction": [], "elasticity": []
        }

        for i, sample in enumerate(samples):
            result = self.evaluate_sample(
                image=sample["image"],
                physics_labels_gt=sample["physics_labels"],
                question=sample["question"],
                ground_truth_answer=sample["answer"],
            )
            per_sample_results.append(result)
            n_correct += int(result["correct"])

            for prop in ["mass", "friction", "elasticity"]:
                err = result["elicitation_error"].get(prop, float("nan"))
                if not np.isnan(err):
                    all_elicitation_errors[prop].append(err)

            if (i + 1) % 50 == 0:
                logger.info(
                    f"  Self-ref eval: {i + 1}/{len(samples)} — "
                    f"running accuracy={n_correct / (i + 1):.3f}"
                )

        accuracy = n_correct / max(len(samples), 1)
        mean_elicitation_errors = {
            prop: float(np.mean(v)) if v else float("nan")
            for prop, v in all_elicitation_errors.items()
        }

        summary = {
            "baseline": "self_referential_phycritic",
            "model": self.model_name,
            "n_samples": len(samples),
            "n_correct": n_correct,
            "overall_accuracy": accuracy,
            "mean_elicitation_mae": mean_elicitation_errors,
            "per_sample_results": per_sample_results,
        }

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            compact = {k: v for k, v in summary.items() if k != "per_sample_results"}
            compact["per_sample_correct"] = [r["correct"] for r in per_sample_results]
            compact["per_sample_elicitation_error"] = [
                r["elicitation_error"] for r in per_sample_results
            ]
            with open(save_path, "w") as f:
                json.dump(compact, f, indent=2)
            logger.info(f"Self-ref results saved to {save_path}")

        logger.info(
            f"Self-Ref Baseline — {self.model_name}: "
            f"accuracy={accuracy:.3f}  elicitation MAE={mean_elicitation_errors}"
        )
        return summary


def compare_baselines(
    pcb_results: Dict[str, Any],
    self_ref_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare PCB (ground-truth context) vs self-referential baseline.

    The gap measures how much the model's prediction error degrades performance.

    Returns:
        dict with accuracy_gap and interpretation string.
    """
    pcb_acc = pcb_results.get("overall_accuracy", 0.0)
    self_ref_acc = self_ref_results.get("overall_accuracy", 0.0)
    gap = pcb_acc - self_ref_acc

    if gap > 0.10:
        interpretation = (
            "Large accuracy gap: VLM physics elicitation is significantly noisy. "
            "Spatial encoding bottleneck is likely in the projection or early LLM layers."
        )
    elif gap > 0.03:
        interpretation = (
            "Moderate gap: Some elicitation error but model captures rough physics trends."
        )
    else:
        interpretation = (
            "Small gap: VLM elicitation closely approximates ground truth context. "
            "Model has strong physics understanding from vision alone."
        )

    return {
        "pcb_accuracy": pcb_acc,
        "self_ref_accuracy": self_ref_acc,
        "accuracy_gap": gap,
        "interpretation": interpretation,
    }
