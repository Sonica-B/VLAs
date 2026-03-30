"""
ConservationBench evaluation module.

ConservationBench tests VLMs on physical conservation law reasoning:
  - Conservation of energy (kinetic + potential)
  - Conservation of momentum (before/after collision)
  - Conservation of mass (fluid systems)
  - Conservation of angular momentum (rotational dynamics)

Each test presents an image or image sequence and asks the model to reason
about whether a physical law is upheld or violated, and to quantify the
relevant conserved quantities.

Evaluation metrics:
  - Binary accuracy: correct identification of conservation law compliance (%)
  - Quantity estimation error: |predicted_value - ground_truth| / ground_truth (MAPE)

Reference: ConservationBench (2024).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

CONSERVATION_LAWS = [
    "energy",
    "momentum",
    "mass",
    "angular_momentum",
]


class ConservationBenchEvaluator:
    """Evaluates a VLM on ConservationBench.

    Args:
        dataset_path: Path to ConservationBench dataset directory.
        model: Loaded VLM model.
        processor: Model processor.
        device: Inference device.
        max_new_tokens: Max tokens for generation.

    Example:
        >>> evaluator = ConservationBenchEvaluator("data/conservation_bench/", model, processor)
        >>> results = evaluator.evaluate()
        >>> print(results["binary_accuracy"])
        >>> print(results["mean_quantity_mape"])
    """

    def __init__(
        self,
        dataset_path: str | Path,
        model: Any,
        processor: Any,
        device: str = "cuda",
        max_new_tokens: int = 64,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.model = model
        self.processor = processor
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._questions: List[Dict[str, Any]] = []

    def load_dataset(self) -> None:
        """Load ConservationBench questions from JSONL files."""
        jsonl_files = sorted(self.dataset_path.glob("*.jsonl"))
        for fpath in jsonl_files:
            with open(fpath, "r") as f:
                self._questions.extend(json.loads(line) for line in f if line.strip())
        logger.info(f"Loaded {len(self._questions)} ConservationBench questions")

    def _format_binary_prompt(self, question: Dict[str, Any]) -> str:
        """Format a conservation law compliance question."""
        law = question.get("conservation_law", "energy")
        return (
            f"Physical scenario: {question['description']}\n"
            f"Question: Is conservation of {law} upheld in this scenario?\n"
            f"Answer with 'yes' or 'no' and briefly explain why:"
        )

    def _format_quantity_prompt(self, question: Dict[str, Any]) -> str:
        """Format a quantity estimation question."""
        law = question.get("conservation_law", "energy")
        quantity_name = question.get("quantity_name", "total energy")
        return (
            f"Physical scenario: {question['description']}\n"
            f"Question: What is the {quantity_name} in this system? "
            f"Provide a numerical estimate with units."
        )

    def _parse_binary_answer(self, output: str) -> bool:
        """Parse yes/no answer from model output."""
        output_lower = output.strip().lower()
        if output_lower.startswith("yes"):
            return True
        if output_lower.startswith("no"):
            return False
        # Fallback: look for yes/no anywhere in first 50 chars
        for word in ["yes", "upheld", "conserved", "maintained"]:
            if word in output_lower[:50]:
                return True
        return False

    def _parse_quantity_answer(self, output: str) -> Optional[float]:
        """Attempt to parse a numerical quantity from model output.

        TODO: Implement robust numerical extraction (regex + unit normalization).
        """
        import re
        numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", output)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        return None

    def _compute_mape(self, predicted: Optional[float], ground_truth: float) -> Optional[float]:
        """Mean absolute percentage error for a single prediction."""
        if predicted is None or ground_truth == 0:
            return None
        return abs(predicted - ground_truth) / abs(ground_truth)

    def evaluate(
        self,
        max_questions: Optional[int] = None,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run ConservationBench evaluation.

        Returns:
            Dict with keys:
                - binary_accuracy: % correct yes/no conservation law classification
                - mean_quantity_mape: mean absolute % error on quantity estimation
                - per_law_binary_accuracy: Dict[str, float]
                - per_law_quantity_mape: Dict[str, float]
                - n_total: int
        """
        if not self._questions:
            self.load_dataset()

        questions = self._questions[:max_questions] if max_questions else self._questions
        binary_results = []
        quantity_mapes = []
        law_binary: Dict[str, List[bool]] = {law: [] for law in CONSERVATION_LAWS}
        law_mape: Dict[str, List[float]] = {law: [] for law in CONSERVATION_LAWS}

        for question in tqdm(questions, desc="Evaluating ConservationBench"):
            try:
                from PIL import Image
                image = Image.open(question["image_path"]).convert("RGB")
                law = question.get("conservation_law", "energy")
                task_type = question.get("task_type", "binary")

                if task_type == "binary":
                    prompt = self._format_binary_prompt(question)
                else:
                    prompt = self._format_quantity_prompt(question)

                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}

                import torch
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                    )
                output_text = self.processor.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )

                if task_type == "binary":
                    predicted_bool = self._parse_binary_answer(output_text)
                    gt_bool = bool(question.get("is_conserved", True))
                    is_correct = predicted_bool == gt_bool
                    binary_results.append(is_correct)
                    if law in law_binary:
                        law_binary[law].append(is_correct)
                else:
                    predicted_val = self._parse_quantity_answer(output_text)
                    gt_val = float(question.get("ground_truth_value", float("nan")))
                    mape = self._compute_mape(predicted_val, gt_val)
                    if mape is not None:
                        quantity_mapes.append(mape)
                        if law in law_mape:
                            law_mape[law].append(mape)

            except Exception as e:
                logger.warning(f"ConservationBench error: {e}")

        binary_accuracy = 100.0 * sum(binary_results) / max(len(binary_results), 1)
        mean_quantity_mape = float(np.mean(quantity_mapes)) if quantity_mapes else float("nan")

        per_law_binary = {
            law: 100.0 * sum(v) / max(len(v), 1)
            for law, v in law_binary.items()
            if v
        }
        per_law_mape = {
            law: float(np.mean(v))
            for law, v in law_mape.items()
            if v
        }

        results = {
            "binary_accuracy": binary_accuracy,
            "mean_quantity_mape": mean_quantity_mape,
            "per_law_binary_accuracy": per_law_binary,
            "per_law_quantity_mape": per_law_mape,
            "n_total": len(questions),
            "n_binary": len(binary_results),
            "n_quantity": len(quantity_mapes),
        }

        logger.info(
            f"ConservationBench — Binary accuracy: {binary_accuracy:.1f}%, "
            f"Quantity MAPE: {mean_quantity_mape:.3f}"
        )

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        return results
