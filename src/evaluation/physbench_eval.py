"""
PhysBench evaluation module.

PhysBench tests VLMs on physical commonsense reasoning across multiple categories:
  - Object stability and support
  - Collision outcome prediction
  - Material property inference
  - Fluid dynamics reasoning
  - Rigid body motion

Each question is multiple-choice (4 options). Evaluation metric: accuracy (%).

Reference: PhysBench benchmark (2024).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

PHYSBENCH_CATEGORIES = [
    "stability",
    "collision",
    "material_properties",
    "fluid_dynamics",
    "rigid_body_motion",
    "energy_conservation",
]


class PhysBenchEvaluator:
    """Evaluates a VLM on the PhysBench benchmark.

    Args:
        dataset_path: Path to the PhysBench dataset directory (or JSONL file).
        model: Loaded VLM model.
        processor: Model processor.
        batch_size: Number of questions to evaluate per batch.
        device: Inference device.
        max_new_tokens: Max tokens for the model's answer generation.

    Example:
        >>> evaluator = PhysBenchEvaluator("data/physbench/", model, processor)
        >>> results = evaluator.evaluate()
        >>> print(f"Overall accuracy: {results['overall_accuracy']:.1f}%")
    """

    def __init__(
        self,
        dataset_path: str | Path,
        model: Any,
        processor: Any,
        batch_size: int = 8,
        device: str = "cuda",
        max_new_tokens: int = 32,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.model = model
        self.processor = processor
        self.batch_size = batch_size
        self.device = device
        self.max_new_tokens = max_new_tokens

        self._questions: List[Dict[str, Any]] = []

    def load_dataset(self) -> None:
        """Load PhysBench questions from disk.

        Expected format per question (JSONL):
        {
            "id": "physbench_001",
            "category": "stability",
            "image_path": "...",
            "question": "Will this tower of blocks remain standing?",
            "choices": {"A": "Yes, ...", "B": "No, ...", "C": "...", "D": "..."},
            "answer": "B"
        }
        """
        if self.dataset_path.is_file():
            with open(self.dataset_path, "r") as f:
                self._questions = [json.loads(line) for line in f if line.strip()]
        elif self.dataset_path.is_dir():
            # Assume JSONL files per category
            for cat_file in sorted(self.dataset_path.glob("*.jsonl")):
                with open(cat_file, "r") as f:
                    self._questions.extend(json.loads(line) for line in f if line.strip())
        else:
            raise FileNotFoundError(f"PhysBench dataset not found at {self.dataset_path}")

        logger.info(f"Loaded {len(self._questions)} PhysBench questions")

    def _format_prompt(self, question: Dict[str, Any]) -> str:
        """Format a multiple-choice question as a prompt string."""
        choices_str = "\n".join(
            f"  {letter}. {text}" for letter, text in question["choices"].items()
        )
        return (
            f"Question: {question['question']}\n"
            f"Choices:\n{choices_str}\n"
            f"Answer with only the letter (A, B, C, or D):"
        )

    def _parse_answer(self, model_output: str) -> str:
        """Extract the answer letter from model output."""
        model_output = model_output.strip().upper()
        for letter in ["A", "B", "C", "D"]:
            if model_output.startswith(letter):
                return letter
        # Fallback: find first capital letter
        for char in model_output:
            if char in "ABCD":
                return char
        return "X"  # Invalid

    @staticmethod
    def _load_image(image_path: str) -> Any:
        """Load an image for VLM input."""
        from PIL import Image
        return Image.open(image_path).convert("RGB")

    def evaluate(
        self,
        max_questions: Optional[int] = None,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run evaluation on all PhysBench questions.

        Args:
            max_questions: If set, evaluate only the first N questions (for debugging).
            output_path: If set, save per-question results to this JSONL file.

        Returns:
            results: Dict with keys:
                - overall_accuracy: float (%)
                - per_category_accuracy: Dict[str, float]
                - per_question_results: List of result dicts
                - n_correct: int
                - n_total: int
        """
        if not self._questions:
            self.load_dataset()

        questions = self._questions[:max_questions] if max_questions else self._questions
        per_question_results = []
        category_counts: Dict[str, Dict[str, int]] = {
            cat: {"correct": 0, "total": 0} for cat in PHYSBENCH_CATEGORIES
        }
        n_correct = 0

        for question in tqdm(questions, desc="Evaluating PhysBench"):
            try:
                image = self._load_image(question["image_path"])
                prompt = self._format_prompt(question)

                # TODO: Implement model-specific prompt formatting and inference
                # This stub shows the interface; actual generation depends on model family.
                inputs = self.processor(text=prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items() if hasattr(v, "to")}

                import torch
                with torch.no_grad():
                    output_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                    )
                output_text = self.processor.decode(
                    output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
                )

                predicted = self._parse_answer(output_text)
                correct = predicted == question["answer"]

                if correct:
                    n_correct += 1

                cat = question.get("category", "unknown")
                if cat in category_counts:
                    category_counts[cat]["total"] += 1
                    if correct:
                        category_counts[cat]["correct"] += 1

                per_question_results.append({
                    "id": question.get("id"),
                    "category": cat,
                    "predicted": predicted,
                    "ground_truth": question["answer"],
                    "correct": correct,
                    "model_output": output_text[:200],
                })

            except Exception as e:
                logger.warning(f"Error on question {question.get('id')}: {e}")
                per_question_results.append({
                    "id": question.get("id"),
                    "error": str(e),
                    "correct": False,
                })

        n_total = len(questions)
        overall_accuracy = 100.0 * n_correct / max(n_total, 1)

        per_category_accuracy = {
            cat: (100.0 * v["correct"] / max(v["total"], 1))
            for cat, v in category_counts.items()
            if v["total"] > 0
        }

        results = {
            "overall_accuracy": overall_accuracy,
            "per_category_accuracy": per_category_accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_question_results": per_question_results,
        }

        logger.info(f"PhysBench overall accuracy: {overall_accuracy:.1f}% ({n_correct}/{n_total})")

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")

        return results
