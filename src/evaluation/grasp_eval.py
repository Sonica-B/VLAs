"""
GRASP Level 2 evaluation module.

GRASP (Grid-based Reasoning and Scene Physics) Level 2 focuses on physical
scene understanding tasks that require spatial reasoning about object properties
and interactions — going beyond simple recognition to compositional reasoning.

Level 2 specifically tests:
  - Inferring hidden physical properties from observed dynamics
  - Predicting multi-step physical interactions
  - Reasoning about counterfactual physics scenarios

Evaluation metric: per-task accuracy (%).

Reference: GRASP benchmark (2024).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

logger = logging.getLogger(__name__)

GRASP_L2_SUBTASKS = [
    "hidden_property_inference",
    "multi_step_prediction",
    "counterfactual_physics",
    "spatial_physics_reasoning",
]


class GRASPEvaluator:
    """Evaluates a VLM on GRASP Level 2 tasks.

    Args:
        dataset_path: Path to GRASP Level 2 dataset directory.
        model: Loaded VLM model.
        processor: Model processor.
        device: Inference device.
        max_new_tokens: Max tokens for generation.

    Example:
        >>> evaluator = GRASPEvaluator("data/grasp_l2/", model, processor)
        >>> results = evaluator.evaluate()
        >>> print(results["overall_accuracy"])
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
        """Load GRASP Level 2 questions from disk."""
        jsonl_files = sorted(self.dataset_path.glob("level2_*.jsonl"))
        for fpath in jsonl_files:
            with open(fpath, "r") as f:
                self._questions.extend(json.loads(line) for line in f if line.strip())
        logger.info(f"Loaded {len(self._questions)} GRASP L2 questions")

    def _format_prompt(self, question: Dict[str, Any]) -> str:
        """Format a GRASP L2 question as a prompt.

        GRASP L2 may use open-ended or multiple-choice format depending on subtask.
        """
        q_type = question.get("type", "multiple_choice")
        if q_type == "multiple_choice":
            choices_str = "\n".join(
                f"  {k}. {v}" for k, v in question.get("choices", {}).items()
            )
            return (
                f"Question: {question['question']}\n"
                f"Choices:\n{choices_str}\n"
                f"Answer with only the letter:"
            )
        else:
            return f"Question: {question['question']}\nAnswer:"

    def _parse_answer(self, output: str, q_type: str = "multiple_choice") -> str:
        """Parse model output to extract the answer."""
        output = output.strip()
        if q_type == "multiple_choice":
            for letter in ["A", "B", "C", "D"]:
                if output.upper().startswith(letter):
                    return letter
            for char in output.upper():
                if char in "ABCD":
                    return char
            return "X"
        else:
            return output  # Return full text for open-ended

    def evaluate(
        self,
        max_questions: Optional[int] = None,
        output_path: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Run GRASP Level 2 evaluation.

        Returns:
            Dict with overall_accuracy, per_subtask_accuracy, n_correct, n_total.
        """
        if not self._questions:
            self.load_dataset()

        questions = self._questions[:max_questions] if max_questions else self._questions
        per_question_results = []
        subtask_counts: Dict[str, Dict[str, int]] = {
            s: {"correct": 0, "total": 0} for s in GRASP_L2_SUBTASKS
        }
        n_correct = 0

        for question in tqdm(questions, desc="Evaluating GRASP L2"):
            try:
                from PIL import Image
                image = Image.open(question["image_path"]).convert("RGB")
                q_type = question.get("type", "multiple_choice")
                prompt = self._format_prompt(question)

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

                predicted = self._parse_answer(output_text, q_type)
                gt = question.get("answer", "")

                if q_type == "multiple_choice":
                    correct = predicted.upper() == gt.upper()
                else:
                    # TODO: Implement open-ended answer matching (exact match or LLM judge)
                    correct = predicted.lower().strip() == gt.lower().strip()

                if correct:
                    n_correct += 1

                subtask = question.get("subtask", "unknown")
                if subtask in subtask_counts:
                    subtask_counts[subtask]["total"] += 1
                    if correct:
                        subtask_counts[subtask]["correct"] += 1

                per_question_results.append({
                    "id": question.get("id"),
                    "subtask": subtask,
                    "predicted": predicted,
                    "ground_truth": gt,
                    "correct": correct,
                })

            except Exception as e:
                logger.warning(f"GRASP error on {question.get('id')}: {e}")

        n_total = len(questions)
        overall_accuracy = 100.0 * n_correct / max(n_total, 1)
        per_subtask_accuracy = {
            s: 100.0 * v["correct"] / max(v["total"], 1)
            for s, v in subtask_counts.items()
            if v["total"] > 0
        }

        results = {
            "overall_accuracy": overall_accuracy,
            "per_subtask_accuracy": per_subtask_accuracy,
            "n_correct": n_correct,
            "n_total": n_total,
            "per_question_results": per_question_results,
        }

        logger.info(f"GRASP L2 overall accuracy: {overall_accuracy:.1f}%")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

        return results
