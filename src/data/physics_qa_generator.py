"""
Physics QA pair generator from Physion++ simulation metadata.

Generates question-answer pairs suitable for LoRA fine-tuning VLMs on
physical reasoning tasks. Three QA types:
  1. Stability prediction — "Will this structure remain stable?"
  2. Property ordering — "Which object is heavier?"
  3. Dynamic outcome — "What will happen when these objects collide?"

Output format: JSONL with fields:
  {
    "id": "dominoes/trial_003/qa_001",
    "image_path": "data/physion/dominoes/trial_003/video/frame_0000.png",
    "question": "...",
    "answer": "...",
    "qa_type": "stability|ordering|dynamics",
    "physics_property": "mass|friction|elasticity|stability",
    "ground_truth_values": {...}
  }
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

from src.data.physion_loader import PhysionLoader, PhysionSample


# QA template banks
STABILITY_QUESTIONS = [
    "Will the objects in this scene remain in their current positions, or will they fall?",
    "Is this arrangement of objects stable? Answer with 'stable' or 'unstable'.",
    "Looking at this scene, do you think the objects will topple over?",
    "Will the stack of objects in this image remain standing?",
]

STABILITY_ANSWERS_STABLE = [
    "The arrangement appears stable. The objects are well-supported and unlikely to topple.",
    "Stable. The center of mass is well within the support base.",
    "The objects will remain in place. This configuration is stable.",
]

STABILITY_ANSWERS_UNSTABLE = [
    "The arrangement is unstable. The objects are likely to fall.",
    "Unstable. The upper objects are overhanging the support, and the system will topple.",
    "The objects will fall. The support base is insufficient for the current configuration.",
]

ORDERING_QUESTIONS_MASS = [
    "Looking at the {obj_a} and the {obj_b}, which one is heavier?",
    "If you were to pick up both the {obj_a} and the {obj_b}, which would weigh more?",
    "Compare the {obj_a} and the {obj_b}: which has greater mass?",
]

DYNAMICS_QUESTIONS = [
    "What will happen when the moving object collides with the stationary object?",
    "Based on the trajectory shown, predict the outcome of this interaction.",
    "If this scene continues to play out, what physical event will occur next?",
]


class PhysicsQAGenerator:
    """Generates physics QA pairs from Physion++ simulation metadata.

    Args:
        loader: A PhysionLoader instance providing access to the dataset.
        qa_types: Which QA types to generate. Defaults to all three.
        stability_threshold: Normalized velocity threshold for "unstable" label.
            Scenarios with mean object speed > threshold are labeled unstable.
        seed: Random seed for question template selection.
        max_pairs_per_scenario: Max QA pairs to generate per trial. Default 3.

    Example:
        >>> loader = PhysionLoader("data/physion", split="train")
        >>> generator = PhysicsQAGenerator(loader)
        >>> pairs = list(generator.generate(max_pairs=1000))
        >>> generator.save_jsonl(pairs, "data/physion/qa_train.jsonl")
    """

    def __init__(
        self,
        loader: PhysionLoader,
        qa_types: Optional[List[str]] = None,
        stability_threshold: float = 0.1,
        seed: int = 42,
        max_pairs_per_scenario: int = 3,
    ) -> None:
        self.loader = loader
        self.qa_types = qa_types or ["stability", "ordering", "dynamics"]
        self.stability_threshold = stability_threshold
        self.max_pairs_per_scenario = max_pairs_per_scenario
        random.seed(seed)
        np.random.seed(seed)

    def generate(
        self,
        max_pairs: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Generate QA pairs from the loader's dataset.

        Args:
            max_pairs: Stop after generating this many pairs. None = generate all.

        Yields:
            QA pair dicts in the format described in the module docstring.
        """
        n_generated = 0
        for sample in self.loader:
            if max_pairs is not None and n_generated >= max_pairs:
                return

            pairs = self._generate_for_sample(sample)
            for pair in pairs[: self.max_pairs_per_scenario]:
                if max_pairs is not None and n_generated >= max_pairs:
                    return
                yield pair
                n_generated += 1

    def _generate_for_sample(self, sample: PhysionSample) -> List[Dict[str, Any]]:
        """Generate all applicable QA pairs for a single sample."""
        pairs = []

        if "stability" in self.qa_types:
            stability_pair = self._make_stability_qa(sample)
            if stability_pair:
                pairs.append(stability_pair)

        if "ordering" in self.qa_types and sample.num_objects >= 2:
            ordering_pair = self._make_ordering_qa(sample, property_name="mass")
            if ordering_pair:
                pairs.append(ordering_pair)

        if "dynamics" in self.qa_types:
            dynamics_pair = self._make_dynamics_qa(sample)
            if dynamics_pair:
                pairs.append(dynamics_pair)

        return pairs

    def _make_stability_qa(self, sample: PhysionSample) -> Optional[Dict[str, Any]]:
        """Create a stability prediction QA pair."""
        stability_labels = sample.physics_labels.get("stability")
        if stability_labels is None:
            return None

        # Determine stability ground truth
        is_unstable = float(np.nanmean(stability_labels)) > self.stability_threshold
        answer_pool = STABILITY_ANSWERS_UNSTABLE if is_unstable else STABILITY_ANSWERS_STABLE
        gt_label = "unstable" if is_unstable else "stable"

        qa_id = f"{sample.scenario_id}/qa_stability_{sample.frame_idx:04d}"
        trial_dir = self.loader.root / sample.scenario_id.replace("/", "/")
        image_path = str(trial_dir / "video" / f"frame_{sample.frame_idx:04d}.png")

        return {
            "id": qa_id,
            "image_path": image_path,
            "question": random.choice(STABILITY_QUESTIONS),
            "answer": random.choice(answer_pool),
            "qa_type": "stability",
            "physics_property": "stability",
            "ground_truth_label": gt_label,
            "ground_truth_values": {"stability_score": float(np.nanmean(stability_labels))},
        }

    def _make_ordering_qa(
        self,
        sample: PhysionSample,
        property_name: str = "mass",
    ) -> Optional[Dict[str, Any]]:
        """Create a property-ordering QA pair (e.g., which object is heavier?)."""
        prop_labels = sample.physics_labels.get(property_name)
        if prop_labels is None or len(prop_labels) < 2:
            return None

        # Pick two objects with distinct property values
        valid_pairs = [
            (i, j)
            for i in range(len(prop_labels))
            for j in range(i + 1, len(prop_labels))
            if not (np.isnan(prop_labels[i]) or np.isnan(prop_labels[j]))
            and prop_labels[i] != prop_labels[j]
        ]

        if not valid_pairs:
            return None

        i, j = random.choice(valid_pairs)
        heavier_idx = i if prop_labels[i] > prop_labels[j] else j
        lighter_idx = j if heavier_idx == i else i

        question_template = random.choice(ORDERING_QUESTIONS_MASS)
        question = question_template.format(
            obj_a=f"object {i + 1}", obj_b=f"object {j + 1}"
        )
        answer = (
            f"Object {heavier_idx + 1} is heavier. "
            f"It has a mass of {prop_labels[heavier_idx]:.2f} units, "
            f"compared to {prop_labels[lighter_idx]:.2f} units for object {lighter_idx + 1}."
        )

        qa_id = f"{sample.scenario_id}/qa_ordering_{property_name}_{i}_{j}"
        trial_dir = self.loader.root / sample.scenario_id.replace("/", "/")
        image_path = str(trial_dir / "video" / f"frame_{sample.frame_idx:04d}.png")

        return {
            "id": qa_id,
            "image_path": image_path,
            "question": question,
            "answer": answer,
            "qa_type": "ordering",
            "physics_property": property_name,
            "ground_truth_values": {
                f"object_{i+1}_{property_name}": float(prop_labels[i]),
                f"object_{j+1}_{property_name}": float(prop_labels[j]),
            },
        }

    def _make_dynamics_qa(self, sample: PhysionSample) -> Optional[Dict[str, Any]]:
        """Create a dynamic outcome prediction QA pair."""
        # TODO: Implement dynamics QA generation using trajectory metadata.
        # Requires temporal data (positions over time) — use scenario-level data.
        # For now, return None until temporal loading is validated.
        return None

    @staticmethod
    def save_jsonl(pairs: List[Dict[str, Any]], output_path: str | Path) -> None:
        """Save QA pairs to a JSONL file.

        Args:
            pairs: List of QA pair dicts.
            output_path: Output file path (created if not exists).
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        print(f"Saved {len(pairs)} QA pairs to {output_path}")

    @staticmethod
    def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
        """Load QA pairs from a JSONL file."""
        path = Path(path)
        pairs = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    pairs.append(json.loads(line))
        return pairs
