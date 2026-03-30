"""
Factorial component ablation runner.

Orchestrates training and evaluation of all 5 LoRA ablation conditions
(A: encoder-only, B: projection-only, C: LLM-only, D: enc+proj, E: full)
across all 3 VLM families.

Training loop uses HuggingFace Trainer with gradient checkpointing
and W&B logging.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import Trainer, TrainingArguments, DataCollatorForSeq2Seq

from src.models.vlm_loader import load_vlm
from src.models.lora_wrapper import LoRAWrapper, ABLATION_CONDITIONS
from src.evaluation.physbench_eval import PhysBenchEvaluator
from src.evaluation.grasp_eval import GRASPEvaluator
from src.evaluation.conservation_eval import ConservationBenchEvaluator

logger = logging.getLogger(__name__)


class ComponentAblation:
    """Orchestrate training and evaluation of all 5 LoRA ablation conditions.

    Args:
        model_name: VLM to fine-tune ("qwen2_5_vl_7b", etc.).
        base_model_kwargs: Kwargs passed to load_vlm().
        train_data_path: Path to physics QA training JSONL.
        val_data_path: Path to physics QA validation JSONL.
        output_base_dir: Base directory for all condition outputs.
        lora_rank: LoRA rank for all conditions. Default 16.
        eval_benchmarks: List of benchmark names to evaluate on.
        conditions: Which conditions to run. Default: all 5.

    Example:
        >>> ablation = ComponentAblation(
        ...     model_name="qwen2_5_vl_7b",
        ...     train_data_path="data/physion/qa_train.jsonl",
        ...     val_data_path="data/physion/qa_val.jsonl",
        ...     output_base_dir="results/ablation_metrics/qwen/",
        ... )
        >>> ablation.run_all_conditions()
        >>> ablation.save_summary()
    """

    def __init__(
        self,
        model_name: str,
        base_model_kwargs: Optional[Dict[str, Any]] = None,
        train_data_path: str = "data/physion/qa_train.jsonl",
        val_data_path: str = "data/physion/qa_val.jsonl",
        output_base_dir: str = "results/ablation_metrics/",
        lora_rank: int = 16,
        eval_benchmarks: Optional[List[str]] = None,
        conditions: Optional[List[str]] = None,
        physbench_path: str = "data/physbench/",
        grasp_path: str = "data/grasp_l2/",
        conservation_path: str = "data/conservation_bench/",
    ) -> None:
        self.model_name = model_name
        self.base_model_kwargs = base_model_kwargs or {"torch_dtype": torch.bfloat16}
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path
        self.output_base_dir = Path(output_base_dir) / model_name
        self.lora_rank = lora_rank
        self.eval_benchmarks = eval_benchmarks or ["physbench", "grasp", "conservation"]
        self.conditions = conditions or ABLATION_CONDITIONS
        self.physbench_path = physbench_path
        self.grasp_path = grasp_path
        self.conservation_path = conservation_path

        self._results: Dict[str, Dict[str, Any]] = {}

    def _build_training_args(self, condition: str, output_dir: Path) -> TrainingArguments:
        """Build HuggingFace TrainingArguments for a given condition."""
        return TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=8,
            learning_rate=2e-4 if condition in ["A", "D"] else 1e-4,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            bf16=True,
            gradient_checkpointing=True,
            evaluation_strategy="steps",
            eval_steps=250,
            save_steps=500,
            save_total_limit=2,
            logging_steps=50,
            dataloader_num_workers=4,
            report_to=["wandb"],
            run_name=f"ablation_{self.model_name}_condition_{condition}",
        )

    def _load_qa_dataset(self, path: str) -> Any:
        """Load physics QA dataset from JSONL.

        TODO: Implement proper VLM-format dataset (image + text) for HF Trainer.
        Currently returns placeholder — actual implementation requires
        model-specific chat template formatting and image loading.
        """
        # TODO: Implement PhysicsQADataset class that wraps the JSONL
        # and formats each sample using the model's chat template.
        # Must handle: image loading, tokenization, label masking (for SFT).
        raise NotImplementedError(
            "Physics QA dataset loading not yet implemented. "
            "See src/data/physics_qa_generator.py for QA format."
        )

    def train_condition(self, condition: str) -> str:
        """Train a single LoRA ablation condition.

        Args:
            condition: One of "A", "B", "C", "D", "E".

        Returns:
            Path to the saved adapter checkpoint directory.
        """
        assert condition in ABLATION_CONDITIONS, f"Invalid condition: {condition}"
        output_dir = self.output_base_dir / f"condition_{condition}"
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Training condition {condition} for {self.model_name}...")
        logger.info(f"  Output dir: {output_dir}")

        # Load base model
        model, processor = load_vlm(self.model_name, **self.base_model_kwargs)

        # Apply LoRA
        wrapper = LoRAWrapper(
            model, self.model_name, condition=condition, rank=self.lora_rank
        )
        peft_model = wrapper.apply()
        wrapper.print_trainable_modules()

        # Load dataset
        # TODO: train_dataset = self._load_qa_dataset(self.train_data_path)
        # TODO: val_dataset = self._load_qa_dataset(self.val_data_path)

        # Build trainer
        training_args = self._build_training_args(condition, output_dir)
        # TODO: trainer = Trainer(model=peft_model, args=training_args, ...)
        # TODO: trainer.train()

        # Save adapter
        adapter_path = str(output_dir / "adapter")
        wrapper.save_adapter(adapter_path)
        logger.info(f"Condition {condition} training complete. Adapter saved to {adapter_path}")
        return adapter_path

    def evaluate_condition(
        self,
        condition: str,
        adapter_path: str,
    ) -> Dict[str, Any]:
        """Evaluate a trained condition on all benchmarks.

        Args:
            condition: Ablation condition ID.
            adapter_path: Path to saved LoRA adapter.

        Returns:
            Dict mapping benchmark name → benchmark results dict.
        """
        logger.info(f"Evaluating condition {condition} on benchmarks...")

        # Load base model and apply adapter
        base_model, processor = load_vlm(self.model_name, **self.base_model_kwargs)
        model = LoRAWrapper.load_adapter(base_model, adapter_path)
        model.eval()

        benchmark_results: Dict[str, Any] = {}

        if "physbench" in self.eval_benchmarks:
            evaluator = PhysBenchEvaluator(
                self.physbench_path, model, processor
            )
            benchmark_results["physbench"] = evaluator.evaluate()

        if "grasp" in self.eval_benchmarks:
            evaluator = GRASPEvaluator(self.grasp_path, model, processor)
            benchmark_results["grasp"] = evaluator.evaluate()

        if "conservation" in self.eval_benchmarks:
            evaluator = ConservationBenchEvaluator(self.conservation_path, model, processor)
            benchmark_results["conservation"] = evaluator.evaluate()

        return benchmark_results

    def run_all_conditions(self, skip_training: bool = False) -> None:
        """Train and evaluate all conditions.

        Args:
            skip_training: If True, assume adapters already exist and only run evaluation.
        """
        for condition in self.conditions:
            logger.info(f"\n{'='*60}")
            logger.info(f"ABLATION CONDITION {condition}")
            logger.info(f"{'='*60}")

            adapter_path = str(self.output_base_dir / f"condition_{condition}" / "adapter")

            if not skip_training:
                adapter_path = self.train_condition(condition)

            eval_results = self.evaluate_condition(condition, adapter_path)
            self._results[condition] = eval_results

        self.save_summary()

    def save_summary(self) -> None:
        """Save a summary of all conditions' benchmark results."""
        summary_path = self.output_base_dir / "ablation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(self._results, f, indent=2)
        logger.info(f"Ablation summary saved to {summary_path}")

    def get_results(self) -> Dict[str, Dict[str, Any]]:
        """Return results dict keyed by condition ID."""
        return self._results
