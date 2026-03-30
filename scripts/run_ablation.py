"""
Main ablation training and evaluation script for Phase 2.

Trains all 5 LoRA ablation conditions for a given model and evaluates
on PhysBench, GRASP Level 2, and ConservationBench.

Usage:
    python scripts/run_ablation.py model=qwen2_5_vl_7b ablation=condition_e_full
    python scripts/run_ablation.py model=qwen2_5_vl_7b --all-conditions
    python scripts/run_ablation.py model=qwen2_5_vl_7b --eval-only --checkpoint results/ablation_metrics/qwen/
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ablation.component_ablation import ComponentAblation
from src.ablation.ablation_analyzer import AblationAnalyzer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_CHOICES = ["qwen2_5_vl_7b", "internvl2_5_8b", "llava_onevision_7b"]
CONDITION_CHOICES = ["A", "B", "C", "D", "E"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA component ablation study")
    parser.add_argument(
        "--model", type=str, required=True, choices=MODEL_CHOICES,
        help="VLM model to fine-tune"
    )
    parser.add_argument(
        "--conditions", nargs="+", default=CONDITION_CHOICES,
        choices=CONDITION_CHOICES,
        help="Which ablation conditions to run (default: all 5)"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=16,
        help="LoRA rank for all conditions (default: 16)"
    )
    parser.add_argument(
        "--train-data", type=str, default="data/physion/qa_train.jsonl",
        help="Path to physics QA training JSONL"
    )
    parser.add_argument(
        "--val-data", type=str, default="data/physion/qa_val.jsonl",
        help="Path to physics QA validation JSONL"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/ablation_metrics",
        help="Base output directory"
    )
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Skip training; only run evaluation on existing checkpoints"
    )
    parser.add_argument(
        "--physbench-path", type=str, default="data/physbench/",
        help="Path to PhysBench dataset"
    )
    parser.add_argument(
        "--grasp-path", type=str, default="data/grasp_l2/",
        help="Path to GRASP Level 2 dataset"
    )
    parser.add_argument(
        "--conservation-path", type=str, default="data/conservation_bench/",
        help="Path to ConservationBench dataset"
    )
    parser.add_argument(
        "--load-in-4bit", action="store_true",
        help="Load base model in 4-bit quantization"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Starting ablation study:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Conditions: {args.conditions}")
    logger.info(f"  LoRA rank: {args.lora_rank}")
    logger.info(f"  Eval only: {args.eval_only}")

    import torch
    base_model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "load_in_4bit": args.load_in_4bit,
    }

    ablation = ComponentAblation(
        model_name=args.model,
        base_model_kwargs=base_model_kwargs,
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        output_base_dir=args.output_dir,
        lora_rank=args.lora_rank,
        conditions=args.conditions,
        physbench_path=args.physbench_path,
        grasp_path=args.grasp_path,
        conservation_path=args.conservation_path,
    )

    ablation.run_all_conditions(skip_training=args.eval_only)

    # Print summary
    logger.info("\nAblation study complete. Generating analysis...")
    analyzer = AblationAnalyzer(results_base_dir=args.output_dir)
    analyzer.load_all_results(model_names=[args.model])
    analyzer.print_summary_table(benchmark="physbench")

    # Test key hypothesis
    h3 = analyzer.test_hypothesis_d_beats_c(benchmark="physbench")
    logger.info(
        f"\nH3 (Enc+Proj > LLM-only on PhysBench): "
        f"{'SUPPORTED' if h3['overall_H3_supported'] else 'NOT SUPPORTED'}"
    )


if __name__ == "__main__":
    main()
