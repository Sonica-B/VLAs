"""
Run all benchmark evaluations (PhysBench, GRASP L2, ConservationBench).

Can evaluate the base model or a fine-tuned LoRA checkpoint.

Usage:
    # Evaluate base model
    python scripts/run_evaluation.py --model qwen2_5_vl_7b

    # Evaluate fine-tuned model (LoRA adapter)
    python scripts/run_evaluation.py --model qwen2_5_vl_7b --adapter results/ablation_metrics/qwen/condition_d_enc_proj/adapter

    # Evaluate all models and all ablation conditions
    python scripts/run_evaluation.py --all
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vlm_loader import load_vlm
from src.evaluation.physbench_eval import PhysBenchEvaluator
from src.evaluation.grasp_eval import GRASPEvaluator
from src.evaluation.conservation_eval import ConservationBenchEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAMES = ["qwen2_5_vl_7b", "internvl2_5_8b", "llava_onevision_7b"]


def evaluate_model(
    model,
    processor,
    physbench_path: str,
    grasp_path: str,
    conservation_path: str,
    output_dir: Path,
    max_questions: int = None,
) -> dict:
    """Run all three benchmark evaluations for a loaded model.

    Returns:
        Dict mapping benchmark name → results.
    """
    all_results = {}

    # PhysBench
    if Path(physbench_path).exists():
        logger.info("Running PhysBench evaluation...")
        evaluator = PhysBenchEvaluator(physbench_path, model, processor)
        results = evaluator.evaluate(
            max_questions=max_questions,
            output_path=output_dir / "physbench_results.json",
        )
        all_results["physbench"] = {
            "overall_accuracy": results["overall_accuracy"],
            "per_category_accuracy": results["per_category_accuracy"],
        }
        logger.info(f"  PhysBench: {results['overall_accuracy']:.1f}%")
    else:
        logger.warning(f"PhysBench dataset not found at {physbench_path}")

    # GRASP L2
    if Path(grasp_path).exists():
        logger.info("Running GRASP L2 evaluation...")
        evaluator = GRASPEvaluator(grasp_path, model, processor)
        results = evaluator.evaluate(
            max_questions=max_questions,
            output_path=output_dir / "grasp_results.json",
        )
        all_results["grasp"] = {
            "overall_accuracy": results["overall_accuracy"],
            "per_subtask_accuracy": results["per_subtask_accuracy"],
        }
        logger.info(f"  GRASP L2: {results['overall_accuracy']:.1f}%")
    else:
        logger.warning(f"GRASP dataset not found at {grasp_path}")

    # ConservationBench
    if Path(conservation_path).exists():
        logger.info("Running ConservationBench evaluation...")
        evaluator = ConservationBenchEvaluator(conservation_path, model, processor)
        results = evaluator.evaluate(
            max_questions=max_questions,
            output_path=output_dir / "conservation_results.json",
        )
        all_results["conservation"] = {
            "binary_accuracy": results["binary_accuracy"],
            "mean_quantity_mape": results["mean_quantity_mape"],
        }
        logger.info(f"  ConservationBench: {results['binary_accuracy']:.1f}%")
    else:
        logger.warning(f"ConservationBench dataset not found at {conservation_path}")

    # Save combined summary
    summary_path = output_dir / "evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Summary saved to {summary_path}")

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark evaluations for a VLM")
    parser.add_argument("--model", type=str, choices=MODEL_NAMES + ["all"], default=None)
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter directory (for fine-tuned evaluation)")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--physbench-path", type=str, default="data/physbench/")
    parser.add_argument("--grasp-path", type=str, default="data/grasp_l2/")
    parser.add_argument("--conservation-path", type=str, default="data/conservation_bench/")
    parser.add_argument("--output-dir", type=str, default="results/evaluation/")
    parser.add_argument("--max-questions", type=int, default=None,
                        help="Limit questions per benchmark (for debugging)")
    parser.add_argument("--all", action="store_true", dest="eval_all",
                        help="Evaluate all models on all conditions")
    args = parser.parse_args()

    if args.eval_all:
        models_to_eval = MODEL_NAMES
    elif args.model:
        models_to_eval = [args.model]
    else:
        parser.error("Specify --model or --all")
        return

    import torch

    for model_name in models_to_eval:
        logger.info(f"\nEvaluating: {model_name}")
        model, processor = load_vlm(
            model_name,
            torch_dtype=torch.bfloat16,
            load_in_4bit=args.load_in_4bit,
        )

        # Apply LoRA adapter if provided
        if args.adapter:
            from src.models.lora_wrapper import LoRAWrapper
            model = LoRAWrapper.load_adapter(model, args.adapter)
            logger.info(f"Loaded adapter from: {args.adapter}")

        adapter_label = Path(args.adapter).name if args.adapter else "base"
        output_dir = Path(args.output_dir) / model_name / adapter_label
        output_dir.mkdir(parents=True, exist_ok=True)

        results = evaluate_model(
            model, processor,
            physbench_path=args.physbench_path,
            grasp_path=args.grasp_path,
            conservation_path=args.conservation_path,
            output_dir=output_dir,
            max_questions=args.max_questions,
        )

        # Print summary
        print(f"\n{'='*50}")
        print(f"Model: {model_name} | Adapter: {adapter_label}")
        print(f"{'='*50}")
        for bench, bench_results in results.items():
            print(f"  {bench}:")
            for k, v in bench_results.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.2f}")


if __name__ == "__main__":
    main()
