"""
Ablation result analyzer.

Loads ablation_summary.json files from multiple models and produces:
  - Ranked condition comparison tables
  - Winner identification per model and benchmark
  - Hypothesis testing: does Condition D beat Condition C?
  - Publication-quality bar charts and heat tables
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BENCHMARK_DISPLAY_NAMES = {
    "physbench": "PhysBench",
    "grasp": "GRASP L2",
    "conservation": "ConservationBench",
}

CONDITION_DISPLAY_NAMES = {
    "A": "Enc. Only",
    "B": "Proj. Only",
    "C": "LLM Only",
    "D": "Enc.+Proj.",
    "E": "Full Model",
}

MODEL_DISPLAY_NAMES = {
    "qwen2_5_vl_7b": "Qwen2.5-VL-7B",
    "internvl2_5_8b": "InternVL 2.5-8B",
    "llava_onevision_7b": "LLaVA-OV-7B",
}


class AblationAnalyzer:
    """Analyzes and visualizes results from the component ablation study.

    Args:
        results_base_dir: Directory containing per-model ablation_summary.json files.
            Expected structure: {results_base_dir}/{model_name}/ablation_summary.json
        dpi: Figure DPI. Default 300.

    Example:
        >>> analyzer = AblationAnalyzer("results/ablation_metrics/")
        >>> analyzer.load_all_results()
        >>> df = analyzer.build_results_table()
        >>> print(df)
        >>> fig = analyzer.plot_condition_comparison()
        >>> fig.savefig("results/figures/ablation_comparison.png")
    """

    def __init__(
        self,
        results_base_dir: str | Path,
        dpi: int = 300,
    ) -> None:
        self.results_base_dir = Path(results_base_dir)
        self.dpi = dpi
        # {model_name: {condition: {benchmark: results}}}
        self._all_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def load_all_results(self, model_names: Optional[List[str]] = None) -> None:
        """Load ablation summary JSON files for all available models.

        Args:
            model_names: Specific models to load. If None, auto-discover.
        """
        if model_names is None:
            model_names = [
                d.name for d in self.results_base_dir.iterdir() if d.is_dir()
            ]

        for model_name in model_names:
            summary_path = self.results_base_dir / model_name / "ablation_summary.json"
            if not summary_path.exists():
                logger.warning(f"No ablation summary found for {model_name} at {summary_path}")
                continue
            with open(summary_path, "r") as f:
                self._all_results[model_name] = json.load(f)
            logger.info(f"Loaded results for {model_name}: conditions {list(self._all_results[model_name].keys())}")

    def _extract_scalar(
        self,
        results: Dict[str, Any],
        benchmark: str,
        metric: str = "overall_accuracy",
    ) -> Optional[float]:
        """Extract a scalar metric from nested results dict."""
        bench_results = results.get(benchmark, {})
        if not bench_results:
            return None
        return bench_results.get(metric)

    def build_results_table(
        self,
        benchmark: str = "physbench",
        metric: str = "overall_accuracy",
    ) -> pd.DataFrame:
        """Build a DataFrame of results: rows=conditions, columns=models.

        Args:
            benchmark: Which benchmark to extract.
            metric: Which metric to extract from the benchmark results.

        Returns:
            DataFrame with condition names as index and model names as columns.
        """
        conditions = ["A", "B", "C", "D", "E"]
        models = sorted(self._all_results.keys())

        data = {}
        for model_name in models:
            col_label = MODEL_DISPLAY_NAMES.get(model_name, model_name)
            col_data = {}
            for cond in conditions:
                cond_results = self._all_results[model_name].get(cond, {})
                val = self._extract_scalar(cond_results, benchmark, metric)
                col_data[CONDITION_DISPLAY_NAMES[cond]] = val
            data[col_label] = col_data

        df = pd.DataFrame(data)
        return df

    def identify_winner(self, benchmark: str = "physbench") -> Dict[str, str]:
        """Identify the winning condition for each model on a given benchmark.

        Args:
            benchmark: Benchmark to use for comparison.

        Returns:
            Dict mapping model_name → winning condition ID.
        """
        winners = {}
        for model_name, model_results in self._all_results.items():
            best_cond = None
            best_val = -float("inf")
            for cond, cond_results in model_results.items():
                val = self._extract_scalar(cond_results, benchmark)
                if val is not None and val > best_val:
                    best_val = val
                    best_cond = cond
            winners[model_name] = best_cond or "N/A"
        return winners

    def test_hypothesis_d_beats_c(self, benchmark: str = "physbench") -> Dict[str, Any]:
        """Test H3: Condition D (Enc+Proj) beats Condition C (LLM-only).

        Returns:
            Dict with per-model comparison and overall support for H3.
        """
        results = {}
        n_support = 0
        n_models = 0

        for model_name, model_results in self._all_results.items():
            c_val = self._extract_scalar(model_results.get("C", {}), benchmark)
            d_val = self._extract_scalar(model_results.get("D", {}), benchmark)

            if c_val is not None and d_val is not None:
                n_models += 1
                d_beats_c = d_val > c_val
                if d_beats_c:
                    n_support += 1
                results[model_name] = {
                    "condition_C": c_val,
                    "condition_D": d_val,
                    "delta_D_minus_C": d_val - c_val,
                    "H3_supported": d_beats_c,
                }

        return {
            "per_model": results,
            "n_models_supporting_H3": n_support,
            "n_models_total": n_models,
            "overall_H3_supported": n_support > n_models / 2,
        }

    def plot_condition_comparison(
        self,
        benchmarks: Optional[List[str]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """Generate a grouped bar chart comparing all 5 conditions across models and benchmarks.

        Args:
            benchmarks: List of benchmark names to include. Default: all 3.
            figsize: Figure size in inches.

        Returns:
            matplotlib Figure.
        """
        benchmarks = benchmarks or ["physbench", "grasp", "conservation"]
        n_benchmarks = len(benchmarks)

        if figsize is None:
            figsize = (5 * n_benchmarks, 5)

        fig, axes = plt.subplots(1, n_benchmarks, figsize=figsize, dpi=self.dpi)
        if n_benchmarks == 1:
            axes = [axes]

        conditions = ["A", "B", "C", "D", "E"]
        cond_labels = [CONDITION_DISPLAY_NAMES[c] for c in conditions]
        n_conditions = len(conditions)
        colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#795548"]

        for ax, benchmark in zip(axes, benchmarks):
            models = sorted(self._all_results.keys())
            n_models = len(models)
            x = np.arange(n_conditions)
            width = 0.8 / n_models

            for i, model_name in enumerate(models):
                values = []
                for cond in conditions:
                    cond_results = self._all_results[model_name].get(cond, {})
                    val = self._extract_scalar(cond_results, benchmark)
                    values.append(val if val is not None else 0.0)

                offset = (i - n_models / 2 + 0.5) * width
                model_label = MODEL_DISPLAY_NAMES.get(model_name, model_name)
                ax.bar(x + offset, values, width=width, label=model_label,
                       color=colors[i % len(colors)], alpha=0.85, edgecolor="white")

            ax.set_xticks(x)
            ax.set_xticklabels(cond_labels, fontsize=9, rotation=15, ha="right")
            ax.set_ylabel("Accuracy (%)", fontsize=10)
            ax.set_title(BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark),
                         fontsize=11, fontweight="bold")
            ax.legend(fontsize=8)
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")

            # Highlight Condition D (key condition)
            ax.axvspan(
                x[3] - 0.5, x[3] + 0.5,
                alpha=0.08, color="#9C27B0", zorder=0
            )
            ax.text(x[3], ax.get_ylim()[1] * 0.98, "Key\ncond.",
                    ha="center", va="top", fontsize=6, color="#9C27B0")

        fig.suptitle(
            "Component Ablation: All Conditions × All Models × All Benchmarks",
            fontsize=12, y=1.02
        )
        plt.tight_layout()
        return fig

    def print_summary_table(self, benchmark: str = "physbench") -> None:
        """Print a formatted results table to stdout."""
        df = self.build_results_table(benchmark=benchmark)
        print(f"\n{'='*60}")
        print(f"ABLATION RESULTS — {BENCHMARK_DISPLAY_NAMES.get(benchmark, benchmark)}")
        print(f"{'='*60}")
        print(df.to_string(float_format="{:.1f}".format))

        h3 = self.test_hypothesis_d_beats_c(benchmark)
        print(f"\nH3 (Enc+Proj beats LLM-only): "
              f"{'SUPPORTED' if h3['overall_H3_supported'] else 'NOT SUPPORTED'} "
              f"({h3['n_models_supporting_H3']}/{h3['n_models_total']} models)")
