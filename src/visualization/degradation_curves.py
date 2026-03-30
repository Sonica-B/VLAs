"""
R² degradation curve visualization.

Plots how probe R² (physics decodability) changes across the 4 VLM pipeline stages:
  Stage 1 (encoder output) → Stage 2 (post-projection) → Stage 3 (LLM-8) → Stage 4 (LLM-16)

One line per model × variable combination.
Expected finding: R² degrades from Stage 1 → Stage 2 (projection bottleneck),
then may partially recover in LLM layers if the LLM re-encodes physics from context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


STAGE_LABELS = [
    "Enc. Output\n(Stage 1)",
    "Post-Proj.\n(Stage 2)",
    "LLM Layer 8\n(Stage 3)",
    "LLM Layer 16\n(Stage 4)",
]

MODEL_COLORS = {
    "Qwen2.5-VL-7B": "#2196F3",    # Blue
    "InternVL 2.5-8B": "#4CAF50",  # Green
    "LLaVA-OV-7B": "#FF5722",      # Deep orange
}

VARIABLE_MARKERS = {
    "mass": "o",
    "friction": "s",
    "elasticity": "^",
    "stability": "D",
}

VARIABLE_LINESTYLES = {
    "mass": "-",
    "friction": "--",
    "elasticity": "-.",
    "stability": ":",
}


@dataclass
class ModelVariableResult:
    """R² results for a single (model, variable) combination across pipeline stages."""
    model_name: str
    variable: str
    r2_by_stage: List[float]                     # Length 4 — one per stage
    ci_lower: Optional[List[float]] = None       # 95% CI lower bounds
    ci_upper: Optional[List[float]] = None       # 95% CI upper bounds


class DegradationCurvePlot:
    """Plot R² degradation across VLM pipeline stages.

    Accumulates results from multiple (model, variable) runs and
    generates a publication-quality figure.

    Example:
        >>> plotter = DegradationCurvePlot()
        >>> plotter.add_result(ModelVariableResult(
        ...     model_name="Qwen2.5-VL-7B",
        ...     variable="mass",
        ...     r2_by_stage=[0.32, 0.18, 0.14, 0.11],
        ... ))
        >>> fig = plotter.plot(title="Physics Decodability Across Pipeline Stages")
        >>> fig.savefig("results/figures/r2_degradation.png")
    """

    def __init__(self, figsize: Tuple[float, float] = (10, 5), dpi: int = 300) -> None:
        self.figsize = figsize
        self.dpi = dpi
        self._results: List[ModelVariableResult] = []

    def add_result(self, result: ModelVariableResult) -> None:
        """Add a single (model, variable) result."""
        self._results.append(result)

    def add_model_results(
        self,
        model_name: str,
        variable: str,
        stages: List[str],
        r2_values: List[float],
        ci_lower: Optional[List[float]] = None,
        ci_upper: Optional[List[float]] = None,
    ) -> None:
        """Convenience method to add results by model name and variable.

        Args:
            model_name: Display name of the model.
            variable: Physics variable name.
            stages: List of stage keys (used for ordering — must have length 4).
            r2_values: List of R² values in stage order.
            ci_lower: Optional 95% CI lower bounds per stage.
            ci_upper: Optional 95% CI upper bounds per stage.
        """
        self._results.append(ModelVariableResult(
            model_name=model_name,
            variable=variable,
            r2_by_stage=r2_values,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        ))

    def plot(
        self,
        title: str = "Physics Decodability Across VLM Pipeline Stages",
        separate_subplots_by: str = "variable",  # "variable" or "model"
        show_grid: bool = True,
        annotate_drop: bool = True,
    ) -> plt.Figure:
        """Generate the degradation curve figure.

        Args:
            title: Figure title.
            separate_subplots_by: Split figure into subplots by "variable" or "model".
            show_grid: Show background grid lines.
            annotate_drop: Annotate the projection bottleneck drop (Stage 1 → Stage 2).

        Returns:
            matplotlib Figure.
        """
        if not self._results:
            raise RuntimeError("No results to plot. Call add_result() first.")

        x = np.arange(4)

        if separate_subplots_by == "variable":
            variables = sorted(set(r.variable for r in self._results))
            n_plots = len(variables)
            fig, axes = plt.subplots(1, n_plots, figsize=(self.figsize[0] * n_plots / 4,
                                                           self.figsize[1]),
                                     dpi=self.dpi, sharey=True)
            if n_plots == 1:
                axes = [axes]

            for ax, var in zip(axes, variables):
                var_results = [r for r in self._results if r.variable == var]
                self._plot_single_panel(ax, var_results, x, show_grid=show_grid,
                                        annotate_drop=annotate_drop)
                ax.set_title(var.capitalize(), fontsize=11, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(STAGE_LABELS, fontsize=7)

            axes[0].set_ylabel("Probe R²", fontsize=10)

        else:  # separate by model
            models = sorted(set(r.model_name for r in self._results))
            n_plots = len(models)
            fig, axes = plt.subplots(1, n_plots, figsize=(self.figsize[0] * n_plots / 3,
                                                           self.figsize[1]),
                                     dpi=self.dpi, sharey=True)
            if n_plots == 1:
                axes = [axes]

            for ax, model in zip(axes, models):
                model_results = [r for r in self._results if r.model_name == model]
                self._plot_single_panel(ax, model_results, x, show_grid=show_grid,
                                        annotate_drop=annotate_drop)
                ax.set_title(model, fontsize=10, fontweight="bold")
                ax.set_xticks(x)
                ax.set_xticklabels(STAGE_LABELS, fontsize=7)

            axes[0].set_ylabel("Probe R²", fontsize=10)

        fig.suptitle(title, fontsize=12, y=1.02)

        # Add unified legend
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0),
                       fontsize=8, framealpha=0.9)

        plt.tight_layout()
        return fig

    def _plot_single_panel(
        self,
        ax: plt.Axes,
        results: List[ModelVariableResult],
        x: np.ndarray,
        show_grid: bool = True,
        annotate_drop: bool = True,
    ) -> None:
        """Plot degradation curves for one subplot panel."""
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle="--", zorder=0)

        for result in results:
            y = np.array(result.r2_by_stage, dtype=float)
            color = MODEL_COLORS.get(result.model_name, "gray")
            marker = VARIABLE_MARKERS.get(result.variable, "o")
            ls = VARIABLE_LINESTYLES.get(result.variable, "-")
            label = f"{result.model_name} ({result.variable})"

            ax.plot(x, y, marker=marker, linestyle=ls, color=color,
                    linewidth=2, markersize=6, label=label, zorder=3)

            # Plot confidence intervals if available
            if result.ci_lower and result.ci_upper:
                ax.fill_between(x, result.ci_lower, result.ci_upper,
                                alpha=0.15, color=color)

            # Annotate projection drop (Stage 1 → Stage 2)
            if annotate_drop and len(y) >= 2:
                drop = y[0] - y[1]
                if drop > 0.02:
                    ax.annotate(
                        f"−{drop:.2f}",
                        xy=(0.5, (y[0] + y[1]) / 2),
                        fontsize=6,
                        color=color,
                        ha="center",
                        va="center",
                        arrowprops=None,
                    )

        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

        # Highlight the projection gap between stage 1 and 2
        ax.axvspan(0.5, 1.5, alpha=0.05, color="red", zorder=1)
        ax.text(1.0, ax.get_ylim()[1] * 0.95, "proj.\ngap",
                ha="center", va="top", fontsize=6, color="red", alpha=0.7)

    def save(self, fig: plt.Figure, path: str | Path) -> None:
        """Save figure to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
