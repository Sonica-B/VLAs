"""
Before/after fine-tuning saliency comparison visualization.

Generates side-by-side comparisons of physics saliency maps before and after
LoRA fine-tuning, to answer Phase 3's question: does fine-tuning sharpen
the spatial physics encoding?

Key metrics:
  - Δ spatial precision: change in fraction of top-20% patches overlapping GT masks
  - Δ spatial recall: change in GT mask coverage by top-20% patches
  - Δ R²: change in physics decodability per stage
  - Sharpening ratio: (post R²) / (pre R²)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

from src.visualization.saliency_map import PhysicsSaliencyMap


@dataclass
class SaliencyComparison:
    """Container for a single before/after saliency comparison.

    Attributes:
        image: The input image.
        scores_before: Per-patch R² scores from the base model.
        scores_after: Per-patch R² scores from the fine-tuned model.
        gt_mask: Ground truth object mask (binary [H, W]), for spatial metrics.
        model_name: VLM model identifier.
        variable: Physics variable (e.g., "mass").
        stage: Pipeline stage (e.g., "stage_1_enc_out").
        ablation_condition: LoRA condition used for fine-tuning (e.g., "D").
    """

    image: Image.Image
    scores_before: np.ndarray
    scores_after: np.ndarray
    gt_mask: Optional[np.ndarray] = None
    model_name: str = ""
    variable: str = ""
    stage: str = ""
    ablation_condition: str = ""


class BeforeAfterComparison:
    """Generate before/after fine-tuning saliency comparison figures.

    Args:
        patch_grid_size: Spatial patch grid size. Default 14.
        top_k_fraction: Fraction of patches considered "activated" for spatial metrics.
            Default 0.20 (top 20%).
        dpi: Figure DPI. Default 300.

    Example:
        >>> comparison = SaliencyComparison(
        ...     image=pil_img,
        ...     scores_before=r2_base,
        ...     scores_after=r2_finetuned,
        ...     gt_mask=binary_mask,
        ... )
        >>> viz = BeforeAfterComparison()
        >>> fig = viz.plot(comparison, title="Mass Saliency: Base vs. LoRA-D")
        >>> metrics = viz.compute_spatial_metrics(comparison)
    """

    def __init__(
        self,
        patch_grid_size: int = 14,
        top_k_fraction: float = 0.20,
        dpi: int = 300,
    ) -> None:
        self.patch_grid_size = patch_grid_size
        self.top_k_fraction = top_k_fraction
        self.dpi = dpi
        self._saliency_viz = PhysicsSaliencyMap(patch_grid_size=patch_grid_size, dpi=dpi)

    def plot(
        self,
        comparison: SaliencyComparison,
        title: str = "",
        figsize: Tuple[float, float] = (12, 4),
    ) -> plt.Figure:
        """Generate a 3-panel figure: original | before | after.

        Args:
            comparison: SaliencyComparison data container.
            title: Figure title.
            figsize: Figure size in inches.

        Returns:
            matplotlib Figure.
        """
        fig = plt.figure(figsize=figsize, dpi=self.dpi)
        gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.05)

        W, H = comparison.image.size
        image_arr = np.array(comparison.image)

        # Panel 1: Original image (+ GT mask if available)
        ax0 = fig.add_subplot(gs[0])
        ax0.imshow(image_arr)
        if comparison.gt_mask is not None:
            # Overlay GT mask as a green contour
            from skimage import measure
            contours = measure.find_contours(comparison.gt_mask, 0.5)
            for contour in contours:
                ax0.plot(contour[:, 1], contour[:, 0], "g-", linewidth=1.5, alpha=0.8)
        ax0.set_title("Original Image", fontsize=9)
        ax0.axis("off")

        # Panel 2: Before fine-tuning saliency
        heatmap_before = self._saliency_viz.scores_to_heatmap(comparison.scores_before, (H, W))
        r2_before = float(np.nanmean(comparison.scores_before[comparison.scores_before > 0]))

        ax1 = fig.add_subplot(gs[1])
        ax1.imshow(image_arr)
        im1 = ax1.imshow(heatmap_before, cmap="inferno", alpha=0.65, interpolation="bilinear")
        ax1.set_title(f"Before Fine-tuning\n(R²={r2_before:.3f})", fontsize=9)
        ax1.axis("off")

        # Panel 3: After fine-tuning saliency
        heatmap_after = self._saliency_viz.scores_to_heatmap(comparison.scores_after, (H, W))
        r2_after = float(np.nanmean(comparison.scores_after[comparison.scores_after > 0]))

        ax2 = fig.add_subplot(gs[2])
        ax2.imshow(image_arr)
        im2 = ax2.imshow(heatmap_after, cmap="inferno", alpha=0.65, interpolation="bilinear")
        delta_r2 = r2_after - r2_before
        delta_sign = "+" if delta_r2 >= 0 else ""
        ax2.set_title(
            f"After LoRA-{comparison.ablation_condition}\n"
            f"(R²={r2_after:.3f}, {delta_sign}{delta_r2:.3f})",
            fontsize=9,
        )
        ax2.axis("off")

        # Shared colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im2, cax=cbar_ax, label="R² (physics decodability)")

        subtitle = (
            f"{comparison.model_name} | {comparison.variable} | {comparison.stage}"
            if comparison.model_name else ""
        )
        full_title = f"{title}\n{subtitle}" if subtitle else title
        fig.suptitle(full_title, fontsize=11, y=1.02)

        return fig

    def plot_grid(
        self,
        comparisons: List[SaliencyComparison],
        row_labels: List[str],
        col_labels: List[str],
        suptitle: str = "",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """Generate a grid of before/after comparisons.

        Args:
            comparisons: List of SaliencyComparison objects [N_rows × N_cols].
                         Each entry produces a before/after pair of heatmaps.
            row_labels: Labels for rows (e.g., physics variable names).
            col_labels: Labels for columns (e.g., model names).
            suptitle: Overall figure title.
            figsize: Auto-computed if None.

        Returns:
            matplotlib Figure.
        """
        n_rows = len(row_labels)
        n_cols = len(col_labels)
        # Each comparison shows 2 panels (before + after), so total columns = 2 × n_cols
        panels_per_row = 2 * n_cols

        if figsize is None:
            figsize = (3.0 * panels_per_row, 3.0 * n_rows + 0.5)

        fig, axes = plt.subplots(n_rows, panels_per_row, figsize=figsize, dpi=self.dpi)
        if n_rows == 1:
            axes = axes[np.newaxis, :]

        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                if idx >= len(comparisons):
                    break
                comp = comparisons[idx]
                W, H = comp.image.size
                image_arr = np.array(comp.image)

                # Before panel
                ax_before = axes[r, 2 * c]
                hm_before = self._saliency_viz.scores_to_heatmap(comp.scores_before, (H, W))
                ax_before.imshow(image_arr)
                ax_before.imshow(hm_before, cmap="inferno", alpha=0.65, interpolation="bilinear")
                ax_before.axis("off")
                if r == 0:
                    ax_before.set_title(f"{col_labels[c]}\nBefore", fontsize=8)

                # After panel
                ax_after = axes[r, 2 * c + 1]
                hm_after = self._saliency_viz.scores_to_heatmap(comp.scores_after, (H, W))
                ax_after.imshow(image_arr)
                ax_after.imshow(hm_after, cmap="inferno", alpha=0.65, interpolation="bilinear")
                ax_after.axis("off")
                if r == 0:
                    ax_after.set_title(f"{col_labels[c]}\nAfter LoRA-D", fontsize=8)

            axes[r, 0].set_ylabel(row_labels[r], fontsize=9, fontweight="bold")

        if suptitle:
            fig.suptitle(suptitle, fontsize=12, y=1.01)

        plt.tight_layout()
        return fig

    def compute_spatial_metrics(
        self,
        comparison: SaliencyComparison,
    ) -> Dict[str, float]:
        """Compute spatial precision, recall, and R² delta.

        Args:
            comparison: SaliencyComparison with gt_mask.

        Returns:
            Dict with keys: precision_before, precision_after, delta_precision,
                            recall_before, recall_after, delta_recall,
                            r2_before, r2_after, delta_r2, sharpening_ratio.
        """
        n = self.patch_grid_size
        top_k = int(n * n * self.top_k_fraction)

        def top_k_mask(scores: np.ndarray) -> np.ndarray:
            """Binary [N_patches] mask — True for top-k patches."""
            threshold = np.sort(scores)[-top_k]
            return (scores >= threshold).astype(np.float32)

        def compute_precision_recall(
            patch_mask: np.ndarray, gt_mask: np.ndarray, image_size: Tuple[int, int]
        ) -> Tuple[float, float]:
            """Compute spatial precision and recall vs GT object mask."""
            from scipy.ndimage import zoom as _zoom
            H, W = image_size
            # Upsample patch mask to image resolution
            activated_upsampled = _zoom(
                patch_mask.reshape(n, n).astype(float),
                (H / n, W / n),
                order=0,  # nearest neighbor for binary mask
            )
            activated_binary = (activated_upsampled > 0.5).astype(float)
            gt_binary = (gt_mask > 0).astype(float)

            intersection = (activated_binary * gt_binary).sum()
            precision = intersection / (activated_binary.sum() + 1e-8)
            recall = intersection / (gt_binary.sum() + 1e-8)
            return float(precision), float(recall)

        W, H = comparison.image.size
        mask_before = top_k_mask(comparison.scores_before)
        mask_after = top_k_mask(comparison.scores_after)

        r2_before = float(np.nanmean(comparison.scores_before[comparison.scores_before > 0]))
        r2_after = float(np.nanmean(comparison.scores_after[comparison.scores_after > 0]))

        metrics: Dict[str, float] = {
            "r2_before": r2_before,
            "r2_after": r2_after,
            "delta_r2": r2_after - r2_before,
            "sharpening_ratio": r2_after / (r2_before + 1e-8),
        }

        if comparison.gt_mask is not None:
            prec_before, rec_before = compute_precision_recall(mask_before, comparison.gt_mask, (H, W))
            prec_after, rec_after = compute_precision_recall(mask_after, comparison.gt_mask, (H, W))
            metrics.update({
                "precision_before": prec_before,
                "precision_after": prec_after,
                "delta_precision": prec_after - prec_before,
                "recall_before": rec_before,
                "recall_after": rec_after,
                "delta_recall": rec_after - rec_before,
            })

        return metrics
