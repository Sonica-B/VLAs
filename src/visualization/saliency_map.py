"""
Physics saliency map generation.

Renders per-patch R² scores as a heatmap overlay on the original image.
The 14×14 patch grid is bilinearly upsampled to the original image resolution
and blended as a semi-transparent colormap overlay.

Output: publication-quality matplotlib figure or PIL Image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image
from scipy.ndimage import zoom


class PhysicsSaliencyMap:
    """Generate physics saliency heatmap overlays from per-patch R² scores.

    Args:
        patch_grid_size: Spatial patch grid size (e.g., 14 for 14×14).
        upsample_mode: Interpolation for upsampling — "bilinear" or "nearest".
        colormap: Matplotlib colormap name. Default "inferno".
        alpha: Heatmap transparency (0=invisible, 1=opaque). Default 0.6.
        dpi: Figure DPI for saved outputs. Default 300.

    Example:
        >>> viz = PhysicsSaliencyMap(patch_grid_size=14)
        >>> fig = viz.visualize(image=pil_img, per_patch_scores=r2_scores,
        ...                     title="Mass Saliency — Qwen2.5-VL, Stage 1")
        >>> fig.savefig("results/figures/mass_saliency.png")
    """

    def __init__(
        self,
        patch_grid_size: int = 14,
        upsample_mode: str = "bilinear",
        colormap: str = "inferno",
        alpha: float = 0.6,
        dpi: int = 300,
    ) -> None:
        self.patch_grid_size = patch_grid_size
        self.upsample_mode = upsample_mode
        self.colormap = colormap
        self.alpha = alpha
        self.dpi = dpi

    def scores_to_heatmap(
        self,
        per_patch_scores: np.ndarray,
        target_size: Tuple[int, int],
        normalize: bool = True,
    ) -> np.ndarray:
        """Convert flat per-patch scores to an upsampled heatmap.

        Args:
            per_patch_scores: float32 [N_patches] — R² or other score per patch.
            target_size: (H, W) target image dimensions for upsampling.
            normalize: If True, linearly scale scores to [0, 1].

        Returns:
            heatmap: float32 [H, W] in [0, 1].
        """
        n = self.patch_grid_size
        grid = per_patch_scores.reshape(n, n).astype(np.float32)

        if normalize:
            min_val, max_val = grid.min(), grid.max()
            if max_val > min_val:
                grid = (grid - min_val) / (max_val - min_val)

        H, W = target_size
        zoom_h = H / n
        zoom_w = W / n

        if self.upsample_mode == "bilinear":
            order = 1  # bilinear = scipy zoom order 1
        elif self.upsample_mode == "nearest":
            order = 0
        else:
            order = 1

        heatmap = zoom(grid, (zoom_h, zoom_w), order=order)
        # Clip to [0, 1] after zoom (can produce tiny out-of-range values)
        heatmap = np.clip(heatmap, 0.0, 1.0)
        return heatmap

    def visualize(
        self,
        image: Image.Image,
        per_patch_scores: np.ndarray,
        title: str = "",
        normalize: bool = True,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        show_colorbar: bool = True,
        figsize: Tuple[float, float] = (6, 5),
    ) -> plt.Figure:
        """Generate a matplotlib figure with the saliency heatmap overlaid on the image.

        Args:
            image: PIL Image (original scene image).
            per_patch_scores: float32 [N_patches] — per-patch R² scores.
            title: Figure title.
            normalize: Normalize scores to [0, 1] before mapping to color.
            vmin: Colormap lower bound (overrides normalize if set).
            vmax: Colormap upper bound (overrides normalize if set).
            show_colorbar: If True, add a colorbar.
            figsize: Figure dimensions in inches.

        Returns:
            matplotlib Figure object.
        """
        W, H = image.size
        image_arr = np.array(image)

        heatmap = self.scores_to_heatmap(per_patch_scores, (H, W), normalize=normalize)
        if vmin is not None or vmax is not None:
            # Recompute with explicit bounds
            grid = per_patch_scores.reshape(self.patch_grid_size, self.patch_grid_size)
            heatmap_raw = zoom(grid, (H / self.patch_grid_size, W / self.patch_grid_size), order=1)
            heatmap = np.clip(heatmap_raw, 0.0, None)

        cmap = plt.get_cmap(self.colormap)
        rgba_heatmap = cmap(heatmap)                          # [H, W, 4]
        rgba_heatmap[..., 3] = self.alpha                     # Set alpha channel

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=self.dpi)
        ax.imshow(image_arr)
        im = ax.imshow(
            heatmap,
            cmap=self.colormap,
            alpha=self.alpha,
            vmin=vmin,
            vmax=vmax,
            interpolation="bilinear",
        )

        if show_colorbar:
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("R² (physics property decodability)", fontsize=9)

        ax.set_title(title, fontsize=11, pad=10)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def visualize_grid(
        self,
        images: list,
        scores_list: list,
        row_labels: list,
        col_labels: list,
        suptitle: str = "",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> plt.Figure:
        """Generate a grid of saliency maps (e.g., 3 models × 4 pipeline stages).

        Args:
            images: List of PIL Images [N_rows × N_cols].
            scores_list: List of per-patch score arrays [N_rows × N_cols].
            row_labels: Labels for each row (e.g., model names).
            col_labels: Labels for each column (e.g., pipeline stage names).
            suptitle: Overall figure title.
            figsize: Figure size (auto-computed if None).

        Returns:
            matplotlib Figure.
        """
        n_rows = len(row_labels)
        n_cols = len(col_labels)

        if figsize is None:
            figsize = (3.5 * n_cols, 3.5 * n_rows + 0.5)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, dpi=self.dpi)
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for r in range(n_rows):
            for c in range(n_cols):
                ax = axes[r, c]
                idx = r * n_cols + c
                image = images[idx] if idx < len(images) else images[0]
                scores = scores_list[idx] if idx < len(scores_list) else scores_list[0]

                W, H = image.size
                image_arr = np.array(image)
                heatmap = self.scores_to_heatmap(scores, (H, W))

                ax.imshow(image_arr)
                ax.imshow(heatmap, cmap=self.colormap, alpha=self.alpha, interpolation="bilinear")
                ax.axis("off")

                if r == 0:
                    ax.set_title(col_labels[c], fontsize=9, fontweight="bold")
                if c == 0:
                    ax.set_ylabel(row_labels[r], fontsize=9, fontweight="bold")

        if suptitle:
            fig.suptitle(suptitle, fontsize=12, y=1.01)

        plt.tight_layout()
        return fig

    def save(self, fig: plt.Figure, path: str | Path, close_after: bool = True) -> None:
        """Save figure and optionally close it.

        Args:
            fig: Matplotlib figure.
            path: Output file path (.png, .pdf, .svg supported).
            close_after: Close the figure after saving to free memory.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        if close_after:
            plt.close(fig)
