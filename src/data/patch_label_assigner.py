"""
Patch-level physics label assignment.

Given a segmentation mask (H×W, object IDs) and per-object physics labels,
assigns physics property values to each image patch in the VLM's patch grid.

Two assignment strategies:
  - Object-level properties (mass, friction, elasticity): assign via segmentation
    mask overlap — each patch gets the label of the object covering > 50% of its area.
    Patches with no dominant object get label NaN (background).

  - Scene-level dynamics (stability): compute per-patch "physics activity scores"
    from temporal derivatives of object positions across frames.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.ndimage import zoom


class PatchLabelAssigner:
    """Assigns physics property labels to VLM image patch positions.

    Args:
        patch_grid_size: Number of patches along each spatial dimension.
            For a 14×14 grid, pass 14. Total patches = patch_grid_size².
        overlap_threshold: Fraction of patch area an object must cover to
            "own" that patch. Default 0.5 (majority overlap).
        background_value: Value assigned to background patches (no dominant object).
            Default NaN — probes should exclude these during training.

    Example:
        >>> assigner = PatchLabelAssigner(patch_grid_size=14)
        >>> patch_labels = assigner.assign(object_masks, physics_labels)
        >>> patch_labels.shape  # [196, 4]  (196 patches, 4 properties)
    """

    PROPERTY_ORDER = ["mass", "friction", "elasticity", "stability"]

    def __init__(
        self,
        patch_grid_size: int = 14,
        overlap_threshold: float = 0.5,
        background_value: float = float("nan"),
    ) -> None:
        self.patch_grid_size = patch_grid_size
        self.n_patches = patch_grid_size * patch_grid_size
        self.overlap_threshold = overlap_threshold
        self.background_value = background_value

    def assign(
        self,
        object_masks: np.ndarray,
        physics_labels: Dict[str, np.ndarray],
        stability_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Assign physics labels to all patches.

        Args:
            object_masks: uint8 array [H, W] — pixel value = object ID (0 = background).
                Object IDs are 1-indexed. Max value = num_objects.
            physics_labels: dict mapping property → float32 array [N_objects].
                Index 0 = object with ID 1, etc.
            stability_scores: Optional pre-computed per-patch stability scores [N_patches].
                If None and "stability" is requested, defaults to NaN.

        Returns:
            patch_labels: float32 array [N_patches, 4], columns in PROPERTY_ORDER.
                Background patches have NaN for object-level properties.
        """
        H, W = object_masks.shape
        patch_assignments = self._assign_patches_to_objects(object_masks)

        patch_labels = np.full(
            (self.n_patches, len(self.PROPERTY_ORDER)), fill_value=float("nan"), dtype=np.float32
        )

        for prop_idx, prop_name in enumerate(self.PROPERTY_ORDER):
            if prop_name == "stability":
                if stability_scores is not None:
                    patch_labels[:, prop_idx] = stability_scores
                continue

            if prop_name not in physics_labels:
                continue

            prop_values = physics_labels[prop_name]  # [N_objects]
            for patch_idx in range(self.n_patches):
                obj_id = patch_assignments[patch_idx]
                if obj_id > 0:  # 0 = background
                    # obj_id is 1-indexed; prop_values is 0-indexed
                    obj_array_idx = obj_id - 1
                    if obj_array_idx < len(prop_values):
                        patch_labels[patch_idx, prop_idx] = prop_values[obj_array_idx]

        return patch_labels

    def _assign_patches_to_objects(self, object_masks: np.ndarray) -> np.ndarray:
        """Determine dominant object ID for each patch.

        For each patch in the grid, finds the object ID covering > overlap_threshold
        of the patch area. Returns 0 for background patches.

        Args:
            object_masks: [H, W] uint8 array of object IDs.

        Returns:
            patch_assignments: int array [N_patches] — dominant object ID per patch.
        """
        H, W = object_masks.shape
        patch_h = H // self.patch_grid_size
        patch_w = W // self.patch_grid_size
        patch_area = patch_h * patch_w

        assignments = np.zeros(self.n_patches, dtype=np.int32)

        for row in range(self.patch_grid_size):
            for col in range(self.patch_grid_size):
                patch_idx = row * self.patch_grid_size + col
                r_start = row * patch_h
                r_end = r_start + patch_h
                c_start = col * patch_w
                c_end = c_start + patch_w

                patch_mask = object_masks[r_start:r_end, c_start:c_end]
                obj_ids, counts = np.unique(patch_mask, return_counts=True)

                # Find dominant object (excluding background = 0)
                best_obj_id = 0
                best_count = 0
                for oid, cnt in zip(obj_ids, counts):
                    if oid > 0 and cnt > best_count:
                        best_count = cnt
                        best_obj_id = oid

                if best_count / patch_area >= self.overlap_threshold:
                    assignments[patch_idx] = best_obj_id
                else:
                    assignments[patch_idx] = 0  # background

        return assignments

    def compute_stability_scores(
        self,
        positions_over_time: np.ndarray,
        segmentation_masks: np.ndarray,
        temporal_window: int = 5,
    ) -> np.ndarray:
        """Compute per-patch physics activity scores from temporal dynamics.

        For each patch, computes the mean magnitude of position change for
        the object covering that patch, averaged over the temporal window.
        High scores → high temporal activity (moving/colliding objects).

        Args:
            positions_over_time: float array [T, N_objects, 3] — XYZ positions.
            segmentation_masks: uint8 array [T, H, W] — object masks over time.
            temporal_window: Number of consecutive frames to use for derivative.

        Returns:
            stability_scores: float32 array [N_patches] — activity score per patch.
                Higher = more temporal change = less stable.
        """
        T, N_objects, _ = positions_over_time.shape

        # Compute per-object velocity magnitude [T-1, N_objects]
        velocities = np.diff(positions_over_time, axis=0)   # [T-1, N_objects, 3]
        speed = np.linalg.norm(velocities, axis=-1)          # [T-1, N_objects]

        # Use first frame's masks for patch assignment
        patch_assignments = self._assign_patches_to_objects(segmentation_masks[0])

        # Average speed over temporal window for each object
        t_end = min(temporal_window, T - 1)
        mean_speed = speed[:t_end].mean(axis=0)  # [N_objects]

        # Map per-object speed to per-patch score
        stability_scores = np.zeros(self.n_patches, dtype=np.float32)
        for patch_idx in range(self.n_patches):
            obj_id = patch_assignments[patch_idx]
            if obj_id > 0:
                obj_array_idx = obj_id - 1
                if obj_array_idx < N_objects:
                    stability_scores[patch_idx] = mean_speed[obj_array_idx]

        # Normalize to [0, 1]
        max_score = stability_scores.max()
        if max_score > 0:
            stability_scores /= max_score

        return stability_scores

    def assign_contrastive(
        self,
        object_masks: np.ndarray,
        physics_labels: Dict[str, np.ndarray],
        stability_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Assign binary (above/below median) physics labels to patches.

        An alternative to raw regression labels. For each physics property,
        patches are labelled 1.0 (object above population median) or 0.0
        (object below median). Background patches retain NaN.

        This framing trains a binary linear classifier instead of a regressor,
        which can be more robust when property ranges are narrow.

        Args:
            object_masks: uint8 [H, W] — object IDs (0 = background).
            physics_labels: dict of property → float32 [N_objects].
            stability_scores: Optional [N_patches] pre-computed stability scores.

        Returns:
            binary_labels: float32 [N_patches, 4], values in {0.0, 1.0, NaN}.
        """
        # Get raw continuous labels first
        raw = self.assign(object_masks, physics_labels, stability_scores)  # [N, 4]

        binary = np.full_like(raw, fill_value=float("nan"))

        for prop_idx in range(raw.shape[1]):
            col = raw[:, prop_idx]
            valid_mask = ~np.isnan(col)
            if valid_mask.sum() < 2:
                continue
            median_val = np.median(col[valid_mask])
            binary[valid_mask, prop_idx] = (col[valid_mask] >= median_val).astype(np.float32)

        return binary

    def get_patch_coords(
        self, image_size: int = 448
    ) -> np.ndarray:
        """Return center (x, y) coordinates for each patch in pixel space.

        Args:
            image_size: Square image size in pixels.

        Returns:
            coords: float32 array [N_patches, 2] — (x, y) pixel center of each patch.
        """
        patch_size = image_size // self.patch_grid_size
        coords = []
        for row in range(self.patch_grid_size):
            for col in range(self.patch_grid_size):
                cx = (col + 0.5) * patch_size
                cy = (row + 0.5) * patch_size
                coords.append([cx, cy])
        return np.array(coords, dtype=np.float32)
