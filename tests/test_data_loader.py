"""
Unit tests for PhysionLoader and PatchLabelAssigner.

These tests use synthetic data (no actual Physion++ dataset required)
to verify the correctness of the data pipeline.
"""

import numpy as np
import pytest
from PIL import Image

from src.data.patch_label_assigner import PatchLabelAssigner


class TestPatchLabelAssigner:
    """Tests for PatchLabelAssigner."""

    def setup_method(self):
        self.assigner = PatchLabelAssigner(patch_grid_size=14, overlap_threshold=0.5)

    def make_mask_and_labels(self, n_objects: int = 3, image_size: int = 448):
        """Create a synthetic object mask and physics labels."""
        mask = np.zeros((image_size, image_size), dtype=np.uint8)
        patch_h = image_size // 14
        patch_w = image_size // 14

        # Fill top-left quadrant with object 1
        mask[:image_size // 2, :image_size // 2] = 1
        # Fill bottom-right quadrant with object 2
        mask[image_size // 2:, image_size // 2:] = 2

        physics_labels = {
            "mass": np.array([2.5, 0.8, 1.2], dtype=np.float32),
            "friction": np.array([0.3, 0.7, 0.5], dtype=np.float32),
            "elasticity": np.array([0.9, 0.4, 0.6], dtype=np.float32),
        }
        return mask, physics_labels

    def test_output_shape(self):
        """patch_labels should have shape [196, 4]."""
        mask, labels = self.make_mask_and_labels()
        patch_labels = self.assigner.assign(mask, labels)
        assert patch_labels.shape == (196, 4), f"Expected (196, 4), got {patch_labels.shape}"

    def test_dtype(self):
        """Output should be float32."""
        mask, labels = self.make_mask_and_labels()
        patch_labels = self.assigner.assign(mask, labels)
        assert patch_labels.dtype == np.float32

    def test_object_patches_have_labels(self):
        """Patches dominated by object 1 should have its mass label."""
        mask, labels = self.make_mask_and_labels()
        patch_labels = self.assigner.assign(mask, labels)

        # Top-left corner patch (row=0, col=0) should be object 1 (mass=2.5)
        patch_0_0 = patch_labels[0, 0]  # mass column
        assert not np.isnan(patch_0_0), "Object 1 patch should have a mass label"
        assert pytest.approx(patch_0_0) == 2.5, f"Expected 2.5 (obj 1 mass), got {patch_0_0}"

    def test_background_patches_are_nan(self):
        """Patches with no dominant object should have NaN labels."""
        mask, labels = self.make_mask_and_labels()
        patch_labels = self.assigner.assign(mask, labels)
        # Check for NaN somewhere in the output (background patches exist)
        has_nan = np.any(np.isnan(patch_labels))
        assert has_nan, "Expected NaN values for background patches"

    def test_no_labels_for_missing_property(self):
        """If a property is not in physics_labels, its column should be NaN."""
        mask, _ = self.make_mask_and_labels()
        labels_no_elasticity = {
            "mass": np.array([1.0, 2.0], dtype=np.float32),
        }
        patch_labels = self.assigner.assign(mask, labels_no_elasticity)
        # Elasticity column (index 2) should be all NaN
        assert np.all(np.isnan(patch_labels[:, 2])), "Elasticity column should be all NaN"

    def test_stability_scores_integration(self):
        """Stability scores can be integrated via assign()."""
        mask, labels = self.make_mask_and_labels()
        stability = np.random.rand(196).astype(np.float32)
        patch_labels = self.assigner.assign(mask, labels, stability_scores=stability)
        # Stability column (index 3) should contain the passed scores
        np.testing.assert_array_almost_equal(
            patch_labels[:, 3], stability,
            err_msg="Stability column should match passed stability_scores"
        )

    def test_patch_coords_shape(self):
        """get_patch_coords should return [196, 2]."""
        coords = self.assigner.get_patch_coords(image_size=448)
        assert coords.shape == (196, 2), f"Expected (196, 2), got {coords.shape}"

    def test_patch_coords_range(self):
        """Patch center coordinates should be within [0, image_size]."""
        coords = self.assigner.get_patch_coords(image_size=448)
        assert coords.min() > 0, "Coords should be > 0 (centers, not edges)"
        assert coords.max() < 448, "Coords should be < image_size"

    def test_compute_stability_scores_shape(self):
        """Stability scores output should be [N_patches]."""
        mask, _ = self.make_mask_and_labels()
        positions = np.random.rand(10, 3, 3).astype(np.float32)  # [T=10, N_obj=3, xyz=3]
        masks_over_time = np.stack([mask] * 10, axis=0)

        scores = self.assigner.compute_stability_scores(positions, masks_over_time)
        assert scores.shape == (196,), f"Expected (196,), got {scores.shape}"
        assert scores.min() >= 0.0, "Stability scores should be non-negative"
        assert scores.max() <= 1.0, "Stability scores should be normalized to [0, 1]"

    def test_assign_patches_to_objects_coverage(self):
        """Majority-covered patches should be assigned to an object."""
        mask, labels = self.make_mask_and_labels()
        assignments = self.assigner._assign_patches_to_objects(mask)
        assert assignments.shape == (196,)
        # Should have patches assigned to object 1 and 2
        assert 1 in assignments, "Object 1 should be assigned to some patches"
        assert 2 in assignments, "Object 2 should be assigned to some patches"
        assert 0 in assignments, "Background (0) should appear in some patches"
