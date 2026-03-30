"""
Unit tests for ActivationExtractor.

Uses small mock models to test hook registration, activation capture,
HDF5 I/O, and shape validation — without requiring real VLM weights.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.activation_extractor import ActivationExtractor, STAGE_NAMES


class MockViTBlock(nn.Module):
    """Minimal ViT-like block for testing."""
    def __init__(self, dim=64):
        super().__init__()
        self.attn = nn.Linear(dim, dim)
        self.ffn = nn.Linear(dim, dim)

    def forward(self, x):
        return self.ffn(torch.relu(self.attn(x)))


class MockMerger(nn.Module):
    """Mock projection MLP."""
    def __init__(self, in_dim=64, out_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())

    def forward(self, x):
        return self.mlp(x)


class MockLLMLayer(nn.Module):
    """Mock LLM transformer layer."""
    def __init__(self, dim=128):
        super().__init__()
        self.self_attn = nn.Linear(dim, dim)

    def forward(self, x):
        return (self.self_attn(x),)  # Return tuple like real transformer


class MockVLM(nn.Module):
    """Minimal mock VLM with the structure ActivationExtractor hooks into."""
    def __init__(self):
        super().__init__()

        # Simulate Qwen2.5-VL-7B structure
        self.model = nn.Module()
        # Visual blocks (list-indexable)
        self.model.visual = nn.Module()
        blocks = nn.ModuleList([MockViTBlock(64) for _ in range(32)])
        self.model.visual.blocks = blocks
        self.model.visual.merger = MockMerger(64, 128)

        # LLM layers
        layers = nn.ModuleList([MockLLMLayer(128) for _ in range(28)])
        self.model.model = nn.Module()
        self.model.model.layers = layers


# Patch HOOK_CONFIGS for the mock model
MOCK_HOOK_CONFIGS = {
    "mock_vlm": {
        "stage_1_enc_out": "model.visual.blocks.31",
        "stage_2_post_proj": "model.visual.merger",
        "stage_3_llm_8": "model.model.layers.8",
        "stage_4_llm_16": "model.model.layers.16",
    }
}


class MockActivationExtractor(ActivationExtractor):
    """Subclass that uses the mock model's hook config."""
    HOOK_CONFIGS = MOCK_HOOK_CONFIGS


class TestActivationExtractorHooks:
    """Tests for hook registration and clearing."""

    def setup_method(self):
        self.model = MockVLM()
        # Patch HOOK_CONFIGS to use mock keys
        self.extractor = MockActivationExtractor(self.model, "mock_vlm", patch_grid_size=4)

    def test_register_hooks_creates_handles(self):
        """register_hooks() should create 4 hook handles."""
        self.extractor.register_hooks()
        assert len(self.extractor._hooks) == 4, "Expected 4 hook handles"
        self.extractor.clear_hooks()

    def test_register_hooks_idempotent_error(self):
        """Registering hooks twice without clearing should raise RuntimeError."""
        self.extractor.register_hooks()
        with pytest.raises(RuntimeError, match="already registered"):
            self.extractor.register_hooks()
        self.extractor.clear_hooks()

    def test_clear_hooks_removes_all(self):
        """clear_hooks() should reset _hooks to empty and _captured to None."""
        self.extractor.register_hooks()
        self.extractor.clear_hooks()
        assert len(self.extractor._hooks) == 0
        assert all(v is None for v in self.extractor._captured.values())

    def test_get_module_by_path_list_index(self):
        """_get_module_by_path should handle integer list indices."""
        block = self.extractor._get_module_by_path("model.visual.blocks.31")
        assert isinstance(block, MockViTBlock)

    def test_get_module_by_path_invalid_raises(self):
        """_get_module_by_path should raise AttributeError for invalid paths."""
        with pytest.raises(AttributeError):
            self.extractor._get_module_by_path("model.nonexistent.path")


class TestActivationExtractorHDF5:
    """Tests for HDF5 save/load round-trip."""

    def test_save_and_load_round_trip(self):
        """Activations saved to HDF5 should be recoverable with same values."""
        model = MockVLM()
        extractor = MockActivationExtractor(model, "mock_vlm")

        dummy_activations = {
            "stage_1_enc_out": torch.randn(196, 64),
            "stage_2_post_proj": torch.randn(196, 128),
            "stage_3_llm_8": torch.randn(196, 128),
            "stage_4_llm_16": torch.randn(196, 128),
        }
        physics_labels = {"mass": [1.0, 2.5], "friction": [0.3, 0.7]}

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "test_activations.h5"
            extractor.save_to_hdf5(
                dummy_activations,
                h5_path,
                scenario_id="test_scenario/trial_001",
                physics_labels=physics_labels,
            )

            assert h5_path.exists(), "HDF5 file should have been created"

            loaded_acts, attrs = ActivationExtractor.load_from_hdf5(h5_path)

            assert "stage_1_enc_out" in loaded_acts, "stage_1 should be in loaded activations"
            assert loaded_acts["stage_1_enc_out"].shape == (196, 64)

            np.testing.assert_array_almost_equal(
                loaded_acts["stage_1_enc_out"],
                dummy_activations["stage_1_enc_out"].numpy(),
                decimal=5,
            )

            assert attrs["scenario_id"] == "test_scenario/trial_001"
            assert attrs["model_name"] == "mock_vlm"

    def test_hdf5_compression_preserves_values(self):
        """gzip-compressed HDF5 datasets should preserve values exactly."""
        model = MockVLM()
        extractor = MockActivationExtractor(model, "mock_vlm")
        original = torch.arange(196 * 64, dtype=torch.float32).reshape(196, 64)

        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = Path(tmpdir) / "compression_test.h5"
            extractor.save_to_hdf5({"stage_1_enc_out": original}, h5_path)
            loaded, _ = ActivationExtractor.load_from_hdf5(h5_path)
            np.testing.assert_array_equal(
                loaded["stage_1_enc_out"],
                original.numpy(),
            )


class TestStageName:
    """Tests for stage name constants."""

    def test_four_stages_defined(self):
        """Should have exactly 4 pipeline stages."""
        assert len(STAGE_NAMES) == 4

    def test_stage_names_match_expected(self):
        expected = [
            "stage_1_enc_out",
            "stage_2_post_proj",
            "stage_3_llm_8",
            "stage_4_llm_16",
        ]
        assert STAGE_NAMES == expected
