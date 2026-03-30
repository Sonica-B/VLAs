"""
Unified probe training loop.

Handles loading activations from HDF5, splitting into train/val/test,
training probes, logging results, and saving metrics.

Supports both LinearProbe and MLPProbe via a common interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np

from src.probing.linear_probe import LinearProbe, compute_per_patch_r2
from src.probing.mlp_probe import MLPProbe

logger = logging.getLogger(__name__)

ProbeType = Union[LinearProbe, MLPProbe]

# Physics variable column indices in the patch_labels array
VARIABLE_COLUMN_MAP = {
    "mass": 0,
    "friction": 1,
    "elasticity": 2,
    "stability": 3,
}


class ProbeTrainer:
    """Trains and evaluates a probe on pre-extracted HDF5 activations.

    Args:
        probe: A LinearProbe or MLPProbe instance.
        output_dir: Directory for saving metrics and per-patch scores.
        train_split: Fraction of samples for training.
        val_split: Fraction of samples for validation.
        seed: Random seed for reproducibility.
        device: Compute device for MLPProbe.

    Example:
        >>> probe = LinearProbe(input_dim=1280)
        >>> trainer = ProbeTrainer(probe, output_dir="results/probe_metrics/")
        >>> trainer.fit_from_hdf5(
        ...     activation_dir="results/activations/qwen_debug/",
        ...     stage="stage_1_enc_out",
        ...     target_variable="mass",
        ... )
        >>> metrics = trainer.evaluate()
        >>> print(metrics)
    """

    def __init__(
        self,
        probe: ProbeType,
        output_dir: str = "results/probe_metrics/",
        train_split: float = 0.70,
        val_split: float = 0.15,
        seed: int = 42,
        device: str = "cuda",
    ) -> None:
        self.probe = probe
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_split = train_split
        self.val_split = val_split
        self.seed = seed
        self.device = device

        # Set after data loading
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._X_val: Optional[np.ndarray] = None
        self._y_val: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._current_stage: Optional[str] = None
        self._current_variable: Optional[str] = None
        self._per_patch_scores: Optional[np.ndarray] = None

    def fit_from_hdf5(
        self,
        activation_dir: str | Path,
        stage: str,
        target_variable: str,
        exclude_background: bool = True,
        max_samples: Optional[int] = None,
    ) -> None:
        """Load activations from HDF5 files and train the probe.

        Args:
            activation_dir: Directory containing .h5 activation files.
            stage: Pipeline stage key, e.g., "stage_1_enc_out".
            target_variable: Physics property to predict: "mass", "friction",
                             "elasticity", or "stability".
            exclude_background: If True, mask out background patches (NaN labels).
            max_samples: If set, limit the number of HDF5 files loaded.
        """
        if target_variable not in VARIABLE_COLUMN_MAP:
            raise ValueError(
                f"Unknown target variable: {target_variable!r}. "
                f"Choose from {list(VARIABLE_COLUMN_MAP.keys())}"
            )

        self._current_stage = stage
        self._current_variable = target_variable
        var_col = VARIABLE_COLUMN_MAP[target_variable]

        logger.info(f"Loading activations: stage={stage}, variable={target_variable}")
        X_list, y_list = [], []

        h5_files = sorted(Path(activation_dir).glob("*.h5"))
        if max_samples:
            h5_files = h5_files[:max_samples]

        for h5_path in h5_files:
            try:
                with h5py.File(h5_path, "r") as f:
                    if stage not in f:
                        continue
                    acts = f[stage][:]                      # [N_patches, D]
                    labels_json = f.attrs.get("physics_labels", None)
                    if labels_json is None:
                        continue
                    labels_dict = json.loads(labels_json)  # {variable: [N_objects]}

                    # Patch labels stored as a [N_patches, 4] array in attrs or separate dataset
                    if "patch_labels" in f:
                        patch_labels = f["patch_labels"][:]  # [N_patches, 4]
                    else:
                        # If not stored separately, skip — need patch_label_assigner
                        continue

                    y_patch = patch_labels[:, var_col]      # [N_patches]
                    X_list.append(acts)
                    y_list.append(y_patch)
            except Exception as e:
                logger.warning(f"Failed to load {h5_path}: {e}")
                continue

        if not X_list:
            raise RuntimeError(
                f"No valid HDF5 files found in {activation_dir} with stage '{stage}' "
                "and patch_labels dataset."
            )

        X_all = np.concatenate(X_list, axis=0)    # [N_total_patches, D]
        y_all = np.concatenate(y_list, axis=0)    # [N_total_patches]

        logger.info(f"  Loaded {X_all.shape[0]} patch samples, D={X_all.shape[1]}")

        # Split
        rng = np.random.default_rng(self.seed)
        n = len(y_all)
        idx = rng.permutation(n)
        n_train = int(n * self.train_split)
        n_val = int(n * self.val_split)

        self._X_train = X_all[idx[:n_train]]
        self._y_train = y_all[idx[:n_train]]
        self._X_val = X_all[idx[n_train : n_train + n_val]]
        self._y_val = y_all[idx[n_train : n_train + n_val]]
        self._X_test = X_all[idx[n_train + n_val :]]
        self._y_test = y_all[idx[n_train + n_val :]]

        # Fit probe
        if isinstance(self.probe, MLPProbe):
            self.probe.fit(self._X_train, self._y_train, self._X_val, self._y_val, verbose=True)
        else:
            self.probe.fit(self._X_train, self._y_train)

        logger.info("Probe fitted successfully.")

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the fitted probe on the held-out test set.

        Returns:
            Dict with keys: r2, pearson_r, mse, mae.
        """
        if self._X_test is None:
            raise RuntimeError("Call fit_from_hdf5() first.")
        metrics = self.probe.score(self._X_test, self._y_test)
        logger.info(
            f"Test results [{self._current_stage}, {self._current_variable}]: "
            + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )
        return metrics

    def save_metrics(self, metrics: Dict[str, float], run_name: str = "") -> None:
        """Save evaluation metrics to JSON.

        Args:
            metrics: Output of evaluate().
            run_name: Optional label for the output filename.
        """
        out_path = self.output_dir / f"metrics_{run_name or 'probe'}.json"
        full_record = {
            "stage": self._current_stage,
            "variable": self._current_variable,
            "probe_type": type(self.probe).__name__,
            "metrics": metrics,
        }
        with open(out_path, "w") as f:
            json.dump(full_record, f, indent=2)
        logger.info(f"Metrics saved to {out_path}")

    def compute_and_save_per_patch_scores(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        patch_grid_size: int = 14,
        run_name: str = "",
    ) -> np.ndarray:
        """Compute per-patch R² scores for saliency map generation.

        Args:
            X_all: [N_samples, N_patches, D] activation tensor.
            y_all: [N_samples, N_patches] per-patch label tensor.
            patch_grid_size: Spatial patch grid size.
            run_name: Label for the output filename.

        Returns:
            r2_per_patch: float32 array [N_patches].
        """
        r2_scores = compute_per_patch_r2(
            X_all, y_all, patch_grid_size=patch_grid_size, seed=self.seed
        )
        self._per_patch_scores = r2_scores

        out_path = self.output_dir / f"per_patch_r2_{run_name or 'scores'}.npy"
        np.save(out_path, r2_scores)
        logger.info(f"Per-patch R² scores saved to {out_path}")
        return r2_scores

    def get_per_patch_scores(self) -> Optional[np.ndarray]:
        """Return the most recently computed per-patch R² scores."""
        return self._per_patch_scores
