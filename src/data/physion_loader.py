"""
Physion++ dataset loader.

Physion++ is a physics simulation dataset containing video scenarios with
ground-truth physical properties (mass, friction, elasticity) per object,
plus temporal dynamics (positions, velocities, segmentation masks over time).

Reference:
    Bear et al. (2023). Physion++: Evaluating Physical Scene Understanding
    that Requires Online Inference. NeurIPS 2023.

Dataset structure on disk:
    data/physion/
        {scenario_type}/        # e.g., dominoes, support, contain, ...
            trial_{id}/
                metadata.pkl    # Physics params, object properties
                video/
                    frame_{t}.png
                masks/
                    frame_{t}.png   # Segmentation masks (uint8, object ID per pixel)
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


# Physion++ scenario types as defined in the original dataset
SCENARIO_TYPES = [
    "dominoes",
    "support",
    "contain",
    "roll",
    "link",
    "drape",
    "drop",
    "collide",
]

# Physics property keys expected in metadata.pkl
PHYSICS_PROPERTY_KEYS = ["mass", "friction", "elasticity"]


class PhysionSample:
    """A single Physion++ sample with image, masks, and physics labels.

    Attributes:
        image: PIL image at the chosen frame.
        object_masks: uint8 array [H, W] — pixel value = object ID (0 = background).
        physics_labels: dict mapping property name → per-object array.
        scenario_id: string identifier, e.g., "dominoes/trial_003".
        frame_idx: which frame was selected.
        num_objects: number of dynamic objects in the scene.
    """

    def __init__(
        self,
        image: Image.Image,
        object_masks: np.ndarray,
        physics_labels: Dict[str, np.ndarray],
        scenario_id: str,
        frame_idx: int,
        num_objects: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.image = image
        self.object_masks = object_masks
        self.physics_labels = physics_labels
        self.scenario_id = scenario_id
        self.frame_idx = frame_idx
        self.num_objects = num_objects
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to a JSON-serializable dict (images as paths, not tensors)."""
        return {
            "scenario_id": self.scenario_id,
            "frame_idx": self.frame_idx,
            "num_objects": self.num_objects,
            "physics_labels": {k: v.tolist() for k, v in self.physics_labels.items()},
        }


class PhysionLoader(Dataset):
    """PyTorch Dataset wrapper for the Physion++ dataset.

    Iterates over (scenario, frame) pairs and returns PhysionSample objects.

    Args:
        root: Path to the Physion++ root directory (contains scenario subdirs).
        split: One of "train", "val", "test". Split is computed deterministically
               from scenario index (70/15/15 by default).
        scenario_types: Which scenario types to include. Defaults to all 8.
        frame_stride: Sample every Nth frame from each trial. Default 5.
        image_size: Resize images to this size (square). Default 448.
        cache_metadata: If True, pre-load all metadata.pkl files at init time.
        seed: Random seed for reproducible train/val/test splits.

    Example:
        >>> loader = PhysionLoader("data/physion", split="train")
        >>> sample = loader[0]
        >>> print(sample.physics_labels["mass"])  # [N_objects] float32
        >>> print(sample.object_masks.shape)       # (H, W)
    """

    SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        scenario_types: Optional[List[str]] = None,
        frame_stride: int = 5,
        image_size: int = 448,
        cache_metadata: bool = False,
        seed: int = 42,
    ) -> None:
        assert split in self.SPLIT_RATIOS, f"split must be one of {list(self.SPLIT_RATIOS)}"
        self.root = Path(root)
        self.split = split
        self.scenario_types = scenario_types or SCENARIO_TYPES
        self.frame_stride = frame_stride
        self.image_size = image_size
        self.seed = seed

        # Discover all trial directories
        self._trial_dirs: List[Path] = self._discover_trials()
        # Create (trial_dir, frame_idx) index
        self._index: List[Tuple[Path, int]] = self._build_index()

        # Optional metadata cache
        self._metadata_cache: Dict[Path, Dict] = {}
        if cache_metadata:
            self._preload_metadata()

    def _discover_trials(self) -> List[Path]:
        """Find all trial directories matching the requested scenario types."""
        trials = []
        for scenario_type in self.scenario_types:
            scenario_dir = self.root / scenario_type
            if not scenario_dir.exists():
                continue
            trial_dirs = sorted(scenario_dir.glob("trial_*"))
            trials.extend(trial_dirs)

        # Deterministic split
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(trials))
        n_train = int(len(trials) * self.SPLIT_RATIOS["train"])
        n_val = int(len(trials) * self.SPLIT_RATIOS["val"])

        if self.split == "train":
            selected = indices[:n_train]
        elif self.split == "val":
            selected = indices[n_train : n_train + n_val]
        else:  # test
            selected = indices[n_train + n_val :]

        return [trials[i] for i in selected]

    def _build_index(self) -> List[Tuple[Path, int]]:
        """Build flat index of (trial_dir, frame_idx) pairs."""
        index = []
        for trial_dir in self._trial_dirs:
            frame_files = sorted((trial_dir / "video").glob("frame_*.png"))
            frame_indices = list(range(0, len(frame_files), self.frame_stride))
            for fi in frame_indices:
                index.append((trial_dir, fi))
        return index

    def _preload_metadata(self) -> None:
        """Pre-load all metadata.pkl files into memory."""
        for trial_dir in tqdm(self._trial_dirs, desc="Loading Physion++ metadata"):
            meta_path = trial_dir / "metadata.pkl"
            if meta_path.exists():
                with open(meta_path, "rb") as f:
                    self._metadata_cache[trial_dir] = pickle.load(f)

    def _load_metadata(self, trial_dir: Path) -> Dict[str, Any]:
        """Load metadata for a trial, using cache if available."""
        if trial_dir in self._metadata_cache:
            return self._metadata_cache[trial_dir]
        meta_path = trial_dir / "metadata.pkl"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return meta

    def _extract_physics_labels(
        self, meta: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """Parse metadata.pkl and extract per-object physics labels.

        Returns:
            physics_labels: dict mapping property → float32 array [N_objects].
            num_objects: number of dynamic objects.
        """
        # TODO: Adapt to actual Physion++ metadata schema after inspecting real .pkl files.
        # The schema below is based on the Physion paper description.
        objects = meta.get("objects", {})
        num_objects = len(objects)

        labels: Dict[str, np.ndarray] = {}
        for prop in PHYSICS_PROPERTY_KEYS:
            prop_values = []
            for obj_id in sorted(objects.keys()):
                obj_meta = objects[obj_id]
                # Physion++ stores physics_params as a nested dict
                physics_params = obj_meta.get("physics_params", {})
                value = physics_params.get(prop, float("nan"))
                prop_values.append(float(value))
            labels[prop] = np.array(prop_values, dtype=np.float32)

        return labels, num_objects

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> PhysionSample:
        trial_dir, frame_idx = self._index[idx]
        scenario_id = f"{trial_dir.parent.name}/{trial_dir.name}"

        # Load image
        frame_path = trial_dir / "video" / f"frame_{frame_idx:04d}.png"
        image = Image.open(frame_path).convert("RGB").resize(
            (self.image_size, self.image_size), Image.BILINEAR
        )

        # Load segmentation mask
        mask_path = trial_dir / "masks" / f"frame_{frame_idx:04d}.png"
        mask = np.array(Image.open(mask_path).convert("L").resize(
            (self.image_size, self.image_size), Image.NEAREST
        ), dtype=np.uint8)

        # Load physics labels from metadata
        meta = self._load_metadata(trial_dir)
        physics_labels, num_objects = self._extract_physics_labels(meta)

        return PhysionSample(
            image=image,
            object_masks=mask,
            physics_labels=physics_labels,
            scenario_id=scenario_id,
            frame_idx=frame_idx,
            num_objects=num_objects,
            metadata=meta,
        )

    def load_scenario(self, scenario_id: str) -> Dict[str, Any]:
        """Load all frames and temporal data for a given scenario.

        Args:
            scenario_id: e.g., "dominoes/trial_000"

        Returns:
            dict with keys: frames, masks, object_positions, physics_labels, metadata.
        """
        parts = scenario_id.split("/")
        trial_dir = self.root / parts[0] / parts[1]
        meta = self._load_metadata(trial_dir)
        physics_labels, num_objects = self._extract_physics_labels(meta)

        frame_files = sorted((trial_dir / "video").glob("frame_*.png"))
        frames = [
            np.array(Image.open(f).convert("RGB").resize(
                (self.image_size, self.image_size), Image.BILINEAR
            ))
            for f in frame_files
        ]

        mask_files = sorted((trial_dir / "masks").glob("frame_*.png"))
        masks = [
            np.array(Image.open(f).convert("L").resize(
                (self.image_size, self.image_size), Image.NEAREST
            ), dtype=np.uint8)
            for f in mask_files
        ]

        # TODO: Extract per-frame object positions from metadata
        # meta["frames"] likely contains position data — adapt after inspecting real data
        object_positions = meta.get("object_positions", None)

        return {
            "frames": np.stack(frames),            # [T, H, W, 3]
            "masks": np.stack(masks),              # [T, H, W]
            "object_positions": object_positions,  # [T, N_objects, 3] or None
            "physics_labels": physics_labels,
            "metadata": meta,
            "num_objects": num_objects,
        }

    def __iter__(self) -> Iterator[PhysionSample]:
        for i in range(len(self)):
            yield self[i]
