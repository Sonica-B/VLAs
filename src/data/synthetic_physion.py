"""
Synthetic physics scene generator that mimics the Physion++ dataset format.

Generates scenes with colored geometric shapes on a ground plane, each shape
assigned randomized physics properties (mass, friction, elasticity). Scenes
can be used to test the full probing pipeline without downloading the 50GB
Physion++ dataset.

Usage:
    from src.data.synthetic_physion import SyntheticPhysicsDataset
    dataset = SyntheticPhysicsDataset(n_scenes=100, image_size=224, seed=42)
    sample = dataset[0]
    print(sample.image.size)           # (224, 224)
    print(sample.object_masks.shape)   # (224, 224)
    print(sample.physics_labels.keys()) # dict_keys(['mass', 'friction', 'elasticity'])

    # Save to disk for persistence
    from src.data.synthetic_physion import save_synthetic_dataset
    save_synthetic_dataset("data/synthetic_physion", n_scenes=100)
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

from src.data.physion_loader import PhysionSample


# Background color options for visual variety
_BG_COLORS = [
    (210, 210, 220),
    (220, 215, 210),
    (205, 220, 215),
    (215, 210, 205),
]

# Object shape palette: (r, g, b) base colors
_SHAPE_PALETTE = [
    (220, 80, 80),    # red
    (80, 180, 80),    # green
    (80, 120, 220),   # blue
    (220, 180, 60),   # yellow
    (180, 80, 200),   # purple
    (60, 200, 200),   # cyan
    (220, 130, 60),   # orange
    (150, 220, 80),   # lime
]


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    color: Tuple[int, int, int],
    obj_id: int,
) -> None:
    """Draw a filled circle on the image and corresponding mask region."""
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    outline = tuple(max(0, c - 60) for c in color)
    draw.ellipse(bbox, fill=color, outline=outline, width=2)

    # Fill mask using vectorized numpy
    H, W = mask.shape
    r_start = max(0, cy - radius)
    r_end = min(H, cy + radius + 1)
    c_start = max(0, cx - radius)
    c_end = min(W, cx + radius + 1)

    rows = np.arange(r_start, r_end)
    cols = np.arange(c_start, c_end)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    inside = (rr - cy) ** 2 + (cc - cx) ** 2 <= radius ** 2
    mask[r_start:r_end, c_start:c_end][inside] = obj_id


def _draw_rect(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    cx: int,
    cy: int,
    half_w: int,
    half_h: int,
    color: Tuple[int, int, int],
    obj_id: int,
) -> None:
    """Draw a filled rectangle on the image and mask."""
    H, W = mask.shape
    bbox = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
    outline = tuple(max(0, c - 60) for c in color)
    draw.rectangle(bbox, fill=color, outline=outline, width=2)

    r_start = max(0, cy - half_h)
    r_end = min(H, cy + half_h)
    c_start = max(0, cx - half_w)
    c_end = min(W, cx + half_w)
    mask[r_start:r_end, c_start:c_end] = obj_id


def _tint(base_color: Tuple[int, int, int], factor: float) -> Tuple[int, int, int]:
    """Darken/lighten a color. factor < 1 darkens, factor > 1 lightens."""
    return tuple(min(255, max(0, int(c * factor))) for c in base_color)


def generate_scene(
    idx: int,
    image_size: int = 224,
    num_objects_range: Tuple[int, int] = (2, 4),
    seed: Optional[int] = None,
) -> PhysionSample:
    """Generate a single synthetic physics scene.

    Renders 2-4 colored geometric objects on a light background with a
    simulated ground plane. Each object is assigned random physics properties.

    Args:
        idx: Scene index (used as part of seed if seed is None).
        image_size: Output image size in pixels (square).
        num_objects_range: (min, max) number of objects per scene.
        seed: RNG seed. If None, uses idx * 1337 for reproducibility.

    Returns:
        PhysionSample with populated image, mask, and physics labels.
    """
    rng = np.random.default_rng(seed if seed is not None else idx * 1337 + 42)
    r = random.Random(int(rng.integers(0, 2**31)))

    num_objects = int(rng.integers(num_objects_range[0], num_objects_range[1] + 1))

    # Background
    bg_color = _BG_COLORS[idx % len(_BG_COLORS)]
    image = Image.new("RGB", (image_size, image_size), color=bg_color)
    draw = ImageDraw.Draw(image)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    # Ground plane (bottom 15%)
    ground_y = int(image_size * 0.85)
    ground_color = (110, 90, 70)
    draw.rectangle([0, ground_y, image_size, image_size], fill=ground_color)

    # Physics properties: log-uniform mass [0.1, 5.0], uniform for friction/elasticity
    masses = np.exp(rng.uniform(np.log(0.1), np.log(5.0), size=num_objects)).astype(np.float32)
    frictions = rng.uniform(0.1, 1.0, size=num_objects).astype(np.float32)
    elasticities = rng.uniform(0.1, 1.0, size=num_objects).astype(np.float32)

    usable_area = int(image_size * 0.85)  # Above the ground
    min_size = max(20, image_size // 10)
    max_size = image_size // 4

    for i in range(num_objects):
        obj_id = i + 1
        base_color = _SHAPE_PALETTE[i % len(_SHAPE_PALETTE)]
        # Heavier objects appear slightly darker
        mass_factor = 1.0 - 0.25 * (masses[i] / 5.0)
        color = _tint(base_color, mass_factor)

        # Random size
        size = int(rng.integers(min_size, max_size))

        # Position: ensure object is within usable image area
        cx = int(rng.integers(size + 5, image_size - size - 5))
        cy = int(rng.integers(size + 5, usable_area - size - 5))

        # Shape: circle or rectangle
        if rng.random() > 0.5:
            _draw_circle(draw, mask, cx, cy, size // 2, color, obj_id)
        else:
            half_w = size // 2
            half_h = int(size * rng.uniform(0.5, 1.2)) // 2
            _draw_rect(draw, mask, cx, cy, half_w, half_h, color, obj_id)

    return PhysionSample(
        image=image,
        object_masks=mask,
        physics_labels={
            "mass": masses,
            "friction": frictions,
            "elasticity": elasticities,
        },
        scenario_id=f"synthetic/scene_{idx:04d}",
        frame_idx=0,
        num_objects=num_objects,
        metadata={"generated": True, "idx": idx},
    )


class SyntheticPhysicsDataset(Dataset):
    """In-memory synthetic physics dataset that mimics Physion++ structure.

    Generates scenes on construction and holds them in memory. Use this for
    end-to-end pipeline testing without the real dataset.

    Args:
        n_scenes: Number of scenes to generate. Default 100.
        image_size: Image size in pixels (square). Default 224.
        seed: Global RNG seed. Default 42.
        num_objects_range: (min, max) objects per scene. Default (2, 4).

    Example:
        >>> ds = SyntheticPhysicsDataset(n_scenes=100)
        >>> sample = ds[0]
        >>> print(sample.physics_labels["mass"])  # [N_objects] float32
    """

    def __init__(
        self,
        n_scenes: int = 100,
        image_size: int = 224,
        seed: int = 42,
        num_objects_range: Tuple[int, int] = (2, 4),
    ) -> None:
        self.n_scenes = n_scenes
        self.image_size = image_size
        self.seed = seed
        self.num_objects_range = num_objects_range

        self._scenes: List[PhysionSample] = [
            generate_scene(i, image_size, num_objects_range, seed=seed + i)
            for i in range(n_scenes)
        ]

    def __len__(self) -> int:
        return self.n_scenes

    def __getitem__(self, idx: int) -> PhysionSample:
        return self._scenes[idx]

    def __iter__(self) -> Iterator[PhysionSample]:
        return iter(self._scenes)

    def get_split(self, split: str, seed: int = 42) -> "SyntheticPhysicsDataset":
        """Return a subset dataset for train/val/test split."""
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.n_scenes)
        n_train = int(self.n_scenes * 0.70)
        n_val = int(self.n_scenes * 0.15)

        if split == "train":
            selected = indices[:n_train]
        elif split == "val":
            selected = indices[n_train : n_train + n_val]
        else:
            selected = indices[n_train + n_val :]

        subset = SyntheticPhysicsDataset.__new__(SyntheticPhysicsDataset)
        subset.n_scenes = len(selected)
        subset.image_size = self.image_size
        subset.seed = self.seed
        subset.num_objects_range = self.num_objects_range
        subset._scenes = [self._scenes[i] for i in selected]
        return subset


def save_synthetic_dataset(
    output_dir: str | Path,
    n_scenes: int = 100,
    image_size: int = 224,
    seed: int = 42,
) -> Path:
    """Generate and save a synthetic dataset to disk.

    Creates:
        {output_dir}/images/scene_XXXX.png   — RGB images
        {output_dir}/masks/scene_XXXX.png    — grayscale segmentation masks
        {output_dir}/metadata.json           — list of scene metadata

    Args:
        output_dir: Root directory for the dataset.
        n_scenes: Number of scenes to generate.
        image_size: Image size in pixels.
        seed: RNG seed.

    Returns:
        Path to output_dir.
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    metadata_list: List[Dict[str, Any]] = []

    for i in range(n_scenes):
        sample = generate_scene(i, image_size=image_size, seed=seed + i)

        img_path = images_dir / f"scene_{i:04d}.png"
        mask_path = masks_dir / f"scene_{i:04d}.png"

        sample.image.save(img_path)
        Image.fromarray(sample.object_masks).save(mask_path)

        metadata_list.append({
            "scenario_id": sample.scenario_id,
            "image": str(img_path.name),
            "mask": str(mask_path.name),
            "num_objects": sample.num_objects,
            "physics_labels": {
                k: v.tolist() for k, v in sample.physics_labels.items()
            },
        })

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"Saved {n_scenes} synthetic scenes to {output_dir}")
    return output_dir


class SyntheticPhysionDiskDataset(Dataset):
    """Load a synthetic dataset previously saved to disk by save_synthetic_dataset().

    Args:
        root: Path to the directory created by save_synthetic_dataset().
        image_size: Resize images to this size. Default 224.
    """

    def __init__(self, root: str | Path, image_size: int = 224) -> None:
        self.root = Path(root)
        self.image_size = image_size

        meta_path = self.root / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.root}. "
                                    "Run save_synthetic_dataset() first.")
        with open(meta_path) as f:
            self._metadata: List[Dict[str, Any]] = json.load(f)

    def __len__(self) -> int:
        return len(self._metadata)

    def __getitem__(self, idx: int) -> PhysionSample:
        meta = self._metadata[idx]

        img = Image.open(self.root / "images" / meta["image"]).convert("RGB")
        if img.size != (self.image_size, self.image_size):
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)

        mask = np.array(
            Image.open(self.root / "masks" / meta["mask"]).convert("L").resize(
                (self.image_size, self.image_size), Image.NEAREST
            ),
            dtype=np.uint8,
        )

        physics_labels = {
            k: np.array(v, dtype=np.float32)
            for k, v in meta["physics_labels"].items()
        }

        return PhysionSample(
            image=img,
            object_masks=mask,
            physics_labels=physics_labels,
            scenario_id=meta["scenario_id"],
            frame_idx=0,
            num_objects=meta["num_objects"],
        )
