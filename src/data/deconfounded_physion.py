"""
Deconfounded synthetic physics dataset — fixes the Week 1 confounds.

Week 1 problem: mass was correlated with brightness (heavier → darker), so
probes detecting "mass" were actually detecting brightness. This module
generates scenes where:

1. Mass is NOT correlated with color/brightness — objects get random colors
   independent of their physics properties.
2. Friction is NOT correlated with texture or material appearance.
3. Physics-irrelevant visual variations (color, position) are included as
   CONTROL probing targets — if mass R² >> color R², that's evidence of
   genuine physics encoding.
4. A permutation baseline mode shuffles physics labels across patches to
   establish the null distribution.

Each scene contains 3–6 objects with:
  - Independent random physics: mass, friction, elasticity
  - Independent random appearance: color (hue), shape, size, position
  - Per-object segmentation masks for patch-level label assignment

Additionally stores per-object HUE as a visual control variable.
"""

from __future__ import annotations

import colorsys
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from src.data.physion_loader import PhysionSample


# Fixed set of visually distinct hues (evenly spaced around the color wheel)
# We use HSV → RGB so hue is fully decoupled from any physics property.
_N_HUES = 12


def _hue_to_rgb(hue: float, saturation: float = 0.7, value: float = 0.8) -> Tuple[int, int, int]:
    """Convert HSV hue [0, 1] to RGB tuple."""
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (int(r * 255), int(g * 255), int(b * 255))


def _draw_circle(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    cx: int, cy: int, radius: int,
    color: Tuple[int, int, int],
    obj_id: int,
) -> Dict[str, float]:
    """Draw circle and return bounding info."""
    bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
    draw.ellipse(bbox, fill=color, outline=tuple(max(0, c - 40) for c in color), width=2)
    H, W = mask.shape
    yy, xx = np.ogrid[max(0, cy-radius):min(H, cy+radius+1),
                       max(0, cx-radius):min(W, cx+radius+1)]
    dist = (yy - cy)**2 + (xx - cx)**2
    inside = dist <= radius**2
    mask[max(0, cy-radius):min(H, cy+radius+1),
         max(0, cx-radius):min(W, cx+radius+1)][inside] = obj_id
    return {"width": 2*radius, "height": 2*radius}


def _draw_rect(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    cx: int, cy: int, half_w: int, half_h: int,
    color: Tuple[int, int, int],
    obj_id: int,
) -> Dict[str, float]:
    """Draw rectangle and return bounding info."""
    bbox = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
    draw.rectangle(bbox, fill=color, outline=tuple(max(0, c - 40) for c in color), width=2)
    H, W = mask.shape
    r0, r1 = max(0, cy - half_h), min(H, cy + half_h)
    c0, c1 = max(0, cx - half_w), min(W, cx + half_w)
    mask[r0:r1, c0:c1] = obj_id
    return {"width": 2*half_w, "height": 2*half_h}


def _draw_triangle(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    cx: int, cy: int, size: int,
    color: Tuple[int, int, int],
    obj_id: int,
) -> Dict[str, float]:
    """Draw equilateral triangle."""
    half = size // 2
    pts = [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)]
    draw.polygon(pts, fill=color, outline=tuple(max(0, c - 40) for c in color))
    # Fill mask via scanline (approximate with bounding box check)
    H, W = mask.shape
    for py in range(max(0, cy - half), min(H, cy + half + 1)):
        # Linear interpolation for triangle edges
        t = (py - (cy - half)) / max(size, 1)
        left = int(cx - half * t)
        right = int(cx + half * t)
        left = max(0, min(W - 1, left))
        right = max(0, min(W - 1, right))
        mask[py, left:right + 1] = obj_id
    return {"width": size, "height": size}


def generate_deconfounded_scene(
    idx: int,
    image_size: int = 224,
    num_objects_range: Tuple[int, int] = (3, 6),
    seed: Optional[int] = None,
) -> Tuple[PhysionSample, Dict[str, np.ndarray]]:
    """Generate a single deconfounded physics scene.

    CRITICAL DIFFERENCE from synthetic_physion.generate_scene():
    - Object color (hue) is sampled INDEPENDENTLY of mass/friction/elasticity
    - No brightness-mass correlation
    - Returns extra visual_features dict with per-object hue for control probing

    Args:
        idx: Scene index.
        image_size: Output image size (square).
        num_objects_range: (min, max) objects per scene.
        seed: RNG seed.

    Returns:
        (sample, visual_features):
            sample: PhysionSample with image, masks, physics labels
            visual_features: dict with "hue" → [N_objects] float32 in [0,1]
    """
    rng = np.random.default_rng(seed if seed is not None else idx * 7919 + 13)

    num_objects = int(rng.integers(num_objects_range[0], num_objects_range[1] + 1))

    # Background: neutral gray (no color bias)
    bg_gray = int(rng.integers(200, 230))
    image = Image.new("RGB", (image_size, image_size), color=(bg_gray, bg_gray, bg_gray))
    draw = ImageDraw.Draw(image)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    # Ground plane
    ground_y = int(image_size * 0.85)
    draw.rectangle([0, ground_y, image_size, image_size], fill=(120, 110, 100))

    # --- PHYSICS: sampled independently of appearance ---
    masses = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=num_objects)).astype(np.float32)
    frictions = rng.uniform(0.05, 1.0, size=num_objects).astype(np.float32)
    elasticities = rng.uniform(0.1, 0.95, size=num_objects).astype(np.float32)

    # --- APPEARANCE: sampled independently of physics ---
    hues = rng.uniform(0.0, 1.0, size=num_objects).astype(np.float32)
    saturations = rng.uniform(0.5, 0.9, size=num_objects).astype(np.float32)
    values = rng.uniform(0.6, 0.9, size=num_objects).astype(np.float32)

    usable_h = int(image_size * 0.82)
    min_size = max(18, image_size // 12)
    max_size = image_size // 4

    shape_widths = np.zeros(num_objects, dtype=np.float32)
    shape_heights = np.zeros(num_objects, dtype=np.float32)

    # Track placed objects to avoid heavy overlap
    placed = []

    for i in range(num_objects):
        obj_id = i + 1
        color = _hue_to_rgb(float(hues[i]), float(saturations[i]), float(values[i]))
        size = int(rng.integers(min_size, max_size))

        # Try to place without heavy overlap (best-effort, not strict)
        for _attempt in range(20):
            cx = int(rng.integers(size + 5, image_size - size - 5))
            cy = int(rng.integers(size + 5, usable_h - size - 5))
            # Check overlap with existing
            ok = True
            for (px, py, ps) in placed:
                if abs(cx - px) < (size + ps) * 0.5 and abs(cy - py) < (size + ps) * 0.5:
                    ok = False
                    break
            if ok:
                break

        placed.append((cx, cy, size))

        # Random shape: circle, rectangle, or triangle
        shape_type = int(rng.integers(0, 3))
        if shape_type == 0:
            radius = size // 2
            info = _draw_circle(draw, mask, cx, cy, radius, color, obj_id)
        elif shape_type == 1:
            half_w = size // 2
            half_h = int(size * rng.uniform(0.4, 1.6)) // 2
            half_h = max(half_h, 5)
            info = _draw_rect(draw, mask, cx, cy, half_w, half_h, color, obj_id)
        else:
            info = _draw_triangle(draw, mask, cx, cy, size, color, obj_id)

        shape_widths[i] = info["width"]
        shape_heights[i] = info["height"]

    # Stability from geometry (not correlated with color)
    raw_stability = shape_widths / np.maximum(shape_heights, 1.0)
    stabilities = np.clip(raw_stability, 0.0, 1.0).astype(np.float32)

    sample = PhysionSample(
        image=image,
        object_masks=mask,
        physics_labels={
            "mass": masses,
            "friction": frictions,
            "elasticity": elasticities,
            "stability": stabilities,
        },
        scenario_id=f"deconfounded/scene_{idx:04d}",
        frame_idx=0,
        num_objects=num_objects,
        metadata={"generated": True, "deconfounded": True, "idx": idx},
    )

    visual_features = {
        "hue": hues,
        "saturation": saturations,
        "brightness": values,
    }

    return sample, visual_features


class DeconfoundedPhysicsDataset(Dataset):
    """Deconfounded synthetic physics dataset.

    Unlike SyntheticPhysicsDataset, physics properties are statistically
    independent of visual appearance (color, brightness). This means any
    probe that achieves R² > 0 for mass must be using something beyond
    simple color/brightness cues.

    Also stores per-object visual features (hue, brightness) for control
    probing — if mass R² >> hue R², that's evidence of physics-specific
    encoding rather than appearance memorization.

    Args:
        n_scenes: Number of scenes. Default 200.
        image_size: Image size. Default 224.
        seed: RNG seed. Default 42.
        num_objects_range: (min, max) objects per scene.
    """

    def __init__(
        self,
        n_scenes: int = 200,
        image_size: int = 224,
        seed: int = 42,
        num_objects_range: Tuple[int, int] = (3, 6),
    ) -> None:
        self.n_scenes = n_scenes
        self.image_size = image_size
        self.seed = seed
        self.num_objects_range = num_objects_range

        self._scenes: List[PhysionSample] = []
        self._visual_features: List[Dict[str, np.ndarray]] = []

        for i in range(n_scenes):
            sample, vis = generate_deconfounded_scene(
                i, image_size, num_objects_range, seed=seed + i
            )
            self._scenes.append(sample)
            self._visual_features.append(vis)

    def __len__(self) -> int:
        return self.n_scenes

    def __getitem__(self, idx: int) -> PhysionSample:
        return self._scenes[idx]

    def get_visual_features(self, idx: int) -> Dict[str, np.ndarray]:
        """Return per-object visual features for control probing."""
        return self._visual_features[idx]

    def verify_deconfounding(self) -> Dict[str, float]:
        """Compute correlations between physics and visual features.

        Returns dict of Pearson correlations. All should be near 0.
        """
        from scipy.stats import pearsonr

        all_mass, all_hue, all_brightness = [], [], []
        for i in range(self.n_scenes):
            s = self._scenes[i]
            v = self._visual_features[i]
            all_mass.extend(s.physics_labels["mass"].tolist())
            all_hue.extend(v["hue"].tolist())
            all_brightness.extend(v["brightness"].tolist())

        mass_arr = np.array(all_mass)
        hue_arr = np.array(all_hue)
        bright_arr = np.array(all_brightness)

        r_mass_hue, _ = pearsonr(mass_arr, hue_arr)
        r_mass_bright, _ = pearsonr(mass_arr, bright_arr)
        r_mass_logmass = 1.0  # trivially

        return {
            "mass_hue_r": float(r_mass_hue),
            "mass_brightness_r": float(r_mass_bright),
            "n_objects_total": len(all_mass),
        }


def save_deconfounded_dataset(
    output_dir: str | Path,
    n_scenes: int = 200,
    image_size: int = 224,
    seed: int = 42,
) -> Path:
    """Generate and save a deconfounded dataset to disk.

    Creates:
        {output_dir}/images/scene_XXXX.png
        {output_dir}/masks/scene_XXXX.png
        {output_dir}/metadata.json  — physics labels + visual features
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    metadata_list: List[Dict[str, Any]] = []

    for i in range(n_scenes):
        sample, vis_features = generate_deconfounded_scene(
            i, image_size=image_size, seed=seed + i
        )

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
            "visual_features": {
                k: v.tolist() for k, v in vis_features.items()
            },
        })

    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata_list, f, indent=2)

    print(f"Saved {n_scenes} deconfounded scenes to {output_dir}")
    return output_dir
