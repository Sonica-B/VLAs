"""
Realistic material-based physics dataset — bridge between deconfounded shapes and Physion++.

The deconfounded dataset (colored shapes with random physics) gave negative R² because
VLMs cannot infer physics from arbitrary colors. In the real world, physics properties
correlate with MATERIAL APPEARANCE:
  - Metal objects (silver/gray, reflective) → heavy, low friction
  - Wood objects (brown, grainy texture) → medium weight, medium friction
  - Rubber objects (black/dark, matte) → medium weight, high friction
  - Foam objects (white/yellow, soft-looking) → light, medium friction
  - Glass objects (transparent/blue-tinted, smooth) → medium weight, low friction
  - Stone objects (gray, rough) → heavy, low friction
  - Plastic objects (bright colored, smooth) → light, low friction

VLMs trained on web data SHOULD encode these material-physics associations. If global
pooled probes achieve high R² on this data, it validates that VLMs learn physics from
material appearance (the "Pixels to Principles" hypothesis).

Each scene contains 2-4 objects of different materials on a surface. Objects have
realistic material-like textures (simulated via procedural patterns) and physics
properties that correlate with their material type.

Crucially, there IS noise in the physics-material mapping (not deterministic), which
prevents probes from just memorizing exact material → property lookups. The correlation
is realistic but imperfect, matching the real world.
"""

from __future__ import annotations

import colorsys
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

from src.data.physion_loader import PhysionSample


# ============================================================================
# Material definitions with realistic physics ranges
# ============================================================================

MATERIALS = {
    "metal": {
        "mass_range": (5.0, 10.0),       # heavy
        "friction_range": (0.15, 0.35),   # smooth, low friction
        "elasticity_range": (0.3, 0.5),   # moderate bounce
        "colors": [
            (180, 180, 195),  # silver
            (160, 165, 175),  # steel
            (140, 145, 155),  # dark steel
            (190, 185, 170),  # brass-tinted
            (170, 175, 185),  # aluminum
        ],
        "texture": "metallic",
    },
    "wood": {
        "mass_range": (2.0, 5.0),         # medium
        "friction_range": (0.35, 0.60),   # moderate friction
        "elasticity_range": (0.2, 0.4),   # low bounce
        "colors": [
            (160, 120, 70),   # oak
            (140, 100, 55),   # walnut
            (180, 140, 85),   # pine
            (120, 85, 50),    # mahogany
            (170, 130, 75),   # birch
        ],
        "texture": "woody",
    },
    "rubber": {
        "mass_range": (1.5, 4.0),         # medium-light
        "friction_range": (0.70, 0.95),   # very high friction
        "elasticity_range": (0.6, 0.9),   # very bouncy
        "colors": [
            (40, 40, 45),     # black rubber
            (50, 45, 50),     # dark gray rubber
            (60, 30, 30),     # dark red rubber
            (30, 50, 35),     # dark green rubber
            (45, 40, 55),     # dark purple rubber
        ],
        "texture": "matte",
    },
    "foam": {
        "mass_range": (0.1, 1.0),         # very light
        "friction_range": (0.40, 0.65),   # moderate friction
        "elasticity_range": (0.1, 0.3),   # absorbs impact
        "colors": [
            (245, 240, 220),  # off-white foam
            (250, 230, 150),  # yellow foam
            (220, 245, 220),  # green foam
            (240, 210, 200),  # pink foam
            (230, 230, 240),  # light blue foam
        ],
        "texture": "soft",
    },
    "glass": {
        "mass_range": (2.0, 5.0),         # medium
        "friction_range": (0.10, 0.25),   # very smooth
        "elasticity_range": (0.5, 0.7),   # moderate bounce
        "colors": [
            (200, 220, 240),  # clear blue-tinted
            (210, 230, 235),  # light cyan
            (195, 215, 230),  # ice blue
            (220, 235, 225),  # green-tinted
            (225, 220, 235),  # purple-tinted
        ],
        "texture": "glossy",
    },
    "stone": {
        "mass_range": (6.0, 10.0),        # very heavy
        "friction_range": (0.30, 0.55),   # moderate friction
        "elasticity_range": (0.1, 0.25),  # almost no bounce
        "colors": [
            (130, 130, 130),  # gray
            (145, 140, 135),  # sandstone
            (110, 115, 120),  # slate
            (155, 150, 140),  # limestone
            (120, 120, 125),  # dark gray
        ],
        "texture": "rough",
    },
    "plastic": {
        "mass_range": (0.5, 2.0),         # light
        "friction_range": (0.20, 0.40),   # smooth
        "elasticity_range": (0.4, 0.65),  # moderate bounce
        "colors": [
            (220, 50, 50),    # red plastic
            (50, 130, 220),   # blue plastic
            (50, 180, 70),    # green plastic
            (230, 180, 40),   # yellow plastic
            (200, 80, 180),   # pink plastic
        ],
        "texture": "smooth",
    },
}

MATERIAL_NAMES = list(MATERIALS.keys())
MATERIAL_TO_IDX = {name: i for i, name in enumerate(MATERIAL_NAMES)}


# ============================================================================
# Procedural texture rendering
# ============================================================================

def _add_texture(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    obj_id: int,
    texture_type: str,
    base_color: Tuple[int, int, int],
    rng: np.random.Generator,
) -> None:
    """Add material-specific texture patterns to an object region."""
    h, w = mask.shape
    obj_pixels = np.where(mask == obj_id)
    if len(obj_pixels[0]) == 0:
        return

    pixels = np.array(image)

    if texture_type == "metallic":
        # Subtle horizontal gradient streaks for metallic look
        for y, x in zip(obj_pixels[0], obj_pixels[1]):
            noise = int(rng.integers(-15, 16))
            streak = int(8 * np.sin(y * 0.3 + x * 0.1))
            r = max(0, min(255, base_color[0] + noise + streak))
            g = max(0, min(255, base_color[1] + noise + streak))
            b = max(0, min(255, base_color[2] + noise + streak + 5))
            pixels[y, x] = [r, g, b]

    elif texture_type == "woody":
        # Wood grain lines
        grain_freq = rng.uniform(0.05, 0.15)
        grain_offset = rng.uniform(0, 2 * np.pi)
        for y, x in zip(obj_pixels[0], obj_pixels[1]):
            grain = int(15 * np.sin(grain_freq * x + grain_offset + 0.02 * y))
            noise = int(rng.integers(-8, 9))
            r = max(0, min(255, base_color[0] + grain + noise))
            g = max(0, min(255, base_color[1] + grain + noise - 3))
            b = max(0, min(255, base_color[2] + grain // 2 + noise))
            pixels[y, x] = [r, g, b]

    elif texture_type == "matte":
        # Uniform dark with very slight noise
        for y, x in zip(obj_pixels[0], obj_pixels[1]):
            noise = int(rng.integers(-5, 6))
            r = max(0, min(255, base_color[0] + noise))
            g = max(0, min(255, base_color[1] + noise))
            b = max(0, min(255, base_color[2] + noise))
            pixels[y, x] = [r, g, b]

    elif texture_type == "soft":
        # Speckled foam-like texture
        for y, x in zip(obj_pixels[0], obj_pixels[1]):
            speckle = int(rng.integers(-20, 21))
            r = max(0, min(255, base_color[0] + speckle))
            g = max(0, min(255, base_color[1] + speckle))
            b = max(0, min(255, base_color[2] + speckle))
            pixels[y, x] = [r, g, b]

    elif texture_type == "glossy":
        # Translucent/reflective highlights
        for y, x in zip(obj_pixels[0], obj_pixels[1]):
            highlight = int(20 * np.sin(y * 0.2) * np.cos(x * 0.2))
            noise = int(rng.integers(-5, 6))
            r = max(0, min(255, base_color[0] + highlight + noise))
            g = max(0, min(255, base_color[1] + highlight + noise))
            b = max(0, min(255, base_color[2] + highlight + noise + 3))
            pixels[y, x] = [r, g, b]

    elif texture_type == "rough":
        # Random coarse noise for stone
        for y, x in zip(obj_pixels[0], obj_pixels[1]):
            noise = int(rng.integers(-25, 26))
            r = max(0, min(255, base_color[0] + noise))
            g = max(0, min(255, base_color[1] + noise))
            b = max(0, min(255, base_color[2] + noise))
            pixels[y, x] = [r, g, b]

    elif texture_type == "smooth":
        # Clean solid color with minimal noise
        for y, x in zip(obj_pixels[0], obj_pixels[1]):
            noise = int(rng.integers(-3, 4))
            r = max(0, min(255, base_color[0] + noise))
            g = max(0, min(255, base_color[1] + noise))
            b = max(0, min(255, base_color[2] + noise))
            pixels[y, x] = [r, g, b]

    # Write back
    image.paste(Image.fromarray(pixels), (0, 0))


def _draw_3d_shape(
    draw: ImageDraw.ImageDraw,
    mask: np.ndarray,
    cx: int, cy: int, size: int,
    color: Tuple[int, int, int],
    obj_id: int,
    shape_type: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Draw a shape with pseudo-3D shading (darker edges, lighter center)."""
    H, W = mask.shape

    # Slightly darker outline for 3D effect
    dark_color = tuple(max(0, c - 50) for c in color)
    light_color = tuple(min(255, c + 30) for c in color)

    if shape_type == 0:  # Circle/sphere
        radius = size // 2
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        # Draw shadow
        shadow_offset = max(2, radius // 8)
        draw.ellipse(
            [b + shadow_offset for b in bbox],
            fill=(80, 80, 80, 60),
        )
        # Draw main shape
        draw.ellipse(bbox, fill=color, outline=dark_color, width=2)
        # Fill mask
        yy, xx = np.ogrid[max(0, cy-radius):min(H, cy+radius+1),
                           max(0, cx-radius):min(W, cx+radius+1)]
        dist = (yy - cy)**2 + (xx - cx)**2
        inside = dist <= radius**2
        mask[max(0, cy-radius):min(H, cy+radius+1),
             max(0, cx-radius):min(W, cx+radius+1)][inside] = obj_id
        return {"width": float(2*radius), "height": float(2*radius)}

    elif shape_type == 1:  # Rectangle/box
        half_w = size // 2
        half_h = int(size * rng.uniform(0.5, 1.5)) // 2
        half_h = max(half_h, 8)
        bbox = [cx - half_w, cy - half_h, cx + half_w, cy + half_h]
        # Shadow
        shadow_offset = max(2, half_w // 8)
        draw.rectangle(
            [b + shadow_offset for b in bbox],
            fill=(80, 80, 80, 60),
        )
        # Main shape
        draw.rectangle(bbox, fill=color, outline=dark_color, width=2)
        # Mask
        r0, r1 = max(0, cy - half_h), min(H, cy + half_h)
        c0, c1 = max(0, cx - half_w), min(W, cx + half_w)
        mask[r0:r1, c0:c1] = obj_id
        return {"width": float(2*half_w), "height": float(2*half_h)}

    else:  # Triangle/pyramid
        half = size // 2
        pts = [(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)]
        # Shadow
        shadow_offset = max(2, half // 8)
        shadow_pts = [(x + shadow_offset, y + shadow_offset) for x, y in pts]
        draw.polygon(shadow_pts, fill=(80, 80, 80, 60))
        # Main shape
        draw.polygon(pts, fill=color, outline=dark_color)
        # Mask (scanline)
        for py in range(max(0, cy - half), min(H, cy + half + 1)):
            t = (py - (cy - half)) / max(size, 1)
            left = int(cx - half * t)
            right = int(cx + half * t)
            left = max(0, min(W - 1, left))
            right = max(0, min(W - 1, right))
            mask[py, left:right + 1] = obj_id
        return {"width": float(size), "height": float(size)}


# ============================================================================
# Scene generation
# ============================================================================

def generate_realistic_scene(
    idx: int,
    image_size: int = 224,
    num_objects_range: Tuple[int, int] = (2, 4),
    seed: Optional[int] = None,
) -> Tuple[PhysionSample, Dict[str, Any]]:
    """Generate a scene with material-textured objects and correlated physics.

    Objects look like their material type, and physics properties correlate with
    material appearance (metal = heavy, rubber = high friction, etc.) but with
    added noise to prevent trivial lookup.

    Args:
        idx: Scene index.
        image_size: Output image size (square). Default 224.
        num_objects_range: (min, max) objects per scene.
        seed: RNG seed.

    Returns:
        (sample, extra_info):
            sample: PhysionSample with image, masks, physics labels
            extra_info: dict with material types, material_idx per object
    """
    rng = np.random.default_rng(seed if seed is not None else idx * 8731 + 17)

    num_objects = int(rng.integers(num_objects_range[0], num_objects_range[1] + 1))

    # Pick materials for this scene (ensure diversity)
    available = list(MATERIAL_NAMES)
    rng.shuffle(available)
    scene_materials = [available[i % len(available)] for i in range(num_objects)]

    # Background: table/surface
    bg_color = tuple(rng.integers(190, 220, size=3).tolist())
    image = Image.new("RGB", (image_size, image_size), color=bg_color)
    draw = ImageDraw.Draw(image)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    # Draw surface/table
    surface_y = int(image_size * 0.80)
    surface_color = tuple(rng.integers(100, 140, size=3).tolist())
    draw.rectangle([0, surface_y, image_size, image_size], fill=surface_color)

    # Physics arrays
    masses = np.zeros(num_objects, dtype=np.float32)
    frictions = np.zeros(num_objects, dtype=np.float32)
    elasticities = np.zeros(num_objects, dtype=np.float32)
    material_indices = np.zeros(num_objects, dtype=np.int32)

    min_size = max(22, image_size // 8)
    max_size = image_size // 3
    usable_h = int(image_size * 0.78)

    placed = []

    for i in range(num_objects):
        obj_id = i + 1
        mat_name = scene_materials[i]
        mat = MATERIALS[mat_name]
        material_indices[i] = MATERIAL_TO_IDX[mat_name]

        # Sample physics from material-specific ranges WITH noise
        masses[i] = rng.uniform(*mat["mass_range"])
        frictions[i] = rng.uniform(*mat["friction_range"])
        elasticities[i] = rng.uniform(*mat["elasticity_range"])

        # Add cross-material noise (10% probability of outlier)
        if rng.random() < 0.10:
            masses[i] *= rng.uniform(0.5, 1.5)
            masses[i] = np.clip(masses[i], 0.1, 12.0)

        # Pick color for this material
        color_idx = int(rng.integers(0, len(mat["colors"])))
        base_color = mat["colors"][color_idx]

        # Object size
        size = int(rng.integers(min_size, max_size))

        # Placement (avoid heavy overlap)
        for _attempt in range(30):
            cx = int(rng.integers(size + 5, image_size - size - 5))
            cy = int(rng.integers(size + 5, usable_h - size - 5))
            ok = True
            for (px, py, ps) in placed:
                if abs(cx - px) < (size + ps) * 0.55 and abs(cy - py) < (size + ps) * 0.55:
                    ok = False
                    break
            if ok:
                break
        placed.append((cx, cy, size))

        # Draw shape
        shape_type = int(rng.integers(0, 3))
        _draw_3d_shape(draw, mask, cx, cy, size, base_color, obj_id, shape_type, rng)

        # Add material texture
        _add_texture(image, draw, mask, obj_id, mat["texture"], base_color, rng)

    # Stability from geometry
    shape_widths = np.zeros(num_objects, dtype=np.float32)
    shape_heights = np.zeros(num_objects, dtype=np.float32)
    for i in range(num_objects):
        obj_pixels = np.where(mask == (i + 1))
        if len(obj_pixels[0]) > 0:
            shape_heights[i] = float(obj_pixels[0].max() - obj_pixels[0].min() + 1)
            shape_widths[i] = float(obj_pixels[1].max() - obj_pixels[1].min() + 1)
    stabilities = np.clip(
        shape_widths / np.maximum(shape_heights, 1.0), 0.0, 1.0
    ).astype(np.float32)

    sample = PhysionSample(
        image=image,
        object_masks=mask,
        physics_labels={
            "mass": masses,
            "friction": frictions,
            "elasticity": elasticities,
            "stability": stabilities,
        },
        scenario_id=f"realistic/scene_{idx:04d}",
        frame_idx=0,
        num_objects=num_objects,
        metadata={
            "generated": True,
            "realistic_materials": True,
            "idx": idx,
        },
    )

    extra_info = {
        "material_names": scene_materials,
        "material_idx": material_indices,
    }

    return sample, extra_info


class RealisticPhysicsDataset(Dataset):
    """Realistic material-based physics dataset.

    Objects have material-specific appearances (metal, wood, rubber, foam, glass,
    stone, plastic) with physics properties that correlate with material type.
    This is the bridge between deconfounded shapes (no physics signal) and real
    Physion++ data.

    Args:
        n_scenes: Number of scenes. Default 300.
        image_size: Image size. Default 224.
        seed: RNG seed. Default 42.
        num_objects_range: (min, max) objects per scene.
    """

    def __init__(
        self,
        n_scenes: int = 300,
        image_size: int = 224,
        seed: int = 42,
        num_objects_range: Tuple[int, int] = (2, 4),
    ) -> None:
        self.n_scenes = n_scenes
        self.image_size = image_size
        self.seed = seed
        self.num_objects_range = num_objects_range

        self._scenes: List[PhysionSample] = []
        self._extra_info: List[Dict[str, Any]] = []

        for i in range(n_scenes):
            sample, extra = generate_realistic_scene(
                i, image_size, num_objects_range, seed=seed + i
            )
            self._scenes.append(sample)
            self._extra_info.append(extra)

    def __len__(self) -> int:
        return self.n_scenes

    def __getitem__(self, idx: int) -> PhysionSample:
        return self._scenes[idx]

    def get_extra_info(self, idx: int) -> Dict[str, Any]:
        """Return material type info for control probing."""
        return self._extra_info[idx]

    def get_material_physics_correlation(self) -> Dict[str, float]:
        """Verify that material appearance correlates with physics.

        Returns dict of Pearson correlations between material index and physics.
        These should be non-zero (unlike deconfounded data).
        """
        from scipy.stats import pearsonr

        all_mass, all_friction, all_elasticity, all_mat_idx = [], [], [], []
        for i in range(self.n_scenes):
            s = self._scenes[i]
            e = self._extra_info[i]
            all_mass.extend(s.physics_labels["mass"].tolist())
            all_friction.extend(s.physics_labels["friction"].tolist())
            all_elasticity.extend(s.physics_labels["elasticity"].tolist())
            all_mat_idx.extend(e["material_idx"].tolist())

        mass_arr = np.array(all_mass)
        mat_arr = np.array(all_mat_idx, dtype=np.float64)

        r_mass, _ = pearsonr(mat_arr, mass_arr)
        r_friction, _ = pearsonr(mat_arr, np.array(all_friction))
        r_elasticity, _ = pearsonr(mat_arr, np.array(all_elasticity))

        return {
            "material_mass_r": float(r_mass),
            "material_friction_r": float(r_friction),
            "material_elasticity_r": float(r_elasticity),
            "n_objects_total": len(all_mass),
            "n_materials": len(MATERIAL_NAMES),
        }
