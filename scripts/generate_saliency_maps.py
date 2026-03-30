"""
Generate physics saliency map visualizations for all models and variables.

For each (model, variable, pipeline_stage) combination, loads pre-computed
per-patch R² scores and generates heatmap overlays on sample images.

Usage:
    python scripts/generate_saliency_maps.py --model qwen2_5_vl_7b --variable mass
    python scripts/generate_saliency_maps.py --all   # all combinations
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualization.saliency_map import PhysicsSaliencyMap
from src.visualization.degradation_curves import DegradationCurvePlot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_NAMES = ["qwen2_5_vl_7b", "internvl2_5_8b", "llava_onevision_7b"]
VARIABLES = ["mass", "friction", "elasticity", "stability"]
STAGES = ["stage_1_enc_out", "stage_2_post_proj", "stage_3_llm_8", "stage_4_llm_16"]


def generate_single(
    model_name: str,
    variable: str,
    stage: str,
    probe_metrics_dir: Path,
    physion_dir: Path,
    output_dir: Path,
) -> None:
    """Generate a saliency map for one (model, variable, stage) combination."""
    # Load per-patch R² scores
    r2_path = probe_metrics_dir / model_name / variable / stage / f"per_patch_r2_{model_name}_{variable}_{stage}.npy"
    if not r2_path.exists():
        logger.warning(f"Per-patch R² not found: {r2_path}. Skipping.")
        return

    r2_scores = np.load(r2_path)

    # Load a sample image from Physion++
    # Use the first available frame from the dominoes scenario for consistency
    sample_image_path = physion_dir / "dominoes" / "trial_000" / "video" / "frame_0000.png"
    if not sample_image_path.exists():
        logger.warning(f"Sample image not found: {sample_image_path}")
        return

    from PIL import Image
    image = Image.open(sample_image_path).convert("RGB").resize((448, 448))

    # Generate saliency map
    viz = PhysicsSaliencyMap(patch_grid_size=14)
    stage_display = {
        "stage_1_enc_out": "Stage 1: Encoder Out",
        "stage_2_post_proj": "Stage 2: Post-Proj.",
        "stage_3_llm_8": "Stage 3: LLM Layer 8",
        "stage_4_llm_16": "Stage 4: LLM Layer 16",
    }.get(stage, stage)

    title = f"{model_name} | {variable} | {stage_display}"
    fig = viz.visualize(image=image, per_patch_scores=r2_scores, title=title)

    out_path = output_dir / f"saliency_{model_name}_{variable}_{stage}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    viz.save(fig, out_path)
    logger.info(f"Saved: {out_path}")


def generate_overview_grid(
    model_name: str,
    probe_metrics_dir: Path,
    physion_dir: Path,
    output_dir: Path,
) -> None:
    """Generate a 4-variable × 4-stage grid of saliency maps for one model."""
    from PIL import Image

    sample_image_path = physion_dir / "dominoes" / "trial_000" / "video" / "frame_0000.png"
    if not sample_image_path.exists():
        logger.warning(f"Sample image not found for grid: {sample_image_path}")
        return

    image = Image.open(sample_image_path).convert("RGB").resize((448, 448))

    images_list = []
    scores_list = []

    for var in VARIABLES:
        for stage in STAGES:
            r2_path = probe_metrics_dir / model_name / var / stage / f"per_patch_r2_{model_name}_{var}_{stage}.npy"
            if r2_path.exists():
                scores_list.append(np.load(r2_path))
            else:
                scores_list.append(np.zeros(196))
            images_list.append(image)

    viz = PhysicsSaliencyMap(patch_grid_size=14)
    row_labels = [v.capitalize() for v in VARIABLES]
    col_labels = ["Enc. Out", "Post-Proj.", "LLM L8", "LLM L16"]

    fig = viz.visualize_grid(
        images=images_list,
        scores_list=scores_list,
        row_labels=row_labels,
        col_labels=col_labels,
        suptitle=f"Physics Saliency Maps — {model_name}",
    )

    out_path = output_dir / f"saliency_grid_{model_name}.png"
    viz.save(fig, out_path)
    logger.info(f"Saved grid: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate physics saliency map visualizations")
    parser.add_argument("--model", type=str, choices=MODEL_NAMES + ["all"], default="all")
    parser.add_argument("--variable", type=str, choices=VARIABLES + ["all"], default="all")
    parser.add_argument("--stage", type=str, choices=STAGES + ["all"], default="all")
    parser.add_argument("--probe-metrics-dir", type=str, default="results/probe_metrics")
    parser.add_argument("--physion-dir", type=str, default="data/physion")
    parser.add_argument("--output-dir", type=str, default="results/saliency_maps")
    parser.add_argument("--generate-grids", action="store_true",
                        help="Also generate per-model overview grids")
    args = parser.parse_args()

    models = MODEL_NAMES if args.model == "all" else [args.model]
    variables = VARIABLES if args.variable == "all" else [args.variable]
    stages = STAGES if args.stage == "all" else [args.stage]

    probe_metrics_dir = Path(args.probe_metrics_dir)
    physion_dir = Path(args.physion_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in models:
        for variable in variables:
            for stage in stages:
                generate_single(
                    model_name, variable, stage,
                    probe_metrics_dir, physion_dir, output_dir
                )

        if args.generate_grids:
            generate_overview_grid(model_name, probe_metrics_dir, physion_dir, output_dir)


if __name__ == "__main__":
    main()
