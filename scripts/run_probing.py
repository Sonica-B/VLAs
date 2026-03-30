"""
Main probing script for Phase 1.

Orchestrates activation extraction and probe training for one model
and one probe configuration. Designed to be run via Hydra configs.

Usage:
    python scripts/run_probing.py model=qwen2_5_vl_7b probe=linear_probe
    python scripts/run_probing.py model=qwen2_5_vl_7b probe=mlp_probe data.num_samples=500
"""

import logging
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.vlm_loader import load_vlm, get_model_hidden_dims
from src.models.activation_extractor import ActivationExtractor, STAGE_NAMES
from src.data.physion_loader import PhysionLoader
from src.data.patch_label_assigner import PatchLabelAssigner
from src.probing.linear_probe import LinearProbe
from src.probing.mlp_probe import MLPProbe
from src.probing.probe_trainer import ProbeTrainer
from src.visualization.saliency_map import PhysicsSaliencyMap
from src.visualization.degradation_curves import DegradationCurvePlot

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="run_probing", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the full probing pipeline for a single model and probe type.

    Config keys (from Hydra):
        cfg.model.name: Model name (e.g., "qwen2_5_vl_7b")
        cfg.probe.type: Probe type ("linear" or "mlp")
        cfg.data.physion_dir: Path to Physion++ dataset
        cfg.data.num_samples: Number of samples to process (None = all)
        cfg.data.save_activations: Whether to save HDF5 activations
        cfg.data.activation_dir: Where to save activations
        cfg.output.results_dir: Where to save probe metrics
    """
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")

    model_name = cfg.model.name
    probe_type = cfg.probe.type
    physion_dir = cfg.data.get("physion_dir", "data/physion")
    activation_dir = cfg.data.get("activation_dir", f"results/activations/{model_name}")
    results_dir = cfg.output.get("results_dir", f"results/probe_metrics/{model_name}")
    num_samples = cfg.data.get("num_samples", None)
    save_activations = cfg.data.get("save_activations", True)

    # --- Step 1: Load model ---
    logger.info(f"Loading model: {model_name}")
    model, processor = load_vlm(
        model_name,
        load_in_4bit=cfg.model.get("load_in_4bit", False),
        torch_dtype="bfloat16",
    )
    hidden_dims = get_model_hidden_dims(model_name)

    # --- Step 2: Load dataset ---
    logger.info("Loading Physion++ dataset...")
    loader = PhysionLoader(
        physion_dir,
        split=cfg.data.get("split", "train"),
        image_size=cfg.model.get("image_size", 448),
    )
    assigner = PatchLabelAssigner(patch_grid_size=14)

    # --- Step 3: Extract activations ---
    extractor = ActivationExtractor(model, model_name, patch_grid_size=14)

    activation_path = Path(activation_dir)
    activation_path.mkdir(parents=True, exist_ok=True)

    n_processed = 0
    for i, sample in enumerate(loader):
        if num_samples and i >= num_samples:
            break

        h5_path = activation_path / f"{sample.scenario_id.replace('/', '_')}_{sample.frame_idx:04d}.h5"
        if h5_path.exists():
            continue  # Skip if already extracted

        try:
            activations = extractor.extract(sample.image, processor)
            patch_labels = assigner.assign(sample.object_masks, sample.physics_labels)

            if save_activations:
                import h5py
                import numpy as np
                with h5py.File(h5_path, "w") as f:
                    for stage, tensor in activations.items():
                        f.create_dataset(stage, data=tensor.numpy(), compression="gzip")
                    f.create_dataset("patch_labels", data=patch_labels, compression="gzip")
                    f.attrs["scenario_id"] = sample.scenario_id
                    f.attrs["model_name"] = model_name

            n_processed += 1
            if n_processed % 50 == 0:
                logger.info(f"  Processed {n_processed} samples...")

        except Exception as e:
            logger.warning(f"Error processing sample {i}: {e}")

    logger.info(f"Activation extraction complete: {n_processed} samples.")

    # --- Step 4: Train probes ---
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    target_variables = cfg.probe.get("target_variables", ["mass", "friction", "elasticity", "stability"])
    stages_to_probe = cfg.probe.get("pipeline_stages", STAGE_NAMES)

    degradation_plotter = DegradationCurvePlot()

    for variable in target_variables:
        r2_by_stage = []

        for stage in stages_to_probe:
            dim = hidden_dims[stage]

            if probe_type == "linear":
                probe = LinearProbe(input_dim=dim, alpha=cfg.probe.get("alpha", 1.0))
            else:
                probe = MLPProbe(input_dim=dim, hidden_dim=cfg.probe.get("hidden_dim", 512))

            trainer = ProbeTrainer(
                probe,
                output_dir=str(results_path / variable / stage),
            )

            try:
                trainer.fit_from_hdf5(
                    activation_dir=activation_dir,
                    stage=stage,
                    target_variable=variable,
                )
                metrics = trainer.evaluate()
                trainer.save_metrics(metrics, run_name=f"{model_name}_{variable}_{stage}")

                r2_by_stage.append(metrics["r2"])
                logger.info(f"  [{variable}, {stage}] R²={metrics['r2']:.4f}")

            except Exception as e:
                logger.warning(f"  Failed [{variable}, {stage}]: {e}")
                r2_by_stage.append(0.0)

        degradation_plotter.add_model_results(
            model_name=model_name,
            variable=variable,
            stages=stages_to_probe,
            r2_values=r2_by_stage,
        )

    # --- Step 5: Generate degradation curve plot ---
    fig = degradation_plotter.plot(title=f"R² Degradation — {model_name}")
    fig_path = results_path / "r2_degradation_curve.png"
    degradation_plotter.save(fig, fig_path)
    logger.info(f"Degradation curve saved to {fig_path}")

    logger.info("Probing pipeline complete.")


if __name__ == "__main__":
    main()
