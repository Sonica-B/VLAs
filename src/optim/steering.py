"""
Subspace-Contrast Activation Steering (SCAS) for physics-aware VLM inference.

This module implements the zero-shot inference-time intervention described
in the Phase 3 plan. The core technique: amplify the low-variance PCA
subspace of the merger output to compensate for the spatial compression
that preferentially destroys quantitative physics information.

## Mathematical foundation (from Phase 3 proof)

Let f be the merger output in R^D. PCA decomposes the feature space into
V_high (top-k, high variance, contains S_qual) and V_low (remaining,
low variance, contains S_quant). The SCAS intervention:

    f_steered = f + alpha * P_low @ f = (I + alpha * P_low) @ f

This scales V_low components by (1+alpha) while leaving V_high unchanged.

Properties:
    1. S_qual exactly preserved (V_high components untouched)
    2. S_quant amplified by (1+alpha)
    3. Optimal alpha = sqrt(C) - 1 where C is the compression ratio
    4. Architecture-dependent: delta_quant proportional to sqrt(C)

## Two steering methods

Method A (Subspace Amplification):
    Amplify ALL low-variance directions uniformly. Simplest, no supervision.

Method B (Subspace Contrast):
    Compute the quant-minus-qual centroid direction, project onto V_low,
    and amplify only THAT direction. More targeted, uses the quant/qual
    labels from PhysBench.

## Usage

    from src.optim.steering import compute_steering_vector, make_steering_hook

    # 1. Compute (offline, from cached features)
    sv = compute_steering_vector(
        cache_dir="cache/week1/features", model_name="qwen3-vl-8b",
        method="contrast", low_var_k=64,
    )

    # 2. Apply at inference time (zero-shot)
    hook = make_steering_hook(sv, alpha=3.0, target_site="post_proj")
    handle = model.visual.merger.register_forward_hook(hook)
    # ... run model.generate() as normal ...
    handle.remove()
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class SteeringConfig:
    """Configuration for a SCAS intervention."""
    method: str = "contrast"          # "contrast" or "amplify"
    low_var_k: int = 64               # number of low-variance components
    alpha: float = 3.0                # amplification factor
    target_site: str = "post_proj"    # where to inject the steering hook
    normalize: bool = True            # normalize the steering vector to unit norm


def compute_pca_basis(features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA on a feature matrix [N, D].

    Returns:
        components: [D, D] — PCA eigenvectors sorted by DECREASING variance
        explained_variance: [D] — eigenvalues (variance per component)
        mean: [D] — feature mean (subtracted before PCA)
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Center features (PCA is mean-centered by default in sklearn).
    mean = features.mean(axis=0)
    n, d = features.shape
    n_components = min(n - 1, d)
    pca = PCA(n_components=n_components)
    pca.fit(features)
    return pca.components_, pca.explained_variance_, mean


def compute_steering_vector(
    cache_dir: str | Path,
    model_name: str,
    split: str = "val",
    method: str = "contrast",
    low_var_k: int = 64,
    site: str = "post_proj",
    quant_ids: Optional[set] = None,
    normalize: bool = True,
) -> Dict:
    """Compute a physics steering vector from cached features.

    Args:
        cache_dir: path to the Week 1 feature cache root
        model_name: e.g. "qwen3-vl-8b"
        split: "val"
        method: "contrast" (quant centroid - qual centroid, projected onto V_low)
                or "amplify" (projection matrix P_low itself)
        low_var_k: number of low-variance PCA components to use
        site: probe site to compute on (default "post_proj")
        quant_ids: set of sample_ids that are quantitative (if None, loaded from PhysBench)
        normalize: whether to normalize the steering vector to unit norm

    Returns:
        Dict with keys:
            vector: np.ndarray [D] — the steering direction (for "contrast")
                    OR np.ndarray [K, D] — the V_low basis (for "amplify")
            method: str
            low_var_k: int
            feature_dim: int
            explained_variance_low: float — fraction of total variance in V_low
            components: np.ndarray [D, D] — full PCA basis (for reference)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.optim.features import FeatureCache

    cache = FeatureCache(Path(cache_dir) / "features", model_name, split)
    features = cache.load_site(site)  # [N, D]
    index_order = list(cache._index["sample_ids"])

    # Get quant/qual mask.
    if quant_ids is None:
        from src.optim.physbench_split import classify_quantitative

        if split == "train":
            # Load training samples from the cleaned JSONL (disjoint from val).
            import json as _json
            train_path = Path(cache_dir).parent / "week2" / "training_data" / "lora_train_clean.jsonl"
            if not train_path.exists():
                # Fallback: try the non-clean version.
                train_path = Path(cache_dir).parent / "week2" / "training_data" / "lora_train.jsonl"
            if train_path.exists():
                with open(train_path) as _f:
                    samples = [_json.loads(l) for l in _f if l.strip()]
            else:
                # Last resort: load from PhysBench test split.
                from scripts.run_physbench_eval import load_physbench_data
                samples = load_physbench_data("data/physbench", split="test")
                for s in samples:
                    s.setdefault("sample_id", f"test_{s.get('idx', '?')}")
        else:
            from scripts.run_physbench_eval import load_physbench_data
            samples = load_physbench_data("data/physbench", split=split)
            for s in samples:
                s.setdefault("sample_id", f"{split}_{s.get('idx', '?')}")

        quant_ids = {s["sample_id"] for s in samples
                     if classify_quantitative(s) == "quantitative"}

    quant_mask = np.array([sid in quant_ids for sid in index_order])
    qual_mask = ~quant_mask

    # PCA on all features.
    components, variances, mean = compute_pca_basis(features)
    d = features.shape[1]
    n_components = len(variances)

    # Low-variance basis: the LAST K components (lowest variance).
    k = min(low_var_k, n_components)
    low_var_basis = components[-k:]  # [K, D]
    low_var_variance = variances[-k:].sum() / variances.sum()

    if method == "amplify":
        return {
            "vector": low_var_basis,  # [K, D] — the projection basis
            "method": "amplify",
            "low_var_k": k,
            "feature_dim": d,
            "explained_variance_low": float(low_var_variance),
            "components": components,
        }

    elif method == "contrast":
        # Compute quant - qual centroid direction.
        quant_feats = features[quant_mask]
        qual_feats = features[qual_mask]
        contrast_dir = quant_feats.mean(axis=0) - qual_feats.mean(axis=0)  # [D]

        # Project onto low-variance subspace.
        # P_low @ contrast_dir = V_low @ V_low^T @ contrast_dir
        proj_coeffs = low_var_basis @ contrast_dir  # [K]
        projected = proj_coeffs @ low_var_basis  # [D]

        if normalize and np.linalg.norm(projected) > 1e-8:
            projected = projected / np.linalg.norm(projected)

        return {
            "vector": projected,  # [D]
            "method": "contrast",
            "low_var_k": k,
            "feature_dim": d,
            "explained_variance_low": float(low_var_variance),
            "contrast_norm_before_proj": float(np.linalg.norm(contrast_dir)),
            "contrast_norm_after_proj": float(np.linalg.norm(proj_coeffs @ low_var_basis)),
            "projection_retention": float(
                np.linalg.norm(proj_coeffs @ low_var_basis)
                / max(np.linalg.norm(contrast_dir), 1e-8)
            ),
            "components": components,
        }

    else:
        raise ValueError(f"Unknown method: {method}. Use 'contrast' or 'amplify'.")


# ---------------------------------------------------------------------------
# Steering hooks.
# ---------------------------------------------------------------------------

def make_steering_hook(
    steering_info: Dict,
    alpha: float = 3.0,
) -> callable:
    """Create a PyTorch forward hook that applies SCAS to the module output.

    For method="contrast": adds alpha * (output . v_hat) * v_hat to the output,
    where v_hat is the unit steering direction.

    For method="amplify": adds alpha * P_low @ output to the output,
    where P_low is the low-variance projection matrix.

    The hook is registered on a nn.Module (typically model.visual.merger)
    and fires after the module's forward pass. It RETURNS the modified
    output so PyTorch uses the steered activations for all downstream
    computation.

    Usage:
        hook_fn = make_steering_hook(steering_info, alpha=3.0)
        handle = model.visual.merger.register_forward_hook(hook_fn)
        # ... run inference ...
        handle.remove()
    """
    method = steering_info["method"]

    if method == "contrast":
        sv_np = steering_info["vector"]  # [D], unit norm
        sv = torch.tensor(sv_np, dtype=torch.bfloat16)

        def hook(_module, _input, output):
            # output shape: [seq, D] or [batch, seq, D] or tuple
            is_tuple = isinstance(output, tuple)
            t = output[0] if is_tuple else output

            sv_dev = sv.to(t.device)
            # Project each token onto the steering direction.
            # t: [..., D], sv_dev: [D]
            proj = (t @ sv_dev)  # [...] scalar per token
            amplification = proj.unsqueeze(-1) * sv_dev  # [..., D]
            t_steered = t + alpha * amplification

            if is_tuple:
                return (t_steered,) + output[1:]
            return t_steered

        return hook

    elif method == "amplify":
        basis_np = steering_info["vector"]  # [K, D]
        basis = torch.tensor(basis_np, dtype=torch.bfloat16)

        def hook(_module, _input, output):
            is_tuple = isinstance(output, tuple)
            t = output[0] if is_tuple else output

            basis_dev = basis.to(t.device)  # [K, D]
            # Project onto V_low, amplify, add back.
            # P_low @ t = basis^T @ basis @ t
            proj = t @ basis_dev.T  # [..., K]
            amplified = proj @ basis_dev  # [..., D]
            t_steered = t + alpha * amplified

            if is_tuple:
                return (t_steered,) + output[1:]
            return t_steered

        return hook

    else:
        raise ValueError(f"Unknown steering method: {method}")
