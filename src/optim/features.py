"""
One-pass feature extraction at 4 probe sites with mmap-backed disk cache.

Why this exists:
    The naive approach — probe site by site, re-running the full VLM for each —
    is 4x slower than necessary. This module runs the model ONCE per sample and
    pulls activations from all 4 sites via forward hooks, then writes each
    site's tensor to a per-site memory-mapped .npy file.

    Probing scripts then read the .npy files as np.memmap and never re-forward
    the model. On a 200-sample PhysBench val slice this takes forward-pass
    time from ~12 min (4 x 3 min) to ~3 min.

The 4 sites (as defined in the Notion log):
    enc_out     vision encoder final hidden state
    post_proj   after the merger / projector MLP
    llm_8       transformer layer 8 residual stream
    llm_16      transformer layer 16 residual stream

Callers supply (model_name, model) — this module owns the model-family-specific
module path lookup in a single registry, so we never hunt through transformers
internals in application code again.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Probe site registry.
# ---------------------------------------------------------------------------

@dataclass
class ProbeSites:
    """Holds the 4 probe sites for a model, plus metadata for hook registration.

    `paths` maps site_name → dotted module path (e.g. "visual.merger").
    `pooling` maps site_name → either "mean" (spatial mean over tokens), "last"
    (last token), or "full" (keep full [seq, dim] tensor — expensive).
    """
    model_name: str
    paths: Dict[str, str]
    pooling: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Default to mean pooling at every site (cheap, consistent across models).
        for site in self.paths:
            self.pooling.setdefault(site, "mean")

    @classmethod
    def for_model(cls, model_name: str) -> "ProbeSites":
        """Return the default 4-site config for known model families.

        These paths are the ones used in the existing probing scripts; updating
        them here updates every script that imports from `src.optim`.
        """
        m = model_name.lower().replace("_", "-")
        if "qwen3-vl" in m:
            return cls(
                model_name=model_name,
                paths={
                    "enc_out":  "model.visual",
                    "post_proj": "model.visual.merger",
                    "llm_8":    "model.language_model.layers.8",
                    "llm_16":   "model.language_model.layers.16",
                },
            )
        if "qwen2.5-vl" in m or "qwen2-5-vl" in m:
            return cls(
                model_name=model_name,
                paths={
                    "enc_out":  "visual",
                    "post_proj": "visual.merger",
                    "llm_8":    "model.layers.8",
                    "llm_16":   "model.layers.16",
                },
            )
        if "internvl" in m:
            return cls(
                model_name=model_name,
                paths={
                    "enc_out":  "vision_model",
                    "post_proj": "mlp1",
                    "llm_8":    "language_model.model.layers.8",
                    "llm_16":   "language_model.model.layers.16",
                },
            )
        if "gemma" in m:
            return cls(
                model_name=model_name,
                paths={
                    "enc_out":  "vision_tower",
                    "post_proj": "multi_modal_projector",
                    "llm_8":    "language_model.model.layers.8",
                    "llm_16":   "language_model.model.layers.16",
                },
            )
        raise ValueError(
            f"ProbeSites.for_model: unknown model family '{model_name}'. "
            "Add a case to features.py or construct ProbeSites manually."
        )


# ---------------------------------------------------------------------------
# Mmap-backed per-site feature cache.
# ---------------------------------------------------------------------------

class FeatureCache:
    """Disk-backed feature store, one .npy file per (model, split, site).

    Layout:
        cache_dir/{model}_{split}/
            enc_out.npy          shape [N, D]   memmap
            post_proj.npy        shape [N, D]   memmap
            llm_8.npy            shape [N, D]   memmap
            llm_16.npy           shape [N, D]   memmap
            index.json           {"sample_ids": [...], "dims": {site: D, ...}}

    Probing code should open each site with `np.load(path, mmap_mode='r')` —
    no loading the full array into RAM.

    Append-only semantics: `append_batch(sample_ids, site_to_tensor)` extends
    each site's .npy file in place. A crash mid-run leaves a valid prefix that
    can be resumed via `completed_ids()`.
    """

    def __init__(self, cache_dir: Path, model_name: str, split: str, dtype: np.dtype = np.float32):
        self.root = Path(cache_dir) / f"{model_name}_{split}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.index_path = self.root / "index.json"
        self._index = self._load_index()

    def _load_index(self) -> Dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {"sample_ids": [], "dims": {}}

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self._index, indent=2))

    def completed_ids(self) -> set:
        """Sample IDs already written to the cache. Used for resume."""
        return set(self._index["sample_ids"])

    def site_path(self, site: str) -> Path:
        return self.root / f"{site}.npy"

    def append_batch(self, sample_ids: List[str], site_to_tensor: Dict[str, np.ndarray]) -> None:
        """Append a batch of features to every site.

        Every tensor must have shape [batch, dim]. First call sets the dim;
        subsequent calls must match. Uses np.save with append by concatenating
        to an in-memory view then rewriting — cheap at PhysBench scale
        (10k samples × 4 sites × ~4k dim = ~600 MB per site, fits in RAM).

        For truly memory-bound cases, switch to .npy append-mode via numpy-format
        low-level writer; not needed for PhysBench val (200 samples).
        """
        if not sample_ids:
            return
        for site, tensor in site_to_tensor.items():
            if tensor.ndim != 2 or tensor.shape[0] != len(sample_ids):
                raise ValueError(
                    f"FeatureCache.append_batch: site '{site}' tensor shape "
                    f"{tensor.shape} does not match batch size {len(sample_ids)}"
                )
            tensor = tensor.astype(self.dtype, copy=False)
            path = self.site_path(site)
            if path.exists():
                existing = np.load(path, mmap_mode="r")
                if existing.shape[1] != tensor.shape[1]:
                    raise ValueError(
                        f"FeatureCache.append_batch: site '{site}' dim mismatch "
                        f"(cache={existing.shape[1]}, new={tensor.shape[1]})"
                    )
                combined = np.concatenate([np.asarray(existing), tensor], axis=0)
                np.save(path, combined)
                del existing
            else:
                np.save(path, tensor)
                self._index["dims"][site] = int(tensor.shape[1])

        self._index["sample_ids"].extend(sample_ids)
        self._save_index()

    def load_site(self, site: str) -> np.memmap:
        """Open a site's features as a read-only memmap. No RAM cost."""
        return np.load(self.site_path(site), mmap_mode="r")


# ---------------------------------------------------------------------------
# Hook registration.
# ---------------------------------------------------------------------------

def _resolve_module(model: nn.Module, dotted_path: str) -> nn.Module:
    """Resolve a dotted attribute path on a torch module. Raises KeyError if missing."""
    obj: Any = model
    for part in dotted_path.split("."):
        if part.isdigit():
            obj = obj[int(part)]
        else:
            if not hasattr(obj, part):
                raise KeyError(
                    f"Module path '{dotted_path}' not found on {type(model).__name__}: "
                    f"missing attribute '{part}'"
                )
            obj = getattr(obj, part)
    return obj


def _pool(tensor: torch.Tensor, mode: str) -> torch.Tensor:
    """Pool a [batch, seq, dim] or [batch, dim] tensor to [batch, dim]."""
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        if mode == "mean":
            return tensor.mean(dim=1)
        if mode == "last":
            return tensor[:, -1, :]
        if mode == "full":
            return tensor  # caller handles [batch, seq, dim]
        raise ValueError(f"Unknown pooling mode: {mode}")
    # 4-D (vision encoder tokens): flatten spatial, mean-pool.
    flat = tensor.flatten(1, -2) if tensor.ndim > 3 else tensor
    return flat.mean(dim=1)


def register_probe_hooks(
    model: nn.Module,
    sites: ProbeSites,
) -> Tuple[Dict[str, torch.Tensor], List[Any]]:
    """Register forward hooks at every probe site.

    Returns:
        captured:  dict mapping site_name → latest captured tensor (on CPU, fp32).
                   Re-populated on every forward pass.
        handles:   list of hook handles; caller must call .remove() on each
                   when done (use `try/finally` or the `extract_features_one_pass`
                   helper below which handles this automatically).
    """
    captured: Dict[str, torch.Tensor] = {}
    handles: List[Any] = []

    def make_hook(site_name: str, pool_mode: str):
        def hook(_mod, _inputs, output):
            # Output may be tensor, tuple, or ModelOutput — extract the hidden state.
            t = output
            if isinstance(t, tuple):
                t = t[0]
            elif hasattr(t, "last_hidden_state"):
                t = t.last_hidden_state
            elif hasattr(t, "hidden_states") and t.hidden_states is not None:
                t = t.hidden_states[-1]
            if not isinstance(t, torch.Tensor):
                return  # unsupported output type — skip silently
            pooled = _pool(t.detach(), pool_mode)
            captured[site_name] = pooled.to(dtype=torch.float32, device="cpu")
        return hook

    for site, path in sites.paths.items():
        module = _resolve_module(model, path)
        handles.append(module.register_forward_hook(make_hook(site, sites.pooling[site])))

    return captured, handles


# ---------------------------------------------------------------------------
# High-level: one-pass extraction to cache.
# ---------------------------------------------------------------------------

def extract_features_one_pass(
    model: nn.Module,
    sites: ProbeSites,
    forward_fn: Callable[[], Any],
    sample_ids: List[str],
    cache: FeatureCache,
) -> Dict[str, np.ndarray]:
    """Run `forward_fn` once with hooks active; write all 4 sites to cache.

    `forward_fn` is a zero-arg callable the caller supplies — typically a
    closure around `model(**batch_inputs)`. This indirection lets the caller
    own the tokenization/image-processing step while this function owns the
    hook lifecycle and cache write.

    Returns the captured {site: np.ndarray [batch, dim]} dict for downstream
    use (e.g. on-the-fly probing without re-reading from disk).

    Crash-safe: hooks are removed in a finally block; the cache append is
    atomic per batch (no partial writes).
    """
    captured, handles = register_probe_hooks(model, sites)
    try:
        with torch.inference_mode():
            _ = forward_fn()
    finally:
        for h in handles:
            h.remove()

    np_captured: Dict[str, np.ndarray] = {}
    for site, tensor in captured.items():
        arr = tensor.numpy()
        if arr.ndim == 3:
            arr = arr.mean(axis=1)  # safety net for "full" pooling
        np_captured[site] = arr

    # Deduplicate against already-cached samples (resume semantics).
    already = cache.completed_ids()
    new_ids: List[str] = []
    new_idx: List[int] = []
    for i, sid in enumerate(sample_ids):
        if sid not in already:
            new_ids.append(sid)
            new_idx.append(i)
    if new_ids:
        slice_captured = {site: arr[new_idx] for site, arr in np_captured.items()}
        cache.append_batch(new_ids, slice_captured)

    return np_captured
