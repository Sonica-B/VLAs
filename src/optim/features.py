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

import gc
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Windows-safe np.save with retry on transient file-lock errors.
# ---------------------------------------------------------------------------

def _windows_safe_np_save(path: Path, arr: np.ndarray, retries: int = 6,
                           backoff_s: float = 0.15) -> None:
    """Write `arr` to `path` as a .npy file, retrying on Windows PermissionError.

    Windows Defender Real-time Protection briefly holds a read lock on
    newly-created files while it scans them. Rapid successive writes to the
    same path (as happens when we re-save a growing feature cache after every
    PhysBench sample) race with Defender and raise WinError 5 ~5% of the time.

    Retry with exponential backoff handles the transient case cleanly. Total
    worst-case wait is ~0.15 * (2^6 - 1) = ~9.5s before giving up — still
    shorter than a single Qwen3-VL-8B forward pass.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(retries):
        try:
            # np.save appends .npy if the extension is missing; we always pass
            # a path ending in .npy so the saved file matches `path` exactly.
            np.save(path, arr, allow_pickle=False)
            return
        except PermissionError as e:
            last_err = e
            gc.collect()  # release any lingering file handles
            time.sleep(backoff_s * (2 ** attempt))
        except OSError as e:
            # WinError 5 sometimes surfaces as OSError instead of PermissionError.
            if getattr(e, "winerror", None) in (5, 32):
                last_err = e
                gc.collect()
                time.sleep(backoff_s * (2 ** attempt))
            else:
                raise
    raise RuntimeError(
        f"_windows_safe_np_save: failed to write {path} after {retries} retries"
    ) from last_err


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
    """Disk-backed feature store, one .npy file per (sample, site).

    Layout:
        cache_dir/{model}_{split}/
            index.json                      {"sample_ids": [...], "dims": {site: D, ...}}
            enc_out/{sample_id}.npy         shape [dim]  per-sample features
            post_proj/{sample_id}.npy
            llm_8/{sample_id}.npy
            llm_16/{sample_id}.npy

    Per-sample files are the Windows-safe alternative to a single growing
    .npy. A single-file append-then-overwrite approach races with Windows
    Defender real-time scanning and raises PermissionError (WinError 5) or
    OSError EINVAL after the first overwrite attempt. Per-sample files
    write each sample exactly once to a unique path — no overwrite, no
    rename, no race.

    At PhysBench val scale (200 samples × 4 sites × ~4 KB/file = ~3 MB/site
    in ~800 files total) this has negligible overhead vs. monolithic .npy.

    `load_site(site)` assembles a [N, D] ndarray from the per-sample files
    for downstream probing. It's a one-time load into RAM (trivially cheap
    at this scale) rather than a memmap — mmap on Windows has its own
    lock problems we're avoiding.
    """

    def __init__(self, cache_dir: Path, model_name: str, split: str, dtype: np.dtype = np.float32):
        self.root = Path(cache_dir) / f"{model_name}_{split}"
        self.root.mkdir(parents=True, exist_ok=True)
        self.dtype = dtype
        self.index_path = self.root / "index.json"
        self._index = self._load_index()
        self._migrate_legacy_if_needed()

    def _load_index(self) -> Dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text())
        return {"sample_ids": [], "dims": {}}

    def _save_index(self) -> None:
        # Direct write is fine for the index — it's text, small, and writes
        # are single-shot.
        self.index_path.write_text(json.dumps(self._index, indent=2))

    def _migrate_legacy_if_needed(self) -> None:
        """One-time migration of legacy single-file .npy layout to per-sample.

        Detects legacy layout by presence of {site}.npy at root alongside
        sample_ids in the index. Reads each legacy file, writes per-sample
        files into {site}/{sample_id}.npy, then removes the legacy file.
        """
        sample_ids = self._index.get("sample_ids", [])
        if not sample_ids:
            return
        for site in list(self._index.get("dims", {}).keys()):
            legacy_path = self.root / f"{site}.npy"
            per_sample_dir = self.root / site
            if legacy_path.exists() and not per_sample_dir.exists():
                per_sample_dir.mkdir(parents=True, exist_ok=True)
                legacy_arr = np.load(legacy_path, mmap_mode=None)
                if legacy_arr.shape[0] != len(sample_ids):
                    raise RuntimeError(
                        f"FeatureCache migration: legacy {legacy_path} has "
                        f"{legacy_arr.shape[0]} rows but index claims "
                        f"{len(sample_ids)} samples"
                    )
                for i, sid in enumerate(sample_ids):
                    sample_path = per_sample_dir / f"{sid}.npy"
                    if not sample_path.exists():
                        np.save(sample_path, legacy_arr[i])
                # Remove the legacy file after successful migration.
                try:
                    legacy_path.unlink()
                except OSError:
                    pass  # leave it if we can't delete; harmless

    def completed_ids(self) -> set:
        """Sample IDs already written to the cache. Used for resume."""
        return set(self._index["sample_ids"])

    def site_dir(self, site: str) -> Path:
        return self.root / site

    def sample_path(self, site: str, sample_id: str) -> Path:
        return self.site_dir(site) / f"{sample_id}.npy"

    def append_batch(self, sample_ids: List[str], site_to_tensor: Dict[str, np.ndarray]) -> None:
        """Write features for a batch of samples, one file per (sample, site).

        Every tensor must have shape [batch, dim]. First call sets the dim per
        site; subsequent calls must match. Each sample writes to a unique
        path so there is no overwrite and no Windows file-lock race.

        Semantics:
            - Atomicity: each site writes its N files in order. A crash at
              file K of site S leaves K files written for S but 0 files for
              S+1. The index update happens only AFTER all sites have written
              successfully, so a crash keeps the cache self-consistent —
              next run treats these sample_ids as not-yet-cached.
            - Idempotent: rewriting an existing sample file is a no-op
              (we skip if it already exists).
        """
        if not sample_ids:
            return

        # Validate shapes upfront.
        for site, tensor in site_to_tensor.items():
            if tensor.ndim != 2 or tensor.shape[0] != len(sample_ids):
                raise ValueError(
                    f"FeatureCache.append_batch: site '{site}' tensor shape "
                    f"{tensor.shape} does not match batch size {len(sample_ids)}"
                )

        # Write every (site, sample) file. Dim check against index.
        for site, tensor in site_to_tensor.items():
            tensor = tensor.astype(self.dtype, copy=False)
            dim = int(tensor.shape[1])
            known_dim = self._index["dims"].get(site)
            if known_dim is None:
                self._index["dims"][site] = dim
            elif known_dim != dim:
                raise ValueError(
                    f"FeatureCache.append_batch: site '{site}' dim mismatch "
                    f"(cache={known_dim}, new={dim})"
                )
            site_dir = self.site_dir(site)
            site_dir.mkdir(parents=True, exist_ok=True)
            for i, sid in enumerate(sample_ids):
                sample_path = site_dir / f"{sid}.npy"
                if sample_path.exists():
                    continue  # idempotent skip
                # Each sample_path is unique — no overwrite, no rename, no
                # lock race. Direct np.save is safe.
                np.save(sample_path, tensor[i])

        # Only advance the index once every site has written successfully.
        existing_ids = set(self._index["sample_ids"])
        for sid in sample_ids:
            if sid not in existing_ids:
                self._index["sample_ids"].append(sid)
                existing_ids.add(sid)
        self._save_index()

    def load_site(self, site: str) -> np.ndarray:
        """Load all features for a site as a [N, D] ndarray, in index order.

        Not a memmap — we load into RAM. At PhysBench val scale this is ~3 MB
        per site, ~12 MB total across 4 sites. Full-load is both simpler and
        faster at this size (no Windows mmap lock issues).
        """
        sample_ids = self._index["sample_ids"]
        if not sample_ids:
            dim = self._index["dims"].get(site, 0)
            return np.zeros((0, dim), dtype=self.dtype)
        site_dir = self.site_dir(site)
        rows = []
        for sid in sample_ids:
            sample_path = site_dir / f"{sid}.npy"
            if not sample_path.exists():
                raise FileNotFoundError(
                    f"FeatureCache.load_site: missing {sample_path} "
                    f"(index claims sample_id '{sid}' is cached for site '{site}')"
                )
            rows.append(np.load(sample_path, mmap_mode=None))
        return np.stack(rows, axis=0).astype(self.dtype, copy=False)


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
