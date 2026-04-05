"""
VRAM / memory footprint helpers.

Everything here is defensive: functions degrade gracefully on CPU-only boxes
(so the smoke test runs without a GPU) and on older bitsandbytes versions.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# bitsandbytes 4-bit config — the single default for all baseline inference.
# ---------------------------------------------------------------------------

def build_bnb_config(
    load_in_4bit: bool = True,
    load_in_8bit: bool = False,
    compute_dtype: torch.dtype = torch.bfloat16,
):
    """Build a BitsAndBytesConfig tuned for Option C inference.

    Defaults (nf4 + double-quant + bf16 compute) were measured at 6.4 GB VRAM
    for Qwen3-VL-8B on PhysBench val. Do not change these defaults without a
    VRAM re-measurement.

    Returns None if bitsandbytes is unavailable — callers should then load
    the model in bf16 and accept the VRAM hit.
    """
    try:
        from transformers import BitsAndBytesConfig
    except Exception:  # pragma: no cover
        return None

    if load_in_4bit:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    if load_in_8bit:
        return BitsAndBytesConfig(load_in_8bit=True)
    return None


# ---------------------------------------------------------------------------
# VRAM snapshots.
# ---------------------------------------------------------------------------

@dataclass
class VramSnapshot:
    allocated_gb: float
    reserved_gb: float
    max_allocated_gb: float
    device: str

    def __str__(self) -> str:
        return (
            f"VRAM[{self.device}] alloc={self.allocated_gb:.2f}GB "
            f"reserved={self.reserved_gb:.2f}GB "
            f"peak={self.max_allocated_gb:.2f}GB"
        )


def snapshot_vram(device: int = 0, reset_peak: bool = False) -> VramSnapshot:
    """Snapshot current VRAM usage on the given device.

    On CPU-only boxes returns a zero snapshot rather than raising, so that the
    smoke test and non-GPU code paths stay clean.
    """
    if not torch.cuda.is_available():
        return VramSnapshot(0.0, 0.0, 0.0, "cpu")
    allocated = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    if reset_peak:
        torch.cuda.reset_peak_memory_stats(device)
    name = torch.cuda.get_device_name(device)
    return VramSnapshot(allocated, reserved, peak, name)


def format_vram_delta(before: VramSnapshot, after: VramSnapshot) -> str:
    """Human-readable VRAM delta for logging model load/unload events."""
    d_alloc = after.allocated_gb - before.allocated_gb
    d_peak = after.max_allocated_gb - before.max_allocated_gb
    sign = "+" if d_alloc >= 0 else ""
    return (
        f"VRAM delta: alloc {sign}{d_alloc:.2f}GB "
        f"peak {sign}{d_peak:.2f}GB "
        f"(now {after.allocated_gb:.2f}GB allocated)"
    )


# ---------------------------------------------------------------------------
# Hard cleanup between model swaps.
# ---------------------------------------------------------------------------

def hard_cleanup(*objs) -> None:
    """Delete references, run GC, empty CUDA cache, reset peak stats.

    Call this between model loads in sequential multi-model runs. The repo's
    prior pattern was inconsistent — this is the one canonical implementation.

    Example:
        model, processor = load_qwen3_vl(...)
        results = evaluate_model(model, processor, ...)
        hard_cleanup(model, processor)
    """
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    gc.collect()  # twice — first pass may create new garbage via finalizers
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def assert_vram_below(limit_gb: float, device: int = 0, label: str = "") -> None:
    """Raise if allocated VRAM exceeds limit_gb. Used as a post-cleanup sanity check.

    Skipped silently on CPU-only boxes.
    """
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated(device) / 1e9
    if allocated > limit_gb:
        raise RuntimeError(
            f"VRAM check failed{(' [' + label + ']') if label else ''}: "
            f"{allocated:.2f}GB allocated, limit {limit_gb:.2f}GB. "
            "Cleanup did not release GPU memory — check for lingering references."
        )


# ---------------------------------------------------------------------------
# Environment setup helper — apply once per process before importing torch.
# ---------------------------------------------------------------------------

def set_cuda_alloc_env() -> None:
    """Set PYTORCH_CUDA_ALLOC_CONF for expandable segments.

    Has to be called BEFORE any CUDA tensor is created. Scripts should call it
    at module-top before `import torch` runs any allocations.
    """
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Reduce fragmentation under frequent alloc/free cycles during probing.
    os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
