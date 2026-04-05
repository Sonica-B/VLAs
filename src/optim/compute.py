"""
Compute-side optimizations: attention backend, safe torch.compile,
generation config, inference context manager.

Every function here is a no-op or graceful-fallback when the optimization
isn't available on this machine. The goal is: "turn it on if you've got it,
never block if you don't."
"""

from __future__ import annotations

import contextlib
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# Attention backend selection.
# ---------------------------------------------------------------------------

def _has_flash_attn() -> bool:
    try:
        import flash_attn  # noqa: F401
        return True
    except Exception:
        return False


def pick_attn_impl(
    prefer_flash: bool = True,
    allow_sdpa: bool = True,
    model_hint: str = "",
) -> str:
    """Return the best attention implementation string for a given model.

    Priority: flash_attention_2 (if installed) > sdpa > eager. Some model
    families (e.g. InternVL custom code) raise on SDPA — callers can pass
    `allow_sdpa=False` to force eager.

    Returns one of: "flash_attention_2", "sdpa", "eager".
    """
    if prefer_flash and _has_flash_attn():
        return "flash_attention_2"
    if allow_sdpa:
        # bf16/fp16 + compute capability >= 7.0 required; we assume the caller
        # has already verified that. sdpa silently falls back to math impl
        # otherwise, so this is safe.
        return "sdpa"
    return "eager"


# ---------------------------------------------------------------------------
# Safe torch.compile wrapper.
# ---------------------------------------------------------------------------

def try_compile(model, model_name: str = "", mode: str = "reduce-overhead"):
    """Wrap model.forward in torch.compile if it's actually going to work.

    Guards:
        - torch.compile must exist (PyTorch >= 2.0)
        - triton must be importable (required by inductor backend on CUDA)
        - CUDA must be available

    Silently returns the model unchanged on any failure. Prints a one-line
    status so the log explains why compile was skipped.

    `mode` defaults to "reduce-overhead" which gives the best latency for
    batch=1 generative inference. Use "default" for training loops.
    """
    if not hasattr(torch, "compile"):
        print(f"  torch.compile skipped for {model_name}: PyTorch too old")
        return model
    if not torch.cuda.is_available():
        print(f"  torch.compile skipped for {model_name}: no CUDA")
        return model
    try:
        import triton  # noqa: F401
    except ImportError:
        print(f"  torch.compile skipped for {model_name}: triton not available")
        return model

    try:
        model.forward = torch.compile(model.forward, mode=mode, fullgraph=False)
        print(f"  torch.compile enabled for {model_name} (mode={mode})")
    except Exception as e:
        print(f"  torch.compile failed for {model_name}: {type(e).__name__}: {e}")
    return model


# ---------------------------------------------------------------------------
# Generation config for PhysBench (multiple-choice short-answer).
# ---------------------------------------------------------------------------

def fast_gen_config(
    max_new_tokens: int = 16,
    use_cache: bool = True,
) -> dict:
    """Minimal deterministic generation config for multiple-choice eval.

    Keys are chosen to work with every VLM `.generate()` in this codebase:
    Qwen3-VL, Qwen2.5-VL, Gemma, LLaVA-OneVision, InternVL.

    `max_new_tokens=16` covers every PhysBench multiple-choice answer; bump
    to 64 only for short-answer/quantitative questions in QuantiPhy.
    """
    return dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        use_cache=use_cache,
        temperature=1.0,        # ignored under do_sample=False but kept for safety
        top_p=1.0,
        repetition_penalty=1.0,
    )


# ---------------------------------------------------------------------------
# Inference context manager — single place to flip inference_mode + autocast.
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def inference_ctx(dtype: torch.dtype = torch.bfloat16, device_type: str = "cuda"):
    """Context manager that activates inference_mode + autocast simultaneously.

    Using this everywhere guarantees consistent compute dtype across the eval
    pipeline — otherwise some HF models silently upcast to fp32 in specific
    submodules and leak memory.

    Usage:
        with inference_ctx():
            outputs = model.generate(**inputs, **fast_gen_config())
    """
    if device_type == "cuda" and not torch.cuda.is_available():
        device_type = "cpu"
    # autocast only helps on CUDA; on CPU it's a no-op for bf16 and can actually
    # slow things down on older torch versions.
    if device_type == "cuda":
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        with torch.inference_mode():
            yield
