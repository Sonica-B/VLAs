"""
Inference optimization stack for the NeurIPS 2026 Option C pivot.

Every submodule here exists to make the 2-week critical path feasible on a
single local GPU:
    - `vram`        VRAM / memory footprint (bnb, bf16, cleanup, snapshots)
    - `compute`     attention backend, safe torch.compile, generation config
    - `features`    single-forward-pass 4-site probe feature extraction + mmap cache
    - `cache`       on-disk tokenized-prompt cache for PhysBench
    - `resilience`  JSONL append writer, resume-from-checkpoint, traceback logging
    - `physbench_split`  qualitative vs quantitative subset classifier

Design goals (from the project memory):
    1. Everything fits in 16 GB VRAM with 4-bit nf4 double-quant + bf16 compute.
    2. No silent failures — probing never re-forwards the model; eval never
       loses more than 15 minutes on a crash.
    3. SDPA default, flash-attn opportunistic, torch.compile only if triton.
    4. Pre-tokenized prompts cached once, reused across every run.
    5. One forward pass populates all 4 probe sites to disk; probing reads mmap.

Import surface intentionally small — concrete functions, no framework layer.
"""

from .vram import (
    build_bnb_config,
    snapshot_vram,
    hard_cleanup,
    format_vram_delta,
    assert_vram_below,
)
from .compute import (
    pick_attn_impl,
    try_compile,
    fast_gen_config,
    inference_ctx,
)
from .features import (
    ProbeSites,
    FeatureCache,
    register_probe_hooks,
    extract_features_one_pass,
)
from .cache import (
    PromptCache,
)
from .resilience import (
    JsonlAppender,
    configure_traceback_logging,
    resume_completed_ids,
)
from .physbench_split import (
    classify_quantitative,
    split_physbench,
)

__all__ = [
    # vram
    "build_bnb_config",
    "snapshot_vram",
    "hard_cleanup",
    "format_vram_delta",
    "assert_vram_below",
    # compute
    "pick_attn_impl",
    "try_compile",
    "fast_gen_config",
    "inference_ctx",
    # features
    "ProbeSites",
    "FeatureCache",
    "register_probe_hooks",
    "extract_features_one_pass",
    # cache
    "PromptCache",
    # resilience
    "JsonlAppender",
    "configure_traceback_logging",
    "resume_completed_ids",
    # physbench_split
    "classify_quantitative",
    "split_physbench",
]
