"""
LoRA intervention specs for the Week 2 diagnosis -> intervention -> improvement loop.

This module is the single source of truth for WHICH modules get fine-tuned under
each Week 2 intervention condition. It is split from `src/optim/__init__.py`
because the PEFT dependency is only needed during training, not during probing.

## Design rationale (for code reviewers)

The core Week 2 hypothesis is that the Qwen3-VL-8B `Qwen3VLVisionPatchMerger`
(2x2 spatial pool + linear_fc1 + GELU + linear_fc2) is the architectural
choke point that makes quantitative physical information degrade relative to
qualitative information. This was derived from Week 1 evidence:

    Qwen3-VL-8B: 784x std compression at merger, 2/3 H3 hits
    Qwen2.5-VL-7B: 270x compression, 3/3 H3 hits
    InternVL3-8B-hf: 2.4x compression (simple Linear projector), 0/3 H3 hits

If the compression hypothesis is correct, applying LoRA to ONLY the merger's
two Linear layers should recover PhysBench performance on the quantitative
slice MORE than applying LoRA to an equivalent number of parameters in the
LLM attention layers. This is a causal test with a double-dissociation
prediction:

    Condition B (merger-only LoRA):
        + large gain on quant slice
        + small/no gain on qual slice

    Condition C (LLM-only LoRA, first 8 layers Q/V):
        + small gain on quant slice
        + larger gain on qual slice

The target module paths below were verified empirically on Qwen3-VL-8B loaded
in bnb 4-bit nf4 with transformers 5.6.0.dev0 (April 2026). See the commit
message for the exact module tree dump used to derive these paths.

## Why these exact modules

Condition B (merger): `model.visual.merger.linear_fc1` and `linear_fc2` are
the TWO Linear layers inside `Qwen3VLVisionPatchMerger`. linear_fc1 maps
4608 -> 4608 (post-spatial-pool expansion), linear_fc2 maps 4608 -> 4096
(projection into the LLM embedding space). These two Linears are the entire
learnable interface between vision and language under the compression-
hypothesis framing.

Condition C (LLM attention): First 8 transformer layers' Q/V projections.
8 layers x 2 projections = 16 LoRA adapters. Parameter count is deliberately
matched to Condition B (within ~2x) so that any difference in performance
isn't just a function of trainable parameter count.

## Parameter count matching

LoRA rank is tuned per condition to approximately match trainable parameter
count. With r=16 alpha=32:
    B: 2 Linear layers x (4608*16 + 16*4608) ~= 295K params. Use r=32 to
       double this to ~590K.
    C: 16 Linear layers x (4096*16 + 16*4096) ~= 2.1M params. Use r=8 to
       reduce this to ~1.05M.

Actual measured values are reported at training time by PEFT's
`print_trainable_parameters()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LoraIntervention:
    """A single Week 2 intervention condition.

    Attributes:
        id: Single-letter identifier (B / C / D / E) matching the Notion plan.
        name: Short human-readable name.
        target_modules: Substring match list passed to PEFT's LoraConfig.
            PEFT resolves each substring against the flat list of module
            names in the model and applies LoRA to every match.
        rank: LoRA rank (r). Chosen to approximately match trainable
            parameter count across conditions.
        lora_alpha: LoRA scaling factor (alpha / r is the effective LR
            multiplier on the adapter). Default 2x rank.
        lora_dropout: LoRA dropout on the adapter forward path.
        description: One-line rationale for the condition (shown at training
            time for reproducibility).
    """

    id: str
    name: str
    target_modules: List[str]
    rank: int
    lora_alpha: Optional[int] = None
    lora_dropout: float = 0.05
    description: str = ""

    def __post_init__(self):
        if self.lora_alpha is None:
            self.lora_alpha = self.rank * 2

    def to_peft_config(self, task_type: str = "CAUSAL_LM"):
        """Build a PEFT LoraConfig from this intervention spec.

        The import is deferred to avoid hard-dependency on peft for modules
        that only consume intervention metadata (e.g. eval code).
        """
        from peft import LoraConfig, TaskType
        return LoraConfig(
            r=self.rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=list(self.target_modules),
            task_type=getattr(TaskType, task_type),
            bias="none",
        )


# ---------------------------------------------------------------------------
# Qwen3-VL-8B Week 2 intervention registry.
# ---------------------------------------------------------------------------
#
# Condition letters follow the Notion plan:
#   A: Encoder-only (last ViT blocks) -- NOT RUN in Week 2 (Week 1 showed
#      encoder features are intact; LoRA on encoder is low-leverage)
#   B: Projection/merger-only          -- PRIMARY Week 2 condition
#   C: LLM-only                        -- PRIMARY Week 2 comparison
#   D: Encoder + Merger                -- Optional Week 2 follow-up
#   E: Full                            -- Optional ceiling check
#
# Week 2 MVP runs ONLY B and C (the double-dissociation test).

QWEN3_VL_8B_INTERVENTIONS: Dict[str, LoraIntervention] = {
    "B": LoraIntervention(
        id="B",
        name="merger",
        # Substring match -- resolves to exactly:
        #   model.visual.merger.linear_fc1
        #   model.visual.merger.linear_fc2
        # which are the two Linear layers inside Qwen3VLVisionPatchMerger.
        target_modules=[
            "visual.merger.linear_fc1",
            "visual.merger.linear_fc2",
        ],
        rank=32,
        lora_alpha=64,
        description=(
            "LoRA on Qwen3-VL-8B PatchMerger (linear_fc1, linear_fc2). "
            "Tests whether targeting the 784x-compression merger specifically "
            "recovers quantitative physics performance."
        ),
    ),
    "C": LoraIntervention(
        id="C",
        name="llm",
        # First 8 LLM layers, Q and V projections.
        # Keys match `model.language_model.layers.{i}.self_attn.{q_proj,v_proj}`
        # via substring (PEFT does substring matching, so "layers.0.self_attn.q_proj"
        # is sufficient to disambiguate from `deepstack_merger_list` etc.).
        target_modules=[
            *[f"language_model.layers.{i}.self_attn.q_proj" for i in range(8)],
            *[f"language_model.layers.{i}.self_attn.v_proj" for i in range(8)],
        ],
        rank=8,
        lora_alpha=16,
        description=(
            "LoRA on Qwen3-VL-8B first 8 LLM layers Q/V projections. "
            "Comparison baseline: tests whether LLM-side fine-tuning "
            "recovers physics performance comparably to merger-side."
        ),
    ),
    "D": LoraIntervention(
        id="D",
        name="merger+encoder",
        target_modules=[
            "visual.merger.linear_fc1",
            "visual.merger.linear_fc2",
            *[f"visual.blocks.{i}.attn.qkv" for i in range(21, 27)],
        ],
        rank=16,
        lora_alpha=32,
        description=(
            "LoRA on merger + last 6 ViT blocks (indices 21-26). "
            "Tests whether encoder refinement compounds with merger fix."
        ),
    ),
    "E": LoraIntervention(
        id="E",
        name="full",
        target_modules=[
            "visual.merger.linear_fc1",
            "visual.merger.linear_fc2",
            *[f"visual.blocks.{i}.attn.qkv" for i in range(21, 27)],
            *[f"language_model.layers.{i}.self_attn.q_proj" for i in range(8)],
            *[f"language_model.layers.{i}.self_attn.v_proj" for i in range(8)],
        ],
        rank=16,
        lora_alpha=32,
        description=(
            "LoRA across encoder + merger + LLM. Reference ceiling. "
            "Useful only if B/C fail to show a differential -- if the ceiling "
            "is also flat, the experiment's intervention is underpowered."
        ),
    ),
}


def resolve_target_modules(model, intervention: LoraIntervention) -> List[str]:
    """Given a loaded model and an intervention spec, return the list of
    fully-qualified module names that will receive LoRA adapters.

    This is purely for validation / logging -- PEFT does its own substring
    matching at `get_peft_model(...)` time. Running this separately lets the
    training script print the exact list to the log BEFORE instantiating
    the PeftModel, which is useful when debugging "no target modules found"
    errors.
    """
    all_names = {name for name, _ in model.named_modules()}
    resolved: List[str] = []
    missing: List[str] = []
    for pattern in intervention.target_modules:
        hits = sorted(n for n in all_names if pattern in n)
        if hits:
            resolved.extend(hits)
        else:
            missing.append(pattern)
    return resolved if not missing else resolved  # caller handles missing check


def resolution_report(model, intervention: LoraIntervention) -> Dict:
    """Full diagnostic report for an intervention on a loaded model.

    Returns a dict with keys:
        resolved: list of fully-qualified module names that matched
        missing: list of patterns that did not match any module
        by_pattern: {pattern: [matched names]}
    """
    all_names = {name for name, _ in model.named_modules()}
    by_pattern: Dict[str, List[str]] = {}
    resolved: List[str] = []
    missing: List[str] = []
    for pattern in intervention.target_modules:
        hits = sorted(n for n in all_names if pattern in n)
        by_pattern[pattern] = hits
        if hits:
            resolved.extend(hits)
        else:
            missing.append(pattern)
    return {
        "intervention_id": intervention.id,
        "intervention_name": intervention.name,
        "resolved": resolved,
        "missing": missing,
        "by_pattern": by_pattern,
    }
