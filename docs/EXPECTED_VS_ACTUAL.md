# Expected vs Actual Results Tracker

Living document tracking predictions vs measured results for the physics probing
paper (Option C: Where Does Physics Understanding Break Down in VLMs?).

**Model:** Qwen2.5-VL-7B-Instruct (4-bit quantized)
**Benchmark:** PhysBench (val=200, test=10,002)
**Last updated:** 2026-04-04

---

## Phase 1: Baseline PhysBench Evaluation

| Metric | Expected | Actual | Status |
|---|---|---|---|
| PhysBench val accuracy | 35-45% (published Qwen2-VL-7B: 37%) | **60.31%** | BETTER than expected |
| PhysBench test accuracy | 35-45% | **PENDING** (running) | |
| Val: dynamics domain | ~35% | 61.5% | BETTER |
| Val: property domain | ~40% | 63.9% | BETTER |
| Val: relationships domain | ~35% | 66.0% | BETTER |
| Val: scene domain | ~35% | 48.8% | BETTER (but weakest) |
| Model loads in 4-bit on 12GB VRAM | Yes | Yes (~6.4GB) | CONFIRMED |

**Notes:**
- Our val accuracy (60.31%) is significantly higher than published Qwen2-VL-7B (37%).
  This is likely because: (1) we use Qwen2.5-VL not Qwen2-VL, and (2) 4-bit quantization
  may not degrade as much as expected.
- Val set is only 200 questions — test set results will be more authoritative.
- GPT-4o achieves 49.5% on the full test set; our val result exceeds this, but val may
  be easier than test.

---

## Phase 2: Probing (Linear Probe Diagnostics)

Activations extracted at 4 pipeline stages, probed with ridge regression for
mass, friction, elasticity, stability.

| Metric | Expected | Actual | Status |
|---|---|---|---|
| Global R² mass (encoder, stage 1) | 0.3–0.6 (based on P2P paper) | 0.54 | IN RANGE |
| Global R² mass (post-merger, stage 2) | 0.2–0.5 | — | PENDING |
| Global R² mass (LLM-8, stage 3) | 0.2–0.4 | — | PENDING |
| Global R² mass (LLM-16, stage 4) | 0.1–0.3 (degradation expected) | 0.38 | IN RANGE |
| Physics degradation encoder→LLM | Yes (R² drops) | Yes (0.54 → 0.38) | CONFIRMED |
| Merger is bottleneck | Yes (biggest R² drop at merger) | Merger delta > LLM delta | SUPPORTED |
| Friction probing works | R² 0.1–0.4 | — | PENDING |
| Elasticity probing works | R² 0.1–0.4 | — | PENDING |

**Key finding:** Physics information IS present in the encoder (R²=0.54 for mass)
but degrades through the pipeline, consistent with the merger bottleneck hypothesis.

---

## Phase 3: QLoRA Ablation (PREDICTIONS — not yet tested)

These are our **pre-registered hypotheses** before running the experiments.
Actual results will be filled in as conditions complete.

### H3: Merger QLoRA > LLM QLoRA for physics

Central claim of the paper: the visual-language merger (projection MLP) is
where physics understanding is lost, so fine-tuning it should help more than
fine-tuning the LLM.

| Condition | Target | Expected PhysBench | Expected Delta | Actual | Status |
|---|---|---|---|---|---|
| Baseline | — | — | — | 60.31% | MEASURED |
| A: Merger only | visual.merger MLP | 63–65% | +3 to +5% | — | PENDING |
| B: LLM only | First 8 LLM layers Q/V | 61–62% | +1 to +2% | — | PENDING |
| C: Encoder only | Last 6 ViT blocks QKV | 62–64% | +2 to +4% | — | PENDING |
| D: Merger+Encoder | A + C combined | 64–66% | +4 to +6% | — | PENDING |
| E: Full | All components | 63–65% | +3 to +5% | — | PENDING |

**Rationale for predictions:**
- **Condition A (Merger):** If the merger is the bottleneck (Phase 2 evidence supports
  this), then directly improving its projection should yield the biggest per-parameter
  improvement. Expected +3-5% because the merger MLP is small (rank 64 LoRA ≈ 2-4M params)
  but high-leverage.
- **Condition B (LLM):** LLM layers already have physics information (R²=0.38 at layer 16).
  Fine-tuning them teaches better reasoning but doesn't fix the information loss at the merger.
  Expected +1-2% — helpful but not the core issue.
- **Condition C (Encoder):** Encoder has good physics representations (R²=0.54) but they could
  be even better. Expected +2-4% because we're improving what goes INTO the merger.
- **Condition D (Merger+Encoder):** Best of both worlds — improve encoder representations AND
  the projection. Expected strongest improvement (+4-6%).
- **Condition E (Full):** More total params but diluted across components. The LoRA capacity
  is spread thin, so may not match D despite more flexibility. Expected +3-5%.

### Expected probing changes after QLoRA

| Condition | Encoder R² mass | Merger R² mass | LLM-16 R² mass | Net effect |
|---|---|---|---|---|
| A: Merger | No change | ↑ improved | ↑ improved | Fixes bottleneck |
| B: LLM | No change | No change | ↑ slight | Better reasoning only |
| C: Encoder | ↑ improved | ↑ indirect | ↑ slight | Better input signal |
| D: Merger+Encoder | ↑ improved | ↑ improved | ↑ improved | Full pipeline fix |
| E: Full | ↑ improved | ↑ improved | ↑ improved | Diluted improvement |

### Key hypothesis tests

| Hypothesis | Test | Expected Outcome |
|---|---|---|
| H3: Merger > LLM | Compare A vs B PhysBench | A > B by 2-3% |
| H3b: D is best | Compare D vs all others | D highest accuracy |
| H4: Probing tracks performance | Correlate R² delta with PhysBench delta | Positive correlation |
| Physics-specific improvement | Compare physics domains vs non-physics | Physics domains improve more |

---

## Tracking Rules

1. **Before running:** Write predictions with rationale
2. **After running:** Fill in actuals immediately
3. **If surprised:** Document WHY the result differs from expectation
4. **Status codes:**
   - PENDING: Not yet measured
   - IN RANGE: Actual falls within expected range
   - BETTER: Actual exceeds expected (in the good direction)
   - WORSE: Actual falls below expected
   - CONFIRMED: Binary hypothesis confirmed
   - CONTRADICTED: Binary hypothesis contradicted
   - INCONCLUSIVE: Can't determine (noise, too few samples, etc.)

---

## Appendix: Published Baselines (PhysBench paper)

| Model | Published Test Accuracy |
|---|---|
| GPT-4o | 49.5% |
| Gemini-1.5-Pro | 43.2% |
| Qwen2-VL-72B | 46.8% |
| InternVL2-76B | 42.2% |
| Qwen2-VL-7B | 37.0% |
| **Our Qwen2.5-VL-7B (4bit) — val** | **60.31%** |
| **Our Qwen2.5-VL-7B (4bit) — test** | **PENDING** |
