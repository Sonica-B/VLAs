# Week 2 Day 10-14: GPU Experiment Instructions

## Prerequisites

### Hardware
- GPU with 12GB+ VRAM (RTX 5070 Ti verified)
- 32GB RAM
- ~20GB free disk space (model weights + cached activations)

### Software Setup

```bash
# Activate your environment
conda activate vlas  # or your env name

# Install required packages
pip install qwen-vl-utils bitsandbytes accelerate
pip install peft>=0.10.0 transformers>=4.45.0

# Verify CUDA
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Step-by-Step Instructions

### 1. Validate Pipeline (TEST MODE — No GPU needed)

Run this first to confirm everything works end-to-end:

```bash
cd D:/WPI\ Assignments/AlgoVerse/VLAs
python scripts/run_full_week2_gpu.py --test-mode --num-scenes 100
```

**Expected output:** Results in `results/week2_test/` (~2-5 minutes on CPU)
**What to check:**
- `results/week2_test/comprehensive_results.json` exists with R² values
- `results/week2_test/figures/` has 3-4 PNG plots
- No Python errors or crashes

### 2. Run Full VLM Probing (GPU)

```bash
python scripts/run_full_week2_gpu.py --model qwen --num-scenes 500
```

**Expected VRAM usage per step:**
| Step | VRAM | Time Estimate |
|------|------|---------------|
| 1. Generate data | <1 GB | 1-2 min |
| 2. Load model (4-bit) | ~5 GB | 2-5 min (first time downloads ~5GB) |
| 3. Extract activations | ~8-10 GB peak | 30-60 min |
| 4. Train probes | <2 GB (CPU) | 5-10 min |
| 5. Spatial metrics | <2 GB (CPU) | 5-10 min |
| 6. Degradation analysis | <1 GB | <1 min |
| 7. Generate figures | <1 GB | <1 min |
| 8. Save results | <1 GB | <1 min |

**Total estimated time:** 45-90 minutes

**Key outputs:**
- `results/week2_qwen/comprehensive_results.json` — all R², CIs, spatial metrics
- `results/week2_qwen/figures/degradation_curves_with_ci.png` — main result figure
- `results/week2_qwen/figures/saliency_maps_all_stages.png` — spatial heatmaps
- `results/week2_qwen/figures/cross_property_scatter.png` — mass vs hue correlation
- `results/week2_qwen/figures/differential_degradation.png` — projection bottleneck analysis

### 3. Run LoRA Ablation (Overnight)

Run all 5 conditions sequentially:

```bash
python scripts/run_week2_lora_ablation.py --num-scenes 500 --num-epochs 3
```

Or run one condition at a time:

```bash
python scripts/run_week2_lora_ablation.py --condition A --num-scenes 500
python scripts/run_week2_lora_ablation.py --condition B --num-scenes 500 --resume
python scripts/run_week2_lora_ablation.py --condition C --num-scenes 500 --resume
python scripts/run_week2_lora_ablation.py --condition D --num-scenes 500 --resume
python scripts/run_week2_lora_ablation.py --condition E --num-scenes 500 --resume
```

**Expected VRAM:** ~10-11 GB peak per condition
**Expected time:** ~2-4 hours per condition, ~10-20 hours total

**Key outputs:**
- `results/week2_ablation/ablation_results_combined.json` — all before/after R²
- `results/week2_ablation/figures/ablation_*_before_after.png` — per-variable comparisons
- `results/week2_ablation/figures/ablation_delta_heatmap_*.png` — effect size heatmaps
- `results/week2_ablation/lora_checkpoint_*/` — saved LoRA weights per condition

### 4. Test LoRA Ablation (TEST MODE)

```bash
python scripts/run_week2_lora_ablation.py --test-mode --num-scenes 100
```

## Troubleshooting

### CUDA Out of Memory
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
**Fix:** Reduce batch size:
```bash
python scripts/run_full_week2_gpu.py --model qwen --num-scenes 500 --batch-size 2
```
Or use pre-quantized model:
```bash
python scripts/run_full_week2_gpu.py --model qwen-4bit --num-scenes 500
```

### Model Download Fails
The model is ~5GB. If download stalls:
```bash
# Pre-download the model
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct
# Or the pre-quantized version:
huggingface-cli download unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit
```

### bitsandbytes Errors on Windows
```bash
# Use the Windows-compatible version
pip install bitsandbytes-windows
# Or:
pip install bitsandbytes --prefer-binary
```

If bitsandbytes still fails, use the pre-quantized model (`--model qwen-4bit`), which doesn't require runtime quantization.

### Resume After Crash
Both scripts support `--resume` to skip completed work:
```bash
python scripts/run_full_week2_gpu.py --model qwen --num-scenes 500 --resume
python scripts/run_week2_lora_ablation.py --num-scenes 500 --resume
```

### Import Errors
Make sure you're running from the VLAs project root:
```bash
cd D:/WPI\ Assignments/AlgoVerse/VLAs
python scripts/run_full_week2_gpu.py ...
```

### Checking Progress
Both scripts print step-by-step progress with ETA estimates. Look for:
```
STEP 3/8: Extract activations
    Extracted 10/500 scenes
    VRAM: 8.5GB allocated / 9.2GB reserved / 12.0GB total
```

## What Success Looks Like

After running the full pipeline, you should see in `comprehensive_results.json`:
- **Mass R²** at Stage 1 (encoder) should be highest (0.3-0.7 range)
- **Mass R²** should drop at Stage 2 (post-merger) — the "projection bottleneck"
- **Hue R²** should remain relatively stable across stages
- **Permutation R²** should be ~0 (confirming signal is real)
- **Moran's I** for mass should be positive (spatially coherent physics encoding)

The `differential_degradation.png` figure should clearly show whether mass degrades more than hue at the projection layer, which is the core hypothesis of the paper.
