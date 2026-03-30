# Where Does Physics Live in Vision Encoders?
## Spatially Probing and Amplifying Physical Reasoning in VLM Representations

**Target venue:** NeurIPS 2026 (Abstract: May 4, Paper: May 6 AOE)

---

## Abstract

Vision-Language Models (VLMs) trained on internet-scale data exhibit surprising physical reasoning capabilities, yet they systematically fail on tasks requiring precise intuitive physics — predicting stability, contact dynamics, and conservation laws. We investigate *where* physical information is encoded within VLM representation pipelines. Using Physion++, a simulation-grounded dataset with ground-truth physical properties (mass, friction, elasticity, object stability), we train lightweight linear and MLP probes on per-patch activations extracted at four pipeline stages: (1) vision encoder output, (2) post-projection, (3) early LLM layers, and (4) mid LLM layers. We produce **physics saliency maps** revealing which image patches carry physical information, measure information loss across the projection bottleneck, and identify systematic spatial degradation from encoder to LLM. To determine which architectural components are the *leverage points* for improving physical reasoning, we train five LoRA configurations targeting different subsets — encoder only, projection only, LLM only, encoder+projection, and full model — and evaluate on PhysBench, GRASP Level 2, and ConservationBench. Our results reveal whether fine-tuning sharpens spatial physics encoding and which components are most responsible for grounding physical intuition in visual representations.

---

## Research Questions

1. **RQ1 — Spatial Encoding:** Which image patches encode physical properties (mass, friction, elasticity, stability) in VLM vision encoders, and do these match physically relevant spatial regions?

2. **RQ2 — Pipeline Degradation:** How does physical information degrade across the representation pipeline (encoder → projection → LLM layers), and is there a measurable "physics bottleneck" at the projection stage?

3. **RQ3 — Leverage Points:** Which architectural components (encoder, projection, LLM) are the primary leverage points for improving physical reasoning when fine-tuned on physics QA data?

4. **RQ4 — Fine-tuning Effects:** Does physics-specific fine-tuning sharpen spatial encoding (more concentrated on physically relevant patches) or merely shift decision boundaries in LLM layers?

---

## Method Overview

The paper is organized around three experimental phases:

### Phase 1: Physics Saliency Probing
- Extract per-patch activations from 3 VLMs at 4 pipeline stages
- Train linear and MLP probes to predict physics properties per patch
- Generate physics saliency maps (14×14 heatmaps overlaid on images)
- Measure R² degradation curves across pipeline stages
- Compare encoder-frozen vs. unfrozen VLM families

### Phase 2: Component Ablation (Factorial LoRA)
- Fine-tune 5 LoRA conditions per VLM (3 models × 5 conditions = 15 runs)
- Evaluate all conditions on PhysBench, GRASP Level 2, ConservationBench
- Identify minimal sufficient component set for physics improvement

### Phase 3: Before/After Visualization
- Re-run probing on best fine-tuned models
- Quantify sharpening of spatial physics encoding
- Attention analysis: does the LLM attend more to physics-relevant patches after fine-tuning?

---

## Models

| Model | Encoder | Projection | LLM | Encoder Frozen? |
|---|---|---|---|---|
| Qwen2.5-VL-7B | ViT-L (custom) | MLP | Qwen2-7B | No |
| InternVL 2.5-8B | InternViT-300M | MLP | InternLM-7B | Partial |
| LLaVA-OneVision-7B | SigLIP-SO400M | MLP | Qwen2-7B | Yes |

---

## Project Structure

```
VLAs/
├── configs/          # Hydra config files for models, probing, ablation
├── src/
│   ├── data/         # Dataset loading, patch labeling, QA generation
│   ├── models/       # VLM loading, activation extraction, LoRA wrappers
│   ├── probing/      # Linear and MLP probes + training loop
│   ├── visualization/# Saliency maps, degradation curves, comparisons
│   ├── evaluation/   # PhysBench, GRASP, ConservationBench evaluators
│   └── ablation/     # Component ablation runner and analyzer
├── scripts/          # Top-level runnable scripts
├── notebooks/        # EDA and result visualization notebooks
├── tests/            # Unit tests
├── results/          # Generated outputs (gitignored)
└── docs/             # Execution plan and task checklists
```

---

## Setup

### 1. Clone and create environment

```bash
git clone <repo-url>
cd VLAs
conda create -n vla-physics python=3.11 -y
conda activate vla-physics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
pip install -e .
# Optional: flash attention for A100
pip install flash-attn --no-build-isolation
```

### 3. Configure HuggingFace access

```bash
huggingface-cli login  # Required for gated models (Qwen2.5, LLaVA)
```

### 4. Download Physion++ dataset

```bash
python scripts/download_physion.py --output-dir data/physion
```

### 5. Verify model loading

```bash
python -c "from src.models.vlm_loader import load_vlm; load_vlm('qwen2_5_vl_7b')"
```

---

## Usage

### Run probing study (Phase 1)

```bash
# Extract activations for all models
python scripts/run_probing.py model=qwen2_5_vl_7b probe=linear_probe

# Generate saliency maps
python scripts/generate_saliency_maps.py --model qwen2_5_vl_7b --variable mass
```

### Run component ablation (Phase 2)

```bash
# Train all 5 ablation conditions for one model
python scripts/run_ablation.py model=qwen2_5_vl_7b ablation=condition_e_full

# Evaluate trained models
python scripts/run_evaluation.py --checkpoint results/ablation_metrics/qwen_condition_e/
```

### Run full evaluation suite

```bash
python scripts/run_evaluation.py --all
```

---

## Key References

- **Physion++**: Bear et al. (2023). *Physion++: Evaluating Physical Scene Understanding that Requires Online Inference*. NeurIPS 2023.
- **PhysBench**: (2024). *PhysBench: Benchmarking Physical Commonsense Understanding in VLMs*.
- **GRASP**: (2024). *GRASP: A Grid-based Benchmark for Physical Scene Understanding*.
- **ConservationBench**: (2024). *ConservationBench: Testing Physical Conservation Laws in VLMs*.
- **Qwen2.5-VL**: Wang et al. (2024). *Qwen2.5-VL Technical Report*.
- **InternVL 2.5**: Chen et al. (2024). *InternVL2.5: A Practical Guide to Scaling Vision-Language Models*.
- **LLaVA-OneVision**: Li et al. (2024). *LLaVA-OneVision: Easy Visual Task Transfer*.
- **LoRA**: Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. ICLR 2022.

---

## Citation

```bibtex
@article{algoverse2026physicsprobing,
  title={Where Does Physics Live in Vision Encoders? Spatially Probing and Amplifying Physical Reasoning in VLM Representations},
  author={AlgoVerse Research},
  journal={arXiv preprint},
  year={2026}
}
```
