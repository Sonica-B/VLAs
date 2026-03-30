# Week 1 Detailed Task Checklist
## Mar 30 – Apr 5: Foundation & Data Pipeline

---

## Day 1 (Mon, Mar 30): Environment Setup

### [ ] 1.1 Verify CUDA and GPU availability
```bash
nvidia-smi                          # Should show A100 80GB
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# Expected: True, 12.x
```
**Acceptance:** CUDA 12.x available, at least 1 A100 visible.

### [ ] 1.2 Create and activate conda environment
```bash
conda create -n vla-physics python=3.11 -y
conda activate vla-physics
pip install -r requirements.txt
pip install -e .
```
**Acceptance:** `python -c "import transformers, peft, h5py; print('OK')"` exits cleanly.

### [ ] 1.3 HuggingFace authentication
```bash
huggingface-cli login
# Paste token with read access to gated repos
```
**Acceptance:** `huggingface-cli whoami` returns your username.

### [ ] 1.4 Test Qwen2.5-VL-7B loads in 4-bit
```python
from src.models.vlm_loader import load_vlm
model, processor = load_vlm("qwen2_5_vl_7b", load_in_4bit=True)
print(f"Loaded: {sum(p.numel() for p in model.parameters()) / 1e9:.1f}B params")
# Expected: ~7.6B params shown, GPU memory ~6-8GB
```
**Acceptance:** Model loads without OOM, generates a plausible caption for a test image.

---

## Day 2 (Tue, Apr 1): Dataset Download & Parsing

### [ ] 2.1 Download Physion++ dataset
```bash
python scripts/download_physion.py \
    --output-dir data/physion \
    --scenarios all \
    --verify-checksums
# Expected runtime: 2-4 hrs depending on bandwidth (~50GB)
```
**Acceptance:** `data/physion/` contains scenario folders; running `--verify-checksums` passes.

### [ ] 2.2 Inspect metadata structure
```python
import pickle
with open("data/physion/dominoes/trial_000/metadata.pkl", "rb") as f:
    meta = pickle.load(f)
print(meta.keys())
# Expected keys: objects, frames, physics_params, segmentation_masks
print(meta["physics_params"])
# Expected: dict with mass, friction, elasticity per object
```
**Acceptance:** Can access `mass`, `friction`, `elasticity` for each object in at least 3 scenario types.

### [ ] 2.3 Test PhysionLoader
```python
from src.data.physion_loader import PhysionLoader
loader = PhysionLoader("data/physion", split="train")
sample = loader[0]
print(sample.keys())
# Expected: image, object_masks, physics_labels, scenario_id, frame_idx
assert sample["physics_labels"]["mass"].shape[0] == sample["object_masks"].shape[0]
print("PhysionLoader OK")
```
**Acceptance:** Loader returns structured dicts with correct shape alignment between masks and labels.

### [ ] 2.4 Test PatchLabelAssigner
```python
from src.data.patch_label_assigner import PatchLabelAssigner
assigner = PatchLabelAssigner(patch_grid_size=14)
patch_labels = assigner.assign(sample["object_masks"], sample["physics_labels"])
print(patch_labels.shape)  # Expected: [196, num_properties]
print(patch_labels[:5])    # Check values are in plausible ranges
```
**Acceptance:** Output shape `[196, 4]` (mass, friction, elasticity, stability); no NaN values.

---

## Day 3 (Wed, Apr 2): Activation Extraction

### [ ] 3.1 Implement forward hooks for Qwen2.5-VL-7B
```python
from src.models.activation_extractor import ActivationExtractor
extractor = ActivationExtractor(model, model_name="qwen2_5_vl_7b")

# Test with a single image
activations = extractor.extract(image=sample["image"], processor=processor)
for stage, act in activations.items():
    print(f"{stage}: {act.shape}")
# Expected:
#   stage_1_enc_out:   [196, 1280]   (ViT-L hidden dim)
#   stage_2_post_proj: [196, 3584]   (Qwen2 LLM hidden dim)
#   stage_3_llm_8:     [196, 3584]
#   stage_4_llm_16:    [196, 3584]
```
**Acceptance:** All 4 stages return tensors; stage 1 dim matches ViT-L hidden dim (1280); stages 2-4 match LLM hidden dim (3584).

### [ ] 3.2 Validate hook cleanup
```python
# Ensure no memory leak from lingering hooks
extractor.clear_hooks()
import gc; gc.collect(); torch.cuda.empty_cache()
print(f"GPU memory after cleanup: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
# Should be close to baseline (model weights only)
```
**Acceptance:** GPU memory returns to near-baseline after hook cleanup.

### [ ] 3.3 Run extraction on 100-sample subset and save to HDF5
```bash
python scripts/run_probing.py \
    model=qwen2_5_vl_7b \
    probe=linear_probe \
    data.num_samples=100 \
    data.save_activations=true \
    data.activation_dir=results/activations/qwen_debug
```
**Acceptance:** HDF5 files written to `results/activations/qwen_debug/`; can be re-opened and shapes match.

---

## Day 4 (Thu, Apr 3): Patch Label Assignment & HDF5 Storage

### [ ] 4.1 Verify segmentation mask → patch label pipeline on 10 samples
```python
from src.data.physion_loader import PhysionLoader
from src.data.patch_label_assigner import PatchLabelAssigner

loader = PhysionLoader("data/physion", split="train")
assigner = PatchLabelAssigner(patch_grid_size=14)

for i in range(10):
    sample = loader[i]
    labels = assigner.assign(sample["object_masks"], sample["physics_labels"])
    assert not torch.isnan(labels).any(), f"NaN in sample {i}"
    # Check: patches with no object assigned should have label -1 (background)
    background_patches = (labels[:, 0] == -1).sum()
    print(f"Sample {i}: {background_patches}/196 background patches")
```
**Acceptance:** All 10 samples produce valid label tensors; background patches flagged correctly.

### [ ] 4.2 Verify stability score computation
```python
from src.data.patch_label_assigner import PatchLabelAssigner
assigner = PatchLabelAssigner(patch_grid_size=14)

# Load a multi-frame scenario (dominoes collapse)
scenario = loader.load_scenario("dominoes/trial_000")
stability_scores = assigner.compute_stability_scores(
    positions_over_time=scenario["object_positions"],  # [T, N_objects, 3]
    segmentation_masks=scenario["segmentation_masks"]  # [T, H, W]
)
print(stability_scores.shape)  # Expected: [196] — per-patch stability score
import matplotlib.pyplot as plt
plt.imshow(stability_scores.reshape(14, 14), cmap="hot"); plt.savefig("/tmp/stability_debug.png")
# Visual check: high-activity patches should correspond to falling objects
```
**Acceptance:** Stability score heatmap visually highlights moving objects when inspected.

---

## Day 5 (Fri, Apr 4): Linear Probe Implementation & Training

### [ ] 5.1 Train linear probe on Stage 1 (encoder output) for mass
```python
from src.probing.linear_probe import LinearProbe
from src.probing.probe_trainer import ProbeTrainer

probe = LinearProbe(input_dim=1280, alpha=1.0)
trainer = ProbeTrainer(probe, device="cuda")

# Load pre-extracted activations
trainer.fit_from_hdf5(
    activation_path="results/activations/qwen_debug/",
    stage="stage_1_enc_out",
    target_variable="mass",
    exclude_background=True
)

r2 = trainer.evaluate()
print(f"R² (mass, stage 1): {r2:.4f}")
# Expected: r2 > 0.10 (rough target; depends on dataset size)
```
**Acceptance:** R² > 0.05 on held-out test split (above chance baseline).

### [ ] 5.2 Train probe at all 4 stages and compare
```python
stages = ["stage_1_enc_out", "stage_2_post_proj", "stage_3_llm_8", "stage_4_llm_16"]
r2_by_stage = {}
for stage in stages:
    probe = LinearProbe(input_dim=probe_dim[stage], alpha=1.0)
    trainer = ProbeTrainer(probe, device="cuda")
    trainer.fit_from_hdf5("results/activations/qwen_debug/", stage, "mass")
    r2_by_stage[stage] = trainer.evaluate()
    print(f"R² ({stage}): {r2_by_stage[stage]:.4f}")

# Expected: R² should be highest at stage 1 (encoder output)
# and generally decrease toward stage 4 (deeper LLM layers)
```
**Acceptance:** Results logged; R² values form a degradation pattern (not necessarily monotone, but interpretable).

### [ ] 5.3 Implement MLP probe and verify
```python
from src.probing.mlp_probe import MLPProbe

mlp_probe = MLPProbe(input_dim=1280, hidden_dim=512)
trainer = ProbeTrainer(mlp_probe, device="cuda", lr=1e-3, epochs=20)
trainer.fit_from_hdf5("results/activations/qwen_debug/", "stage_1_enc_out", "mass")
r2_mlp = trainer.evaluate()
print(f"MLP R² (mass, stage 1): {r2_mlp:.4f}")
# MLP should be >= linear probe R²
```
**Acceptance:** MLP R² ≥ linear probe R² on same data/stage.

---

## Day 6 (Sat, Apr 5): First Saliency Map Prototype

### [ ] 6.1 Generate physics saliency map for mass
```python
from src.visualization.saliency_map import PhysicsSaliencyMap

saliency_viz = PhysicsSaliencyMap(patch_grid_size=14, upsample_mode="bilinear")

# Load a test sample
sample = loader[0]
image = sample["image"]

# Get per-patch R² values from trained probe
per_patch_r2 = trainer.get_per_patch_scores()  # shape [196]

# Generate heatmap overlay
fig = saliency_viz.visualize(
    image=image,
    per_patch_scores=per_patch_r2,
    title="Physics Saliency: Mass (Qwen2.5-VL, Stage 1)",
    colormap="inferno",
    alpha=0.6
)
fig.savefig("results/figures/debug_saliency_mass_stage1.png", dpi=150)
print("Saliency map saved.")
```
**Acceptance:** PNG file saved; visually shows a 14×14 heatmap overlay on the image; hottest patches correspond to object locations (rough visual check).

### [ ] 6.2 Generate R² degradation curve plot
```python
from src.visualization.degradation_curves import DegradationCurvePlot

plotter = DegradationCurvePlot()
plotter.add_model_results(
    model_name="Qwen2.5-VL-7B",
    variable="mass",
    stages=stages,
    r2_values=[r2_by_stage[s] for s in stages]
)
fig = plotter.plot()
fig.savefig("results/figures/debug_r2_degradation.png", dpi=150)
```
**Acceptance:** PNG file saved; shows a line plot with 4 x-points (pipeline stages) and y=R².

---

## Day 7 (Sun, Apr 6): Debug, Document, Push

### [ ] 7.1 Run full test suite
```bash
pytest tests/ -v --tb=short
# All tests should pass
```
**Acceptance:** 0 test failures (warnings OK).

### [ ] 7.2 Write Week 1 results summary
- Create `docs/WEEK1_RESULTS.md` with:
  - R² values obtained (mass × 4 stages on debug subset)
  - Any architectural surprises (shape mismatches, unexpected activation dims)
  - Notes on Physion++ metadata structure quirks
  - Open questions for Week 2

### [ ] 7.3 Git commit all working code
```bash
git add src/ scripts/ configs/ docs/ tests/
git commit -m "Week 1: activation extraction + linear probe + debug saliency map"
git push origin main
```
**Acceptance:** All Week 1 code is committed and pushed.

### [ ] 7.4 Plan Week 2 extraction jobs
- Estimate GPU-hours needed for full Physion++ extraction (all 3 models)
- Write SLURM job scripts if on cluster
- Schedule jobs to start Monday morning before arrival

---

## Week 1 Go / No-Go Criteria

Before proceeding to Week 2, verify:

| Criterion | Status |
|---|---|
| All 3 VLMs load in bf16 or 4-bit without OOM | [ ] |
| PhysionLoader returns valid physics label dicts | [ ] |
| Activation extraction works for Qwen (all 4 stages, correct shapes) | [ ] |
| HDF5 storage and reload works correctly | [ ] |
| Linear probe trains and produces R² > 0.05 for mass | [ ] |
| First saliency map PNG generated and visually plausible | [ ] |
| All unit tests pass | [ ] |
| Code committed to git | [ ] |
