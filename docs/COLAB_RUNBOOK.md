# Colab A100 Runbook — PhysLens Week A

**Purpose:** run Phase 5 Week A on a Colab Pro A100 while the local GPU is busy.
**Compute:** ~90 min A100 ≈ 18 Colab compute units (fits 100-unit budget).
**Checkpoint safety:** every intermediate result is atomically written to Google Drive; runtime disconnects do not lose progress.

---

## 0. Data-staging fast paths

**Do NOT upload PhysBench** — it's 15 GB locally but public on HF Hub.
Colab downloads it fresh in ~3 min. The notebook's cell 4 handles this.

**Only the Qwen3-VL training-PCA cache is unique to your work**, and it
compresses to **~18 MB** (34 MB uncompressed, 2001 small HDF5 fragments).
Three options for getting it to Colab:

| Path | Local upload | Colab time | Best when |
|---|---|---|---|
| **A: Drive zip (recommended)** | 18 MB via browser drag-drop (seconds) | ~5 s unzip | One-shot run |
| **B: Zero upload** | 0 bytes | ~15 min A100 regen ≈ 3 compute units | You can't upload anything |
| **C: HF Hub relay** | One-time `hf upload` (~30 s) | ~1 min pull | Repeated runs; reused across sessions |

### Path A — Drive zip (fastest for first-time)

On your local machine:

```bash
cd D:/WPI_Assignments/AlgoVerse/VLAs
python scripts/bundle_for_colab.py
# -> ./upload_phys_lens_bundle.zip  (~18 MB, 1.8 s)
```

Open `drive.google.com` → `My Drive` → `PhysLens/` → drag the zip in.
Upload takes seconds on any connection.

In the Colab notebook, run cell "Path A — Drive zip".

### Path B — Zero upload (regenerate on Colab)

Skip all uploads. In the Colab notebook, run cell "Path B — Zero upload".
This invokes `scripts/extract_training_features.py` on the A100, which
takes ~15 minutes (costs ~3 compute units) and automatically backs up the
regenerated cache to Drive so subsequent sessions don't repeat the work.

### Path C — HuggingFace Hub relay (for power users)

One-time local setup:

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli repo create physlens-cache --type dataset --private
cd D:/WPI_Assignments/AlgoVerse/VLAs
huggingface-cli upload <your-username>/physlens-cache \
    cache/week1/features/qwen3-vl-8b_train . \
    --repo-type dataset
```

In the Colab notebook, edit the `HF_DATASET_REPO` variable in the Path C
cell to your repo name. Subsequent Colab sessions pull in ~1 min.

---

## 1. First Colab session (~90 min)

1. Open [notebooks/colab_phys_lens.ipynb](../notebooks/colab_phys_lens.ipynb) in Colab.
2. `Runtime → Change runtime type → A100 GPU`.
3. Run cells **1–4** in order (mount Drive, pull repo, install deps, sync cache).
4. Run cell **5 (smoke test)**. If that prints a sane accuracy within ~3 min, plumbing is OK.
5. Run cell **6 (full 20-seed random-direction control)**. ~70 min on A100.
6. Run cell **9 (aggregation)** to see the Gate 1 verdict.

**Budget used:** ~14 units. Remaining: ~86 units.

## 2. Second Colab session (optional, if Gate 1 passes)

1. Re-run cells 1–4 to re-mount + re-pull.
2. Run cell **7 (α=3 sweep)** — ~5 min.
3. Run cell **8 (PhysLens-OC 4-way contest)** — ~15 min.
4. Run cell **9 (aggregation)** — prints the full headline table, writes `week_a_verdict.json` to Drive.

**Budget used:** ~5 units. Remaining: ~80 units.

## 3. Kill-gate decision flow

After Gate 1 completes:

```
  random_control/qwen3-vl-8b/random_control_summary.json
            │
            ├── kill_gate_fired: false  ──►  proceed to Session 2 (α=3, OC contest)
            │                                proceed to Week B (8-model expansion)
            │
            └── kill_gate_fired: true   ──►  SCAS demoted to appendix
                                             Paper reframes to D&B:
                                              PhysLens-Predict + PhysBench-Diag
                                             Still run Week B for predictor
                                             Skip Session 2
```

---

## 4. What to upload back after Colab runs

From Drive, download the following directories back to your local repo:

```
PhysLens/results/week4/
├── random_control/qwen3-vl-8b/
│   ├── baseline_alpha0.json
│   ├── seed_00.json ... seed_19.json
│   └── random_control_summary.json
├── scas_prereg/scas_sweep_qwen3-vl-8b.json
├── oc_contest/
│   ├── amplify/scas_sweep_qwen3-vl-8b.json
│   ├── contrast/scas_sweep_qwen3-vl-8b.json
│   └── qual_contrast/scas_sweep_qwen3-vl-8b.json
└── week_a_verdict.json
```

Commit them to `results/week4/` on the `physics-steering` branch.

---

## 5. Week B (after Week A gates)

Once the 4 Week A runs are committed, run these ADDITIONAL models on Colab
(one per session to avoid Drive I/O bottlenecks):

| Session | Model | Notebook cell modification |
|---|---|---|
| B-1 | minicpm-v-2.6 | `--model minicpm-v-2.6` in cells 5–8 |
| B-2 | glm-4.5v | `--model glm-4.5v` |
| B-3 | llava-onevision-7b | `--model llava-onevision-7b` |
| B-4 | cogvlm2 | `--model cogvlm2` |

Each session needs: the model to be added to `MODEL_LOADERS` in
`scripts/week3_scas_sweep.py` AND a Week 1 training-PCA extraction first.

**Budget estimate Week B:** ~4 × 10 units = 40 units. Tight but fits.

---

## 6. Safety notes

- Keep-alive JS in the notebook is **off by default**. Only enable during active compute.
- Drive writes are atomic (`.tmp` then `os.replace`). Killing the runtime mid-write leaves a `.tmp` file; re-running the script skips the partial seed.
- If you see `CUDA out of memory` on the A100: the bnb 4-bit config already limits Qwen3-VL-8B to ~6.4 GB. If you still OOM, check that `attn_implementation='sdpa'` is being used (printed at model load) and that no other kernel is resident.
- HF gated models (Qwen2.5-VL, LLaVA): set `HF_TOKEN` in Colab Secrets (`🔑` icon in the left sidebar).

---

## 7. Post-run: update Notion

After the verdict JSON lands on Drive:

1. Read `week_a_verdict.json`.
2. Update the [Phase 5 page](https://www.notion.so/3447e661876181b9831fc19b5c73a124) with:
   - Section 10 "Status Snapshot" → mark Gate 1 as PASSED/FAILED
   - Append actual numbers to the "Hard numbers" table in Section 1.3
3. If any kill gate fired, explicitly mark which paper framing was selected in Section 5.
