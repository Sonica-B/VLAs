# Running VLAs Experiments on WPI Turing Cluster

## Quick Start (5 steps)

### 1. Copy repo to Turing
From your laptop:
```bash
scp -r "D:/WPI Assignments/AlgoVerse/VLAs" username@turing.wpi.edu:~/VLAs
```
Or push to GitHub and clone on Turing.

### 2. Setup environment
```bash
ssh username@turing.wpi.edu
bash ~/VLAs/turing/setup_turing.sh
```

### 3. Download data
```bash
sbatch --wrap="bash ~/VLAs/turing/download_data.sh" --partition=cpu --mem=16G --time=2:00:00
```

### 4. Run baseline (PhysBench evaluation)
```bash
sbatch ~/VLAs/turing/slurm_baseline.sh
```

### 5. Run QLoRA ablation (the core experiment)
```bash
bash ~/VLAs/turing/run_all_conditions.sh
```

## Monitor jobs
```bash
squeue -u $USER           # See running jobs
sacct -j <JOBID>          # Job details
tail -f logs/qlora_*.out  # Live output
scancel <JOBID>           # Cancel a job
scancel -u $USER          # Cancel all your jobs
```

## Run individual conditions
```bash
sbatch turing/slurm_qlora_condition.sh merger
sbatch turing/slurm_qlora_condition.sh llm
sbatch turing/slurm_qlora_condition.sh encoder
sbatch turing/slurm_qlora_condition.sh merger+encoder
sbatch turing/slurm_qlora_condition.sh full
```

## Run Physion++ probing study
```bash
sbatch ~/VLAs/turing/slurm_probing.sh
```

## Pull results back to laptop
```bash
bash turing/transfer_results.sh username@turing.wpi.edu
```

## Storage budget (~50GB)
| Item | Size |
|------|------|
| Model cache (`~/.cache/huggingface/`) | ~14GB |
| Datasets (`~/VLAs/data/`) | ~8GB |
| Virtual environment (`~/vlas_env/`) | ~10GB |
| Results (`~/VLAs/results/`) | ~5GB |
| **Remaining buffer** | **~13GB** |

## A100 advantages over RTX 5070 Ti
- **80GB VRAM** -- no quantization needed (bf16 natively)
- **batch_size=4** instead of 1 (4x effective throughput)
- **~10x faster inference** on large models
- Can run multiple conditions simultaneously if multiple GPUs available

## Partition names
The SLURM scripts use `--partition=gpu` by default. You may need to adjust this
to match Turing's actual partition names. Check available partitions with:
```bash
sinfo -s
```

## Troubleshooting
- **`module not found`**: Make sure you activated the env: `source ~/vlas_env/bin/activate`
- **OOM errors**: Shouldn't happen on A100 80GB with bf16, but reduce `--batch-size` if needed
- **Partition errors**: Run `sinfo -s` to find correct GPU partition name, edit SLURM scripts
- **Storage full**: Check usage with `du -sh ~/vlas_env ~/.cache ~/VLAs`
