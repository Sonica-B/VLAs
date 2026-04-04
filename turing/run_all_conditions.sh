#!/bin/bash
# Submit all QLoRA ablation conditions as SLURM jobs
# Usage: bash ~/VLAs/turing/run_all_conditions.sh

cd ~/VLAs
mkdir -p logs

echo "=== Submitting QLoRA Ablation Experiments ==="

# Step 1: Generate training data (CPU job, must complete first)
echo "Submitting data generation job..."
DATA_JOB=$(sbatch --parsable \
    --job-name=gen_data \
    --partition=cpu \
    --cpus-per-task=4 \
    --mem=16G \
    --time=1:00:00 \
    --output=logs/gen_data_%j.out \
    --error=logs/gen_data_%j.err \
    --wrap="source ~/vlas_env/bin/activate && cd ~/VLAs && python3 scripts/run_qlora_full.py --generate-data-only --output-dir data/physics_qa_full")

echo "Data generation job: ${DATA_JOB}"

# Step 2: Submit GPU jobs with dependency on data generation
for condition in merger llm encoder merger+encoder full; do
    echo "Submitting condition: ${condition} (depends on ${DATA_JOB})"
    sbatch --dependency=afterok:${DATA_JOB} turing/slurm_qlora_condition.sh ${condition}
done

echo ""
echo "=== All jobs submitted ==="
echo "Monitor with: squeue -u \$USER"
echo "Cancel all:   scancel -u \$USER"
