#!/bin/bash
# ============================================================================
# Turing cluster setup — run ONCE to prepare the environment.
# ============================================================================
set -e
echo "=== Turing Environment Setup ==="

# 1. Pull latest code.
if [ -d ".git" ]; then
    echo "Repo exists, pulling latest..."
    git fetch --all
    git checkout physics-steering 2>/dev/null || git checkout -b physics-steering origin/physics-steering
    git pull origin physics-steering || echo "No remote physics-steering yet"
else
    echo "ERROR: Run this from the VLAs repo root on Turing."
    exit 1
fi

# 2. Conda environment.
if ! conda info --envs 2>/dev/null | grep -q "vla_physics"; then
    conda create -n vla_physics python=3.11 -y
fi
source activate vla_physics || conda activate vla_physics

# 3. Dependencies.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes peft
pip install scikit-learn numpy tqdm rich matplotlib
pip install qwen-vl-utils pillow decord einops timm
pip install flash-attn --no-build-isolation 2>/dev/null || echo "flash-attn optional"

# 4. Verify GPU.
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name} ({p.total_memory/1e9:.0f}GB)')
"
echo "=== Setup Complete. Submit jobs with sbatch turing/*.sh ==="
