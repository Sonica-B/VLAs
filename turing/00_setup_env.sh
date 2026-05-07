#!/bin/bash
# ============================================================================
# JOB 0: One-time environment setup on Turing (run BEFORE any other job)
# ============================================================================
# This installs ALL Python dependencies into a user-local pip prefix
# since we don't have sudo and conda may not be configured.
#
# Run interactively (NOT as sbatch): bash turing/00_setup_env.sh
# ============================================================================

set -e
cd ${HOME}/VLAs

echo "=== Turing Environment Setup ($(date)) ==="

# 1. Find available CUDA module.
echo "Searching for CUDA modules..."
module avail cuda 2>&1 | head -20 || echo "module avail failed, trying alternatives..."
module avail CUDA 2>&1 | head -20 || true
module avail nvidia 2>&1 | head -20 || true

# Try loading CUDA (common names on HPC clusters).
for CUDA_MOD in cuda/12.2.2 cuda/12.1.1 cuda/12.0 cuda/11.8 cuda cuda/12 CUDA/12.2 nvidia/cuda/12.2; do
    if module load ${CUDA_MOD} 2>/dev/null; then
        echo "Loaded CUDA module: ${CUDA_MOD}"
        echo "export CUDA_MODULE=${CUDA_MOD}" > ${HOME}/VLAs/.turing_cuda_module
        break
    fi
done

# Check if nvcc is available (CUDA might be in PATH already without module).
if command -v nvcc &>/dev/null; then
    echo "nvcc found: $(nvcc --version | head -1)"
elif command -v nvidia-smi &>/dev/null; then
    echo "nvidia-smi found (CUDA driver present, toolkit may not need module)"
    echo "Driver CUDA: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo 'unknown')"
else
    echo "WARNING: No CUDA found. GPU jobs may still work if PyTorch bundles its own CUDA runtime."
fi

# 2. Load Python.
module load python 2>/dev/null || module load Python 2>/dev/null || echo "Using system python"
echo "Python: $(python --version 2>&1)"
echo "Pip: $(pip --version 2>&1)"

# 3. Install ALL dependencies .
echo ""
echo "Installing Python packages..."
pip install  \
    torch torchvision \
    transformers accelerate bitsandbytes peft \
    scikit-learn numpy scipy \
    tqdm rich matplotlib \
    qwen-vl-utils pillow decord einops timm \
    huggingface_hub \
    2>&1 | tail -20

# Try flash-attn (optional, often fails without a build environment).
pip install flash-attn --no-build-isolation 2>/dev/null \
    && echo "flash-attn installed" \
    || echo "flash-attn skipped (optional, will use SDPA)"

# 4. Verify all critical imports.
echo ""
echo "=== Verifying imports ==="
python -c "
import sys
print(f'Python: {sys.version}')
errors = []
for mod in ['torch', 'numpy', 'sklearn', 'transformers', 'accelerate',
            'bitsandbytes', 'peft', 'tqdm', 'matplotlib', 'PIL',
            'qwen_vl_utils', 'decord', 'einops', 'timm', 'rich']:
    try:
        m = __import__(mod if mod != 'PIL' else 'PIL.Image')
        ver = getattr(m, '__version__', '?')
        print(f'  [OK] {mod}: {ver}')
    except ImportError as e:
        errors.append(mod)
        print(f'  [FAIL] {mod}: {e}')

import torch
print(f'  torch.cuda.available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f'  GPU {i}: {p.name} ({p.total_memory/1e9:.0f}GB)')

if errors:
    print(f'\nFAILED imports: {errors}')
    print('Fix these before submitting jobs.')
    sys.exit(1)
else:
    print('\nAll imports OK.')
"

# 5. Save the working module/env config for use by all sbatch scripts.
echo ""
echo "Saving environment config to .turing_env..."
cat > ${HOME}/VLAs/.turing_env << 'INNER'
# Source this at the top of every sbatch script.
module load python 2>/dev/null || module load Python 2>/dev/null || true

# Load CUDA if a working module was found during setup.
if [ -f ${HOME}/VLAs/.turing_cuda_module ]; then
    source ${HOME}/VLAs/.turing_cuda_module
    module load ${CUDA_MODULE} 2>/dev/null || true
fi

# Ensure packages are on PATH.
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${HOME}/.local/lib/python3.13/site-packages:${PYTHONPATH}"
INNER

echo ""
echo "=== Setup Complete ==="
echo "Run: source .turing_env   (already done automatically by sbatch scripts)"
echo "Then: sbatch turing/01_week1_full_extraction.sh"
