#!/bin/bash
# One-time setup script for WPI Turing cluster
# Run: bash ~/VLAs/turing/setup_turing.sh
#
# Requirements: Python 3.10+, CUDA-capable GPU (A100)
# No admin/sudo needed — everything installs to ~/

set -e

echo "=== Setting up VLAs research environment on Turing ==="

# Check Python
python3 --version || { echo "ERROR: Python3 not found. Load a module or install miniconda."; exit 1; }

# Create virtual environment (no admin needed)
if [ -d ~/vlas_env ]; then
    echo "Virtual environment already exists at ~/vlas_env"
else
    echo "Creating virtual environment..."
    python3 -m venv ~/vlas_env
fi
source ~/vlas_env/bin/activate

# Install PyTorch with CUDA 12.1 (A100 supports CUDA 11.8+)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies
pip install "transformers>=4.45.0" "accelerate>=0.34.0" "peft>=0.10.0"
pip install bitsandbytes  # Optional on A100 — can run bf16 natively
pip install datasets huggingface-hub
pip install qwen-vl-utils
pip install scikit-learn numpy scipy h5py matplotlib seaborn
pip install pillow tqdm wandb einops safetensors
pip install decord  # For video loading

echo ""
echo "=== Setup complete ==="
echo "Activate with: source ~/vlas_env/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Copy VLAs repo to ~/VLAs (scp or git clone)"
echo "  2. Download data: sbatch --wrap='bash ~/VLAs/turing/download_data.sh' --partition=cpu --mem=16G --time=2:00:00"
echo "  3. Run baseline: sbatch ~/VLAs/turing/slurm_baseline.sh"
echo "  4. Run ablation: bash ~/VLAs/turing/run_all_conditions.sh"
