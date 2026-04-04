#!/bin/bash
# Download all datasets needed for experiments
# Run as SLURM job: sbatch --wrap="bash ~/VLAs/turing/download_data.sh" --partition=cpu --mem=16G --time=2:00:00

set -e
source ~/vlas_env/bin/activate
cd ~/VLAs

echo "=== Downloading Physion++ readout data ==="
mkdir -p data
cd data
if [ ! -f physion_readout.zip ]; then
    wget -c https://physion-v2.s3.amazonaws.com/readout_data.zip -O physion_readout.zip
fi
if [ ! -d physion_readout ]; then
    unzip physion_readout.zip -d physion_readout
fi

echo "=== Downloading PhysBench ==="
cd ~/VLAs
python3 scripts/download_physbench.py --data-dir data/physbench

echo "=== Downloading Qwen2.5-VL-7B (will cache in ~/.cache/huggingface/) ==="
python3 -c "
from huggingface_hub import snapshot_download
print('Downloading Qwen2.5-VL-7B-Instruct...')
snapshot_download('Qwen/Qwen2.5-VL-7B-Instruct')
print('Download complete')
"

echo "=== All data downloaded ==="
