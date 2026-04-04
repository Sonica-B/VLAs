#!/bin/bash
# Run FROM your laptop to pull results from Turing
# Usage: bash turing/transfer_results.sh username@turing.wpi.edu

REMOTE=${1:?"Usage: bash turing/transfer_results.sh username@turing.wpi.edu"}

echo "=== Pulling results from Turing ==="
mkdir -p results_turing

scp -r "${REMOTE}:~/VLAs/results/" ./results_turing/
scp -r "${REMOTE}:~/VLAs/logs/" ./results_turing/logs/

echo "Results saved to results_turing/"
echo "Logs saved to results_turing/logs/"
