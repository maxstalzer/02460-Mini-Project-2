#!/bin/bash
#BSUB -q gpuv100
#BSUB -J vae_cov_eval3
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 06:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/eval_output_%J.out
#BSUB -e logs/eval_error_%J.err

export UV_CACHE_DIR="/work3/s215141/.cache/uv"
export TMPDIR="/work3/s215141/.tmp"

echo "Starting CoV Evaluation..."

# Run the evaluation mode! 
# Make sure the experiment folder matches where you saved all 30 models
uv run ensemble_vae.py eval_cov --experiment-folder experiments_50epochs --device cuda --num-reruns 10

echo "Evaluation finished!"