#!/bin/bash
#BSUB -q gpuv100
#BSUB -J vae_ensemble_full
#BSUB -n 1
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 10:00
#BSUB -R "rusage[mem=8GB]"
#BSUB -o logs/output_%J.out
#BSUB -e logs/error_%J.err

# 1. Setup Environment (Fixes the quota issues!)
export UV_CACHE_DIR="/work3/s215141/.cache/uv"
export TMPDIR="/work3/s215141/.tmp"

# Create a logs folder so your main directory doesn't get cluttered
mkdir -p logs

# 2. Define our test parameters
# (Change these back to 50 and 10 for the real run)
EPOCHS=20
NUM_RUNS=10

echo "Starting VAE Ensemble Batch Job..."
echo "Using Device: CUDA"

# 3. The Nested Loop
for decoders in 1 2 3; do
    for run in $(seq 1 $NUM_RUNS); do
        
        # Define the systematic folder name
        EXPERIMENT_DIR="experiments/model_dec${decoders}_run${run}"
        
        echo "---------------------------------------------------"
        echo "Training Architecture: $decoders Decoders"
        echo "Run Iteration: $run / $NUM_RUNS"
        echo "Saving outputs to: $EXPERIMENT_DIR"
        echo "---------------------------------------------------"

        # Execute the python script using uv
        uv run ensemble_vae.py train \
            --num-decoders $decoders \
            --epochs-per-decoder $EPOCHS \
            --experiment-folder $EXPERIMENT_DIR \
            --device cuda

    done
done

echo "All training runs completed successfully!"