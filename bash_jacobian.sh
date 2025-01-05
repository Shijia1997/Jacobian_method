#!/bin/bash

#SBATCH --job-name=syn     # Job name
#SBATCH --output=syn.log    # Standard output and error log
#SBATCH --nodes=1                  # Run all processes on a single node
#SBATCH --ntasks=1                 # Run a single task
#SBATCH --cpus-per-task=24         # Number of CPU cores per task
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=128G                 # Job memory request
#SBATCH --time=200:00:00           # Time limit hrs:min:sec
#SBATCH --partition=gpua100            # Partition (queue)

# Load necessary modules
source activate shijia_env

# Debug: Print GPU allocation
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Run the Python script
python syn_CN_MCI_AD.py
