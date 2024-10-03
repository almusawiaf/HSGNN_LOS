#!/bin/bash

#SBATCH --output=logs/X_output_%j.txt           # Standard output
#SBATCH --error=logs/X_error_%j.txt             # Standard error

#SBATCH --nodes=1                               # Request one node
#SBATCH --ntasks=1                              # Request one task (process)
#SBATCH --cpus-per-task=1                      # Number of CPU cores per task
#SBATCH --mem=250G                              # Allocate memory

#SBATCH --partition=gpu-a100                    # Use the GPU-A100 partition
#SBATCH --gres=gpu:1                            # Request 1 GPU

module load cuda/12.3                           # Load CUDA module

conda activate envCUDA                          # Activate your environment with CuPy installed

output_dir="/lustre/home/almusawiaf/PhD_Projects/HGNN_Project2/Data_Generation2/experiments/${NUM_DISEASES}_Diagnoses/${DISEASE_FILE}/${num_Sample}"
mkdir -p $output_dir

jupyter nbconvert --to notebook --execute matrix_multiplication.ipynb --output=$output_dir/X_matrix_multiplication_${DISEASE_FILE}_${NUM_TOP_DISEASES}D_${experiment_name}.ipynb
