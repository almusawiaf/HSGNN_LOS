#!/bin/bash

#SBATCH --output=logs/X_output_%j.txt           # Standard output
#SBATCH --error=logs/X_error_%j.txt             # Standard error

#SBATCH --nodes=1                          
#SBATCH --ntasks=1                         

#SBATCH --cpus-per-task=16
#SBATCH --mem=250G

#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:2

#SBATCH --job-name=GCN_250

# Set environment variables
export num_Sample=10000
export NUM_DISEASES=203
export NUM_TOP_DISEASES=10
export DISEASE_FILE='DMPLB2'
export num_Meta_Path=43
export num_epochs=250
export experiment_name='DMPLB2'

output_dir="/lustre/home/almusawiaf/PhD_Projects/HGNN_Project2/GNN_Models/experiments/${NUM_DISEASES}_Diagnoses/${DISEASE_FILE}/${num_Sample}"
mkdir -p $output_dir


# Run the second notebook in the background
# jupyter nbconvert --to notebook --execute GCN.ipynb --output=$output_dir/X_GCN.ipynb 
jupyter nbconvert --to notebook --execute SAGE.ipynb --output=$output_dir/X_SAGE_${DISEASE_FILE}_${NUM_TOP_DISEASES}D_${experiment_name}.ipynb 

