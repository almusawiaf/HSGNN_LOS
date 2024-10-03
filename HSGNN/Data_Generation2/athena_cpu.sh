#!/bin/bash

#SBATCH --output=logs/X_output_%j.txt           # Standard output
#SBATCH --error=logs/X_error_%j.txt             # Standard error

#SBATCH --nodes=1                          # Request one node
#SBATCH --ntasks=1                         # Request one task (process)

#SBATCH --cpus-per-task=16                 # Number of CPU cores per task
#SBATCH --mem=250G                         # Allocate memory (512 GB in this case)

#SBATCH --job-name=data_generation_15000

# Set environment variables
export MIMIC_Path='/home/almusawiaf/MyDocuments/PhD_Projects/Data/MIMIC_resources'
export disease_data_path='/home/almusawiaf/PhD_Projects/HGNN_Project2/Data'

export NUM_DISEASES=203
export DISEASE_FILE='DMPLB2'
export similarity_type='PC'

export num_Sample=45555
export r_u_sampling='False'
export SNF_ing='False'
experiment_name="45555_patients"


# Create a new directory with the job name and timestamp
output_dir="/lustre/home/almusawiaf/PhD_Projects/HGNN_Project2/Data_Generation2/experiments"

# output_dir="output_${SLURM_JOB_NAME}_$(date +%Y%m%d_%H%M%S)"
mkdir -p $output_dir

# Pipeline of actions
jupyter nbconvert --to notebook --execute main_cpu.ipynb                  --output $output_dir/main_cpu_${NUM_DISEASES}_${num_Sample}_${DISEASE_FILE}_${experiment_name}.ipynb
# jupyter nbconvert --to notebook --execute b_data_preparation.ipynb      --output $output_dir/b_data_preparation.ipynb
# # jupyter nbconvert --to notebook --execute c_StructureSimilarity.ipynb --output $output_dir/c_StructureSimilarity.ipynb
# # jupyter nbconvert --to notebook --execute d_SNF.ipynb                 --output $output_dir/d_SNF.ipynb
# # jupyter nbconvert --to notebook --execute e_convert_SNF_to_edge.ipynb --output $output_dir/e_convert_SNF_to_edge.ipynb
# jupyter nbconvert --to notebook --execute f_Y_superclass.ipynb          --output $output_dir/f_Y_superclass.ipynb
