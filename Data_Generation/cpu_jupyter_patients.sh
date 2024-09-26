#!/bin/bash

#SBATCH --output=%j_output.txt            # Standard output
#SBATCH --error=%j_error.txt              # Standard error

#SBATCH --nodes=1                         # Request one node
#SBATCH --ntasks=1                        # Request one task (process)
#SBATCH --cpus-per-task=16                # Request four CPU cores per task
#SBATCH --qos=short
#SBATCH --mem=2800G


jupyter nbconvert --to notebook --execute c_generate_data.ipynb --output=X_c_generate_data.ipynb 


