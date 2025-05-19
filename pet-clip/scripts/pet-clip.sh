#!/bin/bash
#SBATCH --job-name=tiennh       # Job name
#SBATCH --output=w_result_ct-clip.txt      # Output file
#SBATCH --error=w_error_ct-clip.txt        # Error file
#SBATCH --ntasks=1               # Number of tasks (processes)
#SBATCH --cpus-per-task=4        # Number of CPU cores per task
#SBATCH --mem=4G                 # Memory per node (4 GB)
#SBATCH --gpus=1                 # Number of GPUs per node

accelerate launch  --multi_gpu --num_processes=4 scripts/run_train_cvit.py