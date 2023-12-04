#!/bin/bash
#SBATCH --partition=yoda
#SBATCH --ntasks 1
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00


# Activate Anaconda work environment
source /home/${USER}/miniconda3/etc/profile.d/conda.sh
conda activate listen2yourheart


python3 listen2yourheart/src/training/pretrain.py $@

