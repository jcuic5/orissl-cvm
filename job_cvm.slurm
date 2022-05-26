#!/bin/bash
#SBATCH --partition=general #--partition=cor,general
#SBATCH --qos=medium
#SBATCH --job-name=cvm_vigor # create a short name for your job
#SBATCH --cpus-per-task=4
#SBATCH --mem=8000
#SBATCH --time=12:00:00 # total run time limit (HH:MM:SS)
#SBATCH --gpus-per-node=1 #--gpus-per-node=v100:1
##SBATCH --constraint=gpumem32

module use /opt/insy/modulefiles
module load miniconda
conda activate vissl

python train_cvm.py