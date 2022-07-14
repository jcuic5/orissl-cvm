#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodelist=gpu09
#SBATCH --qos=medium
#SBATCH --job-name=cl_vigor_sat # create a short name for your job
#SBATCH --cpus-per-task=2
#SBATCH --mem=16000
#SBATCH --time=30:00:00 # total run time limit (HH:MM:SS)
#SBATCH --gpus-per-node=1 #--gpus-per-node=v100:1
##SBATCH --constraint=gpumem32

module use /opt/insy/modulefiles
module load miniconda
conda activate torchc11.3

python train_cl.py