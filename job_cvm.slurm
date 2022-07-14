#!/bin/bash
#SBATCH --partition=general
#SBATCH --nodelist=gpu11
#SBATCH --qos=long
#SBATCH --job-name=cvm_vigor # create a short name for your job
#SBATCH --cpus-per-task=32
#SBATCH --mem=128000
#SBATCH --time=50:00:00 # total run time limit (HH:MM:SS)
#SBATCH --gpus-per-node=a40:1 #--gpus-per-node=v100:1
##SBATCH --constraint=gpumem32

module use /opt/insy/modulefiles
module load miniconda
conda activate torchc11.3

python train_cvm.py --config_path configs/train_cvm.yaml
# python train_cvm.py --config_path work_dirs/Jun27-1644/train_cvm_pretrain.yaml