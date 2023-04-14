#!/bin/bash
#SBATCH --job-name=train_mixvpr
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1

#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32000m 
#SBATCH --time=33:00:00

#SBATCH --account=kskin1
#SBATCH --partition=spgpu
#SBATCH --gpus-per-task=1

nproc
nvidia-smi
source /home/advaiths/miniconda3/etc/profile.d/conda.sh
conda activate clip
python /home/advaiths/clip-slcd/MixVPR/main.py $1