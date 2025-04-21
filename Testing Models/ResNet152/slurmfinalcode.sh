#!/bin/sh

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=23:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=final
#SBATCH --mail-type=END
#SBATCH --mail-user=felipe.castanogonzal@ucalgary.ca
#SBATCH --output=2final%j.out

echo starting slurm

date
id

echo start initialization

source /home/felipe.castanogonzal/software/miniconda3/bin/activate my_env

echo finished initializing

# Execute python code
python 2finalcode.py

echo ending slurm

date
