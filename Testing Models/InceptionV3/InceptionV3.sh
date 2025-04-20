#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=8:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=InceptionV3
#SBATCH --mail-type=END
#SBATCH --mail-user=grace.kennedy1@ucalgary.ca
#SBATCH --output=slurmoutput_%j.out

echo starting slurm

date
id

echo start initialization
conda activate DATA622
echo finished initializing

# Execute python code
python InceptionV3.py

echo ending slurm
date
