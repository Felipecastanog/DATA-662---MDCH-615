#!/bin/sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=FINpro
#SBATCH --mail-type=END
#SBATCH --mail-user=carlos.arteagadeanda@ucalgary.ca
#SBATCH --output=output_test_%j.out
#SBATCH --error=error_%j.err

echo "Starting Slurm job"

date 
id

echo "Start initialization"

which python
conda env list

# Initialize conda
conda init bash
source ~/.bashrc

# Activate the conda environment
conda activate 3env

# Install compatible version of libstdc++
conda install -c conda-forge gxx_linux-64==11.1.0 -y

# Add conda's libstdc++ to the library path
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

echo "Finished initializing"

# Add a print statement to confirm the script starts running
echo "Running Python script"

python /home/carlos.arteagadeanda/Training_Xcep.py

date
