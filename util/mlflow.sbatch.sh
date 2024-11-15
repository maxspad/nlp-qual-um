#!/bin/bash
#SBATCH --job-name=mlflow
#SBATCH --partition=standard
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x.log

#conda activate conda/nlp-qual-um
srun -n 1 mlflow server -h 0.0.0.0


