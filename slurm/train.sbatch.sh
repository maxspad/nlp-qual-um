#!/bin/bash
#SBATCH --job-name=train-arr
##SBATCH --partition=standard
#SBATCH --partition=spgpu
#SBATCH --time=00-00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G
#SBATCH --array=0-100%5
#SBATCH --output=/home/maxspad/scratch/job_outputs/%A-%a.log

##SBATCH --job-name=train-test
##SBATCH --partition=standard
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=2
##SBATCH --mem=16G

echo "Starting..."

srun python -u src/randomtrial.py \
    --hf_model_family='google-bert' \
    --hf_model_name='bert-base-uncased' \
    --mlflow_tracking_uri='~/scratch/mlruns' \
    --mlflow_experiment_name="rsearch_$date_prefix" \
    --trainer_args.output_dir='~/scratch/hf_outputs' \
    --trainer_args.logging_dir='~/scratch/hf_logging_dir' 

# srun python src/train.py \
#     --hf_model_family='google-bert' \
#     --hf_model_name='bert-base-uncased'
