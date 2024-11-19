#!/bin/bash

# Simple script which launches a randomtrial.py batch job
# It sets some environment variables first which is helpful

# Set up python environment
echo "CWD is $(pwd)"
echo "Loading python..."
module load python/3.11
source venv/bin/activate

# Set a date prefix for the mlflow experiment
date_prefix=$(date +%Y-%m-%d_%M-%H-%S)
echo "date_prefix=$date_prefix"
export MLFLOW_EXPERIMENT_NAME=${date_prefix}_rsearch

# Set mlflow storage location
export MLFLOW_TRACKING_URI="sqlite:///$SCRATCH/mlruns.db"

# Set output and logging dirs
export OUTPUT_DIR="$SCRATCH/experiments/$MLFLOW_EXPERIMENT_NAME"
export TRAINER_ARGS__OUTPUT_DIR="$OUTPUT_DIR/hf_outputs"
export TRAINER_ARGS__LOGGING_DIR="$OUTPUT_DIR/hf_logging_dir"

# Set log levels
export SCRIPT_LOG_LEVEL="DEBUG"
export TRAINER_ARGS__LOG_LEVEL="debug"

# Choose model family
export HF_MODEL_FAMILY='google-bert'
export HF_MODEL_NAME='bert-base-uncased'


#python -c "from src import config; print(config.TrainConfig())"
echo "Launching batch job..."
slurm_output_dir=$OUTPUT_DIR/job_logs
echo "Job outputs will be in $slurm_output_dir"

export JOB_EXTRA_ARGS=$@
echo "Extra args are $JOB_EXTRA_ARGS"

sbatch -v \
    --output=$slurm_output_dir/%A-%a.log \
    --partition=debug \
    --time=00-00:15:00 \
    --cpus-per-task=2 \
    --nodes=1 \
    --ntasks-per-node=1 \
    --mem=16G \
    util/rtrial_exp.sbatch.sh






