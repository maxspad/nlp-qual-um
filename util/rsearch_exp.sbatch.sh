#!/bin/bash

echo "Job ID: $SLURM_JOBID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

echo "Starting..."
export MLFLOW_RUN_NAME="$SLURM_JOBID-$SLURM_ARRAY_TASK_ID"
echo "Run name: $MLFLOW_RUN_NAME"

python -u src/randomtrial.py $JOB_EXTRA_ARGS
