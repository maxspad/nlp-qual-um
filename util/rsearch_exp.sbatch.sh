#!/bin/bash

echo "Job ID: $SLURM_JOBID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

echo "Starting..."
python -u src/randomtrial.py $JOB_EXTRA_ARGS

# srun python src/train.py \
#     --hf_model_family='google-bert' \
#     --hf_model_name='bert-base-uncased'
