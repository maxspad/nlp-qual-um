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
##SBATCH --array=0-10%5
#SBATCH --array=0-100%5
#SBATCH --output=/scratch/maxspad_root/maxspad0/maxspad/job_outputs/%A-%a.log

##SBATCH --job-name=train-test
##SBATCH --partition=standard
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=2
##SBATCH --mem=16G

module load python/3.11
source venv/bin/activate

echo "Starting..."
echo "Job ID: $SLURM_JOBID"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"

mlflow_tracking_uri="$SCRATCH/mlruns"
#mlflow_tracking_uri="sqlite:///$SCRATCH/mlruns.db"
echo "mlflow_tracking_uri=$mlflow_tracking_uri"
mlflow_experiment_name="rsearch_$date_prefix"
output_dir="$SCRATCH/hf_outputs"
logging_dir="$SCRATCH/hf_logging_dir"

python -u src/randomtrial.py \
    --hf_model_family='google-bert' \
    --hf_model_name='bert-base-uncased' \
    --trainer_args.log_level=info \
    --script_log_level=DEBUG \
    --mlflow_tracking_uri=$mlflow_tracking_uri \
    --mlflow_experiment_name=$mlflow_experiment_name \
    --trainer_args.output_dir=$output_dir \
    --trainer_args.logging_dir=$logging_dir 

# srun python src/train.py \
#     --hf_model_family='google-bert' \
#     --hf_model_name='bert-base-uncased'
