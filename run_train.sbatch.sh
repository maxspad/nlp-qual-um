#!/bin/bash
#SBATCH --job-name=raytrain
#SBATCH --partition=spgpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gpus=2


srun python src/stupidtrain.py \
	--train_config.dataset_path=/gpfs/accounts/maxspad_root/maxspad0/maxspad/nlp-qual-um/data/processed/hf_dataset/ \
	--run_config.storage_path=$(pwd)/../ray_storage_path \
	--train_config.mlflow_tracking_uri=$(pwd)/../mlruns \
	--train_config.trainer_args.disable_tqdm=True \
	--run_config.verbose=2 \
	--tune_config.num_samples=250 \
	--tune_config.metric=eval_balanced_accuracy \
	--tune_config.mode=max \
	--train_config.hf_model_family=google-bert \
	--train_config.hf_model_name=bert-base-cased \
	--train_config.mlflow_experiment_name=2024-11-13_01-39

