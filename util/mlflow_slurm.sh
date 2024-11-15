
echo "Starting mlflow..."
conda activate nlp-qual-um
jobid=$(sbatch --parsable mlflow.sbatch.sh)

echo "Jobid is $jobid"
squeue -j $jobid -o "%N"