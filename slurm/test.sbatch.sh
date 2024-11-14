#!/bin/bash
#SBATCH --job-name=test
#SBATCH --partition=spgpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=16G

echo 'Starting'
echo 'Getting nodes...'
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP head: $ip_head"

echo "Starting HEAD at $head_node"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" \
    --port=$port --num-cpus "${SLURM_CPUS_PER_TASK}" \
    --num-gpus "${SLURM_GPUS_PER_NODE:-0}" --block &

sleep 5
worker_num=$((SLURM_JOB_NUM_NODES - 1))
echo "Starting $worker_num workers"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
# 

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" \
            --num-gpus "${SLURM_GPUS_PER_NODE:-0}" \
            --block &
    sleep 5
done

python -u src/stupidtrain.py