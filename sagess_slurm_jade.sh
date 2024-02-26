#!/bin/bash

#SBATCH --partition="small"
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=24

export CUDA_VISIBLE_DEVICES=$SLURM_JOB_GPUS

echo "IDs of GPUs available: $CUDA_VISIBLE_DEVICES"
 
echo "No of GPUs available: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"

echo "No of CPUs available: $SLURM_CPUS_PER_TASK" 

echo "nproc output: $(nproc)"

nvidia-smi

sleep 10

# Unique Job ID, either the Slurm job ID or Slurm array ID and task ID when an
# array job
if [ "$SLURM_ARRAY_JOB_ID" ]; then
    job_id="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
else
    job_id="$SLURM_JOB_ID"
fi

# Set user ID and name of project
repo="sagess"

dataset="EmailEUCore"
#dataset="Cora"
#dataset="Wiki"
#dataset="Ego_Facebook"
#dataset="SBM"

# print out config file
cat configs/dataset/${dataset}.yaml

# Path to scratch directory on host
scratch_host="/raid/local_scratch"
scratch_root="$scratch_host/${USER}/${job_id}"
# Files and directories to copy to scratch before the job
inputs="."
# File and directories to copy from scratch after the job
outputs="outputs"
# Singularity container
container="./container/${repo}_ws.sif"
# Singularity 'exec' command
container_command="./hpc_run_script.sh"
# Command to execute
run_command="singularity exec
  --nv
  --bind $scratch_root:/scratch_mount
  --pwd /scratch_mount
  --env CUDA_VISIBLE_DEVICES=0
  --env DATASET=${dataset}
  $container
  $container_command"


##########
# Set up scratch
##########

# Copy inputs to scratch
mkdir -p "$scratch_root"
for item in $inputs; do
    echo "Copying $item to scratch_root"
    cp -r "$item" "$scratch_root"
done

##########
# Monitor and run job
##########

# Monitor GPU usage
nvidia-smi dmon -o TD -s um -d 1 > "dmon_$job_id".txt &
gpu_watch_pid=$!

# run the application
start_time=$(date -Is --utc)
$run_command
end_time=$(date -Is --utc)

# Stop GPU monitoring
kill $gpu_watch_pid

# Print summary
echo "executed: $run_command"
echo "started: $start_time"
echo "finished: $end_time"

##########
# Copy outputs
##########

# Copy output from scratch_root
for item in $outputs; do
    echo "Copying $item from scratch_root"
    cp -r "$scratch_root/$item" ./
done

# Clean up scratch_root directory
rm -rf "$scratch_root"
