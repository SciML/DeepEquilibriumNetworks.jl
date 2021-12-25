#!/bin/bash

# Slurm Sbatch Options
#SBATCH --gres=gpu:volta:2
#SBATCH -n 6 -N 1
#SBATCH --time=72:00:00
#SBATCH -a 1-4
#SBATCH -o slurm_logs/mnist_mdeq-%j-%a.log

cd /home/gridsan/apal/research/FastDEQ.jl

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

# Force precompilation
julia --project=experiments/mnist -e "using Dates, FastDEQ, MLDatasets, MPI, Serialization, Statistics, Plots, Random, Wandb, ParameterSchedulers"

# Run the job
mpiexecjl -np 6 julia --project=experiments/mnist experiments/mnist/mnist_mdeq.jl $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
