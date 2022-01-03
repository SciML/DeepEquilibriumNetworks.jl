#!/bin/bash

# Slurm Sbatch Options
#SBATCH --gres=gpu:volta:1
#SBATCH -N 1
#SBATCH --time=144:00:00
#SBATCH -o slurm_logs/mpgnn_deq-%j.log

cd /home/gridsan/apal/research/FastDEQ.jl

echo "SLURM_ARRAY_TASK_ID: " $1
echo "Number of Tasks: " $2

# Force precompilation
julia --project=experiments/mp-gnn experiments/mp-gnn/train.jl $1 $2
