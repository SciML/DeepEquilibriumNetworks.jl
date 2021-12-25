#!/bin/bash

# Slurm Sbatch Options
#SBATCH --gres=gpu:volta:2
#SBATCH -n 6 -N 1
#SBATCH --time=144:00:00
#SBATCH -a 1-4
#SBATCH -o slurm_logs/cifar_mdeq-%j-%a.log

cd /home/gridsan/apal/research/FastDEQ.jl

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "Number of Tasks: " $SLURM_ARRAY_TASK_COUNT

# Force precompilation
julia --project=experiments/cifar -e "using Dates, FastDEQ, MLDatasets, MPI, Serialization, Statistics, ParameterSchedulers, Plots, Random, Wandb, ParameterSchedulers, MLDataPattern"

mpiexecjl -np 6 julia --project=experiments/cifar experiments/cifar/cifar10_deq.jl $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
