module FastDEQExperiments

using CUDA
using Dates
using FastBroadcast
using FastDEQ
using FluxMPI
using Format
using Formatting
using Functors
using Lux
using MPI
using NNlib
using OneHotArrays
using Optimisers
using OrdinaryDiffEq
using ParameterSchedulers
using Random
using Setfield
using Statistics
using Zygote

import DataLoaders: LearnBase
import MLDataPattern
import MLUtils

# logging utilities
include("logging.jl")
# get_model_config
include("config.jl")
# get_model
include("models.jl")
# random utilities
include("utils.jl")

# Exports
export AverageMeter, CSVLogger, ProgressMeter, print_meter

export get_experiment_configuration

export construct_optimiser, get_model

export accuracy, invoke_gc, is_distributed, logitcrossentropy, mae, mse, relieve_gc_pressure, should_log, update_lr

end