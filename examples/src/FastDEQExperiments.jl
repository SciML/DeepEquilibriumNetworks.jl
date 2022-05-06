module FastDEQExperiments

using FastDEQ,
    DataLoaders,
    Random,
    OrdinaryDiffEq,
    FluxMPI,
    Format,
    Funtors,
    Lux,
    MLDatasets,
    Optimisers,
    MPI,
    CUDA,
    Setfield,
    ParameterSchedulers,
    NNlib,
    Zygote

import Flux: OneHotArray, onecold, onehotbatch, onehot
import Flux.Losses: logitcrossentropy, mse
import MLUtils: shuffleobs
import MLDataPattern, MLUtils

# Memory Management
relieve_gc_pressure(::Union{Nothing,<:AbstractArray}) = nothing
relieve_gc_pressure(x::CuArray) = CUDA.unsafe_free!(x)
relieve_gc_pressure(t::Tuple) = relieve_gc_pressure.(t)
relieve_gc_pressure(x::NamedTuple) = fmap(relieve_gc_pressure, x)

function invoke_gc()
    GC.gc(true)
    # CUDA.reclaim()
    return nothing
end

# PrettyTableLogger
include("logging.jl")
# get_model_config
include("config.jl")
# train, loss_function
include("train.jl")
# get_model
include("models.jl")
# get_dataloaders
include("dataloaders.jl")


# Fallback since DataLoaders.jl still relies on MLDataPattern
MLDataPattern.nobs(x) = MLUtils.numobs(x)
MLDataPattern.getobs(d::Union{MLUtils.ObsView,MLDatasetsImageData,DistributedDataContainer}, i::Int64) =
    MLUtils.getobs(d, i)


end