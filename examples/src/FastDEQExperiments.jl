module FastDEQExperiments

using FastDEQ,
    ExplicitFluxLayers,
    Random,
    Flux,
    OrdinaryDiffEq,
    FluxMPI,
    Format,
    MLDatasets,
    MLDataUtils,
    DataLoaders,
    Optimisers,
    MPI,
    CUDA,
    Setfield,
    ParameterSchedulers
import LearnBase: ObsDim
import MLDataUtils: nobs, getobs

const EFL = ExplicitFluxLayers

# FIXME: Remove once FastDEQ has been updated to use latest EFL
Base.keys(::Nothing) = ()

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

end