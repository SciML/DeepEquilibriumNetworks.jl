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
    CUDA
import LearnBase: ObsDim
import MLDataUtils: nobs, getobs

const EFL = ExplicitFluxLayers

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