module FastDEQExperiments

using FastDEQ, ExplicitFluxLayers, Random, Flux, OrdinaryDiffEq, FluxMPI, Format, MLDatasets, MLDataUtils, DataLoaders, Optimisers
import LearnBase: ObsDim
import MLDataUtils: nobs, getobs

const EFL = ExplicitFluxLayers

# PrettyTableLogger
include("logging.jl")
# train, loss_function
include("train.jl")
# get_model
include("models.jl")
# get_dataloaders
include("dataloaders.jl")

end