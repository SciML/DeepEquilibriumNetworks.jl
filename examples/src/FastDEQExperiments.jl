module FastDEQExperiments

using FastDEQ, ExplicitFluxLayers, Random, Flux, OrdinaryDiffEq, FluxMPI, Format, MLDatasets, MLDataUtils, DataLoaders, Optimisers
import LearnBase: ObsDim
import MLDataUtils: nobs, getobs

const EFL = ExplicitFluxLayers

# get_model
include("models.jl")
# PrettyTableLogger
include("logging.jl")
# get_dataloaders
include("dataloaders.jl")

include("train.jl")

end