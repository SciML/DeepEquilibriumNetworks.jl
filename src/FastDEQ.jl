module FastDEQ

using CUDA
using DiffEqSensitivity
using Flux
using LinearAlgebra
using OrdinaryDiffEq
using SteadyStateDiffEq
using UnPack
using Zygote


include("utils.jl")
include("layers.jl")
include("solvers.jl")


export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork
export BroydenCache, broyden

end
