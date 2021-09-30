module FastDEQ

using CUDA
using DiffEqSensitivity
using Flux
using LinearAlgebra
using OrdinaryDiffEq
using SteadyStateDiffEq
using UnPack
using Zygote


abstract type AbstractDeepEquilibriumModel end

function construct_iterator end


include("utils.jl")
include("layers.jl")
include("solvers.jl")


export DeepEquilibriumNetwork, SkipDeepEquilibriumNetwork
export BroydenCache, broyden

end
